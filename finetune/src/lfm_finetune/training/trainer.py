"""SFTTrainer assembly, training loop and checkpoint verification.

Sequence (each step can fail fast with an actionable message):

1. stage 0-4 preflight -- hardware, model, tokenizer, dataset, LoRA audit
2. build the run directory and freeze the resolved config
3. print formatted/tokenised samples with the loss mask visible
4. build TRL ``SFTConfig`` + ``SFTTrainer``
5. train (or resume)
6. save the adapter, verify it reloads, write final metadata
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

from ..config.schema import ExperimentConfig
from ..data.schema import Conversation
from ..model.loader import LoadedModel, load_model
from ..model.lora import assert_effective_batch_size_log, attach_lora
from ..utils.errors import CheckpointError, DatasetError, TrainingError
from ..utils.hardware import assert_vram_sufficient, audit
from ..utils.io import ensure_dir, write_json
from ..utils.logging import get_logger, setup_logging
from ..utils.seeding import set_seed
from .args import build_sft_config, describe_training_plan
from .collator import SFTCollator, TokenizedDataset, TorchDatasetAdapter, tokenize_conversations
from .metadata import RunDirectory, local_metric_summary, resolve_reporting, utc_now
from .tokenization import ChatFormatter, format_example_for_display, verify_template_consistency

log = get_logger("training.trainer")


@dataclass
class TrainingResult:
    """Everything the CLI reports after a run."""

    run_dir: Path
    adapter_path: Path | None
    metrics: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    resumed_from: Path | None = None
    trainable_params: int = 0
    total_params: int = 0
    metadata_path: Path | None = None
    verified: bool = False

    def summary(self) -> str:
        lines = [
            f"run directory       {self.run_dir}",
            f"adapter             {self.adapter_path or '(none)'}",
            f"duration            {self.duration_seconds:.1f}s",
            f"parameters          {self.trainable_params:,} trainable / "
            f"{self.total_params:,} total",
        ]
        for key in ("final_train_loss", "best_eval_loss", "final_eval_loss"):
            if self.metrics.get(key) is not None:
                lines.append(f"{key:19s} {self.metrics[key]:.4f}")
        lines.append(f"metadata            {self.metadata_path}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #

def preflight(
    config: ExperimentConfig,
    *,
    load_base_model: bool = True,
) -> tuple[LoadedModel | None, Any, dict[str, Any]]:
    """Run every cheap fail-fast check. Returns ``(loaded, hardware, report)``.

    Nothing here writes to disk or contacts a training API.
    """
    from ..data.io import detect_format

    hardware = audit()
    log.info("Hardware audit:\n%s", "\n".join("  " + ln for ln in hardware.summary_lines()))

    for warning in hardware.warnings:
        log.warning("HARDWARE: %s", warning)

    # -- dataset present -------------------------------------------------- #
    if not config.dataset.use_processed and config.dataset.path:
        path = Path(config.dataset.path)
        if not path.exists():
            raise DatasetError(
                f"dataset.path does not exist: {path}",
                hint="Run `lfm-finetune prepare --config ...` after placing a .jsonl in "
                "data/raw/, or point dataset.path at your own file.",
            )

    loaded = load_model(config, hardware=hardware, for_training=True) if load_base_model else None

    if loaded is not None:
        # -- context vs sequence length ---------------------------------- #
        ctx = loaded.inventory.max_position_embeddings
        max_seq = config.tokenization.max_seq_length
        if ctx and max_seq > ctx:
            raise TrainingError(
                f"tokenization.max_seq_length={max_seq} exceeds the model context {ctx}."
            )
        trained_ctx = 32768
        if max_seq > trained_ctx:
            log.warning(
                "max_seq_length=%d exceeds the 32,768 tokens LFM2.5 was trained on. "
                "Positions beyond that are out of distribution and will degrade.",
                max_seq,
            )

        # -- VRAM feasibility ------------------------------------------- #
        if config.lora.enabled:
            trainable = _estimate_trainable(loaded, config)
            try:
                assert_vram_sufficient(
                    hardware,
                    param_count=loaded.num_parameters,
                    dtype=loaded.dtype,
                    batch_size=config.training.per_device_train_batch_size,
                    grad_accum=config.training.gradient_accumulation_steps,
                    max_seq_length=config.tokenization.max_seq_length,
                    trainable_params=trainable,
                    gradient_checkpointing=(
                        config.model.gradient_checkpointing
                        if config.training.gradient_checkpointing is None
                        else config.training.gradient_checkpointing
                    ),
                    world_size=1,
                )
            except Exception as exc:
                if config.training.smoke_test:
                    log.warning("Skipping VRAM check during smoke test: %s", exc.message)
                else:
                    raise

    report = {
        "hardware": hardware.to_dict(),
        "model_loaded": loaded is not None,
        "dataset_path": str(config.dataset.path) if config.dataset.path else None,
        "checked_at": utc_now(),
    }
    del detect_format
    return loaded, hardware, report


def _estimate_trainable(loaded: LoadedModel, config: ExperimentConfig) -> int:
    """Predict LoRA trainable count before attaching, for the VRAM estimate."""
    from ..model.discovery import discover_lora_targets

    if not config.lora.enabled:
        return loaded.num_parameters
    try:
        targets, _, diagnostics = discover_lora_targets(
            loaded.model, model_id=loaded.model_id
        )
    except Exception:
        return int(loaded.num_parameters * 0.01)

    by_suffix = {}
    for name, module in loaded.model.named_modules():
        if type(module).__name__ != "Linear":
            continue
        suffix = ".".join(name.split(".")[-2:])
        if suffix in targets:
            by_suffix[name] = (module.in_features, module.out_features)

    total = 0
    r = config.lora.r
    for _, (fan_in, fan_out) in by_suffix.items():
        total += r * fan_in + r * fan_out
    log.debug("Predicted %d LoRA parameters across %d modules", total, len(by_suffix))
    del diagnostics
    return total


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def train(
    config: ExperimentConfig,
    *,
    splits: dict[str, list[Conversation]] | None = None,
    output_dir: Path | None = None,
    resume_from: Path | str | None = None,
    smoke_test: bool = False,
    dry_run: bool = False,
    on_step: Callable[[int, dict[str, Any]], None] | None = None,
) -> TrainingResult:
    """Run (or resume) a LoRA SFT run.

    Args:
        splits: Prepared splits. Loaded from disk when omitted.
        output_dir: Overrides the auto-derived run directory.
        resume_from: Checkpoint dir, or ``"auto"`` for the latest in the run dir.
        smoke_test: Tiny CPU-safe run that exercises the whole code path.
        dry_run: Stop after preflight + sample inspection; train nothing.
        on_step: Callback for streaming metrics (used by the CLI).
    """
    started = time.time()
    smoke_test = smoke_test or config.training.smoke_test

    # ---- stage 0: preflight ------------------------------------------- #
    loaded, hardware, preflight_report = preflight(config, load_base_model=True)
    assert loaded is not None

    # ---- stage 1: dataset --------------------------------------------- #
    splits = splits or _load_splits_from_config(config)
    if not splits.get("train"):
        raise DatasetError(
            "No training examples available.",
            hint="Run `lfm-finetune prepare --config ...` and check "
            "data/validated/report.json for why rows were rejected.",
        )
    if config.training.load_best_model_at_end and not splits.get("validation"):
        raise DatasetError(
            "training.load_best_model_at_end=true but there is no validation split.",
            hint="Set dataset.splitting.validation_ratio > 0 and re-run `prepare`.",
        )

    # ---- stage 2: run directory --------------------------------------- #
    # Resuming continues the SAME run directory, so metrics.jsonl keeps one
    # continuous history and the resolved config that produced the run stays
    # attached to it. A new directory per resume would fragment the record and
    # make `--resume auto` ambiguous.
    resume_target = resume_from or config.training.resume_from_checkpoint
    if output_dir is None and resume_target and resume_target != "auto":
        candidate = Path(resume_target).expanduser()
        if candidate.exists():
            # .../checkpoints/<run>/checkpoint-N -> .../checkpoints/<run>
            inferred = candidate.parent if candidate.name.startswith("checkpoint-") else candidate
            output_dir = inferred
            log.info("Resuming into the original run directory: %s", inferred)

    run_dir = RunDirectory(
        output_dir or config.checkpointing.dir,
        config.run.name,
        exact_dir=output_dir,
    )
    setup_logging(config.logging.level, log_file=run_dir.log_path, force=True)
    log.info("Run directory: %s", run_dir.path)
    run_dir.write_config(config)

    # ---- stage 3: formatter + samples --------------------------------- #
    formatter = ChatFormatter(loaded.tokenizer, config.tokenization)
    formatter.assistant_only_loss = config.training.assistant_only_loss

    consistency = verify_template_consistency(
        loaded.tokenizer, formatter, list(splits["train"][:16])
    )
    if not consistency["ok"]:
        raise TrainingError(
            "Chat-template consistency check failed:\n  "
            + json.dumps(consistency["failures"][:3], indent=2),
            hint="See docs/TRAINING.md#lfm25-base-template-caveat. Fix the template or set "
            "training.assistant_only_loss=false.",
        )
    log.info(
        "Template consistency OK over %d examples (mean loss coverage %.1f%%)",
        consistency["checked"],
        (consistency["mean_loss_coverage"] or 0) * 100,
    )

    train_tokens = tokenize_conversations(
        splits["train"], formatter,
        assistant_only_loss=config.training.assistant_only_loss, split="train",
    )
    eval_tokens = tokenize_conversations(
        splits.get("validation", []), formatter,
        assistant_only_loss=config.training.assistant_only_loss, split="validation",
    ) if splits.get("validation") else TokenizedDataset(examples=[])

    _print_samples(train_tokens, formatter, n=3 if not smoke_test else 2)

    # ---- stage 4: LoRA ------------------------------------------------- #
    model = loaded.model
    if config.lora.enabled:
        model, audit_result, targets, inventory, lora_diagnostics = attach_lora(
            model, config.lora, model_id=loaded.model_id
        )
        lora_info = {
            "config": config.lora.model_dump(mode="json"),
            "audit": audit_result.to_dict(),
            "resolved_target_modules": targets,
            "diagnostics": lora_diagnostics,
        }
    else:
        audit_result = None
        lora_info = {"enabled": False, "audit": None, "resolved_target_modules": []}
        log.warning(
            "lora.enabled=false: this will update ALL model parameters. For LFM2.5 at "
            "1.2B this is a full fine-tune and needs far more VRAM than a LoRA run."
        )

    assert_effective_batch_size_log(
        config.training.per_device_train_batch_size,
        config.training.gradient_accumulation_steps,
        world_size=1,
    )

    if dry_run:
        log.info("dry_run=true: stopping before trainer construction.")
        return TrainingResult(
            run_dir=run_dir.path,
            adapter_path=None,
            duration_seconds=time.time() - started,
            total_params=loaded.num_parameters,
            trainable_params=audit_result.trainable_params if audit_result else loaded.num_parameters,
            metadata_path=run_dir.path,
        )

    # ---- stage 5: trainer --------------------------------------------- #
    import os

    os.environ["LFM_RESOLVED_DEVICE"] = loaded.device
    os.environ["LFM_RESOLVED_DTYPE"] = loaded.dtype

    seed_state = set_seed(
        config.training.seed,
        deterministic_algorithms=config.training.deterministic_algorithms,
    ) if config.reproducibility.save_seed_state else None

    resolved_reports, reporting_extra = resolve_reporting(
        config.training.report_to, run_dir.run_name
    )
    if resolved_reports != config.training.report_to:
        config.training.report_to = resolved_reports

    sft_config = build_sft_config(
        config, run_dir.path, smoke_test=smoke_test, extra=reporting_extra.get("wandb")
    )
    log.info("Training plan:\n%s", describe_training_plan(config, run_dir.path))

    try:
        from trl import SFTTrainer
    except ImportError as exc:
        raise TrainingError(
            "trl is not installed.",
            hint="Run `uv sync` inside finetune/.",
        ) from exc

    collator = SFTCollator(pad_token_id=formatter.pad_token_id)
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=TorchDatasetAdapter(train_tokens.examples),
        eval_dataset=TorchDatasetAdapter(eval_tokens.examples) if eval_tokens.examples else None,
        data_collator=collator,
        processing_class=loaded.tokenizer,
        peft_config=None,  # already attached and audited
    )

    # ---- metadata ------------------------------------------------------ #
    dataset_info = {
        "source": str(config.dataset.path) if config.dataset.path else None,
        "format": config.dataset.format,
        "use_processed": config.dataset.use_processed,
        "num_train": len(splits.get("train", [])),
        "num_validation": len(splits.get("validation", [])),
        "num_test": len(splits.get("test", [])),
        "tokenization": {
            "train": train_tokens.to_manifest(),
            "validation": eval_tokens.to_manifest(),
        },
        "manifest": getattr(config, "_dataset_manifest", None),
    }
    metadata = run_dir.build_metadata(
        config=config,
        seed_state=seed_state,
        model_info={
            "id": loaded.model_id,
            "revision": loaded.revision,
            "architecture": loaded.inventory.architecture,
            "model_type": loaded.inventory.model_type,
            "parameters": loaded.num_parameters,
            "dtype": loaded.dtype,
            "device": loaded.device,
            "attn_implementation": loaded.attn_implementation,
            "context_length": loaded.inventory.max_position_embeddings,
            "vocab_size": loaded.inventory.vocab_size,
            "tie_embedding": loaded.inventory.tie_embedding,
        },
        dataset_info=dataset_info,
        lora_info=lora_info,
        tokenization_info={
            "chat_template_source": loaded.chat_template_source,
            "template_fingerprint": consistency.get("checked") and train_tokens.tokenizer_fingerprint,
            "append_eos_token": config.tokenization.append_eos_token,
            "eos_token_id": formatter.eos_token_id,
            "bos_token_id": loaded.tokenizer.bos_token_id,
            "pad_token_id": formatter.pad_token_id,
            "max_seq_length": config.tokenization.max_seq_length,
            "truncation_strategy": config.tokenization.truncation_strategy,
            "loss_masking": "assistant-only (prefix-length)" if config.training.assistant_only_loss
                            else "full-sequence (BOS masked)",
        },
        hardware=hardware,
        extra={"preflight": preflight_report},
    )
    run_dir.write_metadata(metadata)

    # ---- stage 6: train ------------------------------------------------ #
    checkpoint_arg = _resolve_resume(config, run_dir, resume_from)
    if checkpoint_arg:
        log.info("Resuming from checkpoint: %s", checkpoint_arg)

    # Always stream metrics to metrics.jsonl. The callback is registered even when
    # no CLI sink is supplied: local logging is the pipeline's baseline
    # observability guarantee and must not depend on how `train()` was called.
    # Only the console rendering is conditional.
    if on_step is not None:
        _drop_printer_callback(trainer)
    trainer.add_callback(_StreamingCallback(run_dir, on_step))

    log.info("Starting training...")
    result = trainer.train(resume_from_checkpoint=checkpoint_arg)

    metrics = dict(result.metrics) if hasattr(result, "metrics") else {}
    streamed = local_metric_summary(run_dir.read_metrics())
    for key, value in {**streamed, **metrics}.items():
        if value is not None:
            run_dir.append_metric({"summary": key, "value": value})

    # ---- stage 7: adapter export -------------------------------------- #
    adapter_path: Path | None = None
    if config.lora.enabled:
        adapter_path = _save_adapter(trainer, run_dir, loaded, config)
        verified = _verify_adapter_reload(adapter_path, loaded, config)
    else:
        save_dir = ensure_dir(run_dir.path / "full_model")
        trainer.save_model(str(save_dir))
        loaded.tokenizer.save_pretrained(str(save_dir))
        adapter_path = save_dir
        verified = (save_dir / "config.json").exists()
        log.info("Full fine-tune: model saved to %s", save_dir)

    run_dir.update_metadata(
        completed_at=utc_now(),
        duration_seconds=round(time.time() - started, 2),
        final_metrics={**streamed, **metrics},
        adapter_path=str(adapter_path) if adapter_path else None,
        adapter_verified=verified,
        resumed_from=str(checkpoint_arg) if checkpoint_arg else None,
        world_size=1,
    )

    return TrainingResult(
        run_dir=run_dir.path,
        adapter_path=adapter_path,
        metrics={**streamed, **metrics},
        duration_seconds=time.time() - started,
        resumed_from=Path(checkpoint_arg) if checkpoint_arg else None,
        trainable_params=audit_result.trainable_params if audit_result else loaded.num_parameters,
        total_params=audit_result.total_params if audit_result else loaded.num_parameters,
        metadata_path=run_dir.metadata_path,
        verified=verified,
    )


def _drop_printer_callback(trainer: Any) -> None:
    """Remove HF's ``PrinterCallback`` so metrics are printed exactly once.

    ``Trainer`` installs ``PrinterCallback`` whenever ``report_to`` is empty. It
    dumps a raw dict of stringified metrics to stdout for every ``on_log``. Since
    this pipeline already streams structured metrics through
    :class:`_StreamingCallback` *and* writes them to ``metrics.jsonl`` and
    ``train.log``, keeping it means every metric line appears twice, one of them
    unreadable. Removing it also means the full-precision values are the ones
    persisted instead of the ``%.4g``-truncated display copy.
    """
    from transformers import PrinterCallback

    handler = getattr(trainer, "callback_handler", None)
    callbacks = list(getattr(handler, "callbacks", []) or [])
    printer = [c for c in callbacks if isinstance(c, PrinterCallback)]
    for callback in printer:
        handler.remove_callback(callback)
    if printer:
        log.debug("Removed %d PrinterCallback instance(s) (metrics rendered by the pipeline)",
                  len(printer))


class _StreamingCallback(TrainerCallback):
    """Streams trainer log records into the run's local ``metrics.jsonl``.

    Must subclass ``transformers.TrainerCallback``: the callback handler
    dispatches ``on_train_begin``/``on_step``/``on_log``/``on_save`` on every
    registered callback, so a duck-typed object without those methods raises
    ``AttributeError`` on the first event.

    Local JSONL logging is unconditional, so no external service is ever
    required to observe a run.
    """

    def __init__(
        self,
        run_dir: RunDirectory,
        sink: Callable[[int, dict[str, Any]], None] | None = None,
    ) -> None:
        self.run_dir = run_dir
        self.sink = sink

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if not logs:
            return
        record = {"step": int(getattr(state, "global_step", 0)), **logs}
        self.run_dir.append_metric(record)
        if self.sink is not None:
            self.sink(record["step"], logs)


def _load_splits_from_config(config: ExperimentConfig) -> dict[str, list[Conversation]]:
    from ..data.pipeline import load_splits

    directory = (
        Path(config.dataset.processed_dir) if config.dataset.processed_dir
        else Path("data") / "splits"
    )
    if not directory.exists():
        raise DatasetError(
            f"Prepared splits not found at {directory}.",
            hint="Run `lfm-finetune prepare --config ...` first, or set "
            "dataset.use_processed=true with dataset.processed_dir.",
        )
    splits = load_splits(directory)
    if not splits:
        raise DatasetError(f"No split files ({'train,validation,test'}.jsonl) found in {directory}")
    return splits


def _print_samples(tokens: TokenizedDataset, formatter: ChatFormatter, *, n: int = 3) -> None:
    """Show fully formatted examples with the supervised tokens marked."""
    if not tokens.examples:
        return
    log.info("=== %d formatted training examples (loss mask shown) ===", min(n, len(tokens.examples)))
    for example in tokens.examples[:n]:
        log.info("\n%s", format_example_for_display(example, formatter.tokenizer))
    log.info(
        "Across %d examples: mean length %.1f tokens, mean supervised coverage %.1f%%",
        len(tokens.examples), tokens.mean_length, tokens.mean_loss_coverage * 100,
    )


def _resolve_resume(
    config: ExperimentConfig, run_dir: RunDirectory, resume_from: Path | str | None
) -> str | None:
    candidate = resume_from or config.training.resume_from_checkpoint
    if not candidate:
        return None
    if candidate == "auto":
        latest = run_dir.latest_checkpoint()
        if latest is None:
            log.warning("resume_from_checkpoint='auto' but no checkpoint-* found; starting fresh.")
            return None
        return str(latest)
    path = Path(candidate)
    if not path.exists():
        raise CheckpointError(
            f"resume_from_checkpoint does not exist: {path}",
            hint="List available checkpoints with `ls <run_dir>`; use 'auto' to pick the "
            "latest one in the current run directory.",
        )
    required = ["trainer_state.json"]
    missing = [f for f in required if not (path / f).exists()]
    if missing:
        raise CheckpointError(
            f"Checkpoint {path} is incomplete (missing {missing}).",
            hint="A resumable checkpoint needs trainer_state.json, optimizer.pt, "
            "scheduler.pt, rng_state.pth and the adapter weights.",
        )
    return str(path)


def _save_adapter(
    trainer: Any, run_dir: RunDirectory, loaded: LoadedModel, config: ExperimentConfig
) -> Path:
    """Save the LoRA adapter (lightweight, preferred artifact)."""
    adapter_dir = ensure_dir(run_dir.adapter_dir)
    try:
        trainer.save_model(str(adapter_dir))
    except Exception as exc:
        raise CheckpointError(
            f"Saving the LoRA adapter failed: {exc}",
            hint="Check that lora.enabled=true and that the run directory is writable.",
        ) from exc

    if config.checkpointing.save_tokenizer:
        loaded.tokenizer.save_pretrained(str(adapter_dir))
    write_json(
        adapter_dir / "export_info.json",
        {
            "base_model_id": loaded.model_id,
            "base_model_revision": loaded.revision,
            "lora": config.lora.model_dump(mode="json"),
            "chat_template_source": loaded.chat_template_source,
            "tokenizer_fingerprint": None,
            "created_at": utc_now(),
            "run_name": run_dir.run_name,
            "note": "Reload with PeftModel.from_pretrained(base_model, adapter_config_dir). "
                    "The tokenizer is included so inference needs no network access.",
        },
    )
    from ..utils.io import dir_size_bytes

    size_mb = dir_size_bytes(adapter_dir) / 1024**2
    log.info("Saved LoRA adapter to %s (%.1f MiB)", adapter_dir, size_mb)
    return adapter_dir


def _verify_adapter_reload(
    adapter_dir: Path, loaded: LoadedModel, config: ExperimentConfig
) -> bool:
    """Prove the saved adapter is complete and loadable.

    A ``.safetensors`` file existing is not evidence that an adapter loads. This
    re-injects the saved adapter config into a **meta-device** skeleton of the
    base architecture and asserts that LoRA parameters actually appear.

    Two things this deliberately avoids:

    * Calling ``PeftModel.from_pretrained`` on ``loaded.model``, which is already
      a ``PeftModel`` -- that double-wraps it and warns
      "Already found a peft_config attribute ... multiple adapters".
    * Materialising a second copy of the base weights, which would double peak
      memory at exactly the moment a large run is most likely to be near its
      memory ceiling.
    """
    from ..export.adapter import verify_adapter_loads

    weights_file = next(iter(sorted(adapter_dir.glob("adapter_model*.safetensors"))), None)
    weights_file = weights_file or next(iter(sorted(adapter_dir.glob("adapter_model*.bin"))), None)
    if weights_file is None:
        raise CheckpointError(
            f"No adapter weight file in {adapter_dir}.",
            hint="Expected adapter_model.safetensors. Re-run the training step that saves "
            "the adapter, or set checkpointing.save_adapter_only=false.",
        )

    config_file = adapter_dir / "adapter_config.json"
    if not config_file.exists():
        raise CheckpointError(
            f"{adapter_dir} is missing adapter_config.json, so the adapter cannot be "
            f"reloaded by PEFT.",
            hint="Copy the entire adapter directory rather than individual files.",
        )

    try:
        verify_adapter_loads(
            adapter_dir,
            base_model_id=loaded.model_id,
            model=getattr(loaded.model, "get_base_model", lambda: loaded.model)(),
        )
    except Exception as exc:
        raise CheckpointError(
            f"Saved adapter at {adapter_dir} failed verification against "
            f"{loaded.model_id}: {exc}",
            hint="Delete the adapter directory and re-run training. If the base model id "
            "or revision differs from the one in adapter_config.json, that is the cause.",
        ) from exc
    log.info("Adapter reload verified from %s", weights_file.name)
    return True


def latest_checkpoint_for(run_dir: Path) -> Path | None:
    """Public helper: newest resumable checkpoint inside a run directory."""
    dirs = sorted(
        run_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.rsplit("-", 1)[-1]) if p.name.rsplit("-", 1)[-1].isdigit() else 0,
    )
    return dirs[-1] if dirs else None