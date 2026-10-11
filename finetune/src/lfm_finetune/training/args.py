"""Translation from the typed experiment config into a TRL ``SFTConfig``.

Kept separate from :mod:`lfm_finetune.training.trainer` so that the mapping from
"what the operator wrote in YAML" to "what TRL receives" can be unit-tested
without loading a model.
"""

from __future__ import annotations

import os
from typing import Any

from ..config.schema import ExperimentConfig
from ..utils.errors import TrainingError
from ..utils.logging import get_logger

log = get_logger("training.args")


def _torch_dtype(dtype: str) -> Any:
    import torch

    name = {"bfloat16": "bfloat16", "bf16": "bfloat16",
            "float16": "float16", "fp16": "float16",
            "float32": "float32", "fp32": "float32"}.get(dtype.lower(), dtype.lower())
    return getattr(torch, name, None)


def build_sft_config(
    config: ExperimentConfig,
    output_dir: Any,
    *,
    world_size: int = 1,
    smoke_test: bool = False,
    extra: dict[str, Any] | None = None,
) -> Any:
    """Construct ``trl.SFTConfig`` for this experiment.

    Notes on non-obvious mappings:

    * ``bf16``/``fp16`` are set from the *resolved* device dtype, not from
      ``model.dtype='auto'``, so a CPU run never claims mixed precision.
    * ``max_length`` is passed through ``dataset_kwargs`` in TRL 1.9 because we
      pre-tokenise the data ourselves and pass a collator; TRL must not
      re-prepare it.
    * ``packing`` stays off by default: packing concatenates examples into one
      window, which erases per-example boundaries and lets a single optimiser
      step span two unrelated agricultural domains.
    """
    from trl import SFTConfig

    training = config.training
    model = config.model

    if smoke_test:
        return _smoke_sft_config(config, output_dir, extra=extra)

    device = os.environ.get("LFM_RESOLVED_DEVICE", "").lower()
    resolved_dtype = os.environ.get("LFM_RESOLVED_DTYPE", "").lower() or model.dtype.lower()
    bf16 = resolved_dtype in ("bfloat16", "bf16")
    fp16 = resolved_dtype in ("float16", "fp16")

    grad_accum = max(1, training.gradient_accumulation_steps // max(1, world_size)) \
        if world_size > 1 else training.gradient_accumulation_steps
    effective = training.per_device_train_batch_size * grad_accum * world_size
    log.info(
        "Batch plan: per_device=%d grad_accum=%d world_size=%d -> effective batch %d",
        training.per_device_train_batch_size, grad_accum, world_size, effective,
    )

    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": training.num_train_epochs,
        "max_steps": training.max_steps if training.max_steps > 0 else -1,
        "per_device_train_batch_size": training.per_device_train_batch_size,
        "per_device_eval_batch_size": training.per_device_eval_batch_size
        or training.per_device_train_batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": training.learning_rate,
        "weight_decay": training.weight_decay,
        "warmup_ratio": training.warmup_ratio,
        "lr_scheduler_type": training.lr_scheduler_type,
        "max_grad_norm": training.max_grad_norm,
        "logging_steps": training.logging_steps,
        "logging_strategy": "steps",
        "eval_strategy": training.eval_strategy,
        "save_strategy": training.save_strategy,
        "bf16": bf16,
        "fp16": fp16,
        "optim": training.optim,
        "adam_beta1": training.adam_beta1,
        "adam_beta2": training.adam_beta2,
        "adam_epsilon": training.adam_epsilon,
        "dataloader_num_workers": training.dataloader_num_workers,
        "dataloader_pin_memory": training.dataloader_pin_memory
        and not device.startswith("cpu"),
        "remove_unused_columns": training.remove_unused_columns,
        "report_to": training.report_to or [],
        "seed": training.seed,
        "data_seed": training.seed,
        "gradient_checkpointing": (
            model.gradient_checkpointing
            if training.gradient_checkpointing is None
            else training.gradient_checkpointing
        ),
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "save_total_limit": training.save_total_limit,
        "hub_strategy": "every_save" if config.export.push_to_hub else "no",
    }

    if training.eval_steps:
        kwargs["eval_steps"] = training.eval_steps
    if training.save_steps:
        kwargs["save_steps"] = training.save_steps
    if training.load_best_model_at_end:
        kwargs["load_best_model_at_end"] = True
        kwargs["metric_for_best_model"] = training.metric_for_best_model
        kwargs["greater_is_better"] = training.greater_is_better
    if training.early_stopping_patience:
        kwargs["early_stopping_patience"] = training.early_stopping_patience
    if training.lr_scheduler_kwargs:
        kwargs["lr_scheduler_kwargs"] = dict(training.lr_scheduler_kwargs)
    if config.run.tags:
        kwargs["run_name"] = training.run_name or f"{config.run.name}-" + "-".join(config.run.tags)
    elif training.run_name:
        kwargs["run_name"] = training.run_name

    # We tokenise externally and pass a collator, so TRL must not re-prepare.
    kwargs["dataset_kwargs"] = {"skip_prepare_dataset": True}
    kwargs["packing"] = bool(training.packing)
    kwargs["max_length"] = config.tokenization.max_seq_length

    if extra:
        kwargs.update(extra)

    try:
        return SFTConfig(**kwargs)
    except TypeError as exc:
        raise TrainingError(
            f"TRL's SFTConfig rejected the generated arguments: {exc}\n"
            f"  generated keys: {sorted(kwargs)}",
            hint="This usually means the installed trl version differs from the pinned "
            "1.9.0. Check with `uv run python -c 'import trl; print(trl.__version__)'`.",
        ) from exc


def _smoke_sft_config(
    config: ExperimentConfig, output_dir: Any, *, extra: dict[str, Any] | None = None
) -> Any:
    """Minimal CPU-safe config: tiny batches, no checkpoint rotation, no eval."""
    from trl import SFTConfig

    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "max_steps": 3,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": config.training.learning_rate,
        "logging_steps": 1,
        "eval_strategy": "no",
        # A checkpoint MUST be written: the smoke test exists to prove that
        # save -> resume -> reload works, and save_strategy="no" would leave that
        # path untested while still reporting success.
        "save_strategy": "steps",
        "save_steps": 2,
        "save_total_limit": 1,
        "bf16": False,
        "fp16": False,
        "optim": "adamw_torch",
        "gradient_checkpointing": False,
        "report_to": [],
        "seed": config.training.seed,
        "data_seed": config.training.seed,
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": False,
        "remove_unused_columns": False,
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "packing": False,
        "max_length": min(config.tokenization.max_seq_length, 256),
    }
    if extra:
        kwargs.update(extra)
    return SFTConfig(**kwargs)


def describe_training_plan(
    config: ExperimentConfig, output_dir: Any, *, world_size: int = 1
) -> str:
    """Operator-facing training plan printed before training starts."""
    t = config.training
    steps_per_epoch = "?"
    eff = t.per_device_train_batch_size * t.gradient_accumulation_steps * world_size
    max_steps = t.max_steps if t.max_steps > 0 else f"{t.num_train_epochs} epochs"
    return "\n".join(
        [
            f"output directory    {output_dir}",
            "trainer             trl.SFTTrainer (LoRA/PEFT)",
            f"objective           {'assistant-only' if t.assistant_only_loss else 'full-sequence'} causal LM",
            f"epochs / max steps  {t.num_train_epochs} / {max_steps}"
            + (f"  (~{steps_per_epoch} steps/epoch)" if steps_per_epoch != "?" else ""),
            f"per-device batch    {t.per_device_train_batch_size} x {t.gradient_accumulation_steps} "
            f"accum x {world_size} device(s) = {eff} effective",
            f"learning rate       {t.learning_rate} ({t.lr_scheduler_type}, "
            f"warmup {t.warmup_ratio:.1%}, decay {t.weight_decay})",
            f"grad clip           {t.max_grad_norm}",
            f"optimiser           {t.optim}",
            f"eval                {t.eval_strategy} every {t.eval_steps} steps"
            + (f", best model: {t.metric_for_best_model}" if t.load_best_model_at_end else ""),
            f"checkpoint          {t.save_strategy} every {t.save_steps} steps, "
            f"keep {t.save_total_limit}",
            f"early stopping      patience {t.early_stopping_patience or 'disabled'}",
            f"max sequence length {config.tokenization.max_seq_length} tokens",
        ]
    )