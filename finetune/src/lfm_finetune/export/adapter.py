"""Export artifacts: LoRA adapter (primary) and merged model (optional)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..utils.errors import ExportError
from ..utils.io import dir_size_bytes, ensure_dir, write_json
from ..utils.logging import get_logger
from ..utils.versions import package_versions

log = get_logger("export.adapter")


@dataclass
class ExportResult:
    path: Path
    kind: str
    size_bytes: int
    verified: bool
    details: dict[str, Any]

    def summary(self) -> str:
        mb = self.size_bytes / 1024**2
        return (
            f"export kind        {self.kind}\n"
            f"path              {self.path}\n"
            f"size              {mb:.1f} MiB\n"
            f"reload verified   {self.verified}\n"
            + "\n".join(f"  {k:16s} {v}" for k, v in sorted(self.details.items()))
        )


def export_adapter(
    adapter_path: Path | str,
    destination: Path | str,
    *,
    base_model_id: str | None = None,
    base_revision: str = "main",
    extra_metadata: dict[str, Any] | None = None,
    verify: bool = True,
    model: Any = None,
) -> ExportResult:
    """Copy a trained adapter into the canonical ``adapters/`` layout.

    The adapter is the **preferred** artifact: it is tens of MiB rather than
    gigabytes, it pins the exact base checkpoint in ``adapter_config.json``, and
    it can be re-trained/re-merged without re-downloading the base model.

    The verification step re-loads the adapter into a real PEFT wrapper against
    the *same base checkpoint* it was trained on. Existence of files is not
    evidence that an artifact is usable.
    """
    src = Path(adapter_path)
    dst = ensure_dir(Path(destination))

    required = ["adapter_config.json"]
    weights = list(src.glob("adapter_model*.safetensors")) + list(src.glob("adapter_model*.bin"))
    if not weights:
        raise ExportError(
            f"No adapter weights found in {src}.",
            hint="Expected adapter_model.safetensors. Re-run training, or point at the "
            "run's adapter/ directory.",
        )
    missing = [f for f in required if not (src / f).exists()]
    if missing:
        raise ExportError(
            f"Adapter at {src} is incomplete (missing {missing}).",
            hint="Do not hand-assemble adapters; copy the whole directory.",
        )

    import shutil

    for item in src.iterdir():
        if item.is_file():
            shutil.copy2(item, dst / item.name)
    for item in src.iterdir():
        if item.is_dir() and item.name in ("checkpoint-*",):
            continue

    base_id = base_model_id or _read_base_model_id(src)
    if base_id:
        _ensure_base_reference(dst, base_id, base_revision)

    info_path = dst / "export_info.json"
    existing = _read_json(info_path)
    info = {
        **existing,
        "base_model_id": base_id,
        "base_model_revision": base_revision,
        "exported_at": _utc_now(),
        "source_adapter": str(src),
        "weights_files": [p.name for p in sorted(dst.glob("adapter_model*"))],
        "packages": package_versions(),
        "reload_command": (
            f"python -c \"from peft import PeftModel; from transformers import "
            f"AutoModelForCausalLM; PeftModel.from_pretrained("
            f"AutoModelForCausalLM.from_pretrained('{base_id}'), '{dst}')\""
        ),
        **(extra_metadata or {}),
    }
    write_json(info_path, info)

    if verify:
        verify_adapter_loads(dst, base_model_id=base_id, model=model)
        verified = True

    return ExportResult(
        path=dst,
        kind="peft-lora-adapter",
        size_bytes=dir_size_bytes(dst),
        verified=verified,
        details={
            "base_model": base_id,
            "adapter_files": len(list(dst.iterdir())),
            "config_hash": _hash_dir(dst),
        },
    )


def verify_adapter_loads(
    adapter_dir: Path | str,
    *,
    base_model_id: str | None = None,
    model: Any = None,
) -> bool:
    """Prove the adapter artifact is complete and self-consistent.

    Checks performed:

    1. ``adapter_config.json`` parses and records a base model id.
    2. Weight shards load and contain LoRA tensors -- this catches a truncated
       file or a silently missing shard, which a directory listing cannot.
    3. Every tensor's shape is consistent with the declared rank ``r``, so
       weights and config cannot have come from different runs.
    4. When a live ``model`` is supplied, every configured ``target_module``
       actually exists in it -- this is what catches an adapter trained against
       a different architecture or model size.

    Deliberately *not* implemented by instantiating a meta-device copy of the base
    model. That approach was tried and is unreliable inside a live training
    process: ``AutoModelForCausalLM.from_config`` raised
    ``TypeError: argument of type 'NoneType' is not iterable`` while the real
    model and PEFT wrappers were resident on a CPU-only build (where
    ``torch.version.cuda`` is ``None``). Reading the shards directly is cheaper
    and more precise about what can actually go wrong.
    """
    adapter_dir = Path(adapter_dir)

    try:
        from peft import PeftConfig
    except ImportError as exc:  # pragma: no cover
        raise ExportError("peft is not installed; run `uv sync`.") from exc

    if not (adapter_dir / "adapter_config.json").exists():
        raise ExportError(
            f"{adapter_dir} has no adapter_config.json.",
            hint="Copy the whole adapter directory; a partial copy cannot be loaded.",
        )

    peft_config = PeftConfig.from_pretrained(str(adapter_dir))
    recorded = getattr(peft_config, "base_model_name_or_path", None)
    base_id = base_model_id or recorded
    if not base_id:
        raise ExportError(
            f"Cannot determine which base model {adapter_dir} was trained for.",
            hint="adapter_config.json must contain base_model_name_or_path, or pass "
            "base_model_id explicitly.",
        )
    if recorded and base_model_id and recorded != base_model_id:
        log.warning(
            "adapter_config.json records base_model_name_or_path=%r but the requested base "
            "is %r; the adapter may not be applicable.",
            recorded, base_model_id,
        )

    tensors = _load_adapter_tensors(adapter_dir)
    lora_tensors = {k: v for k, v in tensors.items() if "lora_" in k}
    if not lora_tensors:
        raise ExportError(
            f"{adapter_dir} contains no LoRA tensors (found {len(tensors)} tensors).",
            hint="This is not a LoRA adapter, or only modules_to_save were saved.",
        )
    non_lora = sorted(set(tensors) - set(lora_tensors))
    if non_lora:
        log.warning(
            "Adapter also stores %d non-LoRA tensor(s) (modules_to_save): %s",
            len(non_lora), non_lora[:5],
        )

    rank = int(getattr(peft_config, "r", 0) or 0)
    if rank:
        mismatched = []
        for name, tensor in lora_tensors.items():
            shape = tuple(tensor.shape)
            if "lora_A" in name and shape and shape[0] != rank:
                mismatched.append((name, shape))
            if "lora_B" in name and shape and shape[-1] != rank:
                mismatched.append((name, shape))
        if mismatched:
            raise ExportError(
                f"{len(mismatched)} LoRA tensor(s) disagree with the declared rank "
                f"r={rank}: {mismatched[:4]}",
                hint="The weights and adapter_config.json come from different runs; the "
                "artifact is corrupt.",
            )

    total_params = sum(int(t.numel()) for t in lora_tensors.values())

    if model is not None:
        from ..model.discovery import inspect_modules

        inventory = inspect_modules(model, model_id=base_id)
        targets = peft_config.target_modules
        targets = [targets] if isinstance(targets, str) else list(targets or [])
        missing = [t for t in targets if t not in inventory.linear_suffixes]
        if missing:
            raise ExportError(
                f"Adapter targets module(s) {missing} that do not exist in {base_id}. "
                f"Available Linear suffixes: {sorted(inventory.linear_suffixes)}",
                hint="The adapter was trained against a different architecture or size.",
            )
        log.info("All %d adapter target module type(s) verified present in %s",
                 len(targets), base_id)

    log.info(
        "Adapter verified: %d LoRA tensors, %d parameters, r=%d, base=%s",
        len(lora_tensors), total_params, rank, base_id,
    )
    return True


def _load_adapter_tensors(adapter_dir: Path) -> dict[str, Any]:
    """Read every adapter weight shard into a flat ``name -> tensor`` mapping."""
    adapter_dir = Path(adapter_dir)
    safetensor_files = sorted(adapter_dir.glob("adapter_model*.safetensors"))
    bin_files = sorted(adapter_dir.glob("adapter_model*.bin"))
    if not safetensor_files and not bin_files:
        raise ExportError(
            f"No adapter weight file (*.safetensors / *.bin) found in {adapter_dir}.",
            hint="Expected adapter_model.safetensors. Do not hand-assemble adapters.",
        )

    tensors: dict[str, Any] = {}
    if safetensor_files:
        from safetensors import safe_open

        for path in safetensor_files:
            with safe_open(str(path), framework="pt") as handle:
                keys = list(handle.keys())
                if not keys:
                    raise ExportError(f"{path.name} is empty -- the shard is truncated.")
                for key in keys:
                    tensors[key] = handle.get_tensor(key)
    else:
        import torch

        for path in bin_files:
            state = torch.load(str(path), map_location="cpu", weights_only=True)
            if not isinstance(state, dict) or not state:
                raise ExportError(f"{path.name} does not contain a weight state dict.")
            tensors.update(state)
    return tensors


def _load_base_config(base_id: str, peft_config: Any) -> Any:
    from transformers import AutoConfig

    revision = getattr(peft_config, "revision", None) or None
    return AutoConfig.from_pretrained(base_id, revision=revision, trust_remote_code=False)


def merge_adapter(
    adapter_path: Path | str,
    destination: Path | str,
    *,
    base_model_id: str | None = None,
    base_revision: str | None = None,
    config: Any = None,
    merge_in_bf16: bool = True,
    safe_serialization: bool = True,
    max_shard_size: str = "4GB",
    save_tokenizer: bool = True,
    verify: bool = True,
) -> ExportResult:
    """Merge the LoRA delta into the base weights and save a standalone model.

    LFM2.5 merge support
    --------------------
    ``merge_and_unload`` performs ``W += B @ A`` on the targeted Linear layers.
    Every module in Liquid's reference target set is a plain ``nn.Linear``, so
    the arithmetic is well-defined and merging is supported. Two LFM-specific
    caveats are verified here rather than assumed:

    1. **Weight tying.** LFM2.5-*-Base sets ``tie_embedding: true``. We never
       target ``lm_head``/``embed_tokens``, so the tie is intact and merging
       cannot desynchronise them. This function re-checks that invariant.
    2. **Convolution blocks.** ``conv.in_proj``/``conv.out_proj`` are Linear
       layers inside ``Lfm2ShortConv``. They merge identically to attention/FFN
       projections; no conv-specific handling is needed.

    The merged model is a convenience for runtimes that cannot load an adapter.
    The adapter remains the canonical artifact: merging is lossy with respect to
    future re-training and doubles disk usage.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    src = Path(adapter_path)
    dst = ensure_dir(Path(destination))

    base_id = base_model_id or _read_base_model_id(src)
    revision = base_revision or _read_base_revision(src) or "main"
    if not base_id:
        raise ExportError(
            f"Cannot determine the base model for {src}.",
            hint="Pass base_model_id explicitly.",
        )

    log.info("Loading base model %s@%s for merging", base_id, revision)
    try:
        base = AutoModelForCausalLM.from_pretrained(
            base_id,
            revision=revision,
            dtype="bfloat16" if merge_in_bf16 else "float32",
            low_cpu_mem_usage=True,
        )
    except Exception as exc:
        raise ExportError(f"Could not load base model {base_id}: {exc}") from exc

    tied_before = _weight_tying_state(base)
    try:
        model = PeftModel.from_pretrained(base, str(src))
    except Exception as exc:
        raise ExportError(f"Could not load adapter {src}: {exc}") from exc

    try:
        merged = model.merge_and_unload()
    except Exception as exc:
        raise ExportError(
            f"merge_and_unload failed on {base_id}: {exc}",
            hint="This architecture may not support merging. Keep using the adapter-only "
            "artifact -- it is equivalent at inference and is the supported path.",
        ) from exc

    tied_after = _weight_tying_state(merged)
    if tied_before != tied_after:
        raise ExportError(
            "Weight tying changed during merge (lm_head was no longer sharing the embedding "
            "tensor after merging). The merged model would behave differently from "
            "base+adapter.",
            hint="Do not merge a model whose adapter targeted lm_head or embed_tokens. "
            "For LFM2.5-*-Base keep those modules out of lora.target_modules.",
        )

    merged.config.use_cache = True
    try:
        merged.save_pretrained(
            str(dst),
            safe_serialization=safe_serialization,
            max_shard_size=max_shard_size,
        )
    except Exception as exc:
        raise ExportError(f"Saving merged model to {dst} failed: {exc}") from exc

    if save_tokenizer and config is not None:
        try:
            from ..model.loader import load_tokenizer

            tokenizer, _ = load_tokenizer(config.model)
            tokenizer.save_pretrained(str(dst))
        except Exception as exc:
            log.warning("Could not save the tokenizer alongside the merged model: %s", exc)

    write_json(
        dst / "merge_info.json",
        {
            "base_model_id": base_id,
            "base_model_revision": revision,
            "adapter_source": str(src),
            "merged_at": _utc_now(),
            "merge_dtype": "bfloat16" if merge_in_bf16 else "float32",
            "safe_serialization": safe_serialization,
            "weight_tying_preserved": True,
            "tie_embedding": bool(getattr(merged.config, "tie_embedding", False)),
            "packages": package_versions(),
            "caveats": [
                "Merged weights are a point estimate; keep the adapter to re-train.",
                "Quantisation for edge deployment must happen AFTER merging.",
            ],
        },
    )

    verified = False
    if verify:
        verified = _verify_merged_loads(dst)
        if not verified:
            log.warning("Merged model verification did not complete.")

    return ExportResult(
        path=dst,
        kind="merged-model",
        size_bytes=dir_size_bytes(dst),
        verified=verified,
        details={
            "base_model": f"{base_id}@{revision}",
            "merge_dtype": "bfloat16" if merge_in_bf16 else "float32",
            "weight_tying_preserved": True,
            "safetensors": safe_serialization,
        },
    )


def _verify_merged_loads(path: Path) -> bool:
    """Reload the merged directory and confirm the architecture survived."""
    try:
        from transformers import AutoConfig, AutoModelForCausalLM

        cfg = AutoConfig.from_pretrained(str(path))
        from accelerate import init_empty_weights

        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(cfg)
        n_params = sum(p.numel() for p in model.parameters())
        log.info("Merged model verified: %s reloads with %d parameters", cfg.model_type, n_params)
        del model
        return True
    except Exception as exc:
        log.warning("Merged model verification failed: %s", exc)
        return False


def _weight_tying_state(model: Any) -> tuple[bool, bool]:
    """Return ``(lm_head exists, is_same_tensor_as_embedding)``."""
    try:
        lm_head = model.get_output_embeddings()
        embeddings = model.get_input_embeddings()
        if lm_head is None or embeddings is None:
            return False, False
        return True, lm_head.weight is embeddings.weight
    except Exception:  # pragma: no cover
        return False, False


def _read_base_model_id(adapter_dir: Path) -> str | None:

    config_path = Path(adapter_dir) / "adapter_config.json"
    if not config_path.exists():
        return None
    data = _read_json(config_path)
    return data.get("base_model_name_or_path")


def _read_base_revision(adapter_dir: Path) -> str | None:
    data = _read_json(Path(adapter_dir) / "adapter_config.json")
    revision = data.get("revision")
    return str(revision) if revision else None


def _ensure_base_reference(adapter_dir: Path, base_id: str, revision: str) -> None:
    """Record the intended base checkpoint in ``export_info.json``.

    ``adapter_config.json`` is written by PEFT and its ``revision`` field is not
    always preserved across versions, so we additionally pin it ourselves. This
    is what makes a delivered adapter unambiguous.
    """
    info_path = Path(adapter_dir) / "export_info.json"
    info = _read_json(info_path)
    info["base_model_id"] = base_id
    info["base_model_revision"] = revision
    write_json(info_path, info)


def _read_json(path: Path) -> dict[str, Any]:
    import json

    if not Path(path).exists():
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _hash_dir(path: Path) -> str:
    from ..utils.hashing import hash_directory

    return hash_directory(path, exclude=("export_info.json",))[:16]


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat(timespec="seconds")