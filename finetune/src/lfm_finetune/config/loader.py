"""YAML configuration loading: inheritance, env overrides, validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..utils.errors import ConfigError
from ..utils.io import read_yaml
from ..utils.logging import get_logger
from .schema import ExperimentConfig, deep_merge, env_overrides, parse_config_obj

log = get_logger("config.loader")

#: Key used in a child config to pull in one or more parent configs.
INHERIT_KEY = "extends"

#: Search order when a config path is relative and not found in CWD.
DEFAULT_CONFIG_DIRS = ("configs", "finetune/configs")


def _resolve(path: str | Path) -> Path:
    """Resolve a config path, also searching the conventional config dirs."""
    p = Path(path)
    if p.exists():
        return p.resolve()
    if not p.is_absolute():
        for root in (Path.cwd(), *Path.cwd().parents):
            for sub in DEFAULT_CONFIG_DIRS:
                candidate = root / sub / p
                if candidate.exists():
                    return candidate.resolve()
    raise ConfigError(
        f"Configuration file not found: {path}",
        hint="Pass a path relative to the project root, e.g. "
        "--config configs/agriculture/lfm25_1.2b.yaml",
    )


def load_raw_with_inheritance(
    path: Path, *, _seen: set[Path] | None = None, _depth: int = 0
) -> tuple[dict[str, Any], list[Path]]:
    """Load a YAML config, recursively resolving ``extends`` chains.

    The merge is deep: a child may override a single nested key without
    restating its siblings. Lists are replaced, not concatenated (see
    :func:`deep_merge` for why).

    Returns:
        ``(merged_mapping, [resolved_paths_in_application_order])``
    """
    seen = _seen if _seen is not None else set()
    if path in seen:
        chain = " -> ".join(str(p) for p in [*seen, path])
        raise ConfigError(f"Circular `extends` chain detected: {chain}")
    if _depth > 32:
        raise ConfigError(
            f"`extends` chain deeper than 32 levels at {path}",
            hint="Your config inheritance is almost certainly wrong.",
        )
    seen.add(path)

    raw = read_yaml(path)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(
            f"Configuration root must be a mapping, got {type(raw).__name__} in {path}"
        )

    parents = raw.pop(INHERIT_KEY, None)
    if parents is None:
        return raw, [path]

    if isinstance(parents, (str, Path)):
        parents = [parents]
    if not isinstance(parents, list):
        raise ConfigError(
            f"`{INHERIT_KEY}` must be a string or list of strings, got "
            f"{type(parents).__name__} in {path}"
        )

    merged: dict[str, Any] = {}
    applied: list[Path] = []
    for parent_ref in parents:
        parent_path = _resolve_relative(parent_ref, path)
        parent_cfg, parent_chain = load_raw_with_inheritance(parent_path, _seen=seen, _depth=_depth + 1)
        merged = deep_merge(merged, parent_cfg)
        applied.extend(parent_chain)

    merged = deep_merge(merged, raw)
    applied.append(path)
    return merged, applied


def _resolve_relative(ref: str | Path, child: Path) -> Path:
    """Resolve an ``extends`` reference relative to the child, then to CWD/configs."""
    ref_path = Path(ref)
    candidates = [ref_path] if ref_path.is_absolute() else [child.parent / ref_path]
    if not ref_path.is_absolute():
        candidates += [Path.cwd() / ref_path, *[d / ref_path for d in DEFAULT_CONFIG_DIRS]]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise ConfigError(
        f"Config {child} extends '{ref}', which does not exist.",
        hint=f"Looked in: {', '.join(str(c) for c in candidates)}",
    )


def load_config(
    path: str | Path,
    *,
    apply_env_overrides: bool = True,
    overrides: dict[str, Any] | None = None,
    validate: bool = True,
) -> tuple[ExperimentConfig, dict[str, Any]]:
    """Load, merge, override and validate a configuration.

    Args:
        path: Path to the top-level experiment YAML.
        apply_env_overrides: Merge ``LFM_SECTION__KEY`` environment variables.
        overrides: Explicit programmatic overrides (CLI ``--set``), highest priority.
        validate: Run full pydantic + cross-field validation.

    Returns:
        ``(config, provenance)`` where provenance records the resolved file chain,
        the raw merged mapping and the applied overrides, for the run manifest.
    """
    resolved = _resolve(path)
    merged, chain = load_raw_with_inheritance(resolved)

    applied_overrides: dict[str, Any] = {}
    if apply_env_overrides:
        env = env_overrides()
        if env:
            merged = deep_merge(merged, env)
            applied_overrides["env"] = env
    if overrides:
        merged = deep_merge(merged, _clean_overrides(overrides))
        applied_overrides["cli"] = overrides

    config = (
        parse_config_obj(merged, source=str(resolved))
        if validate
        else ExperimentConfig.model_construct(**merged)  # type: ignore[arg-type]
    )

    provenance = {
        "config_path": str(resolved),
        "extends_chain": [str(p) for p in chain],
        "raw_resolved": merged,
        "overrides": applied_overrides or None,
        "fingerprint": config.fingerprint(),
    }
    log.info(
        "Loaded config %s (extends: %d parent file(s), fingerprint %s)",
        resolved.name,
        len(chain) - 1,
        provenance["fingerprint"][:12],
    )
    return config, provenance


def _clean_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    """Normalise ``--set a.b=1`` style overrides into a nested dict."""

    def _cast(value: str) -> Any:
        from .schema import _coerce_env

        return _coerce_env(value)

    out: dict[str, Any] = {}
    for dotted, value in overrides.items():
        parts = dotted.split(".")
        cursor = out
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = _cast(value) if isinstance(value, str) else value
    return out


def parse_set_args(items: list[str] | None) -> dict[str, Any]:
    """Turn ``["training.learning_rate=1e-4", "lora.r=32"]`` into a nested dict."""
    if not items:
        return {}
    parsed: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ConfigError(
                f"Malformed --set '{item}'; expected key.path=value",
                hint="Example: --set training.learning_rate=1e-4 --set lora.r=32",
            )
        key, _, value = item.partition("=")
        parsed[key.strip()] = value.strip()
    return _clean_overrides(parsed)


def save_resolved(config: ExperimentConfig, destination: Path) -> Path:
    """Freeze the fully-resolved config so a run never depends on mutable YAML."""
    from ..utils.io import write_yaml

    return write_yaml(destination, config.to_dict())


def describe(config: ExperimentConfig) -> str:
    """Human-readable one-block summary for CLI banners."""
    d = config.to_dict()
    t = config.training
    eff = t.per_device_train_batch_size * t.gradient_accumulation_steps
    lines = [
        f"run               {config.run.name}  (domain: {config.run.domain})",
        f"model             {d['model']['name']}@{d['model']['revision']}  "
        f"dtype={d['model']['dtype']} device={d['model']['device']}",
        f"dataset           {d['dataset']['path']}  format={d['dataset']['format']}",
        f"splits            train={d['dataset']['train_split']} "
        f"val={d['dataset']['validation_split']} test={d['dataset']['test_split']}",
        f"max_seq_length    {config.tokenization.max_seq_length}",
        f"lora              r={config.lora.r} alpha={config.lora.alpha} "
        f"dropout={config.lora.dropout} bias={config.lora.bias} "
        f"target_modules={config.lora.target_modules if isinstance(config.lora.target_modules, str) else 'explicit'}",
        f"optimizer         lr={t.learning_rate} wd={t.weight_decay} "
        f"scheduler={t.lr_scheduler_type} warmup={t.warmup_ratio} clip={t.max_grad_norm}",
        f"batching          micro={t.per_device_train_batch_size} "
        f"accum={t.gradient_accumulation_steps} -> effective={eff} (per device)",
        f"schedule          epochs={t.num_train_epochs} max_steps={t.max_steps} "
        f"logging={t.logging_steps} eval={t.eval_strategy}/{t.eval_steps} "
        f"save={t.save_strategy}/{t.save_steps}",
        f"loss              assistant_only={t.assistant_only_loss}",
        f"seed              train={t.seed} split={config.dataset.splitting.seed}",
    ]
    return "\n".join(lines)