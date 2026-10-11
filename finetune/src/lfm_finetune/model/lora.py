"""LoRA/PEFT configuration, attachment and trainable-parameter auditing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config.schema import LoraConfigModel
from ..utils.errors import LoRAError
from ..utils.logging import get_logger
from .discovery import (
    EXCLUDED_BY_DEFAULT,
    ModuleInventory,
    describe_lora_plan,
    discover_lora_targets,
)

log = get_logger("model.lora")


@dataclass
class ParameterAudit:
    """Result of the post-attachment trainable-parameter audit."""

    total_params: int
    trainable_params: int
    frozen_params: int
    trainable_pct: float
    expected_min: int
    max_allowed_pct: float
    target_modules: list[str]
    target_module_types: list[str]
    lora_a_params: int
    lora_b_params: int
    other_trainable: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (
            self.trainable_params >= self.expected_min
            and self.trainable_pct <= self.max_allowed_pct
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_params": self.total_params,
            "trainable_params": self.trainable_params,
            "frozen_params": self.frozen_params,
            "trainable_pct": round(self.trainable_pct, 4),
            "expected_min_trainable_params": self.expected_min,
            "max_allowed_trainable_pct": self.max_allowed_pct,
            "lora_A_params": self.lora_a_params,
            "lora_B_params": self.lora_b_params,
            "target_modules": self.target_modules,
            "target_module_types": self.target_module_types,
            "other_trainable": self.other_trainable,
            "warnings": self.warnings,
            "passed": self.ok,
        }

    def summary(self) -> str:
        return (
            f"total parameters      {self.total_params:,}\n"
            f"trainable parameters  {self.trainable_params:,} "
            f"({self.trainable_pct:.3f}%)\n"
            f"frozen parameters     {self.frozen_params:,}\n"
            f"  LoRA A matrices     {self.lora_a_params:,}\n"
            f"  LoRA B matrices     {self.lora_b_params:,}\n"
            f"target module types   {len(self.target_module_types)}\n"
            f"target module count   {len(self.target_modules)}"
        )


def resolve_target_modules(
    lora_cfg: LoraConfigModel,
    model: Any,
    *,
    model_id: str = "<unknown>",
) -> tuple[list[str], ModuleInventory, dict[str, Any]]:
    """Decide the final target module list.

    ``target_modules: auto`` -> discovered from the model graph.
    ``target_modules: [...]`` -> used verbatim, but **verified** to exist.
    """
    if isinstance(lora_cfg.target_modules, str) and lora_cfg.target_modules == "auto":
        return discover_lora_targets(
            model,
            model_id=model_id,
            exclude=EXCLUDED_BY_DEFAULT,
        )

    explicit = list(lora_cfg.target_modules or [])
    if not explicit:
        raise LoRAError(
            "lora.target_modules is an empty list.",
            hint="Set lora.target_modules=auto to discover the modules from the model.",
        )

    from .discovery import inspect_modules

    inventory = inspect_modules(model, model_id=model_id)
    present_suffixes = set(inventory.linear_suffixes)

    missing = [m for m in explicit if m not in present_suffixes]
    if missing:
        raise LoRAError(
            f"lora.target_modules names modules that do not exist in {model_id}: {missing}.\n"
            f"  Available Linear module suffixes: {sorted(present_suffixes)}",
            hint="On LFM2/LFM2.5 the attention output projection is 'self_attn.out_proj' "
            "(not 'o_proj') and the FFN is 'feed_forward.w1/w2/w3' (not "
            "'gate_proj/up_proj/down_proj'). Run `lfm-finetune inspect-model` to see the "
            "real module names, or set lora.target_modules=auto.",
        )

    diagnostics = {
        "selected": explicit,
        "selected_counts": {m: inventory.linear_suffixes[m] for m in explicit},
        "num_target_instances": sum(inventory.linear_suffixes[m] for m in explicit),
        "source": "explicit",
        "excluded_present": [],
        "is_hybrid": bool(inventory.conv_modules),
        "layer_types": inventory.layer_types,
    }
    return explicit, inventory, diagnostics


def build_peft_config(
    lora_cfg: LoraConfigModel, resolved_target_modules: list[str] | None = None
) -> Any:
    """Translate the typed config into a :class:`peft.LoraConfig`.

    Args:
        lora_cfg: The validated ``lora`` section.
        resolved_target_modules: The concrete module suffixes returned by
            :func:`resolve_target_modules`. **Must be passed** when
            ``lora.target_modules`` is ``"auto"``: handing PEFT the literal string
            ``"auto"`` makes it search for a module named "auto", find nothing, and
            fail with a misleading "target modules not found" error.

    Raises:
        LoRAError: if PEFT is unavailable or the configuration is incoherent.
    """
    try:
        from peft import LoraConfig, TaskType
    except ImportError as exc:  # pragma: no cover
        raise LoRAError(
            "peft is not installed.",
            hint="Run `uv sync` (or `uv sync --extra dev`) inside finetune/.",
        ) from exc

    if resolved_target_modules is not None:
        target_modules: Any = list(resolved_target_modules)
    else:
        targets = lora_cfg.target_modules
        if isinstance(targets, str):
            if targets == "auto":
                raise LoRAError(
                    "build_peft_config called with lora.target_modules='auto' and no "
                    "resolved module list.",
                    hint="Always resolve the targets first with resolve_target_modules(), "
                    "which inspects the live model graph.",
                )
            target_modules = targets
        else:
            target_modules = list(targets)

    if not target_modules:
        raise LoRAError(
            "Resolved LoRA target module list is empty.",
            hint="Run `lfm-finetune inspect-model -c <config>` to see the available "
            "Linear module suffixes.",
        )

    init = lora_cfg.init_lora_weights
    init_value = {"true": True, "gaussian": True, "eva": "eva"}[init]

    try:
        return LoraConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.alpha,
            lora_dropout=lora_cfg.dropout,
            bias=lora_cfg.bias,
            task_type=TaskType(lora_cfg.task_type),
            target_modules=target_modules,
            modules_to_save=list(lora_cfg.modules_to_save) or None,
            inference_mode=False,
            use_rslora=lora_cfg.use_rslora,
            use_dora=lora_cfg.use_dora,
            fan_in_fan_out=lora_cfg.fan_in_fan_out,
            init_lora_weights=init_value,
            layers_to_transform=lora_cfg.layers_to_transform,
            layers_pattern=lora_cfg.layers_pattern,
        )
    except Exception as exc:
        raise LoRAError(
            f"PEFT rejected the LoRA configuration: {exc}",
            hint="Check lora.r (1-256), lora.alpha, and that lora.target_modules is "
            "'auto' or a list of real module suffixes.",
        ) from exc


def attach_lora(
    base_model: Any,
    lora_cfg: LoraConfigModel,
    *,
    model_id: str = "<unknown>",
    prepare_for_kbit: bool = False,
) -> tuple[Any, ParameterAudit, list[str], ModuleInventory, dict[str, Any]]:
    """Attach LoRA to a base model and audit the result.

    Returns:
        ``(peft_model, audit, target_module_types, inventory, diagnostics)``

    Raises:
        LoRAError: when the adapter did not attach, when nothing is trainable,
            or when an implausible share of the model became trainable.
    """
    targets, inventory, diagnostics = resolve_target_modules(lora_cfg, base_model, model_id=model_id)

    if prepare_for_kbit:
        try:
            from peft import prepare_model_for_kbit_training

            base_model = prepare_model_for_kbit_training(
                base_model, use_gradient_checkpointing=False
            )
            log.info("Model prepared for k-bit training (all base weights cast to fp32/bf16)")
        except ImportError as exc:
            raise LoRAError(
                "Quantised training requested but bitsandbytes/peft k-bit helpers are unavailable.",
                hint="Install the optional extra: `uv sync --extra quant`, or set "
                "lora.enabled with model.quantisation=none.",
            ) from exc

    peft_config = build_peft_config(lora_cfg, resolved_target_modules=targets)

    try:
        from peft import get_peft_model

        model = get_peft_model(base_model, peft_config)
    except Exception as exc:
        raise LoRAError(
            f"peft.get_peft_model failed: {exc}",
            hint="This usually means lora.target_modules matched nothing. Run "
            "`lfm-finetune inspect-model` to list the real Linear module names.",
        ) from exc

    audit_result = audit_trainable_parameters(model, lora_cfg, targets)
    log.info("\n%s", audit_result.summary())

    if isinstance(lora_cfg.target_modules, str):
        log.info("\n%s", describe_lora_plan(inventory, targets, diagnostics))

    return model, audit_result, targets, inventory, diagnostics


def audit_trainable_parameters(
    model: Any, lora_cfg: LoraConfigModel, targets: list[str]
) -> ParameterAudit:
    """Verify that exactly the intended parameters are trainable.

    This is the single most important guard in the whole pipeline. Without it,
    a mistyped target list can either train nothing (silent no-op) or, worse,
    unfreeze the tied embedding/lm_head and turn a LoRA run into a partial
    full fine-tune.
    """
    total = 0
    trainable = 0
    lora_a = 0
    lora_b = 0
    other: list[str] = []
    matched_targets: list[str] = []
    matched_types: set[str] = set()

    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        if not param.requires_grad:
            continue
        trainable += n
        if "lora_A" in name:
            lora_a += n
        elif "lora_B" in name:
            lora_b += n
        else:
            other.append(name)
        if "lora_" in name or "modules_to_save" in name:
            matched_targets.append(name)
            for target in targets:
                if target in name:
                    matched_types.add(target)
                    break

    pct = 100.0 * trainable / max(1, total)
    warnings: list[str] = []

    if other:
        preview = other[:8]
        warnings.append(
            f"{len(other)} trainable parameters are not LoRA weights: {preview}"
            + (" ..." if len(other) > 8 else "")
        )

    if trainable == 0:
        raise LoRAError(
            "Zero parameters are trainable after attaching LoRA. Training would silently "
            "do nothing.",
            hint="lora.target_modules matched no module. Inspect the real names with "
            "`lfm-finetune inspect-model`.",
        )

    if trainable < lora_cfg.min_trainable_params:
        raise LoRAError(
            f"Only {trainable:,} trainable parameters, below the configured floor of "
            f"{lora_cfg.min_trainable_params:,}. Either too few target modules matched or "
            "lora.r is very small.",
            hint="Increase lora.r, widen lora.target_modules, or lower "
            "lora.min_trainable_params if this is intentional.",
        )

    if pct > lora_cfg.max_trainable_pct:
        raise LoRAError(
            f"{pct:.2f}% of parameters are trainable, above the "
            f"{lora_cfg.max_trainable_pct}% ceiling. That is not a LoRA run.",
            hint="Check lora.target_modules and lora.modules_to_save. On LFM2.5-*-Base, "
            "including lm_head/embed_tokens in modules_to_save breaks the weight tie.",
        )

    if lora_a == 0 or lora_b == 0:
        raise LoRAError(
            f"LoRA A parameters = {lora_a:,}, LoRA B parameters = {lora_b:,}. Both must be "
            "non-zero for a functioning adapter (A is randomly initialised, B starts at zero).",
        )

    if not matched_types:
        warnings.append(
            f"No matched target module type recorded; resolved types: {matched_targets[:6]}"
        )

    # Explicitly verify the base weights stayed frozen.
    base_trainable = [n for n, p in model.named_parameters()
                      if p.requires_grad and "lora_" not in n and "modules_to_save" not in n]
    if base_trainable:
        raise LoRAError(
            f"{len(base_trainable)} non-LoRA parameters are trainable after PEFT "
            f"attachment (first: {base_trainable[:5]}).",
            hint="The base model must stay frozen. Remove them from "
            "lora.modules_to_save / lora.target_modules.",
        )

    return ParameterAudit(
        total_params=total,
        trainable_params=trainable,
        frozen_params=total - trainable,
        trainable_pct=pct,
        expected_min=lora_cfg.min_trainable_params,
        max_allowed_pct=lora_cfg.max_trainable_pct,
        target_modules=sorted(matched_types),
        target_module_types=targets,
        lora_a_params=lora_a,
        lora_b_params=lora_b,
        other_trainable=other,
        warnings=warnings,
    )


def assert_effective_batch_size_log(
    per_device_batch: int, grad_accum: int, world_size: int
) -> int:
    """Compute and log the true effective batch size across all devices."""
    effective = per_device_batch * grad_accum * max(1, world_size)
    log.info(
        "Effective batch size: %d (per_device=%d x grad_accum=%d x world_size=%d). "
        "The optimizer steps once per effective batch.",
        effective, per_device_batch, grad_accum, world_size,
    )
    return effective