"""Model loading, architecture gating, LoRA target discovery and attachment."""

from .discovery import (
    OFFICIAL_REFERENCE,
    ModuleInventory,
    describe_lora_plan,
    discover_lora_targets,
    inspect_modules,
)
from .loader import (
    KNOWN_CHECKPOINTS,
    LoadedModel,
    freeze_base_model,
    load_model,
    load_tokenizer,
    recommend_model,
    require_cuda,
)
from .lora import (
    ParameterAudit,
    assert_effective_batch_size_log,
    attach_lora,
    audit_trainable_parameters,
    build_peft_config,
    resolve_target_modules,
)

__all__ = [
    "KNOWN_CHECKPOINTS",
    "OFFICIAL_REFERENCE",
    "LoadedModel",
    "ModuleInventory",
    "ParameterAudit",
    "assert_effective_batch_size_log",
    "attach_lora",
    "audit_trainable_parameters",
    "build_peft_config",
    "describe_lora_plan",
    "discover_lora_targets",
    "freeze_base_model",
    "inspect_modules",
    "load_model",
    "load_tokenizer",
    "recommend_model",
    "require_cuda",
    "resolve_target_modules",
]