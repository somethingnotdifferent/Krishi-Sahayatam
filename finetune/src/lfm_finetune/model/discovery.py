"""Programmatic LoRA target-module discovery.

Why this module exists
----------------------
LFM2/LFM2.5 is a **hybrid** architecture, not a Llama/Qwen/Mistral clone:

* ``Lfm2DecoderLayer`` alternates ``Lfm2ShortConv`` (a depthwise ``Conv1d``
  sandwiched between two Linear projections) with ``Lfm2Attention`` (GQA).
* Attention output projection is named ``self_attn.out_proj``, **not**
  ``self_attn.o_proj`` as in Llama/Qwen/Mistral.
* The FFN is a SwiGLU with ``feed_forward.w1`` (gate), ``w3`` (up) and
  ``w2`` (down) -- not ``gate_proj``/``up_proj``/``down_proj``.
* ``config.layer_types`` decides, per layer, whether it is ``"conv"`` or
  ``"full_attention"``, so the module inventory legitimately differs between
  layers of the same model.

Copying a generic PEFT recipe (``target_modules=["q_proj","v_proj"]``) silently
matches **nothing** on this architecture or, worse, half-matches after a
name change. Every target module in this project is therefore derived from
``model.named_modules()`` and cross-checked against Liquid AI's own reference
presets.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ..utils.logging import get_logger

log = get_logger("model.discovery")

#: Liquid AI's reference LoRA target sets, from
#: github.com/Liquid4All/leap-finetune -> src/leap_finetune/training/peft/peft_configs.py
#: Used as an *independent cross-check* of what we discover, never as the source.
OFFICIAL_REFERENCE: dict[str, list[str]] = {
    "GLU": ["feed_forward.w1", "feed_forward.w2", "feed_forward.w3"],
    "MHA": [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.out_proj",
    ],
    "CONV": ["conv.in_proj", "conv.out_proj"],
    "VLM_LFM": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                "self_attn.out_proj", "conv.in_proj"],
}
OFFICIAL_FULL = [m for group in ("GLU", "MHA", "CONV") for m in OFFICIAL_REFERENCE[group]]

#: Suffixes that must never receive LoRA on LFM2.5 base checkpoints.
#: ``tie_embedding: true`` in LFM2.5-*-Base/config.json means lm_head.weight and
#: model.embed_tokens.weight are the *same tensor*. Adapting one breaks the tie
#: and silently rewrites the output projection.
EXCLUDED_BY_DEFAULT = ("lm_head", "embed_tokens", "wte", "shared", "norm", "ln_", "gate.weight")

#: Heuristic classification of a Linear module's role. Deliberately explicit
#: lists rather than a ``"linear" in name`` catch-all.
ROLE_SIGNATURES: dict[str, tuple[str, ...]] = {
    "attention_q": ("self_attn.q_proj", "q_proj", "query_proj"),
    "attention_k": ("self_attn.k_proj", "k_proj", "key_proj"),
    "attention_v": ("self_attn.v_proj", "v_proj", "value_proj"),
    "attention_o": ("self_attn.out_proj", "self_attn.o_proj", "o_proj", "out_proj",
                    "dense", "wo"),
    "ffn_gate": ("feed_forward.w1", "w1", "gate_proj", "fc1"),
    "ffn_up": ("feed_forward.w3", "w3", "up_proj", "up"),
    "ffn_down": ("feed_forward.w2", "w2", "down_proj", "down"),
    "conv_in": ("conv.in_proj", "in_proj"),
    "conv_out": ("conv.out_proj", "out_proj"),
    "lm_head": ("lm_head", "output", "embed_out"),
}

_ATTENTION_NAMES = ("self_attn", "attn", "attention", "encoder_attn")


@dataclass
class ModuleInventory:
    """Everything discovered about a model's Linear-layer topology."""

    model_id: str
    architecture: str
    model_type: str
    total_params: int
    vocab_size: int
    hidden_size: int
    num_layers: int
    layer_types: list[str] = field(default_factory=list)
    max_position_embeddings: int = 0
    tie_embedding: bool = False

    linear_suffixes: dict[str, int] = field(default_factory=dict)
    linear_full_names: list[str] = field(default_factory=list)
    attention_modules: list[str] = field(default_factory=list)
    conv_modules: list[str] = field(default_factory=list)
    norm_modules: list[str] = field(default_factory=list)
    embedding_modules: list[str] = field(default_factory=list)
    output_modules: list[str] = field(default_factory=list)
    module_type_counts: dict[str, int] = field(default_factory=dict)
    per_layer_index: dict[int, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "architecture": self.architecture,
            "model_type": self.model_type,
            "total_params": self.total_params,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "layer_types": self.layer_types,
            "max_position_embeddings": self.max_position_embeddings,
            "tie_embedding": self.tie_embedding,
            "linear_suffixes": dict(sorted(self.linear_suffixes.items())),
            "linear_module_count": len(self.linear_full_names),
            "attention_modules": self.attention_modules,
            "conv_modules": self.conv_modules,
            "norm_modules": self.norm_modules,
            "embedding_modules": self.embedding_modules,
            "output_modules": self.output_modules,
            "module_type_counts": dict(sorted(self.module_type_counts.items())),
        }


_LAYER_INDEX_RE = re.compile(r"layers?\.(\d+)\.")


def _leaf_suffix(name: str) -> str:
    """``model.layers.3.self_attn.q_proj`` -> ``self_attn.q_proj``."""
    return ".".join(name.split(".")[-2:])


def _classify(name: str) -> str:
    suffix = _leaf_suffix(name)
    for role, signatures in ROLE_SIGNATURES.items():
        if suffix in signatures:
            return role
    for role, signatures in ROLE_SIGNATURES.items():
        if any(signature in suffix for signature in signatures):
            return role
    return "other"


def inspect_modules(
    model: Any,
    *,
    model_id: str = "<unknown>",
    architecture: str | None = None,
    max_listed: int = 40,
) -> ModuleInventory:
    """Walk ``model.named_modules()`` and build a full topology report."""
    config = getattr(model, "config", None)

    inventory = ModuleInventory(
        model_id=model_id,
        architecture=architecture or (config.architectures or ["?"])[0]
        if config and getattr(config, "architectures", None)
        else (architecture or type(model).__name__),
        model_type=getattr(config, "model_type", "?") if config else "?",
        total_params=sum(p.numel() for p in model.parameters()),
        vocab_size=int(getattr(config, "vocab_size", 0) or 0) if config else 0,
        hidden_size=int(getattr(config, "hidden_size", 0) or 0) if config else 0,
        num_layers=int(getattr(config, "num_hidden_layers", 0) or 0) if config else 0,
        layer_types=list(getattr(config, "layer_types", []) or []) if config else [],
        max_position_embeddings=int(getattr(config, "max_position_embeddings", 0) or 0)
        if config
        else 0,
        tie_embedding=bool(getattr(config, "tie_embedding", False)) if config else False,
    )

    for name, module in model.named_modules():
        kind = type(module).__name__
        inventory.module_type_counts[kind] = inventory.module_type_counts.get(kind, 0) + 1
        full = name or "<root>"

        if "norm" in kind.lower() or _leaf_suffix(full).endswith(("norm", "ln", "ln_f")):
            inventory.norm_modules.append(full)
        if "Conv1d" in kind or "Conv2d" in kind or "ShortConv" in kind:
            inventory.conv_modules.append(full)
        if any(sig in full for sig in _ATTENTION_NAMES) and kind.endswith("Attention"):
            inventory.attention_modules.append(full)
        if "Embedding" in kind:
            inventory.embedding_modules.append(full)
        if _leaf_suffix(full) in ("lm_head", "output") or full.endswith("lm_head"):
            inventory.output_modules.append(full)

        if kind in ("Linear", "Conv1d"):
            suffix = _leaf_suffix(full)
            inventory.linear_suffixes[suffix] = inventory.linear_suffixes.get(suffix, 0) + 1
            inventory.linear_full_names.append(full)
            match = _LAYER_INDEX_RE.search(full)
            if match:
                inventory.per_layer_index.setdefault(int(match.group(1)), []).append(full)

    return inventory


def discover_lora_targets(
    model: Any,
    *,
    model_id: str = "<unknown>",
    include_conv: bool = True,
    include_ffn: bool = True,
    include_attention: bool = True,
    exclude: Iterable[str] = EXCLUDED_BY_DEFAULT,
    cross_check_reference: bool = True,
) -> tuple[list[str], ModuleInventory, dict[str, Any]]:
    """Derive LoRA target module suffixes from the live model graph.

    Args:
        include_conv: Adapt the short-convolution projections. For LFM2.5 these
            (``conv.in_proj``/``conv.out_proj``) carry a meaningful share of the
            model's sequential-mixing capacity; leaving them frozen means LoRA
            can only change attention and the FFN.
        include_ffn: Adapt the SwiGLU projections.
        include_attention: Adapt q/k/v/out projections.
        exclude: Leaf names to never target (defaults exclude ``lm_head``, which
            is weight-tied to the embeddings on LFM2.5-*-Base).
        cross_check_reference: Compare the discovered set against Liquid's
            published presets and report the difference in the diagnostics.

    Returns:
        ``(target_suffixes, inventory, diagnostics)``

    Raises:
        ModelError: when nothing PEFT-compatible is found -- which is exactly
            what happens if someone loads a Llama and reuses an LFM config.
    """
    from ..utils.errors import ModelError

    inventory = inspect_modules(model, model_id=model_id)
    excluded = set(exclude)

    role_to_groups: dict[str, str] = {
        "attention_q": "MHA",
        "attention_k": "MHA",
        "attention_v": "MHA",
        "attention_o": "MHA",
        "ffn_gate": "GLU",
        "ffn_up": "GLU",
        "ffn_down": "GLU",
        "conv_in": "CONV",
        "conv_out": "CONV",
    }

    selected: dict[str, int] = {}
    for suffix, count in inventory.linear_suffixes.items():
        leaf = suffix.split(".")[-1]
        if leaf in excluded or suffix in excluded:
            continue
        if suffix == "lm_head":
            continue
        role = _classify(suffix)
        group = role_to_groups.get(role)
        if group is None:
            continue
        if group == "MHA" and not include_attention:
            continue
        if group == "GLU" and not include_ffn:
            continue
        if group == "CONV" and not include_conv:
            continue
        selected[suffix] = count

    if not selected:
        raise ModelError(
            f"No PEFT-compatible modules discovered in {model_id} "
            f"(architecture {inventory.architecture}, model_type {inventory.model_type}).",
            hint=f"Linear modules found: {sorted(inventory.linear_suffixes) or 'none'}. "
            "If this is genuinely not an LFM2/LFM2.5 checkpoint, pass "
            "lora.target_modules explicitly and set "
            "model.allow_unsupported_architecture=true.",
        )

    targets = sorted(selected)

    diagnostics: dict[str, Any] = {
        "selected": targets,
        "selected_counts": dict(sorted(selected.items())),
        "num_target_instances": sum(selected.values()),
        "excluded_present": sorted(
            s for s in inventory.linear_suffixes
            if s.split(".")[-1] in excluded or s == "lm_head"
        ),
        "is_hybrid": bool(inventory.conv_modules) or len(set(inventory.layer_types)) > 1,
        "layer_types": inventory.layer_types,
    }

    if cross_check_reference:
        discovered = set(targets)
        reference = set(OFFICIAL_FULL)
        diagnostics["official_reference"] = OFFICIAL_FULL
        diagnostics["matches_official_reference"] = discovered == reference
        diagnostics["not_in_official_reference"] = sorted(discovered - reference)
        diagnostics["official_but_not_discovered"] = sorted(reference - discovered)
        if discovered != reference:
            log.warning(
                "Discovered LoRA targets differ from Liquid AI's reference preset.\n"
                "  only here:      %s\n"
                "  only in refset: %s",
                diagnostics["not_in_official_reference"],
                diagnostics["official_but_not_discovered"],
            )
        else:
            log.info(
                "Discovered LoRA targets exactly match Liquid AI's reference preset (%d module "
                "types, %d instances).",
                len(targets),
                diagnostics["num_target_instances"],
            )

    return targets, inventory, diagnostics


def describe_lora_plan(
    inventory: ModuleInventory, targets: list[str], diagnostics: dict[str, Any]
) -> str:
    """Operator-facing summary of the LoRA topology."""
    lines = [
        f"architecture        {inventory.architecture} (model_type={inventory.model_type})",
        f"hybrid (conv+attn)  {diagnostics.get('is_hybrid')}",
        f"layer types         {inventory.layer_types or 'n/a'}",
        f"total parameters    {inventory.total_params:,}",
        f"target module types {len(targets)}",
    ]
    for suffix in targets:
        lines.append(
            f"  - {suffix:28s} x{diagnostics['selected_counts'].get(suffix, 0):<3d} "
            f"role={_classify(suffix)}"
        )
    if diagnostics.get("matches_official_reference"):
        lines.append("reference check     exact match with Liquid AI leap-finetune preset")
    else:
        lines.append(
            f"reference check     differs (extra={diagnostics.get('not_in_official_reference')}, "
            f"missing={diagnostics.get('official_but_not_discovered')})"
        )
    if diagnostics.get("excluded_present"):
        lines.append(f"excluded (tied)     {diagnostics['excluded_present']}")
    return "\n".join(lines)