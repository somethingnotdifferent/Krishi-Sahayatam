"""LoRA attachment, parameter accounting, and the fail-fast guards.

Split from ``test_model_loading.py`` so the LoRA invariants can be run in
isolation while iterating on adapter behaviour.
"""

from __future__ import annotations

import copy
import math

import pytest

from lfm_finetune.config.schema import LoraConfigModel
from lfm_finetune.model.discovery import OFFICIAL_FULL, discover_lora_targets
from lfm_finetune.model.lora import (
    assert_effective_batch_size_log,
    attach_lora,
    audit_trainable_parameters,
    build_peft_config,
    resolve_target_modules,
)
from lfm_finetune.utils.errors import LoRAError


@pytest.fixture(scope="module")
def tiny_lfm():
    """A real (tiny) Lfm2ForCausalLM: same module names, negligible memory."""
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained("LiquidAI/LFM2.5-350M-Base", trust_remote_code=False)
    cfg.hidden_size = 64
    cfg.intermediate_size = 128
    cfg.block_ff_dim = 128
    cfg.block_dim = 64
    cfg.num_hidden_layers = 4
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.head_dim = 16
    cfg.vocab_size = 256
    cfg.max_position_embeddings = 256
    cfg.conv_dim = 64
    return AutoModelForCausalLM.from_config(cfg)


def _attach(tiny_lfm, **kwargs):
    """Attach LoRA to a fresh copy and return the full result plus the config."""
    kwargs.setdefault("r", 8)
    kwargs.setdefault("alpha", 2 * kwargs["r"])
    base = copy.deepcopy(tiny_lfm)
    base.requires_grad_(False)
    cfg = LoraConfigModel(dropout=0.0, target_modules="auto", **kwargs)
    return (*attach_lora(base, cfg, model_id="tiny"), cfg)


# --------------------------------------------------------------------------- #
# Target resolution
# --------------------------------------------------------------------------- #

def test_auto_resolution_matches_official_preset(tiny_lfm) -> None:
    targets, _inv, diag = discover_lora_targets(tiny_lfm, model_id="tiny")
    assert set(targets) == set(OFFICIAL_FULL)
    assert diag["matches_official_reference"] is True
    assert diag["is_hybrid"] is True
    assert diag["num_target_instances"] == sum(diag["selected_counts"].values())


def test_target_selection_flags_are_honoured(tiny_lfm) -> None:
    no_conv, _i, _d = discover_lora_targets(tiny_lfm, model_id="tiny", include_conv=False)
    assert not any(t.startswith("conv") for t in no_conv)

    no_ffn, _i, _d = discover_lora_targets(tiny_lfm, model_id="tiny", include_ffn=False)
    assert not any(t.startswith("feed_forward") for t in no_ffn)

    no_attn, _i, _d = discover_lora_targets(tiny_lfm, model_id="tiny", include_attention=False)
    assert not any(t.startswith("self_attn") for t in no_attn)


def test_lm_head_never_targeted(tiny_lfm) -> None:
    targets, _i, diag = discover_lora_targets(tiny_lfm, model_id="tiny")
    assert "lm_head" not in targets
    assert "lm_head" in diag["excluded_present"]


def test_explicit_targets_are_verified(tiny_lfm) -> None:
    cfg = LoraConfigModel(r=4, alpha=8, target_modules=["self_attn.q_proj"])
    targets, _inv, diag = resolve_target_modules(cfg, tiny_lfm, model_id="tiny")
    assert targets == ["self_attn.q_proj"]
    assert diag["source"] == "explicit"


def test_explicit_wrong_names_error_lists_real_ones(tiny_lfm) -> None:
    cfg = LoraConfigModel(r=4, alpha=8, target_modules=["o_proj", "gate_proj"])
    with pytest.raises(LoRAError) as exc:
        resolve_target_modules(cfg, tiny_lfm, model_id="tiny")
    msg = str(exc.value)
    assert "do not exist" in msg
    # The remediation must name the correct LFM modules.
    assert "self_attn.out_proj" in msg
    assert "feed_forward.w1" in msg


def test_auto_is_never_handed_to_peft(tiny_lfm) -> None:
    """PEFT would search for a module literally named 'auto'."""
    cfg = LoraConfigModel(r=4, alpha=8, target_modules="auto")
    targets, _i, _d = resolve_target_modules(cfg, tiny_lfm, model_id="tiny")
    peft_config = build_peft_config(cfg, resolved_target_modules=targets)
    assert "auto" not in peft_config.target_modules
    assert set(peft_config.target_modules) == set(targets)


def test_build_peft_config_requires_resolution() -> None:
    with pytest.raises(LoRAError, match="no resolved module list"):
        build_peft_config(LoraConfigModel(r=4, alpha=8, target_modules="auto"))


# --------------------------------------------------------------------------- #
# Parameter accounting
# --------------------------------------------------------------------------- #

def test_trainable_parameters_are_plausible(tiny_lfm) -> None:
    model, audit, targets, _i, _d, cfg = _attach(tiny_lfm)
    assert 0 < audit.trainable_pct < 50
    assert audit.lora_a_params > 0 and audit.lora_b_params > 0
    assert audit.frozen_params == audit.total_params - audit.trainable_params
    assert set(targets) == set(OFFICIAL_FULL)
    assert audit.ok
    del model


def test_trainable_count_matches_rank_formula(tiny_lfm) -> None:
    """LoRA A is (r x fan_in) and LoRA B is (fan_out x r) per targeted Linear.

    Summing r*(in+out) over the discovered modules must equal the audited total.
    This is the check that would catch a silently mis-targeted module list.
    """
    rank = 8
    expected = 0
    targets, _i, _d = discover_lora_targets(tiny_lfm, model_id="tiny")
    target_set = set(targets)
    for name, module in tiny_lfm.named_modules():
        if type(module).__name__ != "Linear":
            continue
        suffix = ".".join(name.split(".")[-2:])
        if suffix in target_set:
            expected += rank * (module.in_features + module.out_features)

    _model, audit, _t, _i2, _d2, _cfg = _attach(tiny_lfm, r=rank, alpha=2 * rank)
    assert audit.lora_a_params + audit.lora_b_params == expected


def test_base_weights_unchanged_by_attachment(tiny_lfm) -> None:
    import torch

    before = {n: p.detach().clone() for n, p in tiny_lfm.named_parameters()}
    model, _audit, _t, _i, _d, _cfg = _attach(tiny_lfm)
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    changed = [
        n for n, p in inner.named_parameters()
        if n in before and not torch.equal(p.detach(), before[n])
    ]
    assert not changed, f"base weights modified during attachment: {changed[:5]}"


def test_only_lora_params_are_trainable(tiny_lfm) -> None:
    model, _audit, _t, _i, _d, _cfg = _attach(tiny_lfm)
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert "lora_" in name or "modules_to_save" in name, name


def test_rank_doubles_trainable_count(tiny_lfm) -> None:
    small = _attach(tiny_lfm, r=4, alpha=8)[1]
    large = _attach(tiny_lfm, r=8, alpha=16)[1]
    assert large.trainable_params == pytest.approx(small.trainable_params * 2, rel=0.01)


def test_alpha_does_not_change_parameter_count(tiny_lfm) -> None:
    """alpha only rescales the update; it adds no parameters."""
    a = _attach(tiny_lfm, r=8, alpha=16)[1]
    b = _attach(tiny_lfm, r=8, alpha=64)[1]
    assert a.trainable_params == b.trainable_params


def test_rslora_and_dora_flags_propagate(tiny_lfm) -> None:

    base = LoraConfigModel(r=8, alpha=16, use_rslora=True)
    assert build_peft_config(base, ["self_attn.q_proj"]).use_rslora is True
    base = LoraConfigModel(r=8, alpha=16, use_dora=True)
    assert build_peft_config(base, ["self_attn.q_proj"]).use_dora is True


def test_effective_scaling() -> None:
    assert LoraConfigModel(r=16, alpha=32).effective_scaling == 2.0
    assert LoraConfigModel(r=16, alpha=16).effective_scaling == 1.0
    assert LoraConfigModel(r=16, alpha=32, use_rslora=True).effective_scaling == pytest.approx(
        32 / math.sqrt(16)
    )


# --------------------------------------------------------------------------- #
# Fail-fast guards
# --------------------------------------------------------------------------- #

def test_zero_trainable_parameters_rejected(tiny_lfm) -> None:
    """Simulated by pointing the audit at a config whose floor cannot be met."""
    base = copy.deepcopy(tiny_lfm)
    base.requires_grad_(False)
    cfg = LoraConfigModel(r=8, alpha=16, dropout=0.0, target_modules="auto",
                          min_trainable_params=10**15)
    with pytest.raises(LoRAError, match="below the configured floor"):
        attach_lora(base, cfg, model_id="tiny")


def test_excessive_trainable_pct_rejected(tiny_lfm) -> None:
    base = copy.deepcopy(tiny_lfm)
    base.requires_grad_(False)
    cfg = LoraConfigModel(r=8, alpha=16, dropout=0.0, target_modules="auto",
                          max_trainable_pct=0.0001)
    with pytest.raises(LoRAError, match="above the"):
        attach_lora(base, cfg, model_id="tiny")


def test_unfrozen_base_weight_rejected_by_audit(tiny_lfm) -> None:
    """Unfreezing a base weight after attachment must be caught.

    Without this guard a run would be a partial full fine-tune while reporting
    itself as LoRA, and the base checkpoint would no longer be reusable.
    """
    base = copy.deepcopy(tiny_lfm)
    base.requires_grad_(False)
    cfg = LoraConfigModel(r=4, alpha=8, dropout=0.0, target_modules="auto")
    model, _audit, targets, _i, _d = attach_lora(base, cfg, model_id="tiny")

    victim = next(
        p for n, p in model.named_parameters()
        if "lora_" not in n and "embed_tokens" not in n and n.endswith("weight")
    )
    victim.requires_grad_(True)
    with pytest.raises(LoRAError, match="non-LoRA parameters are trainable"):
        audit_trainable_parameters(model, cfg, targets)


def test_modules_to_save_lm_head_is_refused_at_config_layer() -> None:
    """Weight tying makes lm_head untrainable-in-practice for LoRA runs."""
    from pydantic import ValidationError

    from lfm_finetune.config.schema import ExperimentConfig

    with pytest.raises(ValidationError, match="lm_head"):
        ExperimentConfig.model_validate({
            "run": {"name": "x"},
            "model": {"name": "LiquidAI/LFM2.5-1.2B-Base"},
            "dataset": {"path": "d"},
            "lora": {"r": 8, "alpha": 16, "target_modules": "auto",
                     "modules_to_save": ["lm_head"]},
        })


def test_effective_batch_size_logged() -> None:
    assert assert_effective_batch_size_log(4, 4, 1) == 16
    assert assert_effective_batch_size_log(1, 8, 2) == 16
    assert assert_effective_batch_size_log(1, 1, 1) == 1


def test_adapter_round_trips_through_disk(tiny_lfm, tmp_path) -> None:
    """Save an adapter, verify it, and reload it into a fresh model."""
    from lfm_finetune.export.adapter import export_adapter

    model, audit, _t, _i, _d, _cfg = _attach(tiny_lfm)
    adapter_dir = tmp_path / "adapter"
    model.save_pretrained(str(adapter_dir))

    result = export_adapter(adapter_dir, tmp_path / "exported", base_model_id="tiny",
                            verify=True, model=tiny_lfm)
    assert result.verified is True

    from peft import PeftModel

    reloaded = PeftModel.from_pretrained(tiny_lfm, str(adapter_dir))
    # named_parameters(), not parameters(): the latter yields tensors with no name.
    n = sum(p.numel() for _n, p in reloaded.named_parameters() if "lora_" in _n)
    assert n == audit.trainable_params
    del reloaded, model


def test_export_detects_corrupt_rank_mismatch(tiny_lfm, tmp_path) -> None:
    """Weights and config coming from different runs must be rejected."""
    import json

    from lfm_finetune.export.adapter import export_adapter
    from lfm_finetune.utils.errors import ExportError

    model, _audit, _t, _i, _d, _cfg = _attach(tiny_lfm, r=8, alpha=16)
    adapter_dir = tmp_path / "adapter"
    model.save_pretrained(str(adapter_dir))

    config_path = adapter_dir / "adapter_config.json"
    data = json.loads(config_path.read_text())
    data["r"] = data["r"] + 5
    config_path.write_text(json.dumps(data))

    with pytest.raises(ExportError, match="disagree with the declared rank"):
        export_adapter(adapter_dir, tmp_path / "out", base_model_id="tiny", verify=True)
    assert issubclass(ExportError, Exception)


def test_export_requires_weights(tiny_lfm, tmp_path) -> None:
    from lfm_finetune.export.adapter import export_adapter
    from lfm_finetune.utils.errors import ExportError

    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "adapter_config.json").write_text('{"r": 8, "target_modules": ["q_proj"]}')
    with pytest.raises(ExportError, match="No adapter weights found"):
        export_adapter(empty, tmp_path / "out", base_model_id="tiny", verify=False)
    del tiny_lfm