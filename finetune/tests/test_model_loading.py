"""Model loading, architecture gating, and programmatic LoRA module discovery.

The load-bearing claims of this project are checked here against the real
checkpoint:

* the architecture is LFM2/LFM2.5 (``Lfm2ForCausalLM``), a hybrid of
  ``Lfm2ShortConv`` and ``Lfm2Attention`` blocks;
* the LoRA target modules discovered from ``model.named_modules()`` are the ones
  Liquid AI's own reference preset uses;
* the generic Llama/Qwen module names (``o_proj``, ``gate_proj``, ...) do **not**
  exist, so a generic PEFT recipe would match nothing.
"""

from __future__ import annotations

import pytest

from lfm_finetune.config.loader import load_config
from lfm_finetune.config.schema import LoraConfigModel, ModelConfig
from lfm_finetune.model.discovery import (
    EXCLUDED_BY_DEFAULT,
    OFFICIAL_FULL,
    discover_lora_targets,
    inspect_modules,
)
from lfm_finetune.model.lora import attach_lora, build_peft_config, resolve_target_modules
from lfm_finetune.utils.errors import LoRAError, ModelError

REPO = __import__("pathlib").Path(__file__).resolve().parent.parent

# Names from Llama/Qwen/Mistral that do NOT exist on LFM2. Copying these is the
# single most common way to produce an adapter that trains nothing.
GENERIC_PEFT_NAMES = {"o_proj", "gate_proj", "up_proj", "down_proj", "q_proj", "v_proj",
                      "fc1", "fc2"}


@pytest.fixture(scope="module")
def loaded(requires_model):
    """A real LFM2.5 model, loaded once for the whole module."""
    from lfm_finetune.model.loader import load_model

    config, _ = load_config(REPO / "configs/smoke.yaml")
    return load_model(config, for_training=True)


# --------------------------------------------------------------------------- #
# Architecture
# --------------------------------------------------------------------------- #

def test_architecture_is_lfm2(loaded) -> None:
    assert loaded.inventory.architecture.startswith("Lfm2")
    assert loaded.inventory.model_type == "lfm2"


def test_model_is_hybrid_conv_plus_attention(loaded) -> None:
    """LFM2 alternates short-conv and full-attention blocks."""
    inventory = loaded.inventory
    assert inventory.conv_modules, "expected Lfm2ShortConv / Conv1d modules"
    assert inventory.attention_modules, "expected Lfm2Attention modules"
    assert len(set(inventory.layer_types)) > 1, (
        f"expected a mix of layer types, got {inventory.layer_types}"
    )
    assert "conv" in inventory.layer_types
    assert "full_attention" in inventory.layer_types


def test_attention_blocks_are_fewer_than_conv_blocks(loaded) -> None:
    """LFM2.5-1.2B is documented as 10 conv + 6 attention layers."""
    conv = sum(1 for t in loaded.inventory.layer_types if t == "conv")
    attn = sum(1 for t in loaded.inventory.layer_types if t == "full_attention")
    assert conv > attn, f"expected more conv than attention layers, got {conv}/{attn}"


def test_generic_peft_names_are_absent(loaded) -> None:
    present = set(loaded.inventory.linear_suffixes)
    leaked = present & GENERIC_PEFT_NAMES
    assert not leaked, (
        f"unexpected generic module names present: {leaked}. If these ever appear, "
        f"a hard-coded PEFT recipe would silently match the wrong layers."
    )


def test_lfm_specific_names_present(loaded) -> None:
    present = set(loaded.inventory.linear_suffixes)
    for required in ("self_attn.out_proj", "self_attn.q_proj", "feed_forward.w1",
                     "feed_forward.w2", "feed_forward.w3", "conv.in_proj",
                     "conv.out_proj"):
        assert required in present, f"expected LFM2 module {required} to exist"


def test_embeddings_are_tied(loaded) -> None:
    """LFM2.5 base checkpoints tie lm_head to embed_tokens."""
    assert loaded.inventory.tie_embedding is True
    lm_head = loaded.model.get_output_embeddings()
    embeddings = loaded.model.get_input_embeddings()
    assert lm_head.weight is embeddings.weight


def test_base_model_is_frozen_after_load(loaded) -> None:
    trainable = [n for n, p in loaded.model.named_parameters() if p.requires_grad]
    assert not trainable, (
        f"{len(trainable)} parameters still trainable before LoRA is attached; a LoRA run "
        f"must start from a fully frozen base model (first: {trainable[:3]})"
    )


def test_tokenizer_has_chat_template_and_specials(loaded) -> None:
    tok = loaded.tokenizer
    assert tok.chat_template
    assert tok.bos_token_id is not None
    assert tok.eos_token_id is not None
    assert tok.pad_token_id is not None
    assert tok.eos_token_id == 7 or tok.eos_token_id > 1000  # 7 (65k vocab) / 124900 (128k)


def test_unsupported_architecture_rejected() -> None:
    class FakeConfig:
        architectures = ["LlamaForCausalLM"]
        model_type = "llama"

    with pytest.raises(ModelError, match="does not support"):
        from lfm_finetune.model.loader import assert_supported_architecture

        assert_supported_architecture(ModelConfig(name="meta-llama/Llama-3-8B"), FakeConfig())


def test_escape_hatch_allows_unsupported() -> None:
    class FakeConfig:
        architectures = ["LlamaForCausalLM"]
        model_type = "llama"

    from lfm_finetune.model.loader import assert_supported_architecture

    cfg = ModelConfig(name="x", allow_unsupported_architecture=True)
    assert_supported_architecture(cfg, FakeConfig())  # must not raise


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #

def test_inventory_counts_modules(loaded) -> None:
    inv = loaded.inventory
    assert inv.total_params > 0
    assert inv.vocab_size > 0
    assert inv.num_layers > 0
    assert inv.linear_suffixes
    assert inv.norm_modules
    assert inv.embedding_modules


def test_discovery_matches_official_preset(loaded) -> None:
    """The strongest available check: our discovered set equals Liquid AI's."""
    targets, inventory, diagnostics = discover_lora_targets(
        loaded.model, model_id=loaded.model_id
    )
    assert diagnostics["matches_official_reference"], diagnostics
    assert set(targets) == set(OFFICIAL_FULL)


def test_discovery_targets_every_expected_module(loaded) -> None:
    targets, _inventory, diagnostics = discover_lora_targets(
        loaded.model, model_id=loaded.model_id
    )
    assert set(targets) == {
        "feed_forward.w1", "feed_forward.w2", "feed_forward.w3",
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj",
        "conv.in_proj", "conv.out_proj",
    }
    counts = diagnostics["selected_counts"]
    assert counts["self_attn.q_proj"] > 0
    assert counts["feed_forward.w1"] > 0
    assert counts["conv.in_proj"] > 0
    assert diagnostics["num_target_instances"] > 0


def test_discovery_excludes_tied_lm_head(loaded) -> None:
    targets, _inventory, _diag = discover_lora_targets(
        loaded.model, model_id=loaded.model_id
    )
    assert "lm_head" not in targets
    assert "lm_head" in EXCLUDED_BY_DEFAULT


def test_discovery_can_exclude_conv_and_ffn(loaded) -> None:
    conv_only, _i, d1 = discover_lora_targets(
        loaded.model, model_id=loaded.model_id, include_ffn=False
    )
    assert not any(t.startswith("feed_forward") for t in conv_only)
    assert any(t.startswith("conv") for t in conv_only)

    no_conv, _i, d2 = discover_lora_targets(
        loaded.model, model_id=loaded.model_id, include_conv=False
    )
    assert not any(t.startswith("conv") for t in no_conv)
    assert any(t.startswith("feed_forward") for t in no_conv)
    assert d1["num_target_instances"] < diagnostics_total(loaded)


def diagnostics_total(loaded) -> int:
    _t, _i, d = discover_lora_targets(loaded.model, model_id=loaded.model_id)
    return d["num_target_instances"]


def test_discovery_on_unrecognised_model_raises() -> None:
    """With nothing PEFT-shaped in the graph, discovery must fail loudly.

    This is the guard that stops an LFM config being pointed at a model whose
    projections are named something discovery does not understand.
    """
    import torch.nn as nn

    class Unrecognised(nn.Module):
        def __init__(self):
            super().__init__()
            self.thing_a = nn.Linear(4, 4)
            self.thing_b = nn.Linear(4, 4)

    with pytest.raises(ModelError, match="No PEFT-compatible modules"):
        discover_lora_targets(Unrecognised(), model_id="fake")


def test_discovery_recognises_generic_llama_style_names() -> None:
    """The role table also covers Llama/Qwen naming.

    LFM is the target, but discovery is architecture-aware rather than
    LFM-hard-coded, so the same code path yields a sensible answer for a
    conventional model instead of silently matching nothing.
    """
    import torch.nn as nn

    class FakeLlama(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(4, 4)
            self.v_proj = nn.Linear(4, 4)
            self.o_proj = nn.Linear(4, 4)
            self.gate_proj = nn.Linear(4, 4)
            self.down_proj = nn.Linear(4, 4)

    targets, _inv, _diag = discover_lora_targets(FakeLlama(), model_id="fake-llama")
    assert set(targets) == {"q_proj", "v_proj", "o_proj", "gate_proj", "down_proj"}


def test_inspect_modules_works_on_plain_nn_module() -> None:
    import torch.nn as nn

    inv = inspect_modules(nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4)), model_id="plain")
    assert sum(inv.linear_suffixes.values()) == 2
    assert inv.total_params == 2 * (4 * 4 + 4)


# --------------------------------------------------------------------------- #
# Explicit target resolution
# --------------------------------------------------------------------------- #

def test_explicit_targets_verified_against_model(loaded) -> None:
    cfg = LoraConfigModel(r=4, alpha=8, target_modules=["self_attn.q_proj"])
    targets, _inv, diag = resolve_target_modules(cfg, loaded.model, model_id=loaded.model_id)
    assert targets == ["self_attn.q_proj"]
    assert diag["source"] == "explicit"


def test_explicit_nonexistent_target_rejected_with_real_names(loaded) -> None:
    cfg = LoraConfigModel(r=4, alpha=8, target_modules=["o_proj", "gate_proj"])
    with pytest.raises(LoRAError) as exc:
        resolve_target_modules(cfg, loaded.model, model_id=loaded.model_id)
    message = str(exc.value)
    assert "do not exist" in message
    # The error must teach the right names.
    assert "self_attn.out_proj" in message
    assert "feed_forward.w1" in message


def test_auto_is_resolved_not_passed_through(loaded) -> None:
    """PEFT given the literal string 'auto' searches for a module named 'auto'
    and fails with a misleading error. Resolution must happen first."""
    cfg = LoraConfigModel(r=4, alpha=8, target_modules="auto")
    targets, _inv, _diag = resolve_target_modules(cfg, loaded.model, model_id=loaded.model_id)
    assert "auto" not in targets
    peft_config = build_peft_config(cfg, resolved_target_modules=targets)
    # PEFT normalises target_modules to a set; what matters is that it holds the
    # discovered suffixes and never the literal string "auto".
    assert "auto" not in peft_config.target_modules
    assert set(peft_config.target_modules) == set(targets)


def test_build_peft_config_rejects_unresolved_auto() -> None:
    cfg = LoraConfigModel(r=4, alpha=8, target_modules="auto")
    with pytest.raises(LoRAError, match="no resolved module list"):
        build_peft_config(cfg)


# --------------------------------------------------------------------------- #
# Attachment + audit
#
# These use a deliberately tiny *real* LFM2 model built from config, not the
# 350M checkpoint. Building the architecture for real still exercises every
# module-name path, while loading it takes milliseconds -- which matters because
# repeatedly materialising a 1.4 GiB fp32 model on a small-RAM machine stalls the
# whole suite.
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def tiny_lfm():
    """A genuine Lfm2ForCausalLM, a few hundred KB, same module names."""
    from transformers import AutoConfig, AutoModelForCausalLM

    hf_config = AutoConfig.from_pretrained(
        "LiquidAI/LFM2.5-350M-Base", trust_remote_code=False
    )
    hf_config.hidden_size = 64
    hf_config.intermediate_size = 128
    hf_config.block_ff_dim = 128
    hf_config.block_dim = 64
    hf_config.num_hidden_layers = 4
    hf_config.num_attention_heads = 4
    hf_config.num_key_value_heads = 2
    hf_config.head_dim = 16
    hf_config.vocab_size = 256
    hf_config.max_position_embeddings = 256
    hf_config.conv_dim = 64
    return AutoModelForCausalLM.from_config(hf_config)


def test_tiny_model_has_lfm_module_names(tiny_lfm) -> None:
    """The fixture must be a real LFM2 so these tests are meaningful."""
    targets, _inv, diag = discover_lora_targets(tiny_lfm, model_id="tiny")
    assert set(targets) == set(OFFICIAL_FULL)
    assert "o_proj" not in targets


def test_attach_lora_and_audit(tiny_lfm) -> None:
    import copy

    base = copy.deepcopy(tiny_lfm)
    base.requires_grad_(False)
    cfg = LoraConfigModel(r=8, alpha=16, dropout=0.0, target_modules="auto")
    model, audit, targets, _inv, diag = attach_lora(base, cfg, model_id="tiny")

    assert audit.total_params > 0
    assert audit.trainable_params > 0
    assert 0 < audit.trainable_pct < 50
    assert audit.lora_a_params > 0 and audit.lora_b_params > 0
    assert audit.ok
    assert set(targets) == set(OFFICIAL_FULL)
    assert diag["matches_official_reference"]

    non_lora = [
        n for n, p in model.named_parameters()
        if p.requires_grad and "lora_" not in n and "modules_to_save" not in n
    ]
    assert not non_lora, f"base weights unfrozen after PEFT: {non_lora[:5]}"


def test_base_weights_are_bit_identical_after_attachment(tiny_lfm) -> None:
    """LoRA must not alter base weights -- that is the whole point of the method."""
    import copy

    from safetensors.torch import (
        load_file,  # noqa: F401
        )

    before = {n: p.detach().clone() for n, p in tiny_lfm.named_parameters()}
    base = copy.deepcopy(tiny_lfm)
    base.requires_grad_(False)
    cfg = LoraConfigModel(r=4, alpha=8, dropout=0.0, target_modules="auto")
    model, _audit, _t, _i, _d = attach_lora(base, cfg, model_id="tiny")

    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    for name, tensor in inner.named_parameters():
        if name in before:
            assert torch_equal(tensor.detach(), before[name]), f"base weight changed: {name}"


def torch_equal(a, b) -> bool:
    import torch

    return bool(torch.equal(a, b))


def test_audit_rejects_excessive_trainable_pct(tiny_lfm) -> None:
    import copy

    base = copy.deepcopy(tiny_lfm)
    base.requires_grad_(False)
    cfg = LoraConfigModel(r=8, alpha=16, dropout=0.0, target_modules="auto",
                          max_trainable_pct=0.01)
    with pytest.raises(LoRAError, match="above the"):
        attach_lora(base, cfg, model_id="tiny")


def test_audit_rejects_tiny_trainable_floor(tiny_lfm) -> None:
    import copy

    base = copy.deepcopy(tiny_lfm)
    base.requires_grad_(False)
    cfg = LoraConfigModel(r=8, alpha=16, dropout=0.0, target_modules="auto",
                          min_trainable_params=10**12)
    with pytest.raises(LoRAError, match="below the configured floor"):
        attach_lora(base, cfg, model_id="tiny")


def test_audit_rejects_unfrozen_base_weights(tiny_lfm) -> None:
    """If PEFT leaves a base weight trainable, the run is not LoRA-only.

    Simulated by unfreezing after attachment.
    """
    import copy

    from lfm_finetune.model.lora import audit_trainable_parameters

    base = copy.deepcopy(tiny_lfm)
    base.requires_grad_(False)
    cfg = LoraConfigModel(r=4, alpha=8, dropout=0.0, target_modules="auto")
    model, _audit, targets, _i, _d = attach_lora(base, cfg, model_id="tiny")

    first_base_weight = next(
        p for n, p in model.named_parameters()
        if "lora_" not in n and "embed_tokens" not in n and n.endswith("weight")
    )
    first_base_weight.requires_grad_(True)
    with pytest.raises(LoRAError, match="non-LoRA parameters are trainable"):
        audit_trainable_parameters(model, cfg, targets)


def test_trainable_pct_scales_with_rank(tiny_lfm) -> None:
    import copy

    counts = {}
    for rank in (4, 16):
        base = copy.deepcopy(tiny_lfm)
        base.requires_grad_(False)
        cfg = LoraConfigModel(r=rank, alpha=2 * rank, dropout=0.0, target_modules="auto")
        _model, audit, _t, _i, _d = attach_lora(base, cfg, model_id="tiny")
        counts[rank] = audit.trainable_params
    assert counts[16] > counts[4] * 2


def test_effective_scaling_alpha_over_r() -> None:
    assert LoraConfigModel(r=16, alpha=32, use_rslora=False).effective_scaling == 2.0
    import math

    assert LoraConfigModel(r=16, alpha=32, use_rslora=True).effective_scaling == pytest.approx(
        32 / math.sqrt(16)
    )


def test_known_good_checkpoint_is_the_configured_one(loaded) -> None:
    """Pin the model actually under test so a config change cannot silently
    invalidate the discovery assertions above."""
    assert loaded.model_id in (
        "LiquidAI/LFM2.5-350M-Base",
        "LiquidAI/LFM2.5-230M-Base",
        "LiquidAI/LFM2.5-1.2B-Base",
    ), f"unexpected test model {loaded.model_id}"


# --------------------------------------------------------------------------- #
# Hardware
# --------------------------------------------------------------------------- #

def test_hardware_audit_runs() -> None:
    from lfm_finetune.utils.hardware import audit

    report = audit()
    assert report.cpu_count_logical >= 1
    assert report.system_ram_bytes > 0
    assert report.disk_free_bytes >= 0
    assert report.summary_lines()


def test_recommend_model_respects_host_capacity() -> None:
    from lfm_finetune.model.loader import recommend_model
    from lfm_finetune.utils.hardware import audit

    report = audit()
    recs = recommend_model(report, seq_length=1024)
    assert recs
    assert {r["model_id"] for r in recs} >= {"LiquidAI/LFM2.5-1.2B-Base"}
    for rec in recs:
        budget = rec["budget_gb"]
        if not rec["feasible_on_this_host"]:
            assert rec["estimated_peak_gb"] > budget * 0.85, (
                f"{rec['model_id']} marked infeasible but fits the budget"
            )


def test_gpu_training_requested_on_cpu_host_fails() -> None:
    from lfm_finetune.utils.errors import HardwareError
    from lfm_finetune.utils.hardware import HardwareReport, assert_vram_sufficient

    empty = HardwareReport(
        platform="test", machine="x", processor="", python_version="3.12",
        python_executable="python", cpu_count_logical=1, cpu_count_physical=1,
        cpu_model="", system_ram_bytes=1 << 30, system_ram_available_bytes=1 << 30,
        disk_total_bytes=1 << 30, disk_free_bytes=1 << 30, disk_mount="/",
        torch_version="2.11.0", torch_cuda_available=False, torch_cuda_version=None,
        torch_cudnn_version=None, nvidia_driver_version=None,
    )
    with pytest.raises(HardwareError, match="no CUDA device"):
        assert_vram_sufficient(empty, param_count=10**9, dtype="bfloat16", batch_size=1,
                               grad_accum=1, max_seq_length=1024, trainable_params=10**6)


def test_vram_estimate_components() -> None:
    from lfm_finetune.utils.hardware import estimate_training_vram_bytes

    est = estimate_training_vram_bytes(
        param_count=1_200_000_000, dtype="bfloat16", batch_size=4, grad_accum=4,
        max_seq_length=2048, gradient_checkpointing=True, trainable_params=12_000_000,
    )
    assert est["weights_bytes"] == 1_200_000_000 * 2
    assert est["gradients_bytes"] == 12_000_000 * 2
    assert est["optimizer_bytes"] == 12_000_000 * 8
    assert est["peak_bytes"] > est["weights_bytes"]


def test_gradient_checkpointing_reduces_activation_estimate() -> None:
    from lfm_finetune.utils.hardware import estimate_training_vram_bytes

    kwargs = dict(param_count=10**9, dtype="bfloat16", batch_size=4, grad_accum=1,
                  max_seq_length=2048, trainable_params=10**6)
    on = estimate_training_vram_bytes(gradient_checkpointing=True, **kwargs)
    off = estimate_training_vram_bytes(gradient_checkpointing=False, **kwargs)
    assert on["activations_bytes"] < off["activations_bytes"]