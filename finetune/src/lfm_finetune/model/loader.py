"""Fail-fast model and tokenizer loading with architecture gating."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config.schema import ExperimentConfig, ModelConfig
from ..utils.errors import HardwareError, ModelError, TokenizerError
from ..utils.hardware import HardwareReport, audit
from ..utils.logging import get_logger
from .discovery import ModuleInventory, inspect_modules

log = get_logger("model.loader")

#: Architectures this pipeline is validated for.
SUPPORTED_ARCH_PREFIXES = ("Lfm2", "Lfm2ForCausalLM")
SUPPORTED_MODEL_TYPES = ("lfm2", "lfm2-moe", "lfm")
#: Known-good checkpoints and what they are good for. Purely informational; the
#: loader does not restrict to this list, but `recommend_model` consults it.
KNOWN_CHECKPOINTS: dict[str, dict[str, Any]] = {
    "LiquidAI/LFM2.5-1.2B-Base": {
        "params": 1.17e9, "edge_class": "mid-range on-device",
        "use_for": "Domain adaptation for on-device assistants (the MICO-class target).",
    },
    "LiquidAI/LFM2.5-350M-Base": {
        "params": 3.5e8, "edge_class": "low-end on-device / microcontroller-class",
        "use_for": "Cheapest credible LFM2.5; CI smoke tests and CPU validation.",
    },
    "LiquidAI/LFM2.5-230M-Base": {
        "params": 2.3e8, "edge_class": "ultra-low-end",
        "use_for": "Reference point for 'is the domain data or the model the bottleneck?'",
    },
    "LiquidAI/LFM2.5-2.6B-Base": {
        "params": 2.6e9, "edge_class": "server / edge gateway",
        "use_for": "Higher domain accuracy when a ~5 GB footprint is acceptable.",
    },
    "LiquidAI/LFM2.5-8B-A1B-Base": {
        "params": 8.0e9, "edge_class": "server (MoE, 1B active)",
        "use_for": "Not an edge target; useful as a quality ceiling reference.",
    },
}


@dataclass
class LoadedModel:
    """A model+tokenizer pair plus everything discovered about it."""

    model: Any
    tokenizer: Any
    config: ExperimentConfig
    model_id: str
    revision: str
    inventory: ModuleInventory
    dtype: str
    device: str
    attn_implementation: str
    warnings: list[str] = field(default_factory=list)
    chat_template_source: str = "checkpoint"

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def summary(self) -> str:
        inv = self.inventory
        lines = [
            f"model              {self.model_id}@{self.revision}",
            f"architecture       {inv.architecture} (model_type={inv.model_type})",
            f"parameters         {self.num_parameters:,}",
            f"dtype / device     {self.dtype} / {self.device}",
            f"attention impl     {self.attn_implementation}",
            f"context (config)   {inv.max_position_embeddings:,} tokens",
            f"vocabulary         {inv.vocab_size:,}",
            f"hidden / layers    {inv.hidden_size} / {inv.num_layers}",
            f"hybrid layer types {inv.layer_types or 'n/a'}",
            f"chat template      {self.chat_template_source}",
            "module kinds       "
            + ", ".join(f"{k}:{v}" for k, v in sorted(inv.module_type_counts.items())),
        ]
        for w in self.warnings:
            lines.append(f"WARNING            {w}")
        return "\n".join(lines)


def _resolve_dtype(spec: str, hardware: HardwareReport) -> str:
    """Map a dtype spec to something this machine can actually execute.

    ``auto`` means: bfloat16 on a GPU that supports it, float16 on pre-Ampere,
    float32 on CPU. This is the right default because CPU bf16 matmul support
    in PyTorch is uneven across versions and silently slow when it falls back.
    """
    spec = (spec or "auto").lower()
    if spec != "auto":
        return {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}.get(spec, spec)
    if not hardware.gpus:
        return "float32"
    return "bfloat16" if all(g.supports_bf16 for g in hardware.gpus) else "float16"


def resolve_device(spec: str, dtype: str) -> str:
    """Resolve ``device='auto'`` against the requested dtype."""
    import torch

    spec = (spec or "auto").lower()
    if spec != "auto":
        return spec
    if not torch.cuda.is_available():
        return "cpu"
    if dtype == "float32" and torch.cuda.is_available():
        # fp32 on GPU is almost always a mistake; keep it but say so loudly.
        log.warning("dtype=float32 with a CUDA device present -- expect ~2x slower and 2x the VRAM")
    return "cuda"


def assert_supported_architecture(model_cfg: ModelConfig, model_config: Any) -> None:
    """Refuse to train an architecture this pipeline was not validated against."""
    architectures = list(getattr(model_config, "architectures", None) or [])
    model_type = getattr(model_config, "model_type", "") or ""
    supported = any(a.startswith(SUPPORTED_ARCH_PREFIXES) for a in architectures) or (
        model_type in SUPPORTED_MODEL_TYPES
    )
    if supported or model_cfg.allow_unsupported_architecture:
        if not supported:
            log.warning(
                "Architecture %s is NOT an LFM2/LFM2.5 model. Proceeding only because "
                "model.allow_unsupported_architecture=true. LoRA targets must be "
                "verified by hand.",
                architectures or model_type,
            )
        return
    raise ModelError(
        f"Model {model_cfg.name} has architecture {architectures or model_type}, which this "
        f"pipeline does not support (expected one of {SUPPORTED_ARCH_PREFIXES} / "
        f"{SUPPORTED_MODEL_TYPES}).",
        hint="Set model.name to an LFM2/LFM2.5 checkpoint such as "
        "LiquidAI/LFM2.5-1.2B-Base. To force a different architecture, set "
        "model.allow_unsupported_architecture=true and set lora.target_modules explicitly "
        "after inspecting the module names with `lfm-finetune inspect-model`.",
    )


def load_tokenizer(model_cfg: ModelConfig, *, chat_template_override: Path | None = None) -> tuple[Any, str]:
    """Load the tokenizer and validate that it can render conversations.

    Raises:
        TokenizerError: missing tokenizer, missing chat template, or a template
            that cannot render the canonical LFM message dicts.
    """
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_cfg.name,
            revision=model_cfg.revision,
            trust_remote_code=model_cfg.trust_remote_code,
            local_files_only=model_cfg.local_files_only,
        )
    except Exception as exc:
        raise TokenizerError(
            f"Could not load tokenizer for {model_cfg.name}@{model_cfg.revision}: {exc}",
            hint="Check the model id and revision. `huggingface-cli login` if it is gated. "
            "Verify the repo has tokenizer.json by listing its files on the Hub.",
        ) from exc

    source = "checkpoint"
    if chat_template_override is not None:
        override = Path(chat_template_override)
        if not override.exists():
            raise TokenizerError(
                f"tokenization.chat_template points at a missing file: {override}"
            )
        tokenizer.chat_template = override.read_text(encoding="utf-8")
        source = f"override:{override}"
        log.warning("Using an overridden chat template from %s", override)

    if not getattr(tokenizer, "chat_template", None):
        raise TokenizerError(
            f"{model_cfg.name} does not ship a chat template.",
            hint="A base checkpoint without a template cannot be SFT'd on conversations. "
            "Either use a checkpoint that ships chat_template.jinja (all LFM2.5 "
            "checkpoints do) or point tokenization.chat_template at a .jinja file -- see "
            "docs/TRAINING.md.",
        )

    _probe_chat_template(tokenizer, model_cfg.name)
    _report_special_tokens(tokenizer)
    return tokenizer, source


def _probe_chat_template(tokenizer: Any, model_id: str) -> None:
    """Prove the template renders both training and inference shapes."""
    probe = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
        {"role": "assistant", "content": "asst"},
    ]
    try:
        rendered = tokenizer.apply_chat_template(probe, tokenize=False)
    except Exception as exc:
        raise TokenizerError(
            f"chat_template of {model_id} failed to render a system/user/assistant "
            f"conversation: {exc}",
            hint="Inspect the template with "
            "`lfm-finetune inspect-samples --config ...` and check "
            "tokenization.chat_template.",
        ) from exc
    if not rendered:
        raise TokenizerError(f"chat_template of {model_id} rendered an empty string")
    try:
        generation_prompt = tokenizer.apply_chat_template(
            probe[:2], tokenize=False, add_generation_prompt=True
        )
    except Exception as exc:
        raise TokenizerError(
            f"chat_template of {model_id} does not support add_generation_prompt=True: {exc}",
            hint="Training and inference both depend on the generation prompt. Without it, "
            "the model has no way to know it should start answering.",
        ) from exc
    log.debug("chat template render OK (%.80s...)", rendered.replace("\n", "\\n"))
    log.debug("generation prompt render OK (%.80s...)", generation_prompt.replace("\n", "\\n"))


def _report_special_tokens(tokenizer: Any) -> None:
    eos = tokenizer.eos_token
    eos_id = tokenizer.eos_token_id
    log.info(
        "tokenizer: vocab=%d bos=%r(%s) eos=%r(%s) pad=%r(%s)",
        len(tokenizer),
        tokenizer.bos_token, tokenizer.bos_token_id,
        eos, eos_id,
        tokenizer.pad_token, tokenizer.pad_token_id,
    )
    if eos_id is None:
        raise TokenizerError(
            "Tokenizer has no eos_token_id. Training needs a definite end-of-sequence id.",
            hint="Set tokenizer.eos_token_id in the checkpoint's tokenizer_config.json.",
        )
    if not eos:
        # LFM2.5 registers ids 6/7 with an EMPTY `content` field in
        # special_tokens_map.json, so inspecting the raw JSON makes the
        # terminator look absent. The loaded tokenizer still resolves the real
        # marker, and eos_token_id is what actually governs generation -- so this
        # is worth confirming explicitly rather than inferred from the file.
        decoded = ""
        with contextlib.suppress(Exception):  # pragma: no cover
            decoded = tokenizer.decode([eos_id])
        log.warning(
            "eos_token is an empty string (id=%s, decodes to %r). Generation will stop on "
            "the id, which for LFM2.5 is the end-of-turn token; the template already emits "
            "it after the final turn. Verify termination via stop reason rather than by "
            "looking for a visible marker in decoded text.",
            eos_id, decoded,
        )
    else:
        log.info(
            "eos_token=%r (id=%s) resolves to a concrete marker token.",
            eos, eos_id,
        )


def load_model(
    config: ExperimentConfig,
    *,
    hardware: HardwareReport | None = None,
    device_override: str | None = None,
    dtype_override: str | None = None,
    for_training: bool = True,
) -> LoadedModel:
    """Load the base model with every fail-fast check applied.

    Sequence of checks, cheapest first: tokenizer -> config/architecture ->
    dtype/device feasibility -> weights -> context length vs max_seq_length.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    hardware = hardware or audit()
    model_cfg = config.model

    template_path = config.tokenization.chat_template
    tokenizer, template_source = load_tokenizer(
        model_cfg, chat_template_override=Path(template_path) if template_path else None
    )

    try:
        hf_config = AutoConfig.from_pretrained(
            model_cfg.name,
            revision=model_cfg.revision,
            trust_remote_code=model_cfg.trust_remote_code,
            local_files_only=model_cfg.local_files_only,
        )
    except Exception as exc:
        raise ModelError(
            f"Could not load config for {model_cfg.name}@{model_cfg.revision}: {exc}",
            hint="Verify the model id/revision on huggingface.co and that you are "
            "authenticated if the repo is gated.",
        ) from exc

    assert_supported_architecture(model_cfg, hf_config)

    dtype = dtype_override or _resolve_dtype(model_cfg.dtype, hardware)
    device = device_override or resolve_device(model_cfg.device, dtype)

    warnings: list[str] = []
    if device == "cuda" and dtype == "float32":
        warnings.append(
            "float32 on CUDA: roughly 2x the memory and much slower than bfloat16. "
            "Prefer model.dtype=bfloat16 unless you have a specific reason."
        )
    if device == "cpu" and dtype != "float32":
        warnings.append(
            f"Running on CPU with dtype={dtype}; CPU bf16/fp16 matmul support is uneven, "
            "so this is much slower than float32. Using float32."
        )
        dtype = "float32"

    # -- context length gate --------------------------------------------- #
    max_seq = config.tokenization.max_seq_length
    ctx = int(getattr(hf_config, "max_position_embeddings", 0) or 0)
    if ctx and max_seq > ctx:
        raise ModelError(
            f"tokenization.max_seq_length={max_seq} exceeds the checkpoint's "
            f"max_position_embeddings={ctx}.",
            hint=f"Lower tokenization.max_seq_length to <= {ctx}. LFM2.5-1.2B-Base is "
            "trained for 32,768 tokens of context even though config declares 128,000; "
            "for domain SFT, 512-2048 is typical.",
        )

    torch_dtype = getattr(torch, dtype)
    attn = model_cfg.attn_implementation
    if attn == "flash_attention_2":
        attn = _negotiate_flash_attention(model_cfg.name)

    load_kwargs: dict[str, Any] = {
        "revision": model_cfg.revision,
        "dtype": torch_dtype,
        "trust_remote_code": model_cfg.trust_remote_code,
        "low_cpu_mem_usage": model_cfg.low_cpu_mem_usage,
        "local_files_only": model_cfg.local_files_only,
    }
    if attn and attn != "auto":
        load_kwargs["attn_implementation"] = attn
    if model_cfg.device_map:
        load_kwargs["device_map"] = model_cfg.device_map

    log.info("Loading %s (dtype=%s, attn=%s) ...", model_cfg.name, dtype, attn)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_cfg.name, **load_kwargs)
    except Exception as exc:
        raise ModelError(
            f"Could not load weights for {model_cfg.name}@{model_cfg.revision}: {exc}",
            hint="Check free disk (a 1.2B bf16 checkpoint needs ~2.4 GB), and that "
            "transformers is on a version with LFM2 support (>=4.57; this project pins 5.3).",
        ) from exc

    if not model_cfg.device_map:
        model = model.to(device)
    model.config.use_cache = bool(model_cfg.use_cache) and not for_training

    inventory = inspect_modules(model, model_id=model_cfg.name)

    # -- gradient checkpointing ------------------------------------------ #
    gradient_checkpointing = (
        config.model.gradient_checkpointing
        if config.training.gradient_checkpointing is None
        else config.training.gradient_checkpointing
    )
    if gradient_checkpointing and for_training:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            model.config.use_cache = False
            log.info("Gradient checkpointing enabled (use_reentrant=False)")
        except Exception as exc:
            warnings.append(f"gradient checkpointing could not be enabled: {exc}")

    # -- weight tying sanity --------------------------------------------- #
    if inventory.tie_embedding:
        tied = _tied_embeddings(model)
        if tied:
            log.info(
                "lm_head is tied to embed_tokens (config.tie_embedding=true). "
                "LoRA must not target either module."
            )
        warnings.append(
            "checkpoint ties lm_head to embed_tokens; excluded both from LoRA targets. "
            "Adding lm_head to lora.modules_to_save would break the tie."
        )

    if for_training:
        freeze_base_model(model)

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        config=config,
        model_id=model_cfg.name,
        revision=model_cfg.revision,
        inventory=inventory,
        dtype=dtype,
        device=str(next(model.parameters()).device),
        attn_implementation=attn or "default",
        warnings=warnings,
        chat_template_source=template_source,
    )


def _negotiate_flash_attention(model_id: str) -> str:
    """Use flash-attn only if the wheel is actually importable and loadable."""
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        log.warning(
            "attn_implementation=flash_attention_2 requested for %s but flash-attn is not "
            "installed; falling back to sdpa.",
            model_id,
        )
        return "sdpa"
    return "flash_attention_2"


def _tied_embeddings(model: Any) -> bool:
    """Verify that lm_head.weight and embed_tokens.weight really are the same tensor."""
    try:
        lm_head = model.get_output_embeddings()
        embeddings = model.get_input_embeddings()
        if lm_head is None or embeddings is None:
            return False
        return lm_head.weight is embeddings.weight
    except Exception:  # pragma: no cover
        return False


def freeze_base_model(model: Any) -> int:
    """Freeze every base-model parameter. Returns how many were unfrozen.

    Necessary and not paranoia: transformers ships models with
    ``requires_grad=True`` on all parameters, and only ``Trainer`` freezes them
    (and it freezes everything, including the LoRA weights, when it manages the
    optimizer). Since this pipeline attaches PEFT itself and verifies the
    trainable set afterwards, the base weights are frozen explicitly here.

    Getting this wrong is not a subtle degradation: if any base weight stayed
    trainable, the run would be a partial full fine-tune while reporting itself
    as LoRA, and the base model would no longer be reusable.
    """
    unfrozen = sum(1 for p in model.parameters() if p.requires_grad)
    model.requires_grad_(False)
    remaining = [n for n, p in model.named_parameters() if p.requires_grad]
    if remaining:  # pragma: no cover - defensive
        raise ModelError(
            f"{len(remaining)} parameters are still trainable after freezing the base "
            f"model (first: {remaining[:5]}).",
            hint="Something re-enabled requires_grad. Report this; it is not expected.",
        )
    if unfrozen:
        log.info("Froze %d base-model parameters (transformers leaves them trainable by default)",
                 unfrozen)
    return unfrozen


def recommend_model(
    hardware: HardwareReport,
    *,
    seq_length: int,
    prefer_edge: bool = True,
) -> list[dict[str, Any]]:
    """Rank LFM2.5 checkpoints for this machine, with an explicit tradeoff note.

    Feasibility is checked against *system RAM* on a CPU host and against *VRAM*
    on a GPU host. Reporting "feasible" on a CPU-only machine purely because
    there is no VRAM would be actively misleading: a 2.6B fp32 model needs
    ~10 GiB of RAM that a small box does not have.
    """
    from ..utils.hardware import estimate_training_vram_bytes

    recommendations: list[dict[str, Any]] = []
    for model_id, meta in KNOWN_CHECKPOINTS.items():
        params = meta["params"]
        trainable = int(params * 0.012)  # ~1.2% is typical for LoRA SFT on LFM2.5
        on_gpu = bool(hardware.gpus)
        compute_dtype = "bfloat16" if on_gpu else "float32"
        estimate = estimate_training_vram_bytes(
            param_count=int(params),
            dtype=compute_dtype,
            batch_size=4,
            grad_accum=1,
            max_seq_length=seq_length,
            gradient_checkpointing=True,
            trainable_params=trainable,
        )
        peak_gb = estimate["peak_bytes"] / 1024**3
        weights_gb = params * (2 if on_gpu else 4) / 1024**3

        # Leave ~1 GiB of headroom for the OS, tokenizer and Python overhead.
        budget_gb = (
            hardware.total_vram_gb if on_gpu else hardware.system_ram_available_bytes / 1024**3
        )
        feasible = peak_gb <= budget_gb * 0.85

        tradeoffs: list[str] = []
        if not on_gpu:
            tradeoffs.append(
                f"CPU-only host: fp32 weights alone are ~{weights_gb:.1f} GiB and the "
                f"estimated peak is ~{peak_gb:.1f} GiB against "
                f"{budget_gb:.1f} GiB available RAM"
            )
        if prefer_edge and params > 2.6e9:
            tradeoffs.append("larger than typical on-device budgets for MICO-class hardware")
        if prefer_edge and params <= 3.6e8:
            tradeoffs.append("smallest credible LFM2.5; weakest at multi-step diagnosis")

        recommendations.append({
            "model_id": model_id,
            "params": params,
            "target": "vram" if on_gpu else "system_ram",
            "budget_gb": round(budget_gb, 2),
            "estimated_peak_gb": round(peak_gb, 2),
            "weights_gb": round(weights_gb, 2),
            "compute_dtype": compute_dtype,
            "feasible_on_this_host": feasible,
            "edge_class": meta["edge_class"],
            "use_for": meta["use_for"],
            "tradeoffs": tradeoffs,
        })

    recommendations.sort(key=lambda r: (not r["feasible_on_this_host"], r["params"]))
    return recommendations


def require_cuda(hardware: HardwareReport) -> None:
    """Fail fast when GPU training was requested on a host without CUDA."""
    import torch

    if not torch.cuda.is_available() or not hardware.gpus:
        raise HardwareError(
            "This command needs a CUDA device but torch.cuda.is_available() is False.",
            hint="Run on a CUDA host, or use a CPU configuration "
            "(model.device=cpu, model.dtype=float32) with a small checkpoint.",
        )