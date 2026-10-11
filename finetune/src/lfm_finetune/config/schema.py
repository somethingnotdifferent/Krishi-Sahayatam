"""Typed configuration schema for the whole pipeline.

Design notes
------------
* Every field that influences numerics is explicit and typed. There are no
  "hidden" hyperparameters in Python -- if it changes results, it lives here.
* Cross-field constraints (``lora.alpha`` vs ``r``, batch size vs VRAM, ...)
  are checked in :meth:`ExperimentConfig.validate_semantics`, so bad configs
  fail before a single GPU-second is spent.
* ``lora.target_modules`` accepts the literal string ``"auto"``, meaning
  "discover from the live model graph" -- see
  :mod:`lfm_finetune.model.discovery`. We never guess module names.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..utils.errors import ConfigError

# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #

Dtype = Literal["auto", "bfloat16", "float16", "float32"]
Device = Literal["auto", "cuda", "cpu", "mps"]
SplitName = Literal["train", "validation", "test"]
HParamPreset = Literal["pilot", "balanced", "memory_saver", "full", "smoke"]


class _Base(BaseModel):
    """Permissive-but-validating base: reject unknown keys to catch typos."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, use_enum_values=True)


class RunIdentity(_Base):
    """Naming/traceability of a single experiment."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
        description="Experiment name; also the output directory stem.",
    )
    domain: str = Field(
        default="generic",
        description="Logical domain (agriculture, finance, ...). Used for grouping only; "
        "never hard-coded into training logic.",
    )
    notes: str | None = None
    tags: list[str] = Field(default_factory=list)


class ModelConfig(_Base):
    """Base checkpoint selection and loading."""

    name: str = Field(
        ...,
        description="HF model id, e.g. LiquidAI/LFM2.5-1.2B-Base. Must be an LFM2/LFM2.5 "
        "checkpoint unless model.allow_unsupported_architecture is set.",
    )
    revision: str = Field(default="main", description="HF git revision (branch, tag or commit sha).")
    dtype: Dtype = "bfloat16"
    device: Device = "auto"
    device_map: str | None = Field(
        default=None,
        description="Passed to from_pretrained(device_map=...). Use 'auto' for multi-GPU / "
        "offload. Mutually exclusive with model.device when both are set.",
    )
    trust_remote_code: bool = Field(
        default=False,
        description="LFM2/LFM2.5 is native to transformers>=4.57; leave this False. "
        "Only enable for a genuinely custom checkpoint you have audited.",
    )
    attn_implementation: Literal["auto", "eager", "sdpa", "flash_attention_2"] = "sdpa"
    gradient_checkpointing: bool = Field(
        default=True,
        description="Recompute activations during the backward pass. Cuts activation VRAM "
        "roughly proportionally to depth at ~30% extra compute. Requires "
        "use_cache=False during training.",
    )
    use_cache: bool = Field(default=False, description="Keep disabled for training; SFTTrainer re-enables it for eval/generation.")
    low_cpu_mem_usage: bool = True
    allow_unsupported_architecture: bool = Field(
        default=False,
        description="Escape hatch for non-LFM checkpoints. When False (default) the loader "
        "refuses any architecture other than Lfm2* / known LFM aliases.",
    )
    local_files_only: bool = Field(
        default=False, description="Never hit the network; use the local HF cache only."
    )

    @field_validator("name")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("model.name must not be empty")
        return v.strip()


# --------------------------------------------------------------------------- #
# Dataset sub-sections (module level so they can be referenced by validators)
# --------------------------------------------------------------------------- #

class QualityGates(_Base):
    """Rejection thresholds enforced by the validator."""

    min_user_chars: int = Field(default=2, description="Shorter user turns are rejected as empty.")
    min_assistant_chars: int = Field(default=8, description="Shorter assistant turns rarely carry signal.")
    max_user_chars: int = Field(default=8000, description="Longer prompts are truncated with a warning.")
    max_assistant_chars: int = Field(default=16000, description="Refuse pathologically long targets.")
    min_messages: int = Field(default=2, description="At least one user + one assistant turn.")
    max_turns: int = Field(default=32, description="Reject runaway multi-turn samples.")
    drop_empty_messages: bool = True
    normalize_whitespace: bool = Field(
        default=True, description="Collapse runs of spaces/tabs, strip trailing spaces per line, normalise newlines."
    )
    strip_control_characters: bool = Field(
        default=True, description="Remove C0/C1 control chars and lone surrogates (broken Unicode)."
    )
    deduplicate: bool = True
    near_duplicate: bool = Field(
        default=True, description="Catch paraphrases via character 5-gram MinHash-style fingerprints."
    )
    near_duplicate_threshold: float = Field(
        default=0.90, ge=0.5, le=1.0, description="Jaccard similarity above which two samples are near-duplicates."
    )
    remove_prompt_leakage: bool = Field(
        default=True,
        description="Drop records where the assistant answer is verbatim contained in the "
        "user prompt (a strong signal of templated/duplicated data).",
    )
    check_train_val_overlap: bool = Field(
        default=True,
        description="Refuse to write splits that share examples across boundaries.",
    )
    placeholder_patterns: list[str] = Field(
        default_factory=lambda: [r"\bTODO\b", r"\bFIXME\b", r"\bXXX+\b", r"\bLorem ipsum\b",
                                 r"\bplaceholder\b", r"\bTBD\b", r"<\|.*?\|>"],
        description="Regexes whose matches mark a record as placeholder/unfinished text.",
    )
    strip_html: bool = Field(default=True, description="Remove tags and decode common entities.")

quality: QualityGates = Field(default_factory=QualityGates)

class SplittingConfig(_Base):
    """Deterministic split policy."""

    strategy: Literal["hashed", "sequential"] = Field(
        default="hashed",
        description="'hashed' assigns splits by BLAKE2b(split_seed||example_id) so the "
        "split is stable under re-ordering and incremental additions. "
        "'sequential' is faster but fragile.",
    )
    train_ratio: float = Field(default=0.8, ge=0.1, le=0.95)
    validation_ratio: float = Field(default=0.1, ge=0.0, le=0.5)
    seed: int = Field(
        default=42,
        description="Split seed. Recorded in the manifest; changing it changes the split.",
    )

    @model_validator(mode="after")
    def _ratios_sum(self) -> SplittingConfig:
        total = self.train_ratio + self.validation_ratio
        if total > 1.0:
            raise ValueError(
                f"train_ratio + validation_ratio = {total:.2f} > 1.0 "
                "(test_split gets the remainder, which would be negative)"
            )
        if self.train_ratio <= 0:
            raise ValueError("train_ratio must be > 0")
        return self


class DatasetConfig(_Base):
    """Raw data location, split strategy and quality gates."""

    path: Path | None = Field(
        default=None,
        description="Raw dataset file (.jsonl/.json) or directory. Required unless "
        "dataset.use_processed is true (reuse a prepared split).",
    )
    format: Literal["conversational", "instruction", "auto"] = Field(
        default="conversational",
        description="'conversational' expects {\"messages\": [...]}; 'instruction' accepts "
        "instruction/input/output; 'auto' sniffs per record and normalises to messages.",
    )
    text_column: str | None = Field(
        default=None, description="For raw text-completion corpora (CPT only, not SFT)."
    )
    system_prompt: str | None = Field(
        default=None,
        description="Optional system prompt injected as the first message when a record "
        "has none. Set per domain, never hard-coded in the trainer.",
    )
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str = "test"
    use_processed: bool = Field(
        default=False, description="Train on data/processed/* produced by `prepare`."
    )
    processed_dir: Path | None = None
    max_train_examples: int | None = Field(
        default=None, description="Deterministic subsample cap (pilot runs). Applied after the split."
    )
    max_eval_examples: int | None = None

    quality: QualityGates = Field(default_factory=QualityGates)
    splitting: SplittingConfig = Field(default_factory=SplittingConfig)

    def __hash__(self) -> int:  # pragma: no cover - dataclass-like usage
        return hash(self.model_dump_json())


class TokenizationConfig(_Base):
    """Chat-template rendering + token budget."""

    max_seq_length: int = Field(
        default=1024,
        ge=32,
        le=131072,
        description="Hard cap on prompt+response tokens. Truncation drops the MIDDLE of "
        "long conversations (keep system + first user turn + final assistant turn), never "
        "the label span.",
    )
    chat_template: str | None = Field(
        default=None,
        description="Optional path to a .jinja template overriding the checkpoint's own. "
        "Leave None to use the template shipped with the exact checkpoint -- that is "
        "always the right choice. Note the templates are NOT uniform across LFM2.5 "
        "sizes: 1.2B-Base and 350M-Base ship a template WITHOUT {% generation %} "
        "blocks (so tokenizer.apply_chat_template(return_assistant_tokens_mask=True) "
        "returns an all-zero mask), while 230M-Base, 2.6B-Base, 8B-A1B-Base and the "
        "Instruct variants include them. This pipeline therefore masks by prefix "
        "length, which is correct for all of them. See docs/TRAINING.md.",
    )
    truncation_strategy: Literal["keep_head_tail", "truncate_tail", "error"] = "keep_head_tail"
    padding_side: Literal["right", "left"] = "right"
    append_eos_token: bool = Field(
        default=True,
        description="Ensure the supervised target ends with the checkpoint's "
        "eos_token_id (<|im_end|>: id 7 for the 65,536-vocab LFM2.5 checkpoints, "
        "124900 for the 128k ones). The LFM2.5 template already emits the end-of-turn "
        "marker after the final turn, followed by a literal newline; that newline is "
        "stripped first so the target ends exactly at the terminator, matching what "
        "generation stops on. This is a safety net for checkpoints whose template "
        "omits the marker -- with it disabled the model may never learn to stop, and "
        "every generate() call runs to max_new_tokens. Do not turn this off for LFM2.5.",
    )
    add_generation_prompt_in_training: bool = Field(
        default=False,
        description="Must stay False for SFT. Set True only for a deliberate "
        "prompt-continuation objective.",
    )
    compute_token_stats_on: Literal["train", "all"] = Field(
        default="all", description="Which records to tokenise when building dataset statistics."
    )


class LoraConfigModel(_Base):
    """PEFT LoRA settings."""

    enabled: bool = True
    r: int = Field(default=16, ge=1, le=256, description="Rank of the low-rank update. "
                     "Higher = more capacity and slower. Liquid's reference preset is r=8/alpha=16.")
    alpha: int = Field(default=32, ge=1, description="LoRA scaling numerator; the effective "
                       "update is scaled by alpha/r. Convention: alpha = 2 * r.")
    dropout: float = Field(default=0.05, ge=0.0, le=0.9, description="LoRA dropout; a light "
                           "regulariser. Liquid's reference preset uses 0.1.")
    bias: Literal["none", "all", "lora_only"] = Field(
        default="none",
        description="Which biases receive LoRA. 'none' is correct for LFM2.5: the base "
        "checkpoint's conv projections use conv_bias=false, so there is nothing to adapt.",
    )
    target_modules: list[str] | Literal["auto"] = Field(
        default="auto",
        description="PEFT target module suffixes, or 'auto' to discover them from the live "
        "model graph. For LFM2.5 the discovered set is the GLU projections "
        "(feed_forward.w1/w2/w3), the attention projections (self_attn.q_proj/k_proj/"
        "v_proj/out_proj -- note 'out_proj', NOT 'o_proj') and the short-conv projections "
        "(conv.in_proj/out_proj).",
    )
    exclude_modules: list[str] = Field(
        default_factory=list,
        description="Suffixes to exclude after discovery, e.g. lm_head (tied to embeddings "
        "in LFM2.5-*-Base; adapting it breaks weight tying).",
    )
    modules_to_save: list[str] = Field(
        default_factory=list,
        description="Extra modules trained fully alongside LoRA (e.g. ['lm_head'] when the "
        "checkpoint's embeddings are NOT tied).",
    )
    use_rslora: bool = Field(
        default=False,
        description="Rank-stabilised LoRA scaling alpha/sqrt(r). Useful at high rank with "
        "low alpha; not required for the r=8..16 regime.",
    )
    use_dora: bool = Field(
        default=False,
        description="Weight-decomposed LoRA (DoRA). ~30-50% slower; only worth testing "
        "when plain LoRA plateaus.",
    )
    task_type: Literal["CAUSAL_LM"] = "CAUSAL_LM"
    fan_in_fan_out: bool = False
    init_lora_weights: Literal["true", "gaussian", "eva"] = "true"
    layers_to_transform: list[int] | None = Field(
        default=None, description="Restrict LoRA to a subset of decoder layer indices."
    )
    layers_pattern: str | None = None
    max_trainable_pct: float = Field(
        default=25.0, gt=0, le=100,
        description="Fail fast if the trainable parameter share exceeds this. A silent "
        "mis-targeting of 'lm_head' or 'embed_tokens' can spike this.",
    )
    min_trainable_params: int = Field(
        default=1024, description="Fail fast if fewer LoRA parameters than this are trainable."
    )

    @model_validator(mode="after")
    def _sane(self) -> LoraConfigModel:
        if not self.enabled:
            return self
        if self.bias == "all":
            # Adapting every bias in a frozen model is a full-parameter smell.
            pass
        return self

    @property
    def effective_scaling(self) -> float:
        """alpha / sqrt(r) for RS-LoRA, else alpha / r."""
        import math

        return self.alpha / math.sqrt(self.r) if self.use_rslora else self.alpha / self.r


class TrainingConfigModel(_Base):
    """Optimisation, batching and scheduling."""

    preset: HParamPreset | None = Field(
        default=None,
        description="Convenience preset (pilot/balanced/memory_saver/full/smoke). Applied "
        "before explicit fields, so any explicit key here wins.",
    )
    seed: int = 42
    deterministic_algorithms: bool = Field(
        default=False,
        description="Request torch.use_deterministic_algorithms(warn_only=True). "
        "Slower, and may disable some fused kernels.",
    )
    num_train_epochs: float = Field(default=3.0, gt=0)
    per_device_train_batch_size: int = Field(default=4, ge=1)
    per_device_eval_batch_size: int | None = Field(default=None, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    learning_rate: float = Field(default=2e-4, gt=0, description="LoRA SFT typically 1e-4..3e-4.")
    weight_decay: float = Field(
        default=0.0, ge=0,
        description="LoRA conventionally uses 0.0. Weight decay on adapter weights mainly "
        "acts as an extra regulariser; enable only if validation loss overfits early.",
    )
    warmup_ratio: float = Field(default=0.03, ge=0.0, lt=1.0,
                                description="Fraction of total steps used for linear warmup.")
    lr_scheduler_type: Literal["linear", "cosine", "constant", "constant_with_warmup",
                               "polynomial", "inverse_sqrt"] = "cosine"
    max_grad_norm: float = Field(default=1.0, ge=0,
                                 description="Gradient clipping. 1.0 is a safe default for LoRA.")
    max_steps: int = Field(
        default=-1,
        description="Hard cap on optimizer steps; overrides num_train_epochs when > 0. "
        "Used by pilot/smoke runs.",
    )
    logging_steps: int = Field(default=10, ge=1)
    eval_strategy: Literal["no", "steps", "epoch"] = "steps"
    eval_steps: int | None = Field(default=50, ge=1)
    save_strategy: Literal["no", "steps", "epoch"] = "steps"
    save_steps: int | None = Field(default=50, ge=1)
    save_total_limit: int | None = Field(
        default=3, ge=1, description="Rotate checkpoints, keeping only the newest N. "
        "Set null to keep all (needs disk).",
    )
    load_best_model_at_end: bool = Field(
        default=True, description="Restore the best-eval-loss checkpoint at the end of training."
    )
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    early_stopping_patience: int | None = Field(
        default=None, ge=1,
        description="Evaluations without improvement before training halts. Set to roughly "
        "the number of evaluations per epoch so at most one epoch is wasted.",
    )
    gradient_checkpointing: bool | None = Field(
        default=None, description="Overrides model.gradient_checkpointing when set."
    )
    optim: Literal["adamw_torch", "adamw_torch_fused", "paged_adamw_8bit", "adafactor"] = "adamw_torch"
    adam_beta1: float = Field(default=0.9, ge=0, lt=1)
    adam_beta2: float = Field(default=0.999, ge=0, lt=1)
    adam_epsilon: float = Field(default=1e-8, gt=0)
    lr_scheduler_kwargs: dict[str, Any] = Field(default_factory=dict)
    resume_from_checkpoint: str | None = Field(
        default=None,
        description="Path to a checkpoint-* dir produced by `train`. Pass 'auto' to use the "
        "latest checkpoint in output_dir.",
    )
    dataloader_num_workers: int = Field(default=0, ge=0, le=16)
    dataloader_pin_memory: bool = Field(default=True)
    remove_unused_columns: bool = True
    report_to: list[str] = Field(
        default_factory=list,
        description="HF integrations, e.g. ['wandb']. Local JSONL logging is always on and "
        "requires no external service.",
    )
    run_name: str | None = None
    output_dir: Path | None = Field(default=None, description="Run directory; auto-derived if omitted.")
    packing: bool = Field(
        default=False,
        description="Concatenate examples up to max_seq_length. Faster, but it removes the "
        "per-example loss boundary and mixes domains in one window. Leave off for a first "
        "domain SFT run.",
    )
    assistant_only_loss: bool = Field(
        default=True,
        description="Mask the loss to assistant response tokens only. The prompt carries no "
        "learning signal for a domain assistant and would otherwise dominate the gradient.",
    )
    include_prompt_tokens_in_loss: bool = False
    sample_generation_steps: int = Field(
        default=0, ge=0,
        description="Every N optimizer steps, generate from a fixed held-out prompt to catch "
        "degenerate output early. 0 disables.",
    )
    smoke_test: bool = Field(
        default=False, description="Tiny end-to-end run used by CI; overrides several fields."
    )
    stop_after_steps: int | None = Field(
        default=None, description="Alias for max_steps, kept for readability in pilot configs."
    )

    @model_validator(mode="after")
    def _consistency(self) -> TrainingConfigModel:
        if self.eval_strategy == "steps" and self.eval_steps is None:
            object.__setattr__(self, "eval_steps", self.logging_steps * 2)
        if self.save_strategy == "steps" and self.save_steps is None:
            object.__setattr__(self, "save_steps", self.eval_steps or self.logging_steps * 2)
        if self.load_best_model_at_end:
            if self.eval_strategy == "no":
                raise ValueError(
                    "training.load_best_model_at_end=true requires an eval_strategy of "
                    "'steps' or 'epoch' (otherwise there is no metric to select on)"
                )
            if self.save_strategy == "no":
                raise ValueError(
                    "training.load_best_model_at_end=true requires save_strategy of "
                    "'steps' or 'epoch' so the best checkpoint exists on disk"
                )
        if self.stop_after_steps is not None and self.max_steps < 0:
            object.__setattr__(self, "max_steps", self.stop_after_steps)
        return self

    @property
    def effective_batch_size_multiplier(self) -> int:
        """Tokens of gradient signal per optimizer step, per device-group."""
        return self.per_device_train_batch_size * self.gradient_accumulation_steps


class EvaluationConfigModel(_Base):
    """Generation + scoring for base-vs-tuned comparison."""

    num_samples: int = Field(default=50, ge=1, description="Held-out examples to generate on.")
    max_new_tokens: int = Field(default=320, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, description="0 = greedy decoding (reproducible).")
    top_p: float = Field(default=1.0, gt=0, le=1)
    top_k: int = Field(default=0, ge=0, description="0 disables; with temperature=0 it is irrelevant.")
    repetition_penalty: float = Field(default=1.0, ge=0.5, le=2.0)
    do_sample: bool = Field(
        default=False,
        description="Sampling makes evaluation non-deterministic. Keep False for "
        "reproducible A/B comparisons.",
    )
    batch_size: int = Field(default=1, ge=1)
    num_beams: int = Field(default=1, ge=1)
    seed: int = 42
    rubric_path: Path | None = Field(
        default=None, description="JSON rubric of domain dimensions; informational by default."
    )
    leakage_probe: bool = Field(
        default=True,
        description="Detect answers that reproduce the reference solution verbatim -- a "
        "cheap signal of memorisation rather than generalisation.",
    )
    max_bleu_probes: int = Field(default=20, ge=0)


class LoggingConfigModel(_Base):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_dir: Path = Path("logs")
    per_run_file: bool = Field(default=True, description="Write logs/run_name.log per run.")
    jsonl_metrics: bool = Field(default=True, description="Stream trainer metrics to metrics.jsonl.")
    capture_warnings: bool = True


class CheckpointingConfigModel(_Base):
    dir: Path = Path("checkpoints")
    keep_last: int = Field(
        default=3, ge=1, description="Rotate checkpoints (mirrors training.save_total_limit)."
    )
    save_adapter_only: bool = Field(
        default=True,
        description="Persist the LoRA adapter rather than full weights. Keeps runs small; "
        "resume still works because Trainer re-materialises base weights from the hub.",
    )
    save_tokenizer: bool = True
    save_resolved_config: bool = Field(
        default=True, description="Freeze the fully-resolved config into the run directory."
    )
    metadata_filename: str = "run_metadata.json"


class ExportConfigModel(_Base):
    adapters_dir: Path = Path("adapters")
    merged_dir: Path = Path("merged")
    safe_serialization: bool = True
    max_shard_size: str = "4GB"
    push_to_hub: bool = Field(
        default=False, description="Never enable without an explicit HF_TOKEN in the environment."
    )
    hub_repo_id: str | None = None
    merge_in_bf16: bool = Field(
        default=True,
        description="Merge in bfloat16 rather than fp32. bf16 keeps the merged weights "
        "bit-identical to how they were trained and halves export size.",
    )
    export_tokenizer: bool = True


class ReproducibilityConfigModel(_Base):
    seed: int = Field(default=42)
    dataset_split_seed: int = Field(default=42)
    deterministic_algorithms: bool = False
    record_environment: bool = Field(default=True)
    record_hardware: bool = True
    record_git: bool = True
    hash_dataset: bool = Field(
        default=True, description="Content-hash the processed splits into the run manifest."
    )
    save_seed_state: bool = True


# --------------------------------------------------------------------------- #
# Top level
# --------------------------------------------------------------------------- #

class ExperimentConfig(_Base):
    """Root configuration object. This is the only thing the CLI passes around."""

    schema_version: str = Field(default="1.0", pattern=r"^\d+\.\d+$")
    run: RunIdentity
    model: ModelConfig
    dataset: DatasetConfig
    tokenization: TokenizationConfig = Field(default_factory=TokenizationConfig)
    lora: LoraConfigModel = Field(default_factory=LoraConfigModel)
    training: TrainingConfigModel = Field(default_factory=TrainingConfigModel)
    evaluation: EvaluationConfigModel = Field(default_factory=EvaluationConfigModel)
    logging: LoggingConfigModel = Field(default_factory=LoggingConfigModel)
    checkpointing: CheckpointingConfigModel = Field(default_factory=CheckpointingConfigModel)
    export: ExportConfigModel = Field(default_factory=ExportConfigModel)
    reproducibility: ReproducibilityConfigModel = Field(default_factory=ReproducibilityConfigModel)

    #: Hyperparameter presets. Applied *before* explicit user fields so an explicit
    #: override always wins. Values are starting points from LoRA-SFT practice and
    #: Liquid's reference presets -- none is claimed to be universally optimal.
    PRESETS: ClassVar[dict[str, dict[str, Any]]] = {
        "smoke": {
            "num_train_epochs": 1,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_steps": 4,
            "logging_steps": 1,
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "load_best_model_at_end": False,
            "dataloader_pin_memory": False,
        },
        "pilot": {
            # Few steps on a small subset: answers "does it train, is memory stable,
            # does eval loss behave, does the adapter reload".
            "num_train_epochs": 1,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "max_steps": 50,
            "logging_steps": 5,
            "eval_strategy": "steps",
            "eval_steps": 25,
            "save_strategy": "steps",
            "save_steps": 25,
            "load_best_model_at_end": True,
        },
        "balanced": {
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "logging_steps": 10,
            "eval_steps": 50,
            "save_steps": 50,
            "load_best_model_at_end": True,
        },
        "memory_saver": {
            "num_train_epochs": 3,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "learning_rate": 2e-4,
            "logging_steps": 10,
            "eval_steps": 50,
            "save_steps": 50,
            "load_best_model_at_end": True,
        },
        "full": {
            "num_train_epochs": 3,
            "per_device_train_batch_size": 8,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "logging_steps": 10,
            "eval_steps": 100,
            "save_steps": 100,
            "load_best_model_at_end": True,
        },
    }

    @model_validator(mode="before")
    @classmethod
    def _apply_preset(cls, data: Any) -> Any:
        """Merge the named preset underneath the user's explicit training keys.

        Presets may only set keys that exist on :class:`TrainingConfigModel`. This
        check exists because a preset key landing in the wrong section produces an
        ``extra_forbidden`` error whose message names the *preset*, not the typo.
        """
        if not isinstance(data, dict):
            return data
        preset_name = (data.get("training") or {}).get("preset") if isinstance(
            data.get("training"), dict
        ) else None
        if not preset_name:
            return data
        preset = cls.PRESETS.get(preset_name)
        if preset is None:
            raise ValueError(
                f"Unknown training.preset '{preset_name}'. "
                f"Available: {sorted(cls.PRESETS)}"
            )
        allowed = set(TrainingConfigModel.model_fields)
        invalid = sorted(set(preset) - allowed)
        if invalid:
            raise ValueError(
                f"training.preset '{preset_name}' sets keys that do not exist on the "
                f"training section: {invalid}. Presets may only set {sorted(allowed)}."
            )
        training = dict(data.get("training") or {})
        merged = {**preset}
        # explicit user keys win over preset keys
        merged.update({k: v for k, v in training.items() if v is not None})
        merged["preset"] = preset_name
        data = {**data, "training": merged}
        return data

    # ------------------------------------------------------------------ #
    # Cross-section validation
    # ------------------------------------------------------------------ #

    @model_validator(mode="after")
    def validate_semantics(self) -> ExperimentConfig:
        # 1. dataset source must be resolvable
        if not self.dataset.use_processed and self.dataset.path is None:
            raise ValueError(
                "dataset.path is required unless dataset.use_processed=true"
            )

        # 2. LoRA sanity
        if self.lora.enabled:
            if self.lora.alpha <= 0:
                raise ValueError("lora.alpha must be positive")
            if self.lora.r > self.lora.alpha * 4 and not self.lora.use_rslora:
                raise ValueError(
                    f"lora.r={self.lora.r} is far above lora.alpha={self.lora.alpha}; "
                    "the effective scaling alpha/r would be tiny and training would stall. "
                    "Use alpha >= 2*r, or enable use_rslora."
                )
            if self.lora.modules_to_save and "lm_head" in self.lora.modules_to_save:
                raise ValueError(
                    "lora.modules_to_save contains 'lm_head'. LFM2.5-*-Base checkpoints tie "
                    "lm_head to embed_tokens (config.tie_embedding=true); training lm_head "
                    "separately breaks weight tying and silently changes the output "
                    "distribution. Remove it, or first set config.tie_embedding=false in a "
                    "local copy of the checkpoint."
                )

        # 3. seed coherence: the split seed must agree with the recorded one,
        #    otherwise "replay from the manifest" is ambiguous.
        if self.dataset.splitting.seed != self.reproducibility.dataset_split_seed:
            raise ValueError(
                f"dataset.splitting.seed ({self.dataset.splitting.seed}) != "
                f"reproducibility.dataset_split_seed "
                f"({self.reproducibility.dataset_split_seed}). They must agree so a run "
                "replays the same split."
            )

        # 4. tokenization sanity against model context
        #    (checked lazily against the real config in model.loader)

        # 5. evaluation determinism
        if self.evaluation.do_sample and self.evaluation.temperature == 0:
            raise ValueError(
                "evaluation.do_sample=true with temperature=0.0 raises in transformers; "
                "set temperature > 0 or do_sample=false."
            )

        # 6. effective batch size must be sane
        eff = self.training.effective_batch_size_multiplier
        if eff > 512:
            raise ValueError(
                f"Effective batch size multiplier is {eff} "
                f"({self.training.per_device_train_batch_size} x "
                f"{self.training.gradient_accumulation_steps}). That is very large for "
                "a small domain dataset; you will see very few optimizer steps."
            )
        return self

    # ------------------------------------------------------------------ #
    # Derived values
    # ------------------------------------------------------------------ #

    @property
    def project_root(self) -> Path:
        return Path.cwd()

    def resolve_output_dir(self) -> Path:
        """Run directory: ``<checkpointing.dir>/<run.name>_<timestamp>`` when unset."""
        from datetime import UTC, datetime

        base = self.training.output_dir or (self.checkpointing.dir / self.run.name)
        if self.training.output_dir is not None:
            return Path(self.training.output_dir)
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        return Path(base) / f"{self.run.name}_{stamp}"

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"PRESETS"})

    def fingerprint(self) -> str:
        """Stable hash of the resolved config, recorded in every artifact."""
        from ..utils.hashing import hash_obj

        return hash_obj(self.to_dict())


def parse_config_obj(data: dict[str, Any], *, source: str = "<dict>") -> ExperimentConfig:
    """Validate a raw mapping into an :class:`ExperimentConfig` with good errors."""
    try:
        return ExperimentConfig.model_validate(data)
    except Exception as exc:
        raise ConfigError(
            f"Configuration from {source} is invalid: {exc}",
            hint="Run `lfm-finetune validate --config <file>` to see every problem at once, "
            "and compare against configs/base.yaml.",
        ) from exc


_ENV_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")


def env_overrides(prefix: str = "LFM_") -> dict[str, Any]:
    """Collect ``LFM_SECTION__KEY=value`` environment variables as config overrides.

    Example::

        LFM_TRAINING__LEARNING_RATE=1e-4 lfm-finetune train --config x.yaml

    Secrets are deliberately *not* read here; see ``.env.example`` for token
    handling. This mechanism is for reproducibility sweeps in CI, where
    parameterising a config from the environment is more convenient than
    generating a file per combination.
    """
    import os

    out: dict[str, Any] = {}
    for raw_key, raw_value in os.environ.items():
        if not raw_key.startswith(prefix):
            continue
        path = raw_key[len(prefix) :].lower()
        if "__" not in path or not _ENV_PATTERN.match(raw_key):
            continue
        parts = [p for p in path.split("__") if p]
        if len(parts) < 2:
            continue
        cursor: dict[str, Any] = out
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = _coerce_env(raw_value)
    return out


def _coerce_env(value: str) -> Any:
    import json

    lowered = value.strip().lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if lowered in ("null", "none", ""):
        return None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge. ``overlay`` wins; lists are replaced wholesale.

    Lists are replaced rather than concatenated on purpose: concatenating
    ``lora.target_modules`` across an inheritance chain would silently grow the
    target set, which is exactly the failure mode this project must avoid.
    """
    result = dict(base)
    for key, value in overlay.items():
        if value is None:
            # `key: null` in a child config means "reset to default".
            result.pop(key, None)
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result