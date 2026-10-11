# Krishi Sahayatam / MICO — Agent Context Document

> **Purpose.** This is the single, authoritative context document for this repository. Read it
> completely before changing any code. It is written for a coding agent with no prior knowledge of
> this project.
>
> **Repository:** `Krishi-Sahayatam` · **All real code lives in `finetune/`** · Git: single commit
> `7c319c1 "Initial commit"` on `main`.
>
> **Quick facts:** Python 3.12 · ~11.5k LOC `src/` · ~3.1k LOC `tests/` (≈214 tests) · 19 CLI
> commands · 13 thin script wrappers · pinned `transformers==5.3.x`, `trl==1.9.0`, `peft>=0.15.2`,
> `torch>=2.10,<2.12` · package name `lfm-sft`, console script `lfm-finetune`.

---

## Table of contents

1. [What this project is](#1-what-this-project-is)
2. [Repository layout](#2-repository-layout)
3. [Setup and the 30-second orientation](#3-setup-and-the-30-second-orientation)
4. [The staged workflow](#4-the-staged-workflow)
5. [Architecture and layering](#5-architecture-and-layering)
6. [Configuration system](#6-configuration-system)
7. [Data pipeline](#7-data-pipeline)
8. [Model loading, LFM2.5 facts and LoRA](#8-model-loading-lfm25-facts-and-lora)
9. [Training](#9-training)
10. [Evaluation](#10-evaluation)
11. [Inference](#11-inference)
12. [Export (adapter + merge)](#12-export-adapter--merge)
13. [Preflight — the GO/NO-GO gate](#13-preflight--the-gono-go-gate)
14. [Error model and exit codes](#14-error-model-and-exit-codes)
15. [Artifact layouts](#15-artifact-layouts)
16. [Testing](#16-testing)
17. [Documentation index](#17-documentation-index)
18. [Hard invariants — do not break these](#18-hard-invariants--do-not-break-these)
19. [Current state, validated results and limitations](#19-current-state-validated-results-and-limitations)
20. [Extension recipes](#20-extension-recipes)
21. [Agent operating rules](#21-agent-operating-rules)
22. [Quick reference: common tasks](#22-quick-reference-common-tasks)

---

## 1. What this project is

This is **supervised fine-tuning (SFT) infrastructure** for **MICO**, an agricultural advisory
assistant for smallholder farmers in India, built on **Liquid AI's LFM2 / LFM2.5** models.

It is **not** a training script. It is a pipeline that turns raw agricultural text into a
versioned, evaluated, reproducible, deployable **LoRA adapter** — and that **refuses to start** when
the machine, the model, the data, or the loss mask is wrong.

```text
BASE MODEL
  -> dataset ingestion -> validation -> cleaning -> dedup
  -> leakage-aware train/val/test split
  -> chat-template formatting -> tokenization -> assistant-only loss mask
  -> sample inspection -> baseline eval
  -> LoRA config (discovered, not guessed) -> SFT -> checkpointing
  -> validation/eval -> adapter export -> optional merge -> inference testing
  -> artifact + version tracking
```

**The single most important architectural principle:**

> *Agriculture is **data and configuration**, never code.* The training engine has zero knowledge of
> crops, diseases or India. Adding `configs/finance/` or `configs/legal/` requires new YAML and new
> JSONL — **no changes to the engine**.

Domain specificity lives only in:
- `configs/agriculture/_domain.yaml` (system prompt = behaviour contract, quality thresholds, split ratios), and
- the dataset JSONL (messages + metadata).

---

## 2. Repository layout

```text
Krishi-Sahayatam/
├── README.md                    # one line: "Krishi-Sahayatam"
├── models/                      # EMPTY. Never populated by this project.
└── finetune/                    # <<< ALL THE CODE
    ├── pyproject.toml           # pinned deps, uv-managed, CPU-torch default
    ├── uv.lock
    ├── .env.example             # secrets template (NEVER commit a real token)
    ├── .gitignore
    │
    ├── configs/
    │   ├── base.yaml            # EVERY field documented — this IS the schema reference
    │   ├── lfm_sft.yaml         # LFM2.5-specific overrides
    │   ├── smoke.yaml           # CPU end-to-end validation (350M, fp32)
    │   └── agriculture/
    │       ├── _domain.yaml     # MICO behaviour contract + domain thresholds
    │       ├── lfm25_350m.yaml  # CPU / low-end edge
    │       ├── lfm25_1.2b.yaml  # MICO PRIMARY TARGET
    │       ├── lfm25_1.2b_6gb.yaml  # laptop RTX 4050, 6 GB VRAM
    │       └── lfm25_2.6b.yaml  # edge gateway
    │
    ├── data/                    # runtime data (raw/ is git-ignored)
    │   ├── raw/                 # source corpora + manifest.yaml  (IGNORED)
    │   ├── validated/           # accepted.jsonl, rejected.jsonl, report.json
    │   ├── splits/              # train.jsonl, validation.jsonl, test.jsonl
    │   └── processed/
    │
    ├── datasets/                # COMMITTED: schema + templates + fixtures
    │   ├── schema.json              # canonical record JSON Schema
    │   ├── manifest.schema.json     # provenance manifest JSON Schema
    │   ├── TEMPLATE_dataset.jsonl   # 7 guided example records
    │   ├── TEMPLATE_manifest.yaml   # guided provenance template
    │   ├── README.md                # dataset authoring guide
    │   └── examples/                # synthetic.jsonl, placeholder_agriculture.json
    │
    ├── src/lfm_finetune/        # <<< THE PACKAGE
    │   ├── config/              # schema.py (pydantic), loader.py (YAML inheritance)
    │   ├── data/                # schema, io, validate, dedupe, split, stats, pipeline, synthetic
    │   ├── model/               # loader, discovery, lora
    │   ├── training/            # tokenization, collator, args, trainer, metadata
    │   ├── evaluation/          # metrics, runner
    │   ├── inference/           # generator, cli_loop
    │   ├── export/              # adapter.py (export + merge + verify)
    │   ├── utils/               # errors, logging, seeding, hashing, hardware, io, versions
    │   ├── cli/                 # main.py (19 commands), __main__.py
    │   └── preflight.py         # GO/NO-GO readiness check
    │
    ├── scripts/                 # 13 thin wrappers, all delegate to the CLI
    ├── tests/                   # 8 files, ~214 tests
    ├── docs/                    # 8 markdown documents (see §17)
    │
    ├── checkpoints/             # run directories (git-ignored)
    ├── adapters/                # exported adapters (git-ignored)
    ├── merged/                  # merged models (git-ignored)
    └── evaluations/             # comparison reports (git-ignored)
```

**Git-ignored** (`.gitignore`): `.env`, all weight binaries (`*.safetensors`, `*.bin`, `*.pt`,
`*.gguf`, `*.onnx`), `models/`, `data/raw/*`, `data/{validated,splits,processed}/*`,
`checkpoints/`, `outputs/`, `adapters/`, `merged/`, `evaluations/`, `logs/`, `datasets/**/*.jsonl`
(except `datasets/examples/*.jsonl`).

---

## 3. Setup and the 30-second orientation

```bash
cd finetune
uv python install 3.12
uv sync                       # installs CPU torch wheels by default

uv run lfm-finetune doctor                 # stage 0
uv run lfm-finetune inspect-hardware --recommend
uv run lfm-finetune stages                 # prints the whole staged workflow
```

**The five commands that matter (synthetic fixture path):**

```bash
uv run lfm-finetune init-synthetic --destination data/raw/synthetic
uv run lfm-finetune prepare  -c configs/smoke.yaml
uv run lfm-finetune train    -c configs/smoke.yaml --smoke-test
uv run lfm-finetune evaluate -c configs/agriculture/lfm25_1.2b.yaml
```

**On a CUDA host**, `uv sync` installs CPU torch by default (deliberate — avoids pulling ~3 GB of
unused CUDA wheels). Install CUDA torch explicitly:

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu126 torch
uv sync --no-build-isolation
```

**Auth** is only needed for gated repos. Copy `.env.example` → `.env` and set `HF_TOKEN` if needed.
**Never commit a token.**

**Environment knobs**

| Variable | Effect |
|---|---|
| `LFM_<SECTION>__<KEY>` | Config override, e.g. `LFM_TRAINING__LEARNING_RATE=1e-4`, `LFM_LORA__R=32` |
| `LFM_RUN_SLOW` | `0` skips every test needing a cached checkpoint (default `1` = run them) |
| `LFM_TEST_MODEL` | Override the test checkpoint id (default `LiquidAI/LFM2.5-350M-Base`) |
| `LFM_RESOLVED_DEVICE` / `LFM_RESOLVED_DTYPE` | Set by the trainer before building `SFTConfig`; read by `training/args.py`. Do not set by hand. |
| `LOG_DEBUG_LIBS=1` | Raise third-party loggers from WARNING to DEBUG |
| `NO_COLOR=1` | Disable ANSI colour |
| `HF_HOME` | Relocate the multi-GB model cache |
| `WANDB_API_KEY` / `WANDB_PROJECT` / `WANDB_ENTITY` / `WANDB_MODE` | Optional W&B; silently skipped when the key is absent |

---

## 4. The staged workflow

Every stage is independently runnable. **A failed run must never require re-running earlier stages.**
This is the design's point.

| Stage | Command | Answers |
|---|---|---|
| — | `preflight` | Am I ready at all? Runs gates and prints GO / NO-GO |
| 0 | `doctor`, `inspect-hardware` | Is the environment sane? |
| 1 | `inspect-model` | Does the model match what I think? |
| 2 | `validate`, `prepare`, `split-stats` | Is the data usable and leak-free? |
| 3 | `inspect-template` | Is the loss mask correct? |
| 4 | `inspect-lora` | Are the LoRA modules right? |
| 5 | `train --smoke-test` | Does it train end to end? |
| 6 | `train --pilot` | Is memory stable, does eval loss behave? |
| 7 | `train` | Full SFT |
| 8 | `baseline`, `evaluate` | Did it actually improve? |
| 9 | `export-adapter`, `merge-adapter` | Can I ship it? |
| — | `inference` | Interactive / batched generation |
| util | `show-config`, `recommend-model`, `init-synthetic`, `stages` | |

**All 19 CLI commands:** `doctor`, `inspect-hardware`, `inspect-model`, `validate`, `prepare`,
`split-stats`, `inspect-template`, `inspect-lora`, `train`, `baseline`, `evaluate`, `inference`,
`export-adapter`, `merge-adapter`, `show-config`, `recommend-model`, `init-synthetic`, `preflight`,
`stages`. Plus global `-v/--verbose` and `--version`.

Every config-taking command accepts `-c/--config` and repeatable `-s/--set KEY.PATH=VALUE`.
Most accept `-o/--output <file.json>` and `--json`.

---

## 5. Architecture and layering

```text
cli  ->  config  ->  data  ->  model  ->  training  ->  evaluation
                    |         |            |
                    +---------+            +--> inference, export
                                  utils (used by all)
```

**Invariant:** nothing in `data`, `model`, `training`, `evaluation`, `inference`, `export` imports
the CLI. This lets tests exercise any layer directly and keeps `scripts/*.py` trivial.

### Module responsibilities (with LOC)

| File | LOC | Responsibility |
|---|---|---|
| `cli/main.py` | 1065 | argparse wiring for 19 commands; `_emit`, `_resolve_adapter`, `cmd_*` |
| `config/schema.py` | 839 | 14 pydantic sections, `PRESETS`, cross-field validators, `env_overrides`, `deep_merge` |
| `config/loader.py` | 231 | `extends` chain resolution, provenance, `save_resolved`, `describe` |
| `training/trainer.py` | 684 | 7-stage `train()`, `preflight()`, `_StreamingCallback`, `_verify_adapter_reload` |
| `model/loader.py` | 543 | dtype/device resolution, architecture gating, tokenizer probes, `recommend_model`, `freeze_base_model` |
| `preflight.py` | 529 | `run_preflight()` returning `Check` objects with PASS/FAIL/WARN/SKIP |
| `training/tokenization.py` | 507 | **`ChatFormatter` — the correctness core** |
| `export/adapter.py` | 495 | `export_adapter`, `verify_adapter_loads`, `merge_adapter` |
| `utils/hardware.py` | 475 | `audit()`, `probe_gpus()`, `estimate_training_vram_bytes`, `assert_vram_sufficient` |
| `evaluation/metrics.py` | 471 | rubric, `score_generation`, `token_f1`, `summarise`, `compare_summaries`, `hallucination_flags` |
| `data/pipeline.py` | 457 | `load_and_clean()` + `prepare_dataset()` — the 10-step ordered pipeline |
| `evaluation/runner.py` | 450 | `evaluate_model`, `compare_variants`, `run_baseline_evaluation`, `run_comparison_evaluation` |
| `data/io.py` | 393 | `iter_raw_records`, `detect_format`, `convert_record`, manifest load/validate |
| `model/lora.py` | 377 | `resolve_target_modules`, `build_peft_config`, `attach_lora`, `audit_trainable_parameters` |
| `data/validate.py` | 354 | `TextNormalizer`, `QualityConfig`, `validate_conversation`, `_find_prompt_leakage` |
| `model/discovery.py` | 339 | `inspect_modules`, `discover_lora_targets`, `OFFICIAL_REFERENCE`, `EXCLUDED_BY_DEFAULT` |
| `data/synthetic.py` | 290 | deterministic fixture generator |
| `training/metadata.py` | 281 | `RunDirectory`, `resolve_reporting`, `local_metric_summary` |
| `inference/cli_loop.py` | 257 | `InferenceEngine`, REPL, `run_batch` |
| `data/dedupe.py` | 235 | `exact_key`/`prompt_key`/`answer_key`, `NearDuplicateIndex`, overlap audit |
| `training/collator.py` | 221 | `tokenize_conversations`, `SFTCollator`, `TorchDatasetAdapter` |
| `inference/generator.py` | 219 | `GenerationConfig`, `Generator` (left-padded batches), `run_generation_sweep` |
| `data/schema.py` | 217 | `Message`, `Conversation`, `Rejection`, `DatasetStats`, `COUNT_KEYS` |
| `training/args.py` | 215 | `ExperimentConfig` → `trl.SFTConfig`; `describe_training_plan` |
| `data/split.py` | 215 | `build_plan` (hashed/sequential), `split_conversations`, `verify_no_cross_split_duplicates`, `subsample` |
| `data/stats.py` | 210 | token-level stats with the real tokenizer; `dataset_warning` |
| `utils/*` | ~630 | errors / logging / seeding / hashing / io / versions |
| `evaluation/__init__.py`, `data/__init__.py`, … | small | re-exports |

### Eight load-bearing design decisions

1. **LoRA targets are *discovered*, never hard-coded.** `lora.target_modules: auto` walks
   `model.named_modules()`, classifies each `Linear` by role, and cross-checks against Liquid's own
   presets. Two independent sources must agree.
2. **Loss masking by prefix length**, with the prefix property *asserted*, because LFM2.5 base
   checkpoints lack `{% generation %}` blocks (§8).
3. **Pre-tokenise in Python**; do not let TRL prepare the dataset. `dataset_kwargs:
   {"skip_prepare_dataset": True}` + a hand-written `SFTCollator`.
4. **Deduplicate globally, THEN split.** Dedup-after-split leaves duplicate pairs across the
   boundary — the single most common cause of a suspiciously good validation curve.
5. **Every failure is typed and carries a fix.** One exception class per failure mode, each with a
   message *and* a remediation hint.
6. **Run directories are self-describing.** `resolved_config.yaml` + `run_metadata.json` +
   `metrics.jsonl` + `train.log` + `checkpoint-N/` + `adapter/`. A finished run is reproducible and
   auditable from its own directory.
7. **Local logging is unconditional.** `metrics.jsonl` is written whether or not an integration is
   configured or a console sink is attached. No external service is ever required.
8. **The adapter is the primary artifact.** Tens of MiB, pins the base checkpoint, survives
   re-training. Merging is a verified convenience for runtimes that cannot load an adapter.

---

## 6. Configuration system

`config/schema.py` defines 14 pydantic `BaseModel` sections, all `_Base` with
`extra="forbid", validate_assignment=True` (unknown keys → typo-catching errors).

| Section | Class | Notable fields |
|---|---|---|
| `run` | `RunIdentity` | `name` (pattern `^[A-Za-z0-9][A-Za-z0-9._-]*$`), `domain`, `notes`, `tags` |
| `model` | `ModelConfig` | `name`, `revision`, `dtype`, `device`, `device_map`, `trust_remote_code`, `attn_implementation`, `gradient_checkpointing`, `use_cache`, `low_cpu_mem_usage`, `allow_unsupported_architecture`, `local_files_only` |
| `dataset` | `DatasetConfig` | `path`, `format`, `text_column`, `system_prompt`, splits, `use_processed`, `processed_dir`, `max_train_examples`, `max_eval_examples` |
| `dataset.quality` | `QualityGates` | see §7 thresholds |
| `dataset.splitting` | `SplittingConfig` | `strategy`, `train_ratio`, `validation_ratio`, `seed` |
| `tokenization` | `TokenizationConfig` | `max_seq_length`, `chat_template`, `truncation_strategy`, `padding_side`, `append_eos_token`, `add_generation_prompt_in_training`, `compute_token_stats_on` |
| `lora` | `LoraConfigModel` | `enabled`, `r`, `alpha`, `dropout`, `bias`, `target_modules` (`list \| "auto"`), `exclude_modules`, `modules_to_save`, `use_rslora`, `use_dora`, `task_type`, `init_lora_weights`, `layers_to_transform`, `max_trainable_pct`, `min_trainable_params` |
| `training` | `TrainingConfigModel` | `preset`, `seed`, epochs, batch sizes, `gradient_accumulation_steps`, LR, schedule, clipping, eval/save strategies, `early_stopping_patience`, `optim`, `resume_from_checkpoint`, `report_to`, `packing`, `assistant_only_loss`, `sample_generation_steps`, `smoke_test`, `stop_after_steps` |
| `evaluation` | `EvaluationConfigModel` | `num_samples`, `max_new_tokens`, `temperature`, `top_p`, `top_k`, `repetition_penalty`, `do_sample`, `batch_size`, `num_beams`, `seed`, `rubric_path`, `leakage_probe` |
| `logging` | `LoggingConfigModel` | `level`, `log_dir`, `per_run_file`, `jsonl_metrics`, `capture_warnings` |
| `checkpointing` | `CheckpointingConfigModel` | `dir`, `keep_last`, `save_adapter_only`, `save_tokenizer`, `save_resolved_config`, `metadata_filename` |
| `export` | `ExportConfigModel` | `adapters_dir`, `merged_dir`, `safe_serialization`, `max_shard_size`, `push_to_hub`, `hub_repo_id`, `merge_in_bf16`, `export_tokenizer` |
| `reproducibility` | `ReproducibilityConfigModel` | `seed`, `dataset_split_seed`, `deterministic_algorithms`, `record_environment/hardware/git`, `hash_dataset`, `save_seed_state` |

Type aliases: `Dtype = auto|bfloat16|float16|float32`; `Device = auto|cuda|cpu|mps`;
`SplitName = train|validation|test`; `HParamPreset = pilot|balanced|memory_saver|full|smoke`.

### Inheritance

`extends:` (string or list) merges parents **before** the child. Resolution order: child dir → CWD →
`configs/` → `finetune/configs/`. Circular chains and depth > 32 raise `ConfigError`.

`deep_merge(base, overlay)` rules — **these are deliberate and load-bearing**:

- Dicts merge recursively.
- **Lists are REPLACED, not concatenated.** Concatenating `lora.target_modules` across a chain would
  silently grow the target set — the exact failure this project exists to prevent.
- `key: null` in a child **resets to the schema default** (pops the key).

**Precedence (low → high):**
`pydantic dataclass defaults < base.yaml < lfm_sft.yaml < agriculture/*.yaml < LFM_* env vars < --set KEY.PATH=VALUE`

`training.preset` is applied **after** the YAML merge but **before** explicit keys in the same
`training:` block, so an explicit key always beats the preset. Presets may only set keys that exist
on `TrainingConfigModel`; a preset key landing elsewhere is rejected at load time with a message
naming the preset (not the typo).

### Presets (`ExperimentConfig.PRESETS`)

| Preset | batch × accum | epochs | max_steps | eval/save | `load_best_model_at_end` |
|---|---|---|---|---|---|
| `smoke` | 1×1 | 1 | 4 | epoch | false |
| `pilot` | 2×4 | 1 | 50 | steps/25 | true |
| `balanced` | 4×4 | 3 | — | steps/50 | true |
| `memory_saver` | 1×16 | 3 | — | steps/50 | true |
| `full` | 8×4 | 3 | — | steps/100 | true |

**None is claimed to be optimal.** They are starting points from LoRA-SFT practice and Liquid's
reference presets.

### Cross-field validators (`validate_semantics`)

1. `dataset.path` required unless `dataset.use_processed=true`.
2. LoRA: `alpha > 0`; `r > alpha*4` without `use_rslora` → error; `modules_to_save` containing
   `lm_head` → **refused outright** (breaks weight tying on LFM2.5-*-Base).
3. `dataset.splitting.seed` **must equal** `reproducibility.dataset_split_seed`.
4. `evaluation.do_sample=true` with `temperature=0.0` → error (transformers raises).
5. Effective batch (`per_device × grad_accum`) > 512 → error.

`SplittingConfig` additionally rejects `train_ratio + validation_ratio > 1.0`.
`TrainingConfigModel` additionally: derives `eval_steps`/`save_steps` when null, and rejects
`load_best_model_at_end=true` when `eval_strategy` or `save_strategy` is `"no"`.

### Derived helpers

- `ExperimentConfig.to_dict()` → JSON-safe dict, `PRESETS` excluded.
- `ExperimentConfig.fingerprint()` → `hash_obj(to_dict())`; recorded in every artifact.
- `ExperimentConfig.resolve_output_dir()` → `<checkpointing.dir>/<run.name>_<YYYYmmdd-HHMMSS>` (UTC).
- `LoraConfigModel.effective_scaling` → `alpha/sqrt(r)` if `use_rslora` else `alpha/r`.
- `TrainingConfigModel.effective_batch_size_multiplier` → `per_device × grad_accum`.
- `env_overrides(prefix="LFM_")` parses `LFM_SECTION__KEY=value`; `_coerce_env` handles
  true/yes/on, false/no/off, null/none/``, else `json.loads`, else raw string.
- `parse_set_args(["training.learning_rate=1e-4"])` → nested dict, JSON-coerced.
- `config/loader.describe(config)` → the multi-line banner printed before training.

---

## 7. Data pipeline

### Canonical record

```json
{
  "id": "agri-000123",
  "messages": [
    {"role": "system",    "content": "<domain system prompt>"},
    {"role": "user",      "content": "My tomato plants have dark brown rings on lower leaves."},
    {"role": "assistant", "content": "..."}
  ],
  "metadata": {
    "domain": "agriculture", "category": "diagnosis_style_qa",
    "crop": "tomato", "plant": "Solanum lycopersicum",
    "disease": null, "pest": null, "nutrient_issue": null,
    "growth_stage": "flowering", "symptom": ["dark brown concentric lesions"],
    "region": "IN-KA", "climate": "tropical_subhumid", "soil": "red_soil",
    "irrigation": "drip_3x_week", "treatment": "remove_infested_leaves",
    "confidence": 0.7, "source": "icar-extension-publication",
    "license": "CC-BY-4.0", "language": "en"
  }
}
```

`datasets/schema.json` is the machine-readable version: `required: ["messages"]`, `messages`
`minItems: 2 / maxItems: 32`, roles `system|user|assistant|tool`, plus `allOf` constraints requiring
≥1 user turn and ≥1 assistant turn after the first user turn.

**Metadata is NEVER placed in the prompt** by default. `_inject_metadata()` exists as an explicit
opt-in (`dataset.inject_metadata`) that prefixes the *first user turn* (never the system prompt) with
`[context] crop: …; region: …`. If you use it, you must train **and** evaluate that way or the A/B
comparison is invalid.

### Accepted input shapes (`dataset.format: auto` sniffs per record)

| Input | Normalised to |
|---|---|
| `{"messages": [...]}` | as-is |
| `{"conversations": [{"from": "human", "value": "..."}]}` | ShareGPT shape, role aliases mapped |
| `{"role": ..., "content": ...}` | single-turn conversation |
| `{"instruction", "input", "output"}` | `input` **appended** to `instruction` with `\n\n` |
| `{"prompt", "completion"}` | user / assistant |
| `{"question", "answer"}` | user / assistant |
| `{"input", "output"}` | user / assistant |

Role aliases: `human|usr|prompter → user`; `gpt|bot|chatbot|model → assistant`;
`function|observation|function_call → tool`.

Multimodal content blocks collapse to their `text` fields. Native `tool_calls` are preserved (the
LFM2.5 template renders them, so training on them teaches the model's own format).

`dataset.system_prompt` is injected as `messages[0]` **only when the record has none**.

`iter_raw_records` walks directories in sorted order, accepts `.jsonl`/`.json`/`.jsonlines`, and
**skips metadata files**: `manifest.json`, `manifest.yaml`, `manifest.yml`, `schema.json`,
`readme.json`, `_meta.json`. Lines starting with `//` and blank lines are skipped. Malformed JSON
raises `DatasetError` naming the **line number**.

### The 10-step pipeline (`prepare_dataset`) — this ORDER IS LOAD-BEARING

1. read raw records
2. convert to canonical `Conversation` (format sniffing)
3. text normalisation (`TextNormalizer`)
4. structural + quality validation — **record every rejection**
5. **global exact deduplication** (SHA-256)
6. **global near-duplicate detection** (char 5-gram Jaccard)
7. deterministic content-hashed split
8. cross-split leakage audit (hard fail on exact duplicates)
9. token-level statistics + advisory warnings
10. write `validated/`, `splits/`, and a machine-readable report

Deduplication **before** splitting is the point: dedup-after-split still leaves duplicate pairs
straddling the boundary.

### Normalisation (`TextNormalizer`, applied *before* dedup)

- strip lone surrogates, C0/C1 control chars, zero-width chars
- **NFC** unicode folding (so `café` decomposed == precomposed, so dedup works)
- CRLF/CR → LF
- HTML tag removal + common entity decoding
- collapse multi-spaces, strip trailing whitespace, collapse 3+ newlines to 2

### Quality gates

Defaults in `configs/base.yaml` → agriculture overrides in `_domain.yaml`:

| Setting | base.yaml | agriculture `_domain.yaml` |
|---|---|---|
| `min_user_chars` | 2 | **4** |
| `min_assistant_chars` | 8 | **80** |
| `max_user_chars` | 8000 | (inherits) |
| `max_assistant_chars` | 16000 | **12000** |
| `min_messages` | 2 | 2 |
| `max_turns` | 32 | **16** |
| `near_duplicate_threshold` | **0.90** | **0.85** (tightened, never loosened) |
| `deduplicate` / `near_duplicate` / `remove_prompt_leakage` / `check_train_val_overlap` | true | true |
| `normalize_whitespace` / `strip_control_characters` / `strip_html` | true | true |

`base.yaml` placeholders: `\bTODO\b`, `\bFIXME\b`, `\bXXX+\b`, `\bLorem ipsum\b`, `\bplaceholder\b`,
`\bTBD\b`, `<\|.*?\|>`, `\[insert .*?\]`.
`_domain.yaml` placeholders add `\bN/?A\b` and `\b(as an AI|I am an AI language model)\b`, drop
`XXX/TBD/[insert]`.

**Rejection reasons** (machine-readable, every rejected row written to `rejected.jsonl` with an
excerpt):

| Reason | Trigger |
|---|---|
| `unrecognised_format` | no message list and no instruction pair |
| `invalid_role` | role outside the known set |
| `missing_turn` | no user turn, or no assistant turn after the first user turn |
| `multiple_system_prompts` | >1 system message (template only honours `messages[0]`) |
| `all_messages_empty` | everything empty after normalisation |
| `empty_user_message` | user turn below `min_user_chars` |
| `empty_assistant_response` | answer below `min_assistant_chars` |
| `assistant_response_too_long` | above `max_assistant_chars` |
| `too_many_turns` / `too_few_messages` | outside bounds |
| `placeholder_text` | matches a `placeholder_patterns` regex |
| `prompt_leakage` | answer contained in prompt, **or** ≥2 shared 12-word spans |
| `duplicate_exact` | identical normalised conversation |
| `duplicate_near` | char 5-gram Jaccard ≥ threshold |

**Note on ordering inside `validate_conversation`:** `drop_empty_messages` runs *before* the length
gates, so a whitespace-only user turn is reported as `missing_turn`, not `empty_user_message`.

### Prompt-leakage detection (two cheap signals)

1. the whole normalised answer appears inside a user turn (answer ≥ 40 chars);
2. ≥2 identical **12-word** shingles shared between the answer and the prompts (answer ≥ 24 words).

Signal 2 catches templated datasets where the assistant restates the question before answering.

### Deduplication

- **Exact** — `exact_key`: SHA-256 over `canonical_json([{role, content}, ...])`.
- **Near** — `NearDuplicateIndex`: character 5-gram shingles (BLAKE2b-4 hashed) over an **inverted
  index** with a shared-shingle pre-filter (`shared >= threshold * len(query)`), giving O(candidates)
  rather than O(n²). Character shingles (not words) survive spelling variation and Hindi/English
  code-switching.
- **`conversation_text(conv)` EXCLUDES the system message by default.** This is not cosmetic: every
  record shares the ~350-char MICO system prompt, which would contribute ~350 identical shingles to
  every pair, inflating Jaccard corpus-wide and silently deleting legitimate examples. There are
  three dedicated regression tests for this bug.
- **Shared prompt, different answer** → kept, flagged `metadata._quality_flags: ["shared_prompt"]`.
- **Same answer to >3 different prompts** → flagged `repeated_answer`.
- `prompt_key` = hash of all turns except the last; `answer_key` = hash of the last turn.

### Splitting

- `hashed` (default): `BLAKE2b(seed ‖ exact_key) → uniform float`, compared against
  `[0, train_ratio)` / `[train_ratio, train+val)` / rest. **Content-addressed**: appending examples
  never moves an existing example between splits, so a recorded validation score stays valid.
- `sequential`: faster, fragile under growth. Escape hatch only.
- `train_ratio + validation_ratio <= 1.0` enforced; remainder is the test split.
- **Empty train or validation is a hard error** (`DatasetLeakageError`), never a silent 0-row run.
- `subsample(list, limit, seed)`: seeded shuffle then re-sort by original index → stable subset.
- `find_cross_split_overlap`: `exact` overlaps are **hard failures**; `prompt`/`answer` overlaps are
  reported as *soft* warnings (a shared short answer like "Yes, water twice a week" is benign).

### Statistics & warnings

`compute_stats` / `multi_split_stats` use the **real tokenizer + chat template** when one is supplied
(`--no-token-stats` skips it). Reports `num_examples`, `total_tokens`, `mean/median/p95/max_tokens`,
`roles`, `categories`, `crops`, `languages`, `turns_histogram`, `train_validation_token_ratio`.

`dataset_warning` fires when: < `min_examples` (200) examples, `mean_tokens < 64`,
`mean_assistant_chars < 120`, or anything hits `max_seq_length`. Additionally, if
`num_train_epochs > 1` and `num_examples < 200`, an explicit memorisation warning is appended.

### Provenance manifest

Placed **next to the raw corpus** as `manifest.yaml`, `manifest.yml`, or `manifest.json` (searched in
the dataset directory, else the file's parent).

Required (per `datasets/manifest.schema.json`): `name`, `version`, `source`, `license`, `created`,
`num_examples`, `content_hash`. Strongly recommended: `categories`, `language`, `annotation_method`,
`review_status`, `known_limitations`, `synthetic`.

`prepare` auto-fills `num_examples`, `content_hash`, `prepared_at`; `validate_manifest` returns
problems for `license ∈ {unknown, unspecified, "", none, tbd}`, missing `source`/`created`/
`num_examples`. `preflight` **hard-fails** on an unknown/missing licence.

Accepted licence values: `CC0-1.0`, `CC-BY-4.0`, `CC-BY-SA-4.0`, `ODbL-1.0`, `MIT`, `proprietary`,
`internal-use-only`. **`CC-BY-ND` is not usable as training data.** If sources carry different
licences, keep **separate corpora with separate manifests** — merging makes the strictest licence govern.

### The MICO agriculture categories (12)

`diagnosis_style_qa`, `symptom_reasoning`, `crop_management`, `pest_identification`,
`nutrient_deficiency`, `irrigation_guidance`, `weather_risk_reasoning`, `farmer_conversational`,
`clarification_questions`, `safety_uncertainty`, `indian_terminology`, `hindi_english_mixed`.

The engine treats all categories identically; they exist for coverage auditing and evaluation slicing.

### The MICO behaviour contract (`_domain.yaml` system prompt)

Seven numbered rules, verbatim in the config:

1. **Diagnose, never guess.** Name the 2–3 most plausible causes and say which detail would separate them.
2. **Ask for what you are missing.** Crop, growth stage, region/season, symptom distribution on the plant, recent weather/irrigation/input changes.
3. **State uncertainty explicitly.**
4. **Recommend safe, practical actions** available to a smallholder, as ordered steps with timing.
5. **Respect safety and legal limits.** Name the active ingredient, mention pre-harvest intervals and protective equipment, note local restrictions.
6. **Know your limits.** For an unidentifiable pest/disease or a fast-spreading problem, route to a local extension officer or Kisan Call Centre.
7. **Be concise and plain.** Short sentences, no jargon without a plain-language gloss.

**Critical consistency rule:** if the system prompt says "ask before diagnosing" and 40% of the
training answers diagnose without asking, **the model learns the dataset, not the prompt.**

### Dataset sizing

| n | What it supports |
|---|---|
| < 200 | Pipeline validation only; memorisation guaranteed at 3 epochs |
| 200 | Model learns the *style* only |
| 300–500 | First pilot |
| 500 | First real experiment (Liquid's guidance floor: 500–5,000) |
| 800–1,500 | Genuine domain adaptation |
| 2,000+ | Broad, robust adaptation |
| 3,000 over 2 crops | "The model believes there are 2 crops. **Coverage beats volume.**" |

### Synthetic fixture

`lfm_finetune.data.synthetic` generates arithmetic / string / list tasks with **computed** answers,
plus 3 deliberate edge cases (multi-turn, Unicode `café-नमस्ते`, long). Manifest: `license: CC0-1.0`,
`synthetic: true`, provenance note containing `SYNTHETIC TEST FIXTURE`.
`placeholder_agriculture_records()` returns schema demos marked `placeholder: true, synthetic: true`
with text like `[PLACEHOLDER] …` that asserts **no agronomic facts**. **Never train on it.**

---

## 8. Model loading, LFM2.5 facts and LoRA

### Supported models

| Model id | Params | bf16 weights | Use |
|---|---|---|---|
| `LiquidAI/LFM2.5-230M-Base` | 0.23 B | ~0.5 GiB | Ultra-low-end, CI |
| `LiquidAI/LFM2.5-350M-Base` | 0.35 B | ~0.7 GiB | Low-end edge, CPU validation (**smoke.yaml**) |
| **`LiquidAI/LFM2.5-1.2B-Base`** | **1.17 B** | **~2.4 GiB** | **MICO primary target** |
| `LiquidAI/LFM2.5-2.6B-Base` | 2.6 B | ~5.2 GiB | Edge gateway |
| `LiquidAI/LFM2.5-8B-A1B-Base` | 8 B (MoE, 1B active) | ~16 GiB | Server; **not** an edge target |

**Base, not Instruct.** Liquid's model card recommends `*-Base` for "heavy fine-tuning …
domain-specific (e.g. medical) assistants, training on proprietary data". Starting from Instruct
means using only domain data to un-learn a general instruct prior.

`SUPPORTED_ARCH_PREFIXES = ("Lfm2", "Lfm2ForCausalLM")`;
`SUPPORTED_MODEL_TYPES = ("lfm2", "lfm2-moe", "lfm")`. Anything else raises `ModelError` unless
`model.allow_unsupported_architecture=true`.

### Architecture: LFM2.5 is HYBRID

`Lfm2DecoderLayer` **alternates** `Lfm2ShortConv` (depthwise `Conv1d` sandwiched between two
`nn.Linear`) with `Lfm2Attention` (GQA). `config.layer_types` decides per layer → the module
inventory legitimately differs between layers of the same model. On 1.2B: **10 conv + 6 attention**
layers. Conv blocks outnumber attention blocks — a test asserts this.

### Discovered LoRA targets (`lora.target_modules: auto`)

```text
feed_forward.w1        x16   SwiGLU gate
feed_forward.w2        x16   SwiGLU down
feed_forward.w3        x16   SwiGLU up
self_attn.q_proj       x6   GQA query
self_attn.k_proj       x6   GQA key
self_attn.v_proj       x6   GQA value
self_attn.out_proj     x6   GQA output     <-- NOT "o_proj"
conv.in_proj          x10   short conv in
conv.out_proj         x10   short conv out
                       ---
                       92 instances, 9 module types
```

Cross-checked against `model/discovery.py::OFFICIAL_REFERENCE` (GLU + MHA + CONV, sourced from
[leap-finetune](https://github.com/Liquid4All/leap-finetune)). The report prints
`reference check  exact match with Liquid AI leap-finetune preset`.

**Five facts that break a generic recipe:**

1. `out_proj`, **not** `o_proj`. The documented TRL example `["q_proj","k_proj","v_proj","o_proj"]`
   matches nothing on the output projection.
2. SwiGLU `feed_forward.w1/w2/w3`, not `gate_proj/up_proj/down_proj`.
3. `conv.in_proj` / `conv.out_proj` **are** legitimate PEFT targets (`nn.Linear` inside `Lfm2ShortConv`).
4. `lm_head` is excluded — `config.tie_embedding: true` makes `lm_head.weight` **the same tensor** as
   `model.embed_tokens.weight`.
5. `all-linear` / substring matching is **never** used.

`EXCLUDED_BY_DEFAULT = ("lm_head", "embed_tokens", "wte", "shared", "norm", "ln_", "gate.weight")`.

`discover_lora_targets` raises `ModelError` when nothing PEFT-compatible is found. Explicit target
lists are **verified to exist**; missing names produce an error listing the real suffixes and naming
`self_attn.out_proj` and `feed_forward.w1` as the correct spellings.

`build_peft_config` **refuses** to be called with `target_modules="auto"` and no resolved list —
handing PEFT the literal string makes it search for a module named `"auto"`.

### Parameter audit (`audit_trainable_parameters`) — refuses to train unless all pass

- `trainable > lora.min_trainable_params` (default **1024**)
- `trainable_pct <= lora.max_trainable_pct` (default **25 %**)
- LoRA A **and** B matrices both non-zero
- **no non-LoRA parameter is trainable** (`modules_to_save` excepted)
- a warning is emitted if any trainable param is neither `lora_*` nor `modules_to_save`

Measured on `LFM2.5-350M-Base`:

```text
total parameters      357,482,240
trainable parameters    2,998,272   (0.839 %)
frozen parameters     354,483,968
  LoRA A matrices       1,212,416
  LoRA B matrices       1,785,856
```

Projected for 1.2B (same target set): r=8 → ~9.9 M (0.85 %); **r=16 → ~19.8 M (1.69 %)**; r=32 → ~39.6 M (3.38 %).

### Loading behaviour (`model/loader.py`)

- Order of checks, cheapest first: tokenizer → `AutoConfig` → architecture gate → dtype/device →
  weights → context length vs `max_seq_length`.
- `_resolve_dtype("auto")`: `bfloat16` on a GPU where **all** GPUs support bf16, `float16` on
  pre-Ampere, `float32` on CPU.
- On CPU with a non-fp32 dtype requested → warns and forces `float32`.
- `flash_attention_2` falls back to `sdpa` with a warning if `flash_attn` is not importable.
- Context gate: `max_seq_length > max_position_embeddings` → `ModelError`. LFM2.5 declares 128,000
  but was **trained on 32,768**; a warning fires above 32,768.
- `gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})` and
  `use_cache=False` during training.
- `freeze_base_model(model)` **must** be called: transformers ships every parameter with
  `requires_grad=True`. It counts unfrozen, calls `requires_grad_(False)`, then asserts none remain.
- Weight tying is verified at runtime via `lm_head.weight is embed_tokens.weight`.

### VRAM estimation (`utils/hardware.py`)

```text
weights    = param_count * dtype_bytes
gradients  = trainable_params * dtype_bytes
optimizer  = trainable_params * 4 * 2      # AdamW m, v in fp32
activations= batch*seq * hidden(=2048) * dtype_bytes * (2 if grad_ckpt else 12)
logits     = batch*seq * 65536 * 4          # fp32 upcast for CE loss — often the LARGEST term
```

`assert_vram_sufficient` fails fast with a generated list of concrete reductions
("lower `per_device_train_batch_size` (currently N; gradient accumulation preserves the effective
batch size)", "lower `tokenization.max_seq_length`", "use bfloat16 instead of float32").
`recommend_model` ranks checkpoints against **system RAM on a CPU host** and **VRAM on a GPU host**,
sorted by `(not feasible, params)`; anything over `budget * 0.85` is not feasible.

---

## 8b. THE CRITICAL LFM2.5 FINDINGS — measured, not assumed

Measured against real checkpoints with **transformers 5.3.0**. Reproduce with
`uv run lfm-finetune inspect-template -c configs/smoke.yaml --samples 2`.

### Finding 1 — templates are NOT uniform across sizes

| Checkpoint | template len | `{% generation %}` | `return_assistant_tokens_mask=True` |
|---|---|---|---|
| `LFM2.5-230M-Base` | 4621 | **yes** | works (7 tokens, answer covered) |
| **`LFM2.5-350M-Base`** | 1296 | **no** | **0 tokens masked** |
| **`LFM2.5-1.2B-Base`** | 1296 | **no** | **0 tokens masked** |
| `LFM2.5-2.6B-Base` | 4294 | yes | works |
| `LFM2.5-8B-A1B-Base` | 4294 | yes | works |
| `LFM2.5-1.2B-Instruct` | 5487 | yes | works |

Consequence on the primary target:

```python
tokenizer.apply_chat_template(msgs, tokenize=True, return_assistant_tokens_mask=True)
# -> an ALL-ZERO mask
```

Anyone using TRL `assistant_only_loss=True` on this checkpoint **trains on 100 % of tokens while
believing the prompt was masked.** This is the single most damaging silent bug in SFT. It is why
this pipeline masks by prefix length.

### Finding 2 — the rendered form

Every LFM2.5 template wraps each turn in ChatML control tokens: `<|im_start|>` (**id 6**) before and
`<|im_end|>` (**id 7**) after, with role name and content separated by a newline:

```text
<|startoftext|>system
SYS
user
USER
assistant
ASST
```

`eos_token_id` **is** 7 on the 65,536-vocab checkpoints (230M/350M/1.2B) and **124900** on the
128k ones (2.6B/8B-A1B). **Never hard-code it — read `tokenizer.eos_token_id`.**

Gotcha: `special_tokens_map.json` registers ids 6/7 with an **empty** `content` field, so reading the
raw JSON makes the terminator look absent. The loaded tokenizer resolves the real marker; the **id**
is what stops generation.

### Finding 3 — `add_special_tokens=False` is MANDATORY

The template emits `{{ bos_token }}` itself, so the tokenizer's post-processor prepends a **second** BOS:

```python
tokenizer(rendered, add_special_tokens=False)["input_ids"][:3]  # [1, 6, 24131]   correct
tokenizer(rendered)["input_ids"][:3]                            # [1, 1, 24131]   WRONG
```

`ChatFormatter.encode` always passes `add_special_tokens=False`.

### Finding 4 — the trailing newline must be stripped

The template ends the rendering with `<|im_end|>` **followed by a literal newline**. Inference stops
the instant the model emits `<|im_end|>`, so the newline is never produced. Training on
`answer, <|im_end|>, "\n"` teaches the model to emit the stop token and then keep going.
`ChatFormatter._strip_trailing_newlines` removes it (whitespace only, never below `prompt_len`;
`<|im_end|>` decodes to a non-whitespace string, so the scan stops and a genuinely-empty final answer
is not truncated away).

### Finding 5 — prefix consistency is an ASSUMPTION you must check

`apply_chat_template(messages[:-1], add_generation_prompt=True)` **must** reproduce the beginning of
`apply_chat_template(messages, add_generation_prompt=False)`. `format_conversation` raises
`TokenizerError("Chat template is not prefix-consistent")` if it does not.

---

## 8c. `ChatFormatter` — the correctness core (`training/tokenization.py`)

```text
prompt_ids = template(messages[:-1], add_generation_prompt=True)
full_ids   = template(messages,     add_generation_prompt=False)
if full_ids[:len(prompt_ids)] != prompt_ids:  raise TokenizerError("not prefix-consistent")
strip trailing whitespace-only tokens (never below prompt_len)
append eos_token_id if not already last
labels = [-100] * prompt_len + full_ids[prompt_len:]
if count(labels != -100) == 0:  raise TokenizerError("zero tokens contributing to the loss")
```

`_build_labels` deliberately does **not** shift: HF models shift internally, so `labels[i] ==
input_ids[i]` is correct and masking position `i` removes the prediction of token `i`.
With `assistant_only_loss=False`, `labels = input_ids` with `labels[0] = -100`.

**Truncation `keep_head_tail` — exact formula:**

```text
reserve = max(1, min(answer_len, limit // 2))     # reserve budget for the supervised span
remaining = limit - reserve
head    = min(prompt_len, remaining // 2)         # system prompt + first user turn
tail    = remaining - head                        # end of prompt, keeps generation prompt adjacent
result  = ids[:head] + ids[len(ids) - tail - reserve:]
```

This guarantees **at least one supervised token survives**. An earlier head+tail implementation
produced an all-masked example whenever the prompt alone exceeded `max_seq_length` — those records
trained on nothing while looking perfectly normal in dataset statistics. Regression test:
`test_long_prompt_still_produces_supervision`.

**Expected coverage.** On realistic MICO data with a long system prompt, supervised coverage of
**4–20 % is normal and correct**. A warning fires only above **95 %** — the signature of masking not
working.

**Other members:** `build_prompt_ids` (exact training prefix for `generate`), `generation_kwargs`
(returns **only** `eos_token_id`; `pad_token_id` is deliberately omitted because PEFT's generate
would then receive it twice), `strip_generated`, `format_example_for_display` (prints token id +
decoded piece + a marker column; blank marker = not supervised), `template_fingerprint`,
`verify_template_consistency` (per-example: prompt ids are a prefix, every prompt token masked, no
assistant token masked).

---

## 9. Training

`training/trainer.py::train()` runs 7 stages:

0. **preflight** — hardware audit, dataset presence, `load_model`, context gate, VRAM estimate
1. **dataset** — load splits; error if train is empty or (`load_best_model_at_end` and no validation)
2. **run directory** — create + freeze `resolved_config.yaml` + `setup_logging(log_file=train.log)`.
   Resuming continues the **SAME** directory so `metrics.jsonl` keeps one continuous history.
3. **formatter + samples** — `verify_template_consistency` over 16 train examples; **hard fail** if
   not `ok`. Then `tokenize_conversations` and print 3 fully-formatted examples with the loss mask.
4. **LoRA** — `attach_lora` → `ParameterAudit`; `--dry-run` stops here.
5. **trainer** — `set_seed`, `resolve_reporting`, `build_sft_config`, `SFTTrainer`
6. **train / resume**
7. **adapter save + reload verification** + final metadata

Key wiring details:

- `peft_config=None` is passed to `SFTTrainer` — **PEFT is already attached and audited** upstream.
- `processing_class=loaded.tokenizer` (not the removed `tokenizer=` kwarg).
- `dataset_kwargs={"skip_prepare_dataset": True}` and `max_length` set explicitly, because the data is
  pre-tokenised by this pipeline.
- `gradient_checkpointing_kwargs={"use_reentrant": False}`.
- `dataloader_pin_memory` forced off on CPU.
- `bf16`/`fp16` derived from `LFM_RESOLVED_DTYPE` (set from the *resolved* dtype), so a CPU run never
  claims mixed precision.
- `hf_metric` selection: `load_best_model_at_end` implies `metric_for_best_model=eval_loss`,
  `greater_is_better=False`.
- `_drop_printer_callback(trainer)` removes HF's `PrinterCallback`, which would otherwise print every
  metric line a **second** time with `%.4g` truncation.
- `_StreamingCallback(TrainerCallback)` writes `metrics.jsonl` **unconditionally** (registered even
  when no CLI sink is supplied). It **must** subclass `transformers.TrainerCallback` — the callback
  handler dispatches `on_log` on every registered callback.

### `SFTCollator` and `TorchDatasetAdapter`

`SFTCollator(pad_token_id, pad_to_multiple_of=8)`:
- right padding; pad positions get `attention_mask=0` and `labels=-100`
- validates `len(input_ids) == len(labels) == len(attention_mask)` per row
- raises `TokenizerError` on an empty batch

`TorchDatasetAdapter` yields **plain dicts of int lists**, not the `FormattedExample` dataclass,
because TRL's `SFTTrainer` probes `"prompt" in sample` and a non-iterable dataclass raises
`TypeError`.

### Checkpoint / resume

- `checkpoint-N/` must contain `trainer_state.json`, `optimizer.pt`, `scheduler.pt`, `rng_state.pth`
  and adapter weights, or `_resolve_resume` raises `CheckpointError("is incomplete")`.
- `--resume auto` → `run_dir.latest_checkpoint()` (highest numeric suffix).
- Passing an explicit `--output` **wins** over inferring the directory from the checkpoint path.
- The smoke SFT config deliberately uses `save_strategy="steps"` — with `max_steps=4` an epoch
  boundary may never be reached, and `save_strategy="no"` would leave the save/resume/reload path
  untested while still reporting success.

### Adapter reload verification

`_verify_adapter_reload` does **not** call `PeftModel.from_pretrained` on `loaded.model` (which is
already a `PeftModel` → double-wrap warning) and does **not** materialise a second copy of the base
weights (doubles peak memory at the worst moment). It checks for the weight file and
`adapter_config.json`, then calls `export.adapter.verify_adapter_loads`.

---

## 10. Evaluation

**The honesty rule, repeated in every artifact:**

> *A lower held-out loss is **not** evidence of improved domain capability.* It shows a better fit to
> the domain's text distribution — necessary, not sufficient. The report carries an
> `interpretation_guard` string saying exactly this, and the human rubric sheet ships with `null`
> scores so the missing judgement is visibly missing.

### What IS measured

- **Held-out loss** — mean causal-LM loss over the **test** split, assistant tokens only, using the
  *same* loss mask as training. Identically computed for base and tuned.
- **Objective proxies over generated text** (`score_generation`):

| Metric | Direction |
|---|---|
| `asks_for_information_rate` (hedge marker **or** any `?`) | higher better |
| `overconfidence_rate` / `mean_overconfidence_markers` | **lower** better |
| `mean_uncertainty_markers` | higher better |
| `any_format_rate`, `format_tag_counts` (bullets, numbered list, headings, key-value, bold, JSON) | higher better |
| `mean_token_f1_vs_reference` (unigram F1) | higher better — **lexical overlap, not correctness** |
| `verbatim_reference_match_rate` (≥60 chars reproduced) | **lower** better (memorisation signal) |
| `indian_terminology_rate` | higher better |
| `mentions_professional_help_rate` | higher better |
| `empty_rate`, `mean_questions_per_answer`, `mean/median/p95_length_words` | `n/a` / lower |

`compare_summaries` emits rows with `base`, `tuned`, `delta`, `higher_is_better`, `verdict`
(`improved`/`regressed`/`unchanged`/`n/a`) and a note that these describe objective proxies only.

- **Regressions** (`_detect_regressions`): token-F1 dropped > 0.25, stopped asking for information,
  gained overconfidence markers, lost output structure, produced an empty response.
- **Hallucination flags**: verbatim copy, confident assertion with no clarification request, empty
  generation, degenerate < 5-word generation.
- **Before/after examples** (up to 50).
- **Disjointness**: `assert_eval_disjoint` proves no evaluation prompt appears in the training split;
  a failure is logged as an error and invalidates the comparison.

### What is NOT measured (handed to a human)

Agronomic correctness, safety/chemical correctness, helpfulness/tone, Hindi/English fluency,
calibration. These go into `human_rubric_sheet` with all 7 dimensions pre-listed and anchored, and
**every score field `null`**.

### The 7-dimension MICO rubric (`DEFAULT_AGRICULTURE_RUBRIC`, scores 0–4)

`factual_correctness`, `no_hallucinated_diagnosis`, `clarification_behaviour`,
`recommended_action_quality`, `instruction_following`, `terminology`, `safety_and_uncertainty`.
Each has anchors for 0 / 2 / 4 and an optional `automatic_proxy` naming the correlated metric (triage
aid only). Override with `evaluation.rubric_path`. **Two independent scorers recommended.**

### Protocol

```bash
uv run lfm-finetune baseline  -c <config>   # BEFORE training, on the test split
uv run lfm-finetune train    -c <config>
uv run lfm-finetune evaluate -c <config>   # base vs tuned, same split, same decoding
```

Decoding **must be deterministic** (`do_sample: false`, greedy). "Sampling noise at n=50 swamps the
effect sizes you are looking for."

Outputs in `evaluations/<run>/`: `comparison.json`, `comparison.md`, `base_generations.json`,
`tuned_generations.json`.

Judging order: (1) disjointness `true`, else stop; (2) held-out loss; (3) regressions — a
"stopped asking for information" regression matters more than a token-F1 gain; (4) before/after
examples; (5) rubric sheet, scored blind by an agronomist.

---

## 11. Inference

```bash
uv run lfm-finetune inference -c <config> --run-dir <run>              # interactive REPL
uv run lfm-finetune inference -c <config> --run-dir <run> -p "..." --json   # batch, machine-readable
uv run lfm-finetune inference -c <config> --merged <dir>                # merged model
```

`InferenceEngine.load(config, adapter_path=..., merged_path=...)` — `adapter_path` and `merged_path`
are mutually exclusive (a merged model already contains the adapter's weights).

REPL commands: `/help`, `/system <text>`, `/reset`, `/params`, `/save <path.jsonl>`, `/quit` (and
`/exit`, `/q`).

`Generator._generate_from_ids` uses **left padding** (a decoder-only model must have the real prompt
ending at the final position for every row) and slices outputs at `row[max_len:]` — slicing from the
start of a row's own prompt returns the prompt itself for every unpadded row, i.e. every row of a
batch of one.

`run_batch(..., quiet=True)` suppresses the human transcript so `--json` stdout stays parseable.
In `--json` mode **nothing else may write to stdout**.

---

## 12. Export (adapter + merge)

### Adapter — the canonical artifact (PREFERRED)

```bash
uv run lfm-finetune export-adapter -c <config> --run-dir <run>
# default destination: adapters/<run.name>/
```

```text
adapters/agri_lfm25_1.2b/
  adapter_config.json       r, alpha, target_modules, base_model_name_or_path
  adapter_model.safetensors
  tokenizer.json            bundled: inference needs no network
  tokenizer_config.json
  chat_template.jinja       bundled: train/inference formats stay identical
  export_info.json          base id + revision, package versions, reload command, config hash
```

`verify_adapter_loads` proves the artifact, rather than trusting the file listing:
1. `adapter_config.json` parses and records a base model id
2. every shard loads via `safe_open` and contains `lora_*` tensors (catches truncation)
3. every tensor's shape is consistent with the declared rank `r` (`lora_A.shape[0] == r`,
   `lora_B.shape[-1] == r`) — catches weights and config from different runs
4. when a live `model` is supplied, every configured `target_module` exists in it — catches an
   adapter trained against a different architecture or size

**Deliberately not implemented:** instantiating a meta-device copy of the base model. That was tried
and is unreliable inside a live training process (`AutoModelForCausalLM.from_config` raised
`TypeError: argument of type 'NoneType' is not iterable` while a real model and PEFT wrappers were
resident on a CPU-only build). Reading the shards directly is cheaper and more precise.

### Merge — convenience only

```bash
uv run lfm-finetune merge-adapter -c <config> --run-dir <run> [--float32]
# default destination: merged/<run.name>/
```

`merge_and_unload()` performs `W += B @ A` on the targeted Linears. Every module in Liquid's
reference set — including `conv.in_proj/out_proj` inside `Lfm2ShortConv` — is a plain `nn.Linear`, so
merging is supported. Two LFM-specific things are **verified, not assumed**:

1. **Weight tying** — `_weight_tying_state(base)` is compared before and after; if it changed, the
   merge **aborts**.
2. **Convolution blocks** — no conv-specific handling needed, but they *are* adapted when targeted,
   and the merged output was verified to generate identically to base+adapter under greedy decoding.

Merges in **bfloat16** by default (bit-consistent with training, half the size). Writes
`merge_info.json` with `weight_tying_preserved`, `merge_dtype`, `packages`, and two caveats
("merged weights are a point estimate; keep the adapter to re-train"; "quantisation must happen
AFTER merging"). `_verify_merged_loads` reloads via `accelerate.init_empty_weights()`.

### Sizing

| Model | bf16 merged | Q8 | Q4_K_M | Fits |
|---|---|---|---|---|
| 350M | 0.68 GiB | ~0.36 | ~0.22 | low-end edge, MCU-class gateways |
| 1.2B | 2.4 GiB | ~1.2 | ~0.75 | mid-range on-device |
| 2.6B | 4.9 GiB | ~2.5 | ~1.6 | edge gateway |

Plus KV cache (~0.25 GiB for 1.2B @ 2048 bf16). Runtime support: transformers (reference), vLLM
(base+adapter or merged), llama.cpp (GGUF), ONNX Runtime (ONNX). **Always measure accuracy after
quantisation.** Do not assume a GGUF conversion of a merged LoRA model exists — it is your artifact.

---

## 13. Preflight — the GO/NO-GO gate

`uv run lfm-finetune preflight -c <config> [--deep] [--no-strict]` — checks run cheapest-first and
print `RESULT: GO` or `RESULT: NO-GO`. Exits 1 on blocking failures (so it works as a CI step).
`--deep` additionally loads the checkpoint and attaches LoRA (~1 minute).

Statuses: `PASS` / `FAIL` (blocking) / `WARN` (advisory) / `SKIP` (not checked).

| Check name | Blocks when |
|---|---|
| `environment / imports` | any of torch/transformers/trl/peft/accelerate/safetensors missing |
| `environment / LFM2 support` | `transformers.Lfm2ForCausalLM` absent |
| `config / resolves + validates` | config invalid (raised, not reported) |
| `config / LoRA enabled` | WARN only when `lora.enabled=false` |
| `config / effective batch` | informational |
| `hardware / GPU` | WARN when no CUDA |
| `hardware / dtype` | FAIL when the resolved dtype is unsupported; WARN when CPU-forces-fp32 |
| `dataset / path` | FAIL when missing |
| `dataset / provenance manifest` | **FAIL** when the manifest is absent or the licence is unknown |
| `dataset / manifest fields` | WARN when `source`/`created` missing |
| `dataset / schema + quality` | **FAIL** when acceptance < 50 % (a data-*shape* bug, not a content problem); WARN when < 90 % |
| `dataset / top rejection reasons` | WARN |
| `dataset / size` | WARN when < 500 examples (Liquid's guidance floor) |
| `dataset / per-record provenance` | WARN when records lack `metadata.source`/`license` |
| `splits / prepared` | FAIL when the dir exists but is empty; SKIP when not prepared |
| `splits / provenance` | **FAIL** when `report.json`'s `source` ≠ the config's `dataset.path` |
| `template / renders + prefix-consistent` | FAIL when the render is not prefix-consistent |
| `template / assistant mask strategy` | PASS either way; reports which strategy applies |
| `tokenization / sequence length` | FAIL above the tokenizer's `model_max_length`; WARN above 32,768 |
| `hardware / memory estimate` | PASS/WARN; never blocks |
| `lora / target modules discovered`, `lora / trainable parameter audit` | `--deep` only; audit FAIL blocks |

**`splits / provenance` catches the most expensive mistake available:** `data/splits` is shared
across configs, so changing `dataset.path` without re-running `prepare` leaves the previous
corpus's splits in place and training proceeds happily **on the wrong data**.

---

## 14. Error model and exit codes

`utils/errors.py`. **Nothing may fail with a bare `RuntimeError`.** Each `PipelineError` carries
`message` (what went wrong) and `hint` (how to fix it).

```text
class PipelineError(RuntimeError):
    exit_code = 1
    def __init__(self, message: str, *, hint: str | None = None)
    def render(self) -> str   # "ClassName: message\n\n  Hint: ..."
```

| Class | Exit | Raised when |
|---|---|---|
| `ConfigError` | 2 | invalid/missing config, bad `--set` |
| `EnvironmentError_` | 3 | missing dependency |
| `HardwareError` | 3 | no CUDA when GPU requested, unsupported dtype, VRAM over budget |
| `ModelError` | 4 | load failure, unsupported architecture, context overflow |
| `TokenizerError` | 4 | missing chat template, non-prefix-consistent render, zero supervised tokens |
| `DatasetError` | 5 | unreadable/invalid data, zero valid examples |
| `DatasetLeakageError` (⊂ DatasetError) | 5 | cross-split duplicates, empty split |
| `LoRAError` | 6 | missing target module, zero/implausible trainable params |
| `TrainingError` | 7 | preflight failure / TRL args rejected |
| `CheckpointError` | 8 | missing/incomplete checkpoint, adapter verification failed |
| `ExportError` | 9 | export/merge failure |
| `EvaluationError` | 10 | generation/evaluation failure |

`cli/main.main()` catches `PipelineError` → renders → returns `exc.exit_code`; `KeyboardInterrupt`
→ 130; anything else → traceback + hint to re-run with `-v` → 1.

---

## 15. Artifact layouts

### Run directory (`checkpointing.dir/<run.name>_<YYYYmmdd-HHMMSS>/`)

```text
resolved_config.yaml      # fully-resolved; source YAML no longer needed
run_metadata.json         # versions, git, hardware, seeds, hashes, LoRA audit, dataset info,
                          # tokenization info, vram_estimate, command.argv, adapter_verified
dataset_report.json       # counts, statistics, leakage audit      (written when available)
metrics.jsonl             # streamed metrics, local, no external service
train.log                 # full DEBUG-level log
checkpoint-N/             # trainer_state.json, optimizer.pt, scheduler.pt, rng_state.pth,
                          # adapter_model.safetensors, tokenizer files
adapter/                  # verified LoRA adapter + tokenizer + export_info.json
full_model/               # only when lora.enabled=false
```

**The rule:** a run directory must be sufficient to reproduce and audit the run **without consulting
anything outside itself.** If a field would require reading mutable external state later, it is
frozen at write time.

`run_metadata.json` top-level keys (from a real smoke run):
`adapter_path`, `adapter_verified`, `command.{argv,reproduce}`, `completed_at`,
`config_fingerprint`, `created_at`, `dataset.{format,manifest,num_test,num_train,num_validation,
source,tokenization.{train,validation},use_processed}`, `duration_seconds`,
`environment.{git,hardware,packages,runtime}`, `final_metrics.*`, `host`, `lora.{audit,config,
diagnostics,resolved_target_modules}`, `model.*`, `reproducibility`, `seeds`, `tokenization.*`,
`training_plan`, `vram_estimate`.

### `data/validated/report.json`

`schema_version`, `generated_at`, `config_fingerprint`, `source`, `counts` (from `COUNT_KEYS`),
`rejection_summary` (sorted by frequency), `duplicate_summary`, `near_dedupe_index`, `split_plan`,
`leakage.{hard,soft}`, `statistics`, `warnings`, `dataset_hash` (+ `_short`), `split_hashes`,
`manifest`, `provenance_problems`, `provenance_ok`, `outputs`, `quality_config`.

`COUNT_KEYS = (raw_rows, accepted, rejected, modified, deduplicated_exact, deduplicated_near,
dropped_prompt_leakage, split_train, split_validation, split_test, after_quality_filter)` plus
`split_<name>` entries.

### `evaluations/<run>/comparison.json`

`schema_version`, `generated_at`, `run_name`, `model`, `generation` (incl. `strategy`),
`evaluation_set.disjointness_vs_train`, `held_out_loss` (with `delta`, `relative_change_pct`,
`improved`, `caution`), `objective_proxies.{base,tuned,comparison}`,
`hallucination_flags.{base,tuned}`, `before_after_examples`, `regressions`, `human_rubric_sheet`,
`interpretation_guard`.

---

## 16. Testing

```bash
uv run pytest                       # ~214 tests, ~60 s, CPU only
uv run pytest -m "not train"        # skip tests that run optimizer steps
uv run ruff check src tests scripts # clean
```

`pyproject.toml`: `addopts = "-ra --strict-markers"`, `testpaths = ["tests"]`,
markers `slow` and `train`. Ruff: line-length 100, target py312, `select = ["E","F","I","UP","B","SIM"]`,
`ignore = ["E501"]`.

| File | Tests | Depends on a cached checkpoint | Covers |
|---|---|---|---|
| `test_config.py` | 32 (+param) | no | pydantic validation, `lm_head` refusal, seed coherence, presets, `extends` chains (circular/missing/null-reset/list-replace), `--set` and env overrides, all shipped configs load, **no config uses generic module names** |
| `test_dataset.py` | 43 (+param) | no | format detection/conversion, all 8 rejection reasons, normalisation (NFC, control, HTML, whitespace), dedup/near-dup, hashed-split determinism + stability under insertion, end-to-end `prepare`, synthetic fixture contract, **shared-system-prompt near-dup regression** |
| `test_preflight.py` | 15 | no (uses `local_files_only`) | every gate name/status, healthy→GO, missing path/manifest/unknown licence block, <50 % acceptance blocks, stale splits block, small dataset warns, 6 GB config still valid with effective batch 16 |
| `test_lora.py` | 23 | via `AutoConfig.from_pretrained` (not skippable) | target resolution vs `OFFICIAL_FULL`, flags, `lm_head` exclusion, explicit targets verified, **exact rank formula** `lora_a + lora_b == Σ r*(fan_in+fan_out)`, base weights bit-identical, rank/alpha scaling, all fail-fast guards, adapter round-trip, rank-mismatch corruption detection |
| `test_model_loading.py` | 36 | via `requires_model` (~19) + `AutoConfig` (~9) | architecture is `Lfm2*`/`lfm2`, hybrid with `conv` + `full_attention` layer types, conv > attention, **generic names absent / LFM names present**, embeddings tied, base frozen, discovery matches the official preset, explicit target resolution, attachment + audit on a tiny model, hardware audit + VRAM arithmetic |
| `test_chat_template.py` | 29 | via `requires_model` (~28) | template shipped, prefix consistency, generation prompt ends `assistant\n`, ChatML markers, **`eos_token_id == 7`**, `add_special_tokens=False` halves the doubled BOS, fingerprint stability, masking (prompt masked / answer supervised / full-sequence mode), truncation (supervised span survives, `error` strategy raises, **long prompt still produces supervision**, zero-loss raises, non-prefix template rejected, empty answer → 1 token), train/inference equivalence, **`generation_kwargs` omits `pad_token_id`**, collator padding + mask |
| `test_training_smoke.py` | 14 (all `pytest.mark.train`) | via `requires_model` | full end-to-end on a tiny on-disk checkpoint: preflight, dry-run writes nothing, **smoke training end to end** (run dir contents, adapter dir contents incl. `tokenizer.json`, checkpoint state files, `run_metadata.json` assertions incl. `eos_token_id == 7` and `loss_masking` startswith `assistant-only`), resolved config round-trips to the same fingerprint, **resume continues in place**, `--resume auto`, missing/incomplete checkpoint fail fast, dataloader produces supervised tokens and gradients, inference with adapter, merged model loads and generates with tying preserved, evaluation report with disjointness and `null` human scores |

### The tiny LFM2 fixtures (three definitions, two variants)

```python
cfg = AutoConfig.from_pretrained("LiquidAI/LFM2.5-350M-Base", trust_remote_code=False)
cfg.hidden_size = 64;  cfg.intermediate_size = 128;  cfg.block_ff_dim = 128
cfg.block_dim = 64;     cfg.num_hidden_layers = 4
cfg.num_attention_heads = 4;  cfg.num_key_value_heads = 2;  cfg.head_dim = 16
cfg.conv_dim = 64
# test_lora.py / test_model_loading.py:  vocab_size = 256,  max_position_embeddings = 256
# test_training_smoke.py (saved to disk with a real tokenizer):
#     vocab_size = 65536  ("must cover the tokenizer's highest id"), max_position_embeddings = 4096
```

Using a **real** `Lfm2ForCausalLM` — not a Transformer stub — means module naming, hybrid layer
types, conv blocks, chat template, EOS handling and LoRA targets are all genuinely exercised, at a
few hundred KB instead of 1.4 GiB.

### Skip gates

`conftest.py::requires_model` skips when `LFM_RUN_SLOW` ∈ {`0`,`false`,`False`} **or** the checkpoint
is not in `HF_HOME/hub`. `SMOKE_MODEL` honours `LFM_TEST_MODEL`; the three hard-coded
`"LiquidAI/LFM2.5-350M-Base"` literals inside the tiny-model fixtures do **not**.

`test_preflight.py` and `test_config.py` are marker-free and fully offline.

---

## 17. Documentation index

| Document | Contents |
|---|---|
| `finetune/README.md` | 16-section overview, TL;DR commands, stages, troubleshooting table, references |
| `docs/ARCHITECTURE.md` | Module map, data flow, 8 design decisions, error table, extension points |
| **`docs/DATA_COLLECTION_GUIDE.md`** | **Start here before real training**: 600 lines — templates, collection practice, step-by-step commands, two failures that will happen to you, OOM ladder, the full Go/No-Go checklist |
| `docs/DATASET.md` | Schema, quality gates, provenance, MICO categories, licensing |
| `docs/TRAINING.md` | LFM2.5 chat-template findings table, masking algorithm, LoRA, hyperparameters |
| `docs/EVALUATION.md` | What is / is not measured, base-vs-tuned protocol, rubric, interpreting a loss decrease |
| `docs/DEPLOYMENT.md` | Adapter vs merged, edge runtimes, sizing, deployment checklist |
| `docs/TROUBLESHOOTING.md` | Every error message and its fix, by category |
| `docs/DELIVERABLE_REPORT.md` | Validated results, exact versions/commands, 14 bugs found and fixed, 8 known limitations, next steps |
| `datasets/README.md` | Dataset authoring, licensing, checklist, 7 required behaviours |

---

## 18. Hard invariants — do not break these

These are asserted by tests. Changing them requires changing the tests **and** the reasoning.

1. **Never hand PEFT a generic target list.** `o_proj`, `gate_proj`, `up_proj`, `down_proj`,
   `fc1`, `fc2` do not exist on LFM2/LFM2.5. A test greps every shipped config to enforce this.
2. **`"auto"` must never reach `peft.LoraConfig`.** Always resolve via `resolve_target_modules()`.
3. **Never target `lm_head` or `embed_tokens` on LFM2.5-*-Base** (`tie_embedding: true`). The config
   layer rejects `modules_to_save: [lm_head]`.
4. **Base weights must stay frozen.** `freeze_base_model()` runs on every `for_training=True` load,
   and `audit_trainable_parameters` re-checks after attachment.
5. **`add_special_tokens=False` when tokenising rendered template text** (double-BOS otherwise).
6. **Loss masking by prefix length, with the prefix property asserted.** Never rely on
   `return_assistant_tokens_mask` or TRL `assistant_only_loss`.
7. **Strip the template's trailing newline** so the target ends exactly at the terminator.
8. **Deduplicate globally before splitting.** Never reorder these two steps.
9. **Never drop a row silently.** Every rejection carries a machine-readable reason + excerpt.
10. **Normalise before dedup** (NFC, whitespace, HTML, control chars), or near-identical copies
    survive as separate rows and straddle the split boundary.
11. **The system message is excluded from near-duplicate comparison** unless explicitly requested.
12. **Never concatenate lists in config inheritance** — replace them.
13. **Resuming must continue the same run directory**, so `metrics.jsonl` stays continuous.
14. **`metrics.jsonl` is written unconditionally**, independent of CLI sinks or integrations.
15. **`pad_token_id` must come from exactly one place** (the generation loop), not from both the
    tokenizer and `generation_kwargs()`.
16. **Generation must slice at `row[max_len:]`** with left padding, never at the row's own prompt start.
17. **Human rubric scores stay `null`.** The pipeline does not fake agronomic judgement.
18. **Decode greedily for every A/B comparison.** Sampling makes base and tuned incomparable.
19. **Merge aborts if weight tying changed.** Merged ≡ base+adapter under greedy decoding.
20. **`resolve_output_dir` uses UTC**; `utc_now()` uses `datetime.now(UTC)`.
21. **A `key: null` in a child config means "reset to default"** — a deliberate deep_merge behaviour.
22. **Never write to stdout in `--json` mode** other than the JSON payload.

---

## 19. Current state, validated results and limitations

### Validated on the development host (CPU-only, no GPU)

| Stage | Result |
|---|---|
| 0 `doctor` / `inspect-hardware --recommend` | PASS |
| 1 `inspect-model` | PASS — 9 target types, exact match with Liquid's preset |
| 2 `validate` / `prepare` | PASS — 42 accepted, 0 rejected, 1 dedup, 29/8/5 split |
| 3 `inspect-template` | PASS — assistant-only mask, prefix-consistent |
| 4 `inspect-lora` | PASS — 2,998,272 trainable (0.839 %), base frozen |
| 5 `train --smoke-test` | PASS — 15.8 s, loss 3.31 → adapter verified, checkpoint written |
| 6 resume | PASS — continued in place, step 3 → 4, metrics 15 → 28 lines |
| 7 full SFT | **not runnable** — no CUDA; configs validated + estimates computed |
| 8 `baseline` / `evaluate` | PASS — loss 1.731 → 1.373 (−20.7 %), token-F1 0.391 → 0.800, 0 regressions |
| 9 `export-adapter` / `merge-adapter` | PASS — 16 MiB adapter, 680.7 MiB merged, tying preserved |
| inference | PASS — merged ≡ base+adapter under greedy decoding |
| `pytest` | PASS |
| `ruff` | PASS |

Host: AMD EPYC 7763, 2 logical cores, 7.8 GiB RAM, ~10 GiB free disk,
`torch 2.11.0+cpu`, Python 3.12.15, no CUDA.

### Exact validated versions

| Package | Version |
|---|---|
| python | 3.12.15 |
| torch | 2.11.0+cpu |
| transformers | 5.3.0 |
| trl | 1.9.0 |
| peft | 0.21.2 |
| accelerate | 1.15.0 |
| datasets | 5.0.1 |
| safetensors | 0.8.0 |
| tokenizers | 0.22.2 |
| numpy | 2.5.3 |
| huggingface-hub | 1.33.0 |
| pydantic | 2.13.5 |
| PyYAML | 6.0.3 |
| bitsandbytes | not installed (optional `--extra quant`) |

### Known limitations (8)

1. **No real agriculture data has been trained on.** Validated runs use a synthetic fixture
   (arithmetic/string ops with computed answers). It proves the pipeline; it makes **no claim** about
   agricultural capability.
2. **The dev host has no GPU.** bf16, flash-attention, multi-GPU and the 1.2B/2.6B configs are
   exercised only through preflight arithmetic and fail-fast guards.
3. **Evaluation proxies are not correctness metrics.** Agronomic correctness requires human scoring.
4. **Near-duplicate detection is character-shingle Jaccard** — catches templated paraphrase
   duplication, not semantic similarity.
5. **Bit-exact reproducibility across different GPUs/drivers is not guaranteed** even with fixed
   seeds; the manifest records the environment so a mismatch is detectable.
6. **Distribution shift is untested.** No Hindi/Devanagari data; the normaliser handles control
   chars and NFC, but token-level behaviour on Devanagari is unverified.
7. **QLoRA / quantisation is unexercised** (optional `quant` extra, used by no shipped config).
8. **Edge conversions (GGUF/ONNX) are unverified** — no device available.

### The single next step

**Build the agriculture corpus with provenance, then run pilot → full SFT on a CUDA host.**
No amount of hyperparameter work substitutes for data. Priority order:

1. Assemble **500–2,000 sourced examples** across the 12 categories with a `manifest.yaml` recording
   source, licence, annotation method and known limitations. Prefer extension bulletins and
   agronomist review over LLM generation; if generated, mark `synthetic: true`.
2. `validate` + `prepare`, then read `data/validated/report.json` — fix data problems before spending
   GPU time. Watch `warnings`, `rejection_summary`, `leakage`.
3. `inspect-template --samples 3` — supervised span must be the assistant text plus the terminator.
4. `train --pilot -s dataset.max_train_examples=500` (~10 min) and answer the six pilot questions.
5. Full SFT → `baseline` → `evaluate` → have an agronomist score the rubric sheet **blind** for both.
6. **Only then** tune `lora.r`, `learning_rate`, `max_seq_length`, re-running the comparison each time.

**Do not** spend effort on quantisation, multi-node training, or a larger model until a CUDA host has
produced a pilot whose base-vs-tuned comparison shows a real behavioural change.

### 14 bugs found and fixed during validation (do not reintroduce)

1. `keep_head_tail` truncation masked the entire answer when the prompt alone exceeded
   `max_seq_length`.
2. PEFT received the literal string `"auto"` as target modules.
3. `pad_token_id` passed twice to `PeftModel.generate`.
4. Generation returned the prompt (sliced from the row's prompt start instead of the padded end).
5. HF `PrinterCallback` duplicated every metric line with `%.4g` truncation.
6. Metrics were only written when a CLI sink was attached.
7. Adapter verification used meta-device instantiation, which fails inside a live training process.
8. `--json` output was not machine-readable (human text interleaved).
9. Resume created a **new** run directory, fragmenting `metrics.jsonl`.
10. `Counter` omitted zero-valued keys, so `counts["rejected"]` raised on clean data.
11. `manifest.json` was ingested as a data record.
12. `recommend_model` reported everything "FEASIBLE" on CPU-only hosts.
13. `torch.manual_seed()` returns a Generator, crashing the `%d` log format.
14. transformers leaves parameters `requires_grad=True`; the "frozen base" assertion wrongly failed.

---

## 20. Extension recipes

| Goal | How |
|---|---|
| **New domain** (finance, legal, …) | Add `configs/<domain>/_domain.yaml` (system prompt + thresholds) + `configs/<domain>/<model>.yaml` + a corpus + `manifest.yaml`. **No code changes.** |
| **New model size** | Add `configs/<domain>/lfm25_<size>.yaml`. Discovery adapts automatically. |
| **New quality gate** | Add a check in `data/validate.py::validate_conversation` returning a `Rejection`. |
| **New metric** | Add a function in `evaluation/metrics.py`; include it in `score_generation` + `summarise` + `compare_summaries`'s `default_directions`. |
| **New rubric** | Point `evaluation.rubric_path` at a JSON `{"dimensions": [...]}` file (override `DEFAULT_AGRICULTURE_RUBRIC`). |
| **New export format** | Add to `export/`; reuse `verify_adapter_loads`. |
| **New integration (wandb/mlflow/…)** | Extend `training/metadata.py::resolve_reporting`; never make it required. |
| **New CLI command** | `sub.add_parser(...)` + `cmd_*` in `cli/main.py`; keep the logic in a package module so it stays testable and the CLI stays thin. |
| **New error type** | Subclass `PipelineError` with a distinct `exit_code` in `utils/errors.py`. |

---

## 21. Agent operating rules

1. **Read `configs/base.yaml` before touching `config/schema.py`.** It documents every field and is
   the de-facto schema reference.
2. **Never add domain knowledge to `src/`.** If a change mentions crops, diseases, India or MICO
   behaviour, it belongs in `configs/agriculture/_domain.yaml` or the dataset.
3. **Never weaken a fail-fast guard to make a test pass.** The guards are the product. Every one of
   them exists because a real, expensive, silent failure motivated it (see §19, bug list).
4. **Never hard-code a model name, token id, or module name outside `configs/` and the `KNOWN_*`
   constants.** Read `tokenizer.eos_token_id`; discover module suffixes from `named_modules()`.
5. **Adding a field to a config section means updating all three of:** the pydantic model,
   `configs/base.yaml` (documented), and — if it changes numerics — `docs/TRAINING.md`.
6. **Write a test that pins the invariant**, and prefer the existing three-fixture pattern (tiny real
   `Lfm2ForCausalLM`) over mocks.
7. **Comment the *why*, not the *what*.** This codebase's comments explain measured facts, past bugs
   and rejected alternatives. Match that density and tone — it is the highest-value part of the code.
8. **Never commit weights, datasets, or secrets.** `.gitignore` already covers it; verify before
   staging.
9. **A `pipeline` run must be self-describing.** Any new artifact you introduce should be written
   atomically (`utils/io.atomic_write_text`) and recorded in `run_metadata.json` or `report.json`.
10. **Do not claim domain capability from a pipeline test.** Synthetic-fixture results prove the
    machinery, nothing about agriculture.
11. **When you are unsure of a measured fact** (token id, template content, module name), verify it
    with `inspect-template` / `inspect-model` rather than reasoning about it.
12. **Keep `docs/` in sync.** If you change behaviour, the matching doc section is part of the change.

---

## 22. Quick reference: common tasks

### Run the whole pipeline on the synthetic fixture (CPU)

```bash
cd finetune && uv sync
uv run lfm-finetune init-synthetic --destination data/raw/synthetic
uv run lfm-finetune prepare       -c configs/smoke.yaml
uv run lfm-finetune train         -c configs/smoke.yaml --smoke-test
uv run lfm-finetune baseline      -c configs/smoke.yaml
uv run lfm-finetune evaluate      -c configs/smoke.yaml
uv run lfm-finetune export-adapter -c configs/smoke.yaml
uv run pytest && uv run ruff check src tests scripts
```

### Start a real agriculture run

```bash
mkdir -p data/raw/mico-agriculture
cp datasets/TEMPLATE_manifest.yaml data/raw/mico-agriculture/manifest.yaml
cp datasets/TEMPLATE_dataset.jsonl  data/raw/mico-agriculture/mico-agriculture-v1.jsonl
# ... author the corpus (answers >= 80 chars, source + license on every record) ...

cp configs/agriculture/lfm25_1.2b_6gb.yaml configs/mico-v1.yaml
# then in configs/mico-v1.yaml:
#   extends: agriculture/lfm25_1.2b_6gb.yaml
#   run: {name: mico-agri-v1}
#   dataset: {path: data/raw/mico-agriculture, use_processed: true}

uv run lfm-finetune show-config   -c configs/mico-v1.yaml
uv run lfm-finetune preflight     -c configs/mico-v1.yaml          # RESULT: GO
uv run lfm-finetune prepare       -c configs/mico-v1.yaml
python -m json.tool data/validated/report.json | head -60           # READ THIS
uv run lfm-finetune inspect-template -c configs/mico-v1.yaml --samples 3
uv run lfm-finetune train         -c configs/mico-v1.yaml --pilot -s dataset.max_train_examples=500
uv run lfm-finetune baseline      -c configs/mico-v1.yaml
uv run lfm-finetune train         -c configs/mico-v1.yaml
uv run lfm-finetune evaluate      -c configs/mico-v1.yaml
uv run lfm-finetune export-adapter -c configs/mico-v1.yaml --run-dir <run>
```

### Sweep without editing YAML

```bash
LFM_TRAINING__LEARNING_RATE=1e-4 LFM_LORA__R=32 \
  uv run lfm-finetune train -c configs/agriculture/lfm25_1.2b.yaml

uv run lfm-finetune train -c configs/agriculture/lfm25_1.2b.yaml \
  -s training.learning_rate=1e-4 -s lora.r=32 -s tokenization.max_seq_length=1024
```

### Diagnose

```bash
uv run lfm-finetune doctor
uv run lfm-finetune inspect-hardware --recommend
uv run lfm-finetune recommend-model --seq-length 2048
uv run lfm-finetune stages
uv run lfm-finetune -v <command>                 # full traceback
cat checkpoints/<run>/train.log
cat checkpoints/<run>/metrics.jsonl
cat checkpoints/<run>/run_metadata.json
python -m json.tool data/validated/report.json
head -3 data/validated/rejected.jsonl | python -m json.tool
```

### Hyperparameters at a glance

| Parameter | Default (agriculture) | Notes |
|---|---|---|
| `lora.r` / `lora.alpha` | 16 / 32 | `alpha = 2r`. `alpha << r` stalls training. Liquid's presets: r=8/α=16 and r=16/α=32. |
| `lora.dropout` | 0.05 | Liquid's preset uses 0.1. |
| `lora.bias` | `none` | Correct for LFM2.5 (`conv_bias=false`). |
| `training.learning_rate` | 2e-4 | LoRA SFT range 1e-4…3e-4. Drop to 1e-4 if outputs become repetitive. |
| `training.weight_decay` | 0.0 | LoRA convention. |
| `training.warmup_ratio` | 0.03 | |
| `training.lr_scheduler_type` | `cosine` | |
| `training.max_grad_norm` | 1.0 | |
| `tokenization.max_seq_length` | 2048 | Logits memory scales `batch × seq × 65536`. |
| `model.gradient_checkpointing` | true | ~30 % compute for a large activation saving. |
| `training.assistant_only_loss` | true | Suppresses the prompt gradient. |
| `training.packing` | false | Packing erases per-example boundaries. |
| `evaluation.do_sample` | false | Required for reproducible A/B. |
| effective batch | 16 | `4 × 4` (1.2B), `1 × 16` (350M / 6 GB laptop), `2 × 8` (2.6B). **Keep it at 16 across configs so LR schedules stay comparable.** |

### OOM ladder, in order

1. Close everything else using the GPU; check `nvidia-smi`.
2. **Halve `tokenization.max_seq_length`** (biggest single win — the logits tensor dominates).
3. Lower `per_device_train_batch_size` and raise `gradient_accumulation_steps` **together** (effective
   batch unchanged, LR schedule unchanged).
4. Confirm `model.gradient_checkpointing: true`.
5. Drop to `configs/agriculture/lfm25_350m.yaml`.

**Never** lower the learning rate to fix OOM — batch size and LR are separate knobs, and changing
both makes runs incomparable.

### Common errors → immediate fix

| Error | Fix |
|---|---|
| `Target modules auto not found` | Cannot happen; discovery resolves first. If seen, the config bypassed `resolve_target_modules`. |
| `Zero parameters are trainable` | `target_modules` matched nothing. Run `inspect-model`. |
| `X% of parameters are trainable, above the ceiling` | `modules_to_save` contains `lm_head`/`embed_tokens`. Remove. |
| Generation never stops | `tokenization.append_eos_token: true`; verify with `inspect-template`. |
| `pad_token_id` multiple values | `pad_token_id` re-added to a custom generation path. |
| `dataset.path is required` | Run `prepare`, or set `dataset.use_processed: true`. |
| `Split produced empty validation set` | Tiny corpus. `-s dataset.splitting.train_ratio=0.6 -s dataset.splitting.validation_ratio=0.2`. |
| `duplicate_near: 700` | Templated answers (same paragraph, crop name swapped). Ask: *what would an agronomist actually say about this specific crop?* |
| `Exact duplicate conversations span split boundaries` | Raw data bypassed `prepare` or `quality.deduplicate` is off. |
| `Manifest warning: license: unknown` | Fill in a real SPDX id or `internal-use-only`. |
| `loss: NaN` | LR too high → 1e-4; bf16 on pre-Ampere → fp16/fp32; check `grad_norm` in `metrics.jsonl`. |
| `splits / provenance` FAIL | `dataset.path` changed without re-running `prepare`. |
| `Estimated peak training memory X exceeds usable VRAM Y` | Already failed fast — follow the listed reductions. |

---

## References

- Liquid LFM fine-tuning docs: <https://docs.liquid.ai/lfm/fine-tuning/trl>
- Liquid's training stack: <https://github.com/Liquid4All/leap-finetune>
- LFM2 technical report: arXiv:2511.23404
- LFM2.5 announcement: <https://www.liquid.ai/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai>