# Architecture

How the code is organised, why, and which decisions were load-bearing.

---

## 1. Layout

```text
finetune/
├── pyproject.toml          pinned, uv-managed, CPU-torch default
├── uv.lock                 exact resolution
├── .env.example            secrets template (never a real token)
├── configs/
│   ├── base.yaml           every field, documented — also the schema reference
│   ├── lfm_sft.yaml        LFM2.5-specific defaults (LoRA auto-discovery, EOS)
│   ├── smoke.yaml          CPU end-to-end validation
│   └── agriculture/        _domain.yaml + 350M / 1.2B / 2.6B
├── data/{raw,processed,validated,splits}/
├── datasets/               schema.json, manifest.schema.json, README
├── src/lfm_finetune/
│   ├── config/             typed schema (pydantic) + loader (YAML inheritance)
│   ├── data/               schema, io, validate, dedupe, split, stats, pipeline, synthetic
│   ├── model/              loader, discovery, lora
│   ├── training/           tokenization, collator, args, trainer, metadata
│   ├── evaluation/         metrics, runner
│   ├── inference/          generator, cli_loop
│   ├── export/             adapter (export + merge + verify)
│   ├── utils/              errors, logging, seeding, hashing, hardware, io, versions
│   └── cli/                main (unified argparse CLI)
├── scripts/                thin wrappers over the CLI
├── tests/                  196 tests
├── docs/                   this directory
└── outputs/ checkpoints/ adapters/ merged/ evaluations/ logs/
```

---

## 2. Layering

Dependencies point one way:

```text
cli  ->  config  ->  data  ->  model  ->  training  ->  evaluation
                    |         |            |
                    +---------+            +--> inference, export
                                  utils (used by all)
```

Nothing in `data`, `model`, `training`, `evaluation`, `inference` or `export`
imports the CLI. That is what lets the tests exercise any layer without invoking
a command, and what lets `scripts/*.py` be thin.

Agriculture is not represented anywhere in the engine. Domain knowledge lives in
`configs/agriculture/_domain.yaml` (system prompt, quality thresholds, split
ratios) and in the dataset (messages + metadata). `configs/finance/` would be a
new YAML plus a new corpus.

---

## 3. Design decisions and their justification

### 3.1 LoRA targets are discovered, never written down

`target_modules` defaults to `auto`: `model.discovery.discover_lora_targets`
walks `named_modules()`, classifies each Linear by role, and cross-checks the
result against Liquid's published presets. Two independent sources must agree.

The alternative — a hard-coded list — fails silently. `["q_proj","v_proj"]`
matches nothing on LFM2.5 (the names are `self_attn.q_proj`, which PEFT
matches by suffix, so `q_proj` *does* work; but `o_proj` matches nothing because
the layer is `out_proj`). A partially-matching list produces a training run that
looks fine and adapts almost nothing.

### 3.2 Loss masking by prefix length

TRL's `assistant_only_loss` depends on `{% generation %}` blocks in the chat
template. `LFM2.5-1.2B-Base` and `LFM2.5-350M-Base` do not have them, so the mask
is all zeros. The mask is therefore computed directly and the prefix property is
asserted. See `docs/TRAINING.md` §1–2.

### 3.3 Pre-tokenise in Python, do not let TRL prepare the dataset

`tokenization.skip_prepare_dataset: true` with a hand-written collator. This
puts truncation, loss masking and the reported token statistics in one place, and
makes them inspectable before training starts. It also means swapping model size
does not change data handling behaviour.

### 3.4 Deduplicate globally, *then* split

Deduplicating after splitting still leaves duplicate pairs on opposite sides of
the boundary, which is the most common cause of a suspiciously good validation
curve. Order in `prepare_dataset`: read → convert → normalise → validate →
**dedup** → **split** → leakage audit → statistics.

Split assignment is content-addressed (`BLAKE2b(seed || conversation_hash)`), so
growing the corpus never moves an existing example between splits and a recorded
validation score stays valid.

### 3.5 Every failure is typed and carries a fix

`utils/errors.py` defines one exception per failure class with a distinct exit
code. Messages state what failed *and* how to fix it. Examples:

```text
LoRAError: Adapter targets module(s) ['o_proj'] that do not exist in
LiquidAI/LFM2.5-1.2B-Base.
  Available Linear module suffixes: ['conv.in_proj', ..., 'self_attn.out_proj', ...]
  Hint: On LFM2/LFM2.5 the attention output projection is 'self_attn.out_proj'
  (not 'o_proj') and the FFN is 'feed_forward.w1/w2/w3'. Run
  `lfm-finetune inspect-model` to see the real module names.
```

### 3.6 Run directories are self-describing

`resolved_config.yaml` + `run_metadata.json` + `metrics.jsonl` + `train.log` +
`checkpoint-N/` + `adapter/`. A finished run can be reproduced and audited from
its own directory; it never depends on mutable external YAML. Resuming continues
the *same* directory so the metrics history stays continuous.

### 3.7 Local logging is unconditional

`metrics.jsonl` is written whether or not an integration is configured and
whether or not a console sink is attached. `report_to: ["wandb"]` without
`WANDB_API_KEY` is skipped with an INFO message, never an error — the pipeline
must run without any external service.

### 3.8 Adapter is the primary artifact

It is tens of MiB instead of gigabytes, it pins the exact base checkpoint, and it
survives re-training and re-merging. Merging exists for runtimes that cannot load
an adapter, and is verified (shards load, rank consistency, tie preservation).

---

## 4. Data flow

```text
raw jsonl
  │
  ├─ iter_raw_records        directory walk, skip manifest/schema files
  ├─ convert_record          format sniffing -> Conversation
  ├─ TextNormalizer          NFC, control/zero-width strip, HTML, whitespace
  ├─ validate_conversation   roles, ordering, lengths, placeholders, leakage
  ├─ exact dedup             SHA-256 over the normalised payload
  ├─ near dedup              char 5-gram Jaccard via inverted index
  ├─ build_plan              content-hashed split
  ├─ split_conversations     train / validation / test
  ├─ leakage audit           hard-fail on cross-split exact duplicates
  ├─ statistics              token counts with the real tokenizer
  └─ report.json             counts, reasons, warnings, hashes, manifest
```

Counts for accepted / rejected / modified / deduplicated / per-split are all in
`report.json`, and rejected rows are written to
`data/validated/rejected.jsonl` with a reason and an excerpt.

---

## 5. Error handling model

| Class | Exit code | Raised when |
|---|---|---|
| `ConfigError` | 2 | invalid/unreadable config, bad override |
| `EnvironmentError_` | 3 | missing dependency |
| `HardwareError` | 3 | no CUDA when GPU requested, dtype unsupported, VRAM estimate over budget |
| `ModelError` | 4 | load failure, unsupported architecture, context overflow |
| `TokenizerError` | 4 | missing chat template, non-prefix-consistent render, zero supervised tokens |
| `DatasetError` | 5 | unreadable/invalid data, zero valid examples |
| `DatasetLeakageError` | 5 | cross-split duplicates, empty split |
| `LoRAError` | 6 | missing target module, zero/implausible trainable params |
| `TrainingError` | 7 | preflight failure |
| `CheckpointError` | 8 | missing/incomplete checkpoint, adapter verification failure |
| `ExportError` | 9 | export/merge failure |
| `EvaluationError` | 10 | generation/evaluation failure |

All are subclasses of `PipelineError`, which the CLI renders with its hint and
maps to the exit code. Unexpected exceptions are logged with a traceback under
`-v` and return exit code 1.

---

## 6. Extension points

| Goal | How |
|---|---|
| New domain | add `configs/<domain>/`, new corpus, reuse engine |
| New model size | add `configs/<domain>/lfm25_<size>.yaml`; discovery adapts automatically |
| New quality gate | add a check in `data/validate.validate_conversation` |
| New metric | add a function in `evaluation/metrics.py`, include it in `summarise` |
| New export format | add to `export/`; reuse `verify_adapter_loads` |
| New integration | extend `training/metadata.resolve_reporting` |
| Without a GPU | `configs/smoke.yaml`, or `configs/agriculture/lfm25_350m.yaml` |

---

## 7. Testing strategy

196 tests, ~60 s, CPU only.

- **Fast, no model** (`test_config.py`, `test_dataset.py`): pydantic validation,
  YAML inheritance, every shipped config, format conversion, quality gates,
  dedup, split stability under insertion, leakage detection.
- **Real tokenizer** (`test_chat_template.py`): the template facts that matter —
  prefix consistency, `add_special_tokens=False`, EOS handling, masking,
  truncation invariants, train/inference equivalence.
- **Real architecture, tiny tensors** (`test_lora.py`, `test_training_smoke.py`):
  an actual `Lfm2ForCausalLM` built from the real config but shrunk, so module
  naming, hybrid layer types and conv blocks are exercised at ~5 M parameters.
  Training tests therefore run in seconds and cannot exhaust a small host.
- **Real checkpoint** (`test_model_loading.py`, chat-template tests): the 350M
  checkpoint for architecture, discovery and template facts.

Training tests assert the parts that are easy to get quietly wrong: base weights
bit-identical after attachment, no non-LoRA parameter trainable, checkpoint
contains optimizer/scheduler/RNG state, resume continues the same directory,
merged model reloads with tying intact, rubric scores stay `null`.