# Krishi Sahayatam — MICO agriculture LLM

Supervised fine-tuning infrastructure for the **MICO** agricultural advisory
assistant, built on Liquid AI's **LFM2.5** models.

This is not a training script. It is a pipeline that turns raw agricultural text
into a versioned, evaluated, reproducible, deployable LoRA adapter — and refuses
to start when the machine, the model, the data, or the loss mask is wrong.

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

---

## TL;DR — the five commands that matter

```bash
cd finetune
uv sync                                                    # install

uv run lfm-finetune doctor                                 # stage 0
uv run lfm-finetune init-synthetic --destination data/raw/synthetic
uv run lfm-finetune prepare  -c configs/smoke.yaml         # stage 2
uv run lfm-finetune train    -c configs/smoke.yaml --smoke-test   # stage 5
uv run lfm-finetune evaluate -c configs/agriculture/lfm25_1.2b.yaml # stage 8
```

**Starting with real data?** Read
[`docs/DATA_COLLECTION_GUIDE.md`](docs/DATA_COLLECTION_GUIDE.md) first, copy the
two templates from `datasets/`, then run:

```bash
uv run lfm-finetune preflight -c configs/mico-v1.yaml   # GO / NO-GO in one command
```

Replace `data/raw/synthetic` with your own corpus and the same commands run a
real agriculture experiment. Nothing else changes.

---

## 1. What this project does

| Capability | Where |
|---|---|
| Typed config, deep YAML inheritance, env/CLI overrides, fail-fast validation | `src/lfm_finetune/config/` |
| Ingestion, schema validation, quality gates, dedup, near-dup, leakage control, stats, provenance | `src/lfm_finetune/data/` |
| Checkpoint loading, architecture gating, **programmatic LoRA discovery**, parameter audit | `src/lfm_finetune/model/` |
| Chat-template rendering, assistant-only loss masking, SFTTrainer wiring, run manifests | `src/lfm_finetune/training/` |
| Base-vs-tuned comparison, objective proxies, human rubric sheets | `src/lfm_finetune/evaluation/` |
| Interactive + batched generation over base+adapter or merged | `src/lfm_finetune/inference/` |
| Adapter export (primary) and verified merging (optional) | `src/lfm_finetune/export/` |
| One CLI, ten runnable stages | `src/lfm_finetune/cli/` |

Agriculture is **data and configuration**, never code. Adding `configs/finance/`
or `configs/legal/` needs new YAML and new JSONL, no changes to the engine.

---

## 2. Supported models

Validated against these checkpoints (all ship a `chat_template.jinja`):

| Model id | Params | bf16 weights | Use |
|---|---|---|---|
| `LiquidAI/LFM2.5-230M-Base` | 0.23 B | ~0.5 GiB | ultra-low-end, CI |
| `LiquidAI/LFM2.5-350M-Base` | 0.35 B | ~0.7 GiB | low-end edge, CPU validation |
| **`LiquidAI/LFM2.5-1.2B-Base`** | **1.17 B** | **~2.4 GiB** | **MICO primary target** |
| `LiquidAI/LFM2.5-2.6B-Base` | 2.6 B | ~5.2 GiB | edge gateway |
| `LiquidAI/LFM2.5-8B-A1B-Base` | 8 B (MoE) | ~16 GiB | server; not an edge target |

**Base, not instruct.** Liquid's model card recommends `*-Base` specifically for
"heavy fine-tuning ... domain-specific (e.g. medical) assistants, training on
proprietary data". Starting from an instruct checkpoint means using only domain
data to un-learn a general instruct prior — which a few thousand agricultural
examples cannot do reliably. If general chat must survive, mix general instruct
data into the corpus.

Rank the checkpoints for the machine you are on:

```bash
uv run lfm-finetune recommend-model --seq-length 2048
```

---

## 3. Hardware requirements

| Target | Requirement |
|---|---|
| CPU validation / CI | 2 cores, 8 GiB RAM, 4 GiB disk. `configs/smoke.yaml` (350M, fp32). |
| 1.2B LoRA SFT (recommended) | 1× 12 GiB GPU (RTX 3060 12G / 4060 Ti 16G / A10 / A100), bf16 |
| 2.6B LoRA SFT | 1× 24 GiB, or 2× 12 GiB with halved accumulation |
| Inference (1.2B) | ~2.4 GiB bf16 weights; merged 350M ≈ 0.7 GiB, 1.2B ≈ 2.4 GiB |

Memory profile for the 1.2B config (bf16, gradient checkpointing, LoRA r=16,
batch 4 × 2048 tokens): weights 2.34 GiB, gradients ~0.02 GiB, AdamW ~0.04 GiB,
activations ~1.5 GiB, logits ~1.0 GiB → **peak ≈ 5.0 GiB**.

The pipeline estimates this before allocating and fails with concrete reduction
advice if it exceeds usable VRAM — it will not OOM 40 minutes into a run.

---

## 4. Setup

```bash
cd finetune
uv python install 3.12
uv sync                       # CPU wheels by default
```

On a CUDA host, install a CUDA torch build:

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu126 torch
uv sync --no-build-isolation
```

Authentication is only needed for gated repos. Copy `.env.example` to `.env` and
set `HF_TOKEN` if required. **Never commit a token.**

Verify:

```bash
uv run lfm-finetune doctor
uv run lfm-finetune inspect-hardware --recommend
```

---

## 5. Dataset

Canonical format — conversational, documented in `datasets/schema.json`:

```json
{
  "id": "agri-000123",
  "messages": [
    {"role": "system", "content": "<domain system prompt>"},
    {"role": "user", "content": "My tomato plants have ..."},
    {"role": "assistant", "content": "..."}
  ],
  "metadata": {
    "domain": "agriculture", "category": "diagnosis",
    "crop": "tomato", "growth_stage": "flowering", "region": "IN-KA",
    "confidence": 0.7, "source": "icar-extension-publication",
    "license": "CC-BY-4.0", "language": "en"
  }
}
```

`instruction/input/output`, `prompt/completion` and `question/answer` are accepted
and normalised automatically. Metadata is preserved for analysis and **is not
placed in the prompt**. Every example needs a real `source` and `license`; a
manifest with `license: unknown` is flagged loudly in every run manifest.

See `datasets/README.md` and `docs/DATASET.md`.

### Build the dataset

```bash
uv run lfm-finetune validate -c configs/agriculture/lfm25_1.2b.yaml  # dry check
uv run lfm-finetune prepare  -c configs/agriculture/lfm25_1.2b.yaml  # clean+split
```

`prepare` writes:

- `data/splits/{train,validation,test}.jsonl`
- `data/validated/accepted.jsonl`, `rejected.jsonl`, `report.json`

Nothing is ever dropped silently — every rejection carries a machine-readable
reason, and the counts land in `report.json`.

---

## 6. LoRA architecture

LFM2/LFM2.5 is a **hybrid** model and does not use Llama/Qwen module names.
Targets are **discovered from `model.named_modules()`** and cross-checked
against Liquid's own reference presets
([leap-finetune](https://github.com/Liquid4All/leap-finetune)):

```text
feed_forward.w1, feed_forward.w2, feed_forward.w3     SwiGLU   (16 each)
self_attn.q_proj, self_attn.k_proj,
self_attn.v_proj, self_attn.out_proj                  attention (6 each)
conv.in_proj, conv.out_proj                            short conv (10 each)
```

Note `out_proj`, **not** `o_proj`. `lm_head` and `embed_tokens` are excluded:
LFM2.5 base checkpoints set `tie_embedding: true`, so the two are the same
tensor and adapting either breaks the tie. The config layer refuses
`modules_to_save: [lm_head]` outright.

Inspect before you train:

```bash
uv run lfm-finetune inspect-model -c configs/agriculture/lfm25_1.2b.yaml
uv run lfm-finetune inspect-lora  -c configs/agriculture/lfm25_1.2b.yaml
```

`inspect-lora` fails loudly if zero parameters are trainable, if a named target
does not exist, or if the trainable share exceeds `lora.max_trainable_pct`.

---

## 7. Training

Stages, each runnable on its own (`uv run lfm-finetune stages`):

| Stage | Command | Answers |
|---|---|---|
| — | `preflight` | Am I ready at all? Runs stages 0–7 and prints GO / NO-GO |
| 0 | `doctor`, `inspect-hardware` | Is the environment sane? |
| 1 | `inspect-model` | Does the model match what I think? |
| 2 | `validate`, `prepare` | Is the data usable and leak-free? |
| 3 | `inspect-template` | Is the loss mask correct? |
| 4 | `inspect-lora` | Are the LoRA modules right? |
| 5 | `train --smoke-test` | Does it train end to end? |
| 6 | `train --pilot` | Is memory stable, does eval loss behave? |
| 7 | `train` | Full SFT |
| 8 | `baseline`, `evaluate` | Did it actually improve? |
| 9 | `export-adapter`, `merge-adapter` | Can I ship it? |

### Commands

```bash
# pilot (stage 6) -- short run on a subset, before committing to a full run
uv run lfm-finetune train -c configs/agriculture/lfm25_1.2b.yaml --pilot \
  -s dataset.max_train_examples=500

# full SFT (stage 7)
uv run lfm-finetune train -c configs/agriculture/lfm25_1.2b.yaml

# resume (continues the SAME run directory)
uv run lfm-finetune train -c configs/agriculture/lfm25_1.2b.yaml --resume auto
```

Every run directory is self-describing and sufficient to reproduce and audit the
run without consulting anything else:

```text
checkpoints/<run>_<timestamp>/
  resolved_config.yaml     # fully-resolved; source YAML no longer needed
  run_metadata.json        # versions, git, hardware, seeds, hashes, LoRA audit
  dataset_report.json      # counts, statistics, leakage audit
  metrics.jsonl            # streamed metrics (local; no external service)
  train.log                # full DEBUG log
  checkpoint-N/            # resumable: trainer/optimizer/scheduler/RNG state
  adapter/                 # verified LoRA adapter
```

### Hyperparameters

Exposed and commented in `configs/base.yaml`. Presets: `pilot`, `balanced`,
`memory_saver`, `full`, `smoke`. **None is claimed to be optimal** — they are
starting points from LoRA-SFT practice and Liquid's reference presets.

---

## 8. Evaluation

Baseline is measured **before** training, on the test split, which is never
trained on:

```bash
uv run lfm-finetune baseline -c configs/agriculture/lfm25_1.2b.yaml
uv run lfm-finetune evaluate -c configs/agriculture/lfm25_1.2b.yaml
```

`evaluate` writes `evaluations/<run>/comparison.json` and `comparison.md` with
held-out loss, objective proxies, before/after examples, detected regressions,
hallucination flags, and a **blank human rubric sheet** for the seven agriculture
dimensions (factual correctness, hallucinated diagnosis, clarification behaviour,
recommended-action quality, instruction following, terminology, safety).

The pipeline does not compute agronomic correctness, and the rubric scores are
left `null` on purpose. **A lower held-out loss is not evidence of improved
domain capability** — the report says so in every artifact. See
`docs/EVALUATION.md`.

---

## 9. Inference

```bash
uv run lfm-finetune inference -c configs/agriculture/lfm25_1.2b.yaml --run-dir <run>
```

Interactive commands: `/help`, `/system <text>`, `/reset`, `/params`,
`/save <path>`, `/quit`. Non-interactive:

```bash
uv run lfm-finetune inference -c configs/agriculture/lfm25_1.2b.yaml \
  --run-dir <run> -p "..." --json
```

Decoding defaults live in `evaluation:` (greedy, deterministic — required for a
meaningful A/B comparison) and are overridable on the command line.

---

## 10. Export

```bash
# adapter only -- PREFERRED artifact (tens of MiB, pins the exact base checkpoint)
uv run lfm-finetune export-adapter -c configs/agriculture/lfm25_1.2b.yaml --run-dir <run>

# merged standalone model -- convenience for runtimes that cannot load an adapter
uv run lfm-finetune merge-adapter  -c configs/agriculture/lfm25_1.2b.yaml --run-dir <run>
```

Export verifies the artifact rather than trusting the file listing: shards load,
every LoRA tensor's shape matches the declared rank, and every target module
exists in the base checkpoint. Merging additionally verifies that weight tying
survived. `docs/DEPLOYMENT.md` covers edge export paths.

---

## 11. Testing

```bash
uv run pytest                      # 196 tests, ~60 s, CPU only
uv run pytest -m "not train"       # skip the tests that run optimizer steps
uv run ruff check src tests
```

The suite covers configuration, dataset validation, chat-template formatting and
loss masking, model loading, LoRA attachment and the parameter audit,
checkpoint save/reload, adapter reload, merging, and inference generation. Tests
that need a real checkpoint skip cleanly if it is not cached; set
`LFM_RUN_SLOW=0` to skip them unconditionally.

---

## 12. Troubleshooting

`docs/TROUBLESHOOTING.md` covers CUDA/VRAM errors, loss stuck at NaN, generation
that never stops, `target modules not found`, empty validation splits, and more.
The short version:

| Symptom | Cause | Fix |
|---|---|---|
| `Target modules auto not found` | PEFT received the literal string | never happens; discovery resolves first |
| Generation never stops | terminator not trained | `tokenization.append_eos_token: true` |
| Validation loss looks too good | leakage across splits | `prepare` dedups **before** splitting |
| OOM mid-run | batch × seq × vocab | lower batch, raise accumulation, enable checkpointing |
| `dataset.path is required` | config incomplete | run `prepare` or set `use_processed` |

---

## 13. Reproducibility

Seeded Python/NumPy/PyTorch (and CUDA when present), content-hashed splits, and a
run manifest recording exact package versions, git commit, hardware, seeds,
dataset hash and the LoRA audit. `resolved_config.yaml` freezes the fully-resolved
config so an old run never depends on mutable YAML.

Override any value from the environment for sweeps:

```bash
LFM_TRAINING__LEARNING_RATE=1e-4 LFM_LORA__R=32 \
  uv run lfm-finetune train -c configs/agriculture/lfm25_1.2b.yaml
```

---

## 14. Documentation

| Document | Contents |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Module map, data flow, design decisions |
| [`docs/DATA_COLLECTION_GUIDE.md`](docs/DATA_COLLECTION_GUIDE.md) | **Start here before training**: dataset templates, collection practice, step-by-step commands, checklist |
| [`docs/DATASET.md`](docs/DATASET.md) | Schema, quality gates, provenance, MICO categories |
| [`docs/TRAINING.md`](docs/TRAINING.md) | **LFM2.5 chat-template findings**, masking, LoRA, hyperparameters |
| [`docs/EVALUATION.md`](docs/EVALUATION.md) | What is and is not measured, base-vs-tuned protocol |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | Adapter vs merged, edge export paths for LFM2.5 |
| [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) | Error messages and fixes |
| [`datasets/README.md`](datasets/README.md) | Dataset authoring, licensing, checklist |
| [`docs/DELIVERABLE_REPORT.md`](docs/DELIVERABLE_REPORT.md) | Validation results, exact versions/commands, limitations, next steps |

---

## 15. Known limitations

1. **No real agriculture data has been trained on here.** The validated runs use
   a synthetic fixture (arithmetic/string ops, computed answers). They prove the
   pipeline, not domain quality. See `docs/TRAINING.md`.
2. **This machine has no GPU**, so bf16, flash-attention, multi-GPU and the
   1.2B/2.6B configs are exercised only through preflight arithmetic and
   fail-fast guards, not real training. See `docs/TROUBLESHOOTING.md`.
3. **Evaluation proxies are not correctness metrics.** Agricultural correctness
   requires human scoring of the rubric sheet.
4. **Near-duplicate detection is character-shingle Jaccard.** It catches
   templated paraphrase duplication, not semantic similarity.
5. **Bit-exact reproducibility across different GPUs/driver versions is not
   guaranteed** even with fixed seeds; the manifest records the exact
   environment so a mismatch is detectable.

---

## 16. References

- Model card and fine-tuning docs: <https://docs.liquid.ai/lfm/fine-tuning/trl>
- Liquid's training stack: <https://github.com/Liquid4All/leap-finetune>
- LFM2 technical report: arXiv:2511.23404
- LFM2.5 announcement: <https://www.liquid.ai/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai>