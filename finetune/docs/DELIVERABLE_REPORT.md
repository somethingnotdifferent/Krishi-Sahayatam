# Final Deliverable Report

Project: **Krishi Sahayatam / MICO agriculture LLM**
Date: 2026-10-04 · Repo: `finetune/` · 14 029 LOC (10 931 src · 2 756 tests)

---

## 1. Project tree

```text
finetune/
├── pyproject.toml                    pinned stack, CPU-torch default
├── uv.lock                           1380 lines, exact resolution
├── .gitignore .env.example           secrets + weights + corpora excluded
├── README.md
├── configs/
│   ├── base.yaml                     every field documented (schema reference)
│   ├── lfm_sft.yaml                  LFM2.5 defaults (LoRA auto-discovery, EOS)
│   ├── smoke.yaml                    CPU end-to-end validation
│   └── agriculture/
│       ├── _domain.yaml              system prompt, quality gates, splits
│       ├── lfm25_350m.yaml           low-end / CPU
│       ├── lfm25_1.2b.yaml           MICO primary
│       └── lfm25_2.6b.yaml           edge gateway
├── data/{raw,processed,validated,splits}/
├── datasets/{README.md, schema.json, manifest.schema.json, examples/}
├── docs/{ARCHITECTURE,DATASET,TRAINING,EVALUATION,DEPLOYMENT,TROUBLESHOOTING}.md
├── src/lfm_finetune/
│   ├── config/     schema.py (pydantic, 14 sections)  loader.py (YAML inheritance)
│   ├── data/       schema io validate dedupe split stats pipeline synthetic
│   ├── model/      loader discovery lora
│   ├── training/   tokenization collator args trainer metadata
│   ├── evaluation/ metrics runner
│   ├── inference/  generator cli_loop
│   ├── export/     adapter (export + merge + verify)
│   ├── utils/      errors logging seeding hashing hardware io versions
│   └── cli/        main.py (unified CLI, 16 commands)
├── scripts/        13 thin wrappers
├── tests/          196 tests across 6 files + conftest
└── outputs/ checkpoints/ adapters/ merged/ evaluations/ logs/
```

---

## 2. Exact dependency versions

Validated, not floated. The compatibility envelope comes from Liquid's own
stack ([leap-finetune](https://github.com/Liquid4All/leap-finetune)) and the
official [TRL fine-tuning docs](https://docs.liquid.ai/lfm/fine-tuning/trl).

| Package | Version | Pin rationale |
|---|---|---|
| python | 3.12.15 | leap-finetune requires ≥3.12 |
| torch | 2.11.0+cpu | `>=2.10.0,<2.12.0` per leap-finetune |
| transformers | 5.3.0 | `>=5.3.0,<5.4.0` — LFM2 + trl 1.9 validated on the 5.3 line |
| trl | 1.9.0 | `==1.9.0` exact per leap-finetune |
| peft | 0.21.2 | `>=0.15.2` |
| accelerate | 1.15.0 | `>=1.7.0` |
| datasets | 5.0.1 | |
| safetensors | 0.8.0 | |
| tokenizers | 0.22.2 | (transitive) |
| numpy | 2.5.3 | `>=2.0` |
| huggingface-hub | 1.33.0 | `>=1.0` (leap-finetune override) |
| pydantic | 2.13.5 | config validation |
| PyYAML | 6.0.3 | |
| bitsandbytes | *not installed* | optional `--extra quant`, unused by default |

CPU torch is the uv default so a fresh `uv sync` on a non-CUDA box does not pull
~3 GB of unused CUDA wheels; the swap to a CUDA index is documented in the README.

---

## 3. Model selected

**`LiquidAI/LFM2.5-1.2B-Base`** for MICO (configs/agriculture/lfm25_1.2b.yaml).
**`LiquidAI/LFM2.5-350M-Base`** for CPU/CI validation (configs/smoke.yaml).

The repository contained **no model reference** — an empty `models/` directory and
a one-line README. The checkpoint was therefore chosen from inspection plus
Liquid's published guidance.

---

## 4. Why that model, for this hardware

### Model-side reasoning

| Criterion | 1.2B-Base |
|---|---|
| Capacity | 1.17 B params, 28 T training tokens, 10 conv + 6 attention layers — enough to absorb agricultural vocabulary and a response style |
| Edge tier | Liquid's **"on-device"** class; official GGUF and ONNX conversions published |
| VRAM | ~2.4 GiB bf16; LoRA SFT peaks ≈5.0 GiB → a single 12 GB card (RTX 3060 12G, 4060 Ti 16G, A10, A100) or 2×8 GB |
| 2.6B-Base | ~5.2 GiB weights, ≈9.5 GiB peak → edge *gateway*, not handset; 2.2× the training cost |
| 350M-Base | ~0.7 GiB; learns terminology and style well but materially weaker at multi-step causal reasoning — exactly what diagnosis needs |

**Base, not instruct.** Liquid's model card recommends `*-Base` for "heavy
fine-tuning ... domain-specific (e.g. medical) assistants, training on
proprietary data". Starting from Instruct would use only domain data to un-learn a
general instruct prior — not something a few thousand agricultural examples can
do reliably.

### Detected hardware (this host)

```text
GPU                 NONE — torch.cuda.is_available() == False
CPU                 AMD EPYC 7763, 1 physical / 2 logical cores
System RAM          7.8 GiB total, ~4.9 GiB available
Disk                ~10 GiB free (32 GB overlay, 60% used)
Python / torch      3.12.15 / 2.11.0+cpu (no CUDA runtime compiled in)
bf16 / fp16         unsupported — no CUDA device
```

Ranking computed by `lfm-finetune recommend-model --seq-length 2048`
(feasibility against **system RAM** on a CPU host, VRAM on a GPU host):

| Model | Params | Est. peak | Verdict |
|---|---|---|---|
| LFM2.5-230M-Base | 0.23 B | 3.0 GiB | FEASIBLE |
| **LFM2.5-350M-Base** | **0.35 B** | **3.5 GiB** | **FEASIBLE — used here** |
| LFM2.5-1.2B-Base | 1.17 B | 6.6 GiB | NOT FEASIBLE (needs 6.6 GiB, 4.9 available) |
| LFM2.5-2.6B-Base | 2.60 B | 12.2 GiB | NOT FEASIBLE |
| LFM2.5-8B-A1B-Base | 8.00 B | 33.0 GiB | NOT FEASIBLE |

**Conclusion:** 1.2B-Base remains the correct *MICO deployment* target; this host
cannot train it, so the pipeline was validated on 350M-Base on CPU. The 1.2B and
2.6B configs are parsed, schema-validated and memory-estimated, and training them
requires a CUDA host.

---

## 5. Exact LoRA target modules discovered

Derived programmatically from `model.named_modules()`, then cross-checked against
Liquid's reference presets:

```text
reference check     exact match with Liquid AI leap-finetune preset
excluded (tied)     ['lm_head']

feed_forward.w1          x16     SwiGLU gate
feed_forward.w2          x16     SwiGLU down
feed_forward.w3          x16     SwiGLU up
self_attn.q_proj          x6     GQA query
self_attn.k_proj          x6     GQA key
self_attn.v_proj          x6     GQA value
self_attn.out_proj        x6     GQA output     <-- NOT "o_proj"
conv.in_proj             x10     short conv in
conv.out_proj            x10     short conv out
                          ---
                          92 instances, 9 module types
```

Findings that would break a generic recipe:

1. **`out_proj`, not `o_proj`.** `["q_proj","k_proj","v_proj","o_proj"]` — the
   documented TRL example for LFM — matches nothing on the output projection.
2. **SwiGLU, not gate/up/down proj.**
3. **The conv blocks are adaptable.** `Lfm2ShortConv` wraps a depthwise `Conv1d`
   between two `nn.Linear`, so `conv.in_proj/out_proj` are legitimate PEFT targets.
4. **`lm_head` is excluded** because `config.tie_embedding: true` makes it the
   same tensor as `embed_tokens`. The loader verifies the tie at runtime and the
   config layer rejects `modules_to_save: [lm_head]`.
5. **`all-linear` / name-substring matching is never used.**

---

## 6–7. Parameter counts

Measured on `LFM2.5-350M-Base` (validated run):

```text
total parameters      357,482,240      (base 354,483,968 + adapter)
trainable parameters    2,998,272      (0.839 %)
frozen parameters     354,483,968      (100 % of base)
  LoRA A matrices       1,212,416
  LoRA B matrices       1,785,856
```

Projected for `LFM2.5-1.2B-Base` (same target set, `r` as configured):

| `lora.r` | Trainable | Share |
|---|---|---|
| 8 | ~9.9 M | 0.85 % |
| **16** (default) | **~19.8 M** | **1.69 %** |
| 32 | ~39.6 M | 3.38 % |

Enforced by fail-fast guards: trainable > `min_trainable_params` (1024), share ≤
`max_trainable_pct` (25 %), LoRA A and B both non-empty, **no non-LoRA parameter
trainable**, and a test asserting base weights are bit-identical after attachment.

---

## 8. Dataset schema

Machine-readable: `datasets/schema.json`. Summary:

```json
{
  "id": "agri-000123",
  "messages": [
    {"role": "system",    "content": "You are MICO, ..."},
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "metadata": {
    "domain","category","crop","plant","disease","pest","nutrient_issue",
    "growth_stage","symptom[]","region","climate","soil","irrigation",
    "treatment","confidence","source","license","language"
  }
}
```

Accepted and auto-normalised: `conversations` (`from`/`value`), bare
`{"role","content"}`, `instruction/input/output`, `prompt/completion`,
`question/answer`. Metadata is preserved for analysis and **never injected into
the prompt**. Every record needs `source` + `license`; `license: unknown` is
flagged loudly.

---

## 9–15. Commands

```bash
cd finetune && uv sync
```

**9 — Pilot (stage 6)**
```bash
uv run lfm-finetune prepare -c configs/agriculture/lfm25_1.2b.yaml
uv run lfm-finetune train   -c configs/agriculture/lfm25_1.2b.yaml --pilot \
  -s dataset.max_train_examples=500
```

**10 — Full SFT (stage 7)**
```bash
uv run lfm-finetune train -c configs/agriculture/lfm25_1.2b.yaml
```

**11 — Resume**
```bash
uv run lfm-finetune train -c configs/agriculture/lfm25_1.2b.yaml --resume auto
# resumes into the SAME run directory; metrics.jsonl continues
```

**12 — Evaluation (stage 8)**
```bash
uv run lfm-finetune baseline -c configs/agriculture/lfm25_1.2b.yaml   # before training
uv run lfm-finetune evaluate -c configs/agriculture/lfm25_1.2b.yaml   # base vs tuned
```

**13 — Inference**
```bash
uv run lfm-finetune inference -c configs/agriculture/lfm25_1.2b.yaml --run-dir <run>
uv run lfm-finetune inference -c configs/agriculture/lfm25_1.2b.yaml \
  --run-dir <run> -p "..." --json
```

**14 — Adapter export (preferred artifact)**
```bash
uv run lfm-finetune export-adapter -c configs/agriculture/lfm25_1.2b.yaml --run-dir <run>
```

**15 — Merged model (supported and verified)**
```bash
uv run lfm-finetune merge-adapter -c configs/agriculture/lfm25_1.2b.yaml --run-dir <run>
```

---

## 16. Validation executed (all ten stages)

| Stage | Command | Result |
|---|---|---|
| 0 | `doctor` | **PASS** — all imports, LFM2 support present |
| 0 | `inspect-hardware --recommend` | **PASS** — RAM-aware feasibility ranking |
| 1 | `inspect-model` | **PASS** — 9 targets, exact match with Liquid's preset |
| 2 | `validate` / `prepare` | **PASS** — 42 accepted, 0 rejected, 1 dedup, 29/8/5 split |
| 3 | `inspect-template` | **PASS** — assistant-only mask, verified prefix-consistent |
| 4 | `inspect-lora` | **PASS** — 2 998 272 trainable (0.839 %), base frozen |
| 5 | `train --smoke-test` | **PASS** — 15.8 s, loss 3.31, adapter verified, checkpoint written |
| 6 | resume | **PASS** — continued in place, step 3 → 4, metrics 15 → 28 lines |
| 7 | full SFT | **not runnable** — no CUDA; configs validated + estimates computed |
| 8 | `baseline` / `evaluate` | **PASS** — loss 1.731 → 1.373, token-F1 0.391 → 0.800 |
| 9 | `export-adapter` / `merge-adapter` | **PASS** — 16 MiB adapter, 680.7 MiB merged, tying preserved |
| — | inference | **PASS** — merged ≡ base+adapter under greedy decoding |
| — | `pytest` | **PASS — 196 passed in 47 s** |
| — | `ruff check src tests scripts` | **PASS** — clean |
| — | git hygiene | **PASS** — 91 files staged, 0 weights/artifacts/secrets |

### Bugs found and fixed during validation

Several were only discoverable by actually running the pipeline:

1. **`keep_head_tail` truncation masked the entire answer** when the prompt
   alone exceeded `max_seq_length` — those records trained on nothing while
   looking normal in the statistics. Fixed to reserve budget for the supervised
   span; regression test added.
2. **PEFT received the literal string `"auto"`** as target modules instead of the
   resolved list → "Target modules auto not found".
3. **`pad_token_id` passed twice** to `PeftModel.generate`.
4. **Generation returned the prompt**, because output was sliced from the row's
   prompt start rather than the padded-prompt end.
5. **HF `PrinterCallback` duplicated every metric line** with `%.4g` truncation.
6. **Metrics were only written when a CLI sink was attached**, so programmatic
   `train()` calls produced an empty `metrics.jsonl`.
7. **Adapter verification used meta-device instantiation**, which fails inside a
   live training process. Replaced with direct shard + rank + target checks.
8. **`--json` output was not machine-readable** — human text was interleaved.
9. Resume created a **new** run directory, fragmenting `metrics.jsonl`.
10. `Counter` omitted zero-valued keys, so `counts["rejected"]` raised on clean data.
11. `manifest.json` was ingested as a data record.
12. `recommend_model` reported everything "FEASIBLE" on CPU-only hosts.
13. `torch.manual_seed()` returns a Generator, crashing the `%d` log format.
14. transformers leaves parameters `requires_grad=True`; a "frozen base" assertion
    wrongly failed. Now freezes explicitly.

### LFM2.5-specific findings (all measured, not assumed)

| Finding | Consequence |
|---|---|
| `1.2B-Base` and `350M-Base` ship a template **without `{% generation %}`** | `return_assistant_tokens_mask=True` returns **0 masked tokens**; TRL's `assistant_only_loss=True` would silently train on 100 % of tokens. Masking is therefore computed by prefix length and asserted. |
| `230M/2.6B/8B/Instruct` **do** have the blocks | behaviour would differ by model size; prefix masking is uniform |
| Templates emit `<|im_start|>`/`<|im_end|>` (ids 6/7; 124899/124900 on 128 k vocab) | generation must stop on `eos_token_id`, never hard-coded |
| Default `add_special_tokens=True` **doubles the BOS** | formatter always passes `False` |
| Template appends a newline after `<|im_end|>` | target must end exactly at the terminator, so the newline is stripped |
| `special_tokens_map.json` registers ids 6/7 with **empty** content | the terminator looks absent when reading the raw JSON |

---

## 17. Known limitations

1. **No real agriculture data was trained on.** All validated runs use a synthetic
   fixture (arithmetic/string-ops with computed answers). It proves the pipeline;
   it makes **no claim about agricultural capability**.
2. **This host has no GPU.** bf16, flash-attention, multi-GPU and the 1.2B/2.6B
   configs are exercised only through preflight arithmetic and fail-fast guards,
   not real training.
3. **Evaluation proxies are not correctness metrics.** Agronomic correctness
   requires human scoring; the rubric sheet ships with `null` scores deliberately.
4. **Near-duplicate detection is character-shingle Jaccard** — catches templated
   paraphrase, not semantic similarity.
5. **Bit-exact reproducibility across different GPUs/driver stacks is not
   guaranteed** even with fixed seeds; the run manifest records the environment so
   a mismatch is detectable.
6. **Distribution shift is untested.** No Hindi/Devanagari data exists here; the
   normaliser handles control characters and NFC, but token-level behaviour on
   Devanagari is unverified.
7. **QLoRA/quantisation is unexercised.** `bitsandbytes` is an optional extra and
   is deliberately unused.
8. **Edge conversions (GGUF/ONNX) are unverified** — no device available.

---

## 18. Next recommended engineering step

**Build the agriculture corpus with provenance, then run pilot → full SFT on a
CUDA host.** In priority order:

1. **Assemble 500–2000 sourced examples** across the categories in
   `configs/agriculture/_domain.yaml`, with `manifest.yaml` recording source,
   license, annotation method and known limitations. This is the single
   bottleneck: no amount of hyperparameter work substitutes for it. Prefer
   extension bulletins and agronomist review over LLM generation; if generated
   data is used for research, mark it `synthetic: true`.
2. **Run `validate` + `prepare`, then read `data/validated/report.json`** — fix
   data problems before spending GPU time. Watch `warnings`, `rejection_summary`
   and `leakage`.
3. **Confirm the loss mask on real data** with
   `inspect-template --samples 3`: supervised span must be the assistant text
   plus `<|im_end|>`, and coverage should be 5–30 %.
4. **Pilot on a CUDA host** (`--pilot`, ~500 examples, 50 steps) and answer the
   six pilot questions before the full run.
5. **Full SFT**, then `baseline` → `evaluate`, and have an agronomist score the
   rubric sheet blind for both base and tuned.
6. **Then, and only then**, tune `lora.r`, `learning_rate` and
   `max_seq_length` — and re-run the comparison each time.

Do **not** spend effort on quantisation, multi-node training or a larger model
until a CUDA host has produced a pilot whose base-vs-tuned comparison shows a
real behavioural change.