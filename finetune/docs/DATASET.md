# Dataset

How to build a corpus that this pipeline will accept, and what it will refuse.

---

## 1. Canonical format

One record, JSON Lines:

```json
{
  "id": "agri-000123",
  "messages": [
    {"role": "system", "content": "You are MICO, ..."},
    {"role": "user", "content": "My tomato plants have dark brown rings on lower leaves."},
    {"role": "assistant", "content": "Ring-shaped spots starting on lower leaves could be ..."}
  ],
  "metadata": {
    "domain": "agriculture", "category": "diagnosis",
    "crop": "tomato", "plant": "Solanum lycopersicum",
    "disease": null, "pest": null, "nutrient_issue": null,
    "growth_stage": "vegetative",
    "symptom": ["dark brown concentric lesions", "lower leaves first"],
    "region": "IN-KA", "climate": "tropical_subhumid", "soil": "red_soil",
    "irrigation": "drip_3x_week", "treatment": "remove_infested_leaves",
    "confidence": 0.6,
    "source": "icar-extension-publication", "license": "CC-BY-4.0",
    "language": "en"
  }
}
```

Machine-readable: `datasets/schema.json`. Provenance: `datasets/manifest.schema.json`.

### Accepted alternatives

`dataset.format: auto` (default) sniffs every record independently — a mixed
corpus is common and rejecting a whole file because 3 % of rows use a different
shape is hostile.

| Input | Normalised to |
|---|---|
| `{"messages": [...]}` | as-is |
| `{"conversations": [{"from": "human", "value": "..."}]}` | role aliases mapped |
| `{"role": ..., "content": ...}` | single-turn conversation |
| `{"instruction", "input", "output"}` | `input` appended to `instruction` |
| `{"prompt", "completion"}` | user / assistant |
| `{"question", "answer"}` | user / assistant |

Role aliases: `human`/`usr`/`prompter` → user, `gpt`/`bot`/`model` → assistant,
`observation`/`function` → tool.

### Metadata is not in the prompt

Metadata is preserved for analysis, filtering and evaluation slicing, and is
deliberately **not** interpolated into the prompt. A model trained without tags
must not be evaluated with them, or the comparison is invalid. If you do inject
tags, do it for both training and evaluation and record that you did.

---

## 2. Quality gates

Enforced by `prepare`, each rejection recorded with a reason and an excerpt in
`data/validated/rejected.jsonl`.

| Reason | Trigger | Why |
|---|---|---|
| `unrecognised_format` | no message list and no instruction pair | cannot become a conversation |
| `invalid_role` | role outside the known set | template cannot render it |
| `missing_turn` | no user turn, or no assistant turn after a user turn | nothing to supervise |
| `multiple_system_prompts` | >1 system message | the template honours only `messages[0]`; the rest render as literal user turns |
| `all_messages_empty` | everything empty after normalisation | no signal |
| `empty_user_message` | user turn below `min_user_chars` | — |
| `empty_assistant_response` | answer below `min_assistant_chars` | teaches the model to stop immediately |
| `assistant_response_too_long` | above `max_assistant_chars` | runaway target |
| `too_many_turns` / `too_few_messages` | outside bounds | — |
| `placeholder_text` | matches a `placeholder_patterns` regex | unfinished/template text |
| `prompt_leakage` | answer contained in the prompt, or ≥2 shared 12-word spans | templated/parroting data |
| `duplicate_exact` | identical normalised conversation | — |
| `duplicate_near` | char 5-gram Jaccard ≥ threshold | paraphrase duplication |

### Normalisation, applied before dedup

NFC unicode folding, control/zero-width/surrogate stripping, HTML tag removal and
entity decoding, CRLF normalisation, whitespace collapsing, trailing-space
stripping.

Normalising **before** dedup is essential: two copies differing only in trailing
whitespace would otherwise survive as separate rows and split across train and
validation. NFC matters for the same reason — canonically equivalent sequences
("é" as one codepoint vs `e` + combining accent) must dedup.

### Prompt-leakage detection

Two signals, both cheap:

1. the whole normalised answer appears inside a user turn;
2. two or more identical 12-word shingles appear in both the answer and the
   prompt.

The second catches templated datasets where the assistant turn restates the
question before answering — a pattern strongly correlated with the model learning
to echo rather than reason.

---

## 3. Deduplication

- **Exact** — SHA-256 over the normalised payload.
- **Near** — character 5-gram Jaccard over an inverted index, so it is
  O(candidates) rather than O(n²). Character shingles (not words) make it robust
  to spelling variation, inflection and Hindi/English code-switching.
- **Shared prompt, different answer** — kept, but flagged
  `metadata._quality_flags: ["shared_prompt"]`. These are legitimately hard items
  or labelling disagreements and deserve human adjudication.
- **Repeated answer across many prompts** — flagged `repeated_answer`.

Tighten the threshold (default 0.90; the agriculture config uses 0.85) rather
than loosening it. Loosening produces false positives that silently shrink the
training set.

---

## 4. Splitting

```yaml
dataset:
  splitting:
    strategy: hashed        # content-addressed (recommended)
    train_ratio: 0.80
    validation_ratio: 0.10  # test gets the remainder
    seed: 42
```

`hashed` assigns by `BLAKE2b(seed || conversation_hash)`, so adding examples never
moves existing ones between splits — a recorded validation score stays valid.
`sequential` is faster but fragile under dataset growth.

Keep `validation_ratio < 0.2` so a real **test** split exists. The test split is
used for the baseline and the base-vs-tuned comparison and is never trained on.

After splitting, a hard-fail audit checks for exact duplicates across boundaries.
Soft overlaps (shared prompts, shared short answers) are reported with counts.

---

## 5. Statistics

Token counts use the **real tokenizer and chat template**, not a heuristic, so
reported lengths are the tokens actually trained on:

```json
{
  "num_examples": 42, "total_tokens": 3006,
  "mean_tokens": 71.6, "median_tokens": 65, "p95_tokens": 91, "max_tokens": 256,
  "roles": {"assistant": 42, "system": 40, "user": 43},
  "categories": {"arithmetic": 22, "string_ops": 20},
  "crops": {}, "languages": {"en": 42},
  "turns_histogram": {"2": 25, "4": 14, "3": 3}
}
```

Advisory warnings fire for small datasets, very short answers, very short
sequences and truncation at the cap.

---

## 6. Provenance and licensing

Place a manifest next to the raw data as `manifest.yaml`, `manifest.yml` or
`manifest.json`:

```yaml
name: mico-agriculture-core
version: 0.4.0
source: ICAR annual reports 2019-2025; state agriculture university bulletins
license: CC-BY-4.0
created: 2026-01-15
num_examples: 3200
annotation_method: reviewed by 2 agronomists, disagreements adjudicated by a third
language: [en, hi]
review_status: double-reviewed
known_limitations:
  - under-represented in coconut and arecanut
  - symptom descriptions normalised to English, losing some Hindi phrasing
```

`prepare` fills in `num_examples`, `content_hash` and `prepared_at`, and reports
provenance problems into `report.json` and every run manifest. A manifest with
`license: unknown` is flagged loudly.

### Rules

- **Never train on data of unknown provenance.** The pipeline cannot enforce
  this, so it makes the omission visible.
- **Never include examples authored by the model being trained** in a
  production corpus. Record LLM-generated data honestly as such
  (`annotation_method`, `synthetic: true`); it is research material, not a
  knowledge source.
- **Do not download corpora implicitly.** External datasets must be explicitly
  configured, with a manifest you wrote.
- **Do not fabricate agricultural facts.** Placeholders exist only to exercise
  the schema and are marked `synthetic: true, placeholder: true`.
- **`confidence` is a human/curator judgement**, not a model score.

---

## 7. MICO categories

The engine treats all categories identically; the categories exist for coverage
auditing and evaluation slicing.

| Category | Purpose |
|---|---|
| `diagnosis_style_qa` | symptom → plausible causes → next step |
| `symptom_reasoning` | why here and not elsewhere |
| `crop_management` | spacing, staking, rotation, mulching, pruning |
| `pest_identification` | distinguish look-alike damage |
| `nutrient_deficiency` | visual signatures and corrections |
| `irrigation_guidance` | scheduling, method, water stress |
| `weather_risk_reasoning` | rainfall/heat/frost risk to a crop and its pests |
| `farmer_conversational` | open-ended, multi-turn, low-literacy phrasing |
| `clarification_questions` | correct behaviour when information is missing |
| `safety_uncertainty` | chemical limits, pre-harvest intervals, refusal to guess |
| `indian_terminology` | correct local crop/input/unit vocabulary |
| `hindi_english_mixed` | Devanagari-inflected queries (later phase) |

### Dataset quality targets

Liquid's guidance is **500–5000** task examples, matched to production inputs.
Quality and distribution matter more than volume. Prioritise:

1. realistic **user phrasing**, including short and low-literacy questions;
2. answers that follow the system prompt's contract (ask, then reason, then act);
3. symptom *atoms* in metadata so co-occurrence can be analysed;
4. balanced coverage across crops and regions — a corpus dominated by two crops
   teaches a model that only knows two crops.

---

## 8. Workflow

```bash
# 1. write data/raw/<dataset>/<corpus>.jsonl + manifest.yaml
# 2. dry check
uv run lfm-finetune validate -c configs/agriculture/lfm25_1.2b.yaml
# 3. clean, dedup, split, statistics, report
uv run lfm-finetune prepare  -c configs/agriculture/lfm25_1.2b.yaml
# 4. read data/validated/report.json -> counts, warnings, leakage
# 5. inspect the loss mask on real data
uv run lfm-finetune inspect-template -c configs/agriculture/lfm25_1.2b.yaml --samples 3
```

`validate` exits non-zero only with `--fail-on-reject`. Both commands write a
machine-readable report; neither discards anything without a recorded reason.