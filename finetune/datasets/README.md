# Datasets

Authoring guide for Krishi Sahayatam / MICO. Machine-readable definitions:

| File | Purpose |
|---|---|
| **`TEMPLATE_dataset.jsonl`** | **copy this** — 7 records showing every accepted format |
| **`TEMPLATE_manifest.yaml`** | **copy this** — provenance template with inline guidance |
| `schema.json` | canonical record (JSON Schema 2020-12) |
| `manifest.schema.json` | provenance manifest (JSON Schema 2020-12) |
| `examples/` | fixtures for the pipeline |

Full walkthrough: [`../docs/DATA_COLLECTION_GUIDE.md`](../docs/DATA_COLLECTION_GUIDE.md).
Reference: [`../docs/DATASET.md`](../docs/DATASET.md).

---

## Layout

```text
data/raw/<dataset-name>/
    <corpus>.jsonl        # one JSON object per line
    manifest.yaml         # provenance -- REQUIRED
```

`data/raw/`, `data/processed/`, `data/validated/` and `data/splits/` are
git-ignored: real corpora are large and may be licensed. Only
`datasets/examples/` is committed.

---

## Record format

```json
{
  "id": "agri-000001",
  "messages": [
    {"role": "system", "content": "You are MICO, an agricultural advisory assistant..."},
    {"role": "user", "content": "My tomato plants have dark brown rings on the lower leaves."},
    {"role": "assistant", "content": "Ring-shaped spots starting on the lower leaves could be ..."}
  ],
  "metadata": {
    "domain": "agriculture",
    "category": "diagnosis_style_qa",
    "crop": "tomato",
    "plant": "Solanum lycopersicum",
    "disease": null,
    "pest": null,
    "nutrient_issue": null,
    "growth_stage": "vegetative",
    "symptom": ["dark brown concentric lesions", "lower leaves first"],
    "region": "IN-KA",
    "climate": "tropical_subhumid",
    "soil": "red_soil",
    "irrigation": "drip_3x_week",
    "treatment": "remove_infested_leaves",
    "confidence": 0.6,
    "source": "icar-extension-publication",
    "license": "CC-BY-4.0",
    "language": "en"
  }
}
```

Rules the validator enforces:

- at least one `user` turn **followed by** an `assistant` turn;
- at most one `system` message, at index 0;
- no empty content after normalisation;
- answers ≥ `min_assistant_chars` (8 by default, **80** for agriculture).

Also accepted and auto-normalised: `conversations` with `from`/`value`,
bare `{"role", "content"}`, `instruction`/`input`/`output`,
`prompt`/`completion`, `question`/`answer`.

---

## Metadata

All fields optional. Metadata is preserved for analysis and evaluation slicing
and is **not** placed in the prompt.

| Field | Meaning |
|---|---|
| `category` | task category; primary axis for eval slicing (see below) |
| `crop` / `plant` | English common name / scientific name |
| `disease` / `pest` / `nutrient_issue` | cause, or `null` when unknown |
| `growth_stage` | seedling, vegetative, flowering, fruiting, maturity |
| `symptom` | array of symptom **atoms** so co-occurrence can be analysed |
| `region` | subnational code or name |
| `climate`, `soil`, `irrigation` | context |
| `treatment` | recommended action; record the **active ingredient**, not a brand |
| `confidence` | **human/curator** judgement in [0,1]; never a model score |
| `source`, `license`, `language` | provenance |

### Categories

`diagnosis_style_qa` · `symptom_reasoning` · `crop_management` ·
`pest_identification` · `nutrient_deficiency` · `irrigation_guidance` ·
`weather_risk_reasoning` · `farmer_conversational` · `clarification_questions` ·
`safety_uncertainty` · `indian_terminology` · `hindi_english_mixed`

---

## Behavioural contract

The system prompt in `configs/agriculture/_domain.yaml` is a contract. **Answers
must obey it.** If the prompt says "ask before diagnosing" and 40 % of the answers
diagnose without asking, the model learns the dataset, not the prompt.

Required behaviours:

1. Offer 2–3 plausible causes rather than asserting one from a single symptom.
2. Ask for crop, growth stage, region/season, symptom distribution on the plant,
   and recent weather/irrigation/input changes.
3. State uncertainty explicitly.
4. Give safe, practical, ordered actions with timing.
5. Name active ingredients; mention pre-harvest intervals and protective
   equipment; note local restrictions.
6. Recommend a local extension officer or the Kisan Call Centre when the pest or
   disease cannot be identified or is spreading fast.
7. Plain, concise language — short sentences, jargon glossed.

---

## Provenance and licensing

Every component records `source` and `license`. A manifest with
`license: unknown` is flagged loudly in every run manifest.

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
```

**Non-negotiable:**

- Do not train on data of unknown provenance.
- Do not include examples authored by the model being trained in a production
  corpus. If you must, record it honestly (`synthetic: true`) — it is research
  material, not a knowledge source.
- Do not let the pipeline download corpora implicitly. External datasets are
  explicitly configured, with a manifest you wrote.
- Do not fabricate agricultural facts.
- Do not ship a corpus whose licence forbids commercial or derivative use without
  checking with whoever owns the project.

---

## Sizing

Liquid's guidance is **500–5000** task examples matched to production inputs.

| n | What it supports |
|---|---|
| < 200 | pipeline validation only; memorisation is guaranteed at 3 epochs |
| 500 | first real experiment |
| 2000+ | meaningful domain adaptation |

Prioritise realistic user phrasing — including short, low-literacy and
code-switched questions — and balance crops and regions. A corpus dominated by
two crops teaches a model that only knows two crops.

---

## Fixtures

`examples/` contains only pipeline fixtures.

`synthetic.jsonl` is generated by `lfm_finetune.data.synthetic`: arithmetic and
string operations with **computed** answers, so a low loss is meaningful for the
pipeline and says nothing about agriculture.

```bash
uv run lfm-finetune init-synthetic --destination data/raw/synthetic --examples 40
```

`placeholder_agriculture.json` demonstrates the metadata schema. Its text is
explicitly bracketed `[PLACEHOLDER]` and every record is marked
`placeholder: true, synthetic: true`. **It asserts no agronomic facts and must
never be trained on.** Its only purpose is to show how metadata is populated.

---

## Workflow

```bash
# 1. write data/raw/<name>/corpus.jsonl + manifest.yaml
uv run lfm-finetune validate -c configs/agriculture/lfm25_1.2b.yaml  # dry check
uv run lfm-finetune prepare  -c configs/agriculture/lfm25_1.2b.yaml  # clean+split

# 2. read the report before training
cat data/validated/report.json

# 3. confirm the loss mask on YOUR data
uv run lfm-finetune inspect-template -c configs/agriculture/lfm25_1.2b.yaml --samples 3
```

Both commands are non-destructive and record every rejection with a reason.
Nothing is discarded silently.