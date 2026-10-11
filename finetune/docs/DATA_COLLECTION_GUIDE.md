# Data Collection & Pre-Training Guide

Everything you need to do **before** running fine-tuning, in order, with the
exact commands and what each one tells you.

Audience: someone who has to actually produce a working dataset. No farming
background assumed, no ML background assumed.

---

## Contents

1. [The one thing that matters](#1-the-one-thing-that-matters)
2. [What you are building](#2-what-you-are-building)
3. [Step 1 — Set up the folder](#3-step-1--set-up-the-folder)
4. [Step 2 — Write the dataset file](#4-step-2--write-the-dataset-file)
5. [Step 3 — Write the manifest](#5-step-3--write-the-manifest)
6. [Step 4 — Point the config at it](#6-step-4--point-the-config-at-it)
7. [Step 5 — Run preflight](#7-step-5--run-preflight)
8. [Step 6 — Clean and split](#8-step-6--clean-and-split)
9. [Step 7 — Verify the loss mask](#9-step-7--verify-the-loss-mask)
10. [Step 8 — Run the pilot](#10-step-8--run-the-pilot)
11. [How to collect good data](#11-how-to-collect-good-data)
12. [Common rejection reasons](#12-common-rejection-reasons)
13. [Checklist](#13-go--no-go-checklist)

---

## 1. The one thing that matters

**The model is not the bottleneck. The data is.**

The code, the GPU and the hyperparameters are all ready and tested. What
determines whether MICO becomes useful is the quality, breadth and honesty of
the farming data you collect.

| | Effect |
|---|---|
| 200 good examples | Model learns the *style* — asks better questions, answers are structured. Limited farming knowledge. |
| 1,000–3,000 good examples | Genuine domain adaptation. This is the target. |
| 3,000 examples covering 2 crops only | Model believes there are 2 crops. Coverage beats volume. |

---

## 2. What you are building

One file of examples plus one file describing them:

```text
finetune/data/raw/mico-agriculture/
    mico-agriculture-v1.jsonl   <- the examples (JSON Lines)
    manifest.yaml               <- provenance: where from, licence, limitations
```

`data/raw/` is git-ignored. Real corpora are large and may be licensed. Only
`datasets/` is committed.

---

## 3. Step 1 — Set up the folder

```bash
cd finetune
mkdir -p data/raw/mico-agriculture
cp datasets/TEMPLATE_manifest.yaml data/raw/mico-agriculture/manifest.yaml
cp datasets/TEMPLATE_dataset.jsonl data/raw/mico-agriculture/mico-agriculture-v1.jsonl
```

Both templates are already in the repo. The dataset template contains 7 records
showing every accepted format, with `<FILL IN>` markers.

---

## 4. Step 2 — Write the dataset file

### The format

One example per line. No blank lines, no comments, no trailing commas.

```json
{"id": "agri-000001", "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "metadata": {"crop": "tomato", "source": "...", "license": "..."}}
```

### The four required rules

1. **A `user` turn, then an `assistant` turn.** That is the whole learning signal.
2. **At most one `system` turn, and only as the first message.**
3. **The assistant's answer must be at least 80 characters** (agriculture config).
   This is the strictest rule and it exists because *"Apply fungicide."* is not a
   useful answer to a farmer.
4. **`metadata.source` and `metadata.license` on every record.**

### Worked example

```json
{
  "id": "agri-000042",
  "messages": [
    {"role": "user", "content": "my tomato plants have brown rings on the lower leaves. what is this"},
    {"role": "assistant", "content": "Rings on the lower leaves can come from a few different causes, and they are hard to tell apart from one photo of leaves. Two things would narrow it down: whether the centre of each ring goes dry and cracks, and whether the problem started after several days of wet weather. Until then, remove the worst affected leaves and keep water off the foliage, because damp leaves make it spread. If new plants keep showing it, contact your local agriculture extension officer and ask what is registered for tomato in your area this season."}
  ],
  "metadata": {
    "domain": "agriculture",
    "category": "diagnosis_style_qa",
    "crop": "tomato",
    "growth_stage": "vegetative",
    "symptom": ["brown rings", "lower leaves first"],
    "region": "IN-KA",
    "confidence": 0.6,
    "source": "extension-bulletin-2024-karnataka",
    "license": "CC-BY-4.0",
    "language": "en"
  }
}
```

Note what that answer does and does not say. It gives **three** plausible causes
rather than one diagnosis, names the **two details** that would separate them,
gives a **safe action available today**, and **escalates**. That structure is the
contract in `configs/agriculture/_domain.yaml`.

### The five answer structures that matter

| Category | Shape of a good answer |
|---|---|
| `diagnosis_style_qa` | 2–3 plausible causes → the detail that separates them → a safe next step → when to escalate |
| `clarification_questions` | **Asks 1–3 specific missing details. Does NOT diagnose.** |
| `safety_uncertainty` | Refuses to name a chemical without crop/stage/region; names the active ingredient; mentions pre-harvest interval and protective equipment; escalates. **Never invents a dose.** |
| `crop_management` / `irrigation_guidance` | Ordered steps, timing, what to use, what it costs or needs |
| `farmer_conversational` | Same content, plainer language, shorter sentences |

### Other formats accepted

```json
{"instruction": "...", "input": "...", "output": "..."}    -> user / assistant
{"prompt": "...", "completion": "..."}                     -> user / assistant
{"question": "...", "answer": "..."}                       -> user / assistant
{"conversations": [{"from": "human", "value": "..."}]}     -> user / assistant
```

`dataset.format: auto` (the default) detects each line independently, so you can
mix formats in one file.

### Writing tool

Any of these is fine, as long as the output is one JSON object per line:

- A **spreadsheet** → export CSV → a 3-line Python script converts it to JSONL.
  The script is worth writing once.
- **VS Code** with a JSONL extension (syntax highlighting per line).
- **A text editor** — it is just one long line per example.

**Never** hand-type JSON with errors. `preflight` will tell you exactly which
line is broken.

---

## 5. Step 3 — Write the manifest

This is not paperwork. It is the only way to know, a year later, whether you are
allowed to ship the model.

Open the copy you made in Step 1 and fill in:

```yaml
name: mico-agriculture
version: 0.1.0
source: >
  ICAR annual reports 2019-2025; Karnataka and Punjab state agriculture
  university extension bulletins; structured farmer interviews,
  6 districts, 2026 Q1
license: CC-BY-4.0
created: 2026-01-15
categories: [diagnosis_style_qa, clarification_questions, safety_uncertainty]
language: [en]
annotation_method: reviewed by 2 agronomists, disagreements adjudicated by a third
review_status: double-reviewed
known_limitations:
  - under-represented in coconut and arecanut
  - symptom descriptions normalised to English, losing original Hindi phrasing
synthetic: false
```

### The four questions it forces you to answer

| Field | Why it matters |
|---|---|
| **license** | Training on unlicensed data can make your model unusable commercially. `unknown` is treated as a **blocking error**. |
| **source** | Which document or interview each claim came from. Required for audit and for updating when facts change. |
| **annotation_method** | Who wrote the answers — a human, an LLM, or extracted verbatim. Do not describe LLM output as human-authored. |
| **known_limitations** | Bias and gaps that will not go away by being unmentioned. |

### Licence options

| Value | Meaning |
|---|---|
| `CC0-1.0` | No restrictions. Safest to share. |
| `CC-BY-4.0` | Free use, must give credit. Most extension material. |
| `ODbL-1.0` | Free, share-alike. |
| `internal-use-only` | Own interviews and notes. Fine for MICO; blocks redistribution. |

**If several sources carry different licences, use separate corpora with separate
manifests.** Merging them means the strictest licence governs the whole thing.

---

## 6. Step 4 — Point the config at it

Create your own config by extending the agriculture domain config. Do not edit
the shipped files — you want to be able to change models without losing your data
settings.

```bash
cp configs/agriculture/lfm25_1.2b_6gb.yaml configs/mico-v1.yaml
```

Then change just the dataset path:

```yaml
# configs/mico-v1.yaml
extends: agriculture/lfm25_1.2b_6gb.yaml

run:
  name: mico-agri-v1

dataset:
  path: data/raw/mico-agriculture
  use_processed: true
```

Check it resolves:

```bash
uv run lfm-finetune show-config -c configs/mico-v1.yaml
```

---

## 7. Step 5 — Run preflight

**One command that tells you whether you are ready.** This is the command to run
before every training run.

```bash
uv run lfm-finetune preflight -c configs/mico-v1.yaml
```

It checks, cheapest first: environment → config → hardware/dtype → dataset path
→ provenance → schema and quality → split provenance → chat template → memory
estimate.

### Reading the output

```text
[PASS] dataset / schema + quality
       1180 accepted (96.4%), 45 rejected, 62 normalised, 31 exact + 8 near duplicates
[WARN] dataset / size
       1180 examples
       -> Liquid's guidance is 500-5000 task examples. Below ~200 the model
          memorises rather than generalises.
...
RESULT: GO -- no blocking problems. (2 warning(s), 1 skipped)
```

| Line | Meaning |
|---|---|
| `[PASS]` | Fine. |
| `[WARN]` | Your call. Warnings do not stop you, but read them. |
| `[FAIL]` | **Blocking.** `RESULT: NO-GO`. Fix it before spending GPU time. |
| `[SKIP]` | Not checked (e.g. LoRA needs `--deep`, which loads the checkpoint). |

The command prints the exact next commands when the result is GO.

### Four checks worth understanding

**`dataset / schema + quality`** — if fewer than **50 %** of your rows survive,
preflight blocks. That is nearly always a *shape* bug (wrong key name, missing
assistant turn, leftover `<FILL IN>` markers), not a content problem.

**`dataset / provenance manifest`** — blocks if the manifest is missing or the
licence is `unknown`.

**`splits / provenance`** — catches the most expensive mistake available:
changing `dataset.path` without re-running `prepare`. `data/splits/` is shared
across configs, so training would otherwise proceed happily **on the previous
corpus**.

**`template / assistant mask strategy`** — will tell you if your checkpoint lacks
`{% generation %}` blocks. LFM2.5-1.2B-Base does. This pipeline masks by prefix
length instead and asserts the prefix property, so it is handled — but it is
worth seeing the message once.

### Full check including the model

```bash
uv run lfm-finetune preflight -c configs/mico-v1.yaml --deep
```

Loads the checkpoint and attaches LoRA. Takes about a minute. Confirms the
discovered target modules and the trainable-parameter count.

---

## 8. Step 6 — Clean and split

```bash
uv run lfm-finetune prepare -c configs/mico-v1.yaml
```

This reads your raw file and produces:

```text
data/splits/train.jsonl       ~80 %
data/splits/validation.jsonl  ~10 %
data/splits/test.jsonl        ~10 %   <- never trained on
data/validated/accepted.jsonl
data/validated/rejected.jsonl       <- every rejected row, with a reason
data/validated/report.json          <- machine-readable report
```

What it does, in an order that matters:

1. reads your JSONL
2. converts every line to the standard format
3. cleans text (unicode, control characters, whitespace, HTML)
4. applies the quality rules — **nothing is deleted without a recorded reason**
5. removes exact duplicates
6. removes near-duplicates (paraphrases of the same example)
7. splits into train / validation / test
8. checks that no example appears in two splits
9. counts tokens using the real tokenizer and writes statistics

**Deduplication happens before splitting, on purpose.** Duplicates are the most
common cause of a validation score that looks great and means nothing.

### Read the report

```bash
python -m json.tool data/validated/report.json | head -60
```

Look at:

| Key | What it tells you |
|---|---|
| `counts` | accepted / rejected / deduplicated / per-split counts |
| `rejection_summary` | why rows were dropped, most frequent first |
| `warnings` | size warnings, truncation warnings, overfit warnings |
| `statistics.per_split` | token counts, median, p95, max |
| `statistics.categories` | which task categories you actually have |
| `statistics.crops` | crop balance — **check this is not two crops** |
| `leakage.hard` | must be `has_leakage: false` |
| `dataset_hash` | recorded in the run manifest |

If `statistics.categories` shows one category dominating, you have a coverage
problem, not a volume problem.

### Two failures that will happen to you

**`DatasetLeakageError: Split produced empty validation set`**

The hashed splitter assigns by content, so a tiny corpus can put nothing in
validation. With 9 examples and an 80/10/10 split, 10 % of 9 rounds to zero.

```bash
# for a tiny trial corpus
uv run lfm-finetune prepare -c <config> \
  -s dataset.splitting.train_ratio=0.6 -s dataset.splitting.validation_ratio=0.2
```

Real corpora (500+) do not hit this.

**`duplicate_near: 700` — most of your data vanished**

You wrote **templated answers**: the same paragraph with the crop name swapped.
Near-duplicate detection removes them, and that is deliberate — templated answers
teach the model one script rather than agriculture.

Write genuinely different answers per crop. Ask: *what would an agronomist
actually say about this specific crop?*

Note that the shared **system prompt** is excluded from that comparison, so a
common system prompt alone will not trigger it.

---

## 9. Step 7 — Verify the loss mask

This checks that the model will be trained on **answers**, not on questions.

```bash
uv run lfm-finetune inspect-template -c configs/mico-v1.yaml --samples 3
```

It prints your real examples, token by token, with the scored tokens marked:

```text
example id=agri-000042 len=210 prompt_tokens=76 loss_tokens=134 coverage=63.7%
--------------------------------------------------------------------------------
    0 id=1       '<|startoftext|>'
    1 id=6       '<|im_start|>'
    2 id=5273    'user'
   ...
   76 id=1234    'Rings'          <- scored
   ...
Blank 'marker' column = token does NOT contribute to the loss.
```

**What to check:**

- The scored span is the **assistant's answer**.
- The farmer's question is **not** scored.
- The very last scored token is the end-of-turn marker, so the model learns to stop.
- Coverage is roughly 20–70 %. Very low is fine when answers are long; above
  **95 %** means masking is broken and you must stop.

---

## 10. Step 8 — Run the pilot

```bash
uv run lfm-finetune train -c configs/mico-v1.yaml --pilot \
  -s dataset.max_train_examples=500
```

About 10 minutes on your laptop GPU. It answers six questions:

| Question | Where to look |
|---|---|
| Does it train without crashing? | Does it reach the end, or OOM? |
| Is memory stable? | `nvidia-smi` in another terminal; does VRAM climb all the way? |
| Are the LoRA modules right? | `trainable parameters ≈ 1.7 %` in the output |
| Does validation loss behave? | `metrics.jsonl` — should fall, then flatten |
| Does the adapter reload? | `Adapter reload verified` in the log |
| Does inference work? | `uv run lfm-finetune inference -c configs/mico-v1.yaml --run-dir <run>` |

If the pilot is clean, run the real thing:

```bash
uv run lfm-finetune train -c configs/mico-v1.yaml
```

### If it runs out of memory

In order:

```bash
# 1. free the GPU: close browsers with video, Discord screen-share, other python
nvidia-smi

# 2. halve the text length (biggest single win)
uv run lfm-finetune train -c configs/mico-v1.yaml \
  -s tokenization.max_seq_length=1024

# 3. smaller model
uv run lfm-finetune train -c configs/agriculture/lfm25_350m.yaml
```

---

## 11. How to collect good data

### Where farming knowledge legitimately comes from

| Source | What it gives you | Note |
|---|---|---|
| ICAR / KVK bulletins | Crop-specific, region-specific, already reviewed | Check each licence; many are CC-BY |
| State agriculture university extension material | Regional practices, locally correct | Usually the most useful source |
| Package of practices (PoP) documents | Authoritative per-crop calendars and schedules | Published by state universities |
| Government scheme documents | Scheme eligibility, application process | Useful for `farmer_conversational` |
| Farmer interviews | **Real phrasing** — the most valuable and least available | Consent and anonymisation required |
| Kisan Call Centre transcripts | Real questions, real confusion patterns | Anonymise |

**The most underrated source is real farmer phrasing.** Most corpora are written
in textbook English. Farmers write short, misspelt, code-switched messages. A
model trained only on textbook phrasing will fail on its actual users.

### The collection process

1. **Decide the category mix first.** Write down how many of each you want
   before you start collecting, or you will end up with 80 % diagnosis Q&A and
   nothing else.

2. **Collect per source, keep provenance per record.** Write `metadata.source`
   as you go. Retrofitting it later is painful and error-prone.

3. **Review in two passes.** One agronomist writes or extracts; a second checks
   the agronomy. Disagreements go to a third. Record this in
   `annotation_method`.

4. **Do not let one model write the answers for another model.** LLM-generated
   agricultural text is fine for *research experiments* — mark it
   `synthetic: true` — but it is not a knowledge source for a model farmers will
   rely on.

5. **Balance crops and regions deliberately.** Check
   `statistics.crops` in the report after `prepare`.

6. **Never invent a fact.** If you do not know, the record should not exist.

### How much is enough

| Stage | Count | What you get |
|---|---|---|
| First pilot | 300–500 | Proves the pipeline; learns style |
| First real model | 800–1,500 | Genuine domain adaptation |
| Good model | 2,000–5,000 | Broad, robust coverage |

Below ~200 the model memorises rather than generalises, and preflight will warn.

### Ethics and legality — the short version

- Get consent for interviews; anonymise names, village specifics and GPS.
- Never include personal financial or health information.
- Respect the licence of every source. `CC-BY-ND` (no derivatives) is **not**
  usable as training data.
- Keep a record of where every claim came from, so a wrong fact can be traced
  and corrected.

---

## 12. Common rejection reasons

From `report.json → rejection_summary`, with what each really means:

| Reason | What happened | Fix |
|---|---|---|
| `unrecognised_format` | Line is not a message list and not an instruction pair | Check the key names |
| `missing_turn` | No assistant turn after a user turn | Add the answer |
| `empty_assistant_response` | Answer under 80 characters | Real answers, or `min_assistant_chars` |
| `placeholder_text` | Still contains `<FILL IN>`, TODO, Lorem ipsum | You have not finished filling the template |
| `multiple_system_prompts` | Two system messages in one record | Keep only the first |
| `prompt_leakage` | Answer repeats the question verbatim | Templated data; rewrite genuinely |
| `duplicate_exact` | Identical to an earlier record | Expected if you copied a template row |
| `duplicate_near` | A paraphrase of an earlier record | Genuine duplication; keep one |
| `too_many_turns` | More than 16 messages | Split the conversation |

Every rejected row is kept in `data/validated/rejected.jsonl` with the reason and
an excerpt. Look there first — it is usually immediately obvious what went wrong.

```bash
head -3 data/validated/rejected.jsonl | python -m json.tool
```

---

## 13. Go / No-Go checklist

Print this, work down it.

### Data

- [ ] Folder exists: `data/raw/mico-agriculture/`
- [ ] Corpus is one JSON object per line, valid JSON, no blank lines
- [ ] Every record has a `user` turn then an `assistant` turn
- [ ] Every assistant answer is ≥ 80 characters
- [ ] Every record has `metadata.source` and `metadata.license`
- [ ] `manifest.yaml` filled in — `license` is **not** `unknown`
- [ ] `known_limitations` written honestly
- [ ] At least 8 categories represented, not just diagnosis Q&A
- [ ] At least 10 different crops, balanced
- [ ] Answers follow the contract: multiple causes, asks for missing details, safe actions, escalation

### Checks

- [ ] `uv run lfm-finetune validate -c configs/mico-v1.yaml` → acceptance ≥ 90 %
- [ ] `uv run lfm-finetune prepare  -c configs/mico-v1.yaml` → read `report.json`
- [ ] `report.json → leakage.hard.has_leakage` is `false`
- [ ] `report.json → statistics.crops` is not dominated by 2 crops
- [ ] `report.json → warnings` read and understood
- [ ] `uv run lfm-finetune preflight -c configs/mico-v1.yaml` → **RESULT: GO**
- [ ] `uv run lfm-finetune inspect-template -c configs/mico-v1.yaml --samples 3`
      → scored span is the answer, coverage 20–70 %
- [ ] `uv run lfm-finetune preflight -c configs/mico-v1.yaml --deep`
      → trainable ≈ 1.7 %, targets discovered

### Training

- [ ] `uv run lfm-finetune train -c configs/mico-v1.yaml --pilot -s dataset.max_train_examples=500`
      completed, no OOM
- [ ] Validation loss fell then flattened
- [ ] `Adapter reload verified` in the log
- [ ] `uv run lfm-finetune baseline -c configs/mico-v1.yaml` → **before** full training
- [ ] Full run: `uv run lfm-finetune train -c configs/mico-v1.yaml`
- [ ] `uv run lfm-finetune evaluate -c configs/mico-v1.yaml` → read `comparison.md`
- [ ] An agronomist scored the rubric sheet for base and tuned, blind
- [ ] `uv run lfm-finetune export-adapter -c configs/mico-v1.yaml --run-dir <run>`

---

## Related documents

| Document | For |
|---|---|
| [`../docs/DATASET.md`](../docs/DATASET.md) | Full schema, every quality gate, deduplication, provenance rules |
| [`../datasets/schema.json`](../datasets/schema.json) | Machine-readable record schema |
| [`../datasets/manifest.schema.json`](../datasets/manifest.schema.json) | Machine-readable manifest schema |
| [`../datasets/README.md`](../datasets/README.md) | Authoring guide and checklist |
| [`../docs/TRAINING.md`](../docs/TRAINING.md) | LoRA, loss masking, hyperparameters |
| [`../docs/EVALUATION.md`](../docs/EVALUATION.md) | How to tell whether it actually got better |
| [`../docs/TROUBLESHOOTING.md`](../docs/TROUBLESHOOTING.md) | Every error message and its fix |