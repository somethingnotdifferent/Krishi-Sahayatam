# Evaluation

What this pipeline measures, what it deliberately refuses to measure, and the
protocol for deciding whether fine-tuning helped.

---

## 1. The core honesty rule

**A lower held-out loss is not evidence of improved domain capability.**

It shows the model fits the text distribution of that domain better. That is
necessary, not sufficient. A model can drive loss down by memorising, by becoming
terse, or by hedging more. Every artifact this pipeline writes repeats this
warning, and the human rubric sheet ships with `null` scores so that missing
human judgement is visibly missing rather than implied.

```json
"interpretation_guard": "This report compares objective proxies only. Agronomic
correctness, safety and helpfulness require a human to score human_rubric_sheet.
A decrease in held-out loss alone is NOT evidence of improved domain capability."
```

---

## 2. What is measured

### 2.1 Held-out loss

Mean causal-LM loss over the **test** split, assistant tokens only, using the same
loss mask as training. Computed identically for base and tuned models.

Comparable between variants. Not comparable across different masking or
different sequence lengths.

### 2.2 Objective proxies over generated text

| Metric | Definition | Direction |
|---|---|---|
| `asks_for_information_rate` | answer contains a question or an uncertainty/clarification marker | higher is better |
| `overconfidence_rate` | contains absolute-certainty phrasing ("definitely", "always", "guaranteed") | **lower** is better |
| `mean_uncertainty_markers` | count of hedging markers | higher is better |
| `any_format_rate` / `format_tag_counts` | bullets, numbered list, headings, key-value, bold, JSON | higher is better |
| `mean_token_f1_vs_reference` | unigram F1 against the reference answer | higher is better |
| `verbatim_reference_match_rate` | ≥60-char normalised reference reproduced verbatim | **lower** is better (memorisation signal) |
| `indian_terminology_rate` | Indian agricultural terms present | higher is better |
| `mentions_professional_help_rate` | routes to extension officer / Kisan Call Centre | higher is better |
| `mean_questions_per_answer` | `?` count | context-dependent |
| `mean_length_words`, `p95_length_words` | verbosity | context-dependent |
| `empty_rate` | empty generation | **lower** is better |

Every metric is labelled with its direction in the report, and metrics with no
natural direction are marked `n/a` rather than given a fake verdict.

### 2.3 Derived reports

- **before/after examples** — prompt, reference, base output, tuned output, side
  by side.
- **regressions** — per-sample objective declines: lexical overlap dropped
  >0.25, stopped asking for information, gained certainty phrasing, lost output
  structure, produced an empty response.
- **hallucination flags** — verbatim reference copy; confident assertion with no
  request for information; empty or degenerate output.

### 2.4 Evaluation-set disjointness

Every report includes a check that no evaluation prompt also appears in the
training set. Overlap is logged as an error and invalidates the comparison.

---

## 3. What is NOT measured

| Not measured | Why |
|---|---|
| agronomic correctness | requires a qualified human; no automatic proxy is trustworthy |
| safety/chemical correctness | same |
| helpfulness / tone | same |
| Hindi/English fluency | same |
| calibration | would need a confidence-labelled probe set |

These are handed to a human through `human_rubric_sheet` in `comparison.json`:
every sample with every dimension pre-listed, anchored, and scored `null`.

---

## 4. Agriculture rubric

Shipped in `evaluation/metrics.DEFAULT_AGRICULTURE_RUBRIC`; override with
`evaluation.rubric_path`.

| Dimension | What 0 looks like | What 4 looks like |
|---|---|---|
| `factual_correctness` | wrong, or invents a cause confidently | correct for crop/region/stage, no unsupported claims |
| `no_hallucinated_diagnosis` | asserts one diagnosis from one symptom | 2–3 plausible causes, or asks the disambiguating question |
| `clarification_behaviour` | never asks anything | asks precisely the 1–3 details that change the answer |
| `recommended_action_quality` | unsafe or not actionable | concrete, safe, stage-appropriate, names what and when |
| `instruction_following` | ignores the instruction | exactly as requested |
| `terminology` | wrong or imported terminology | correct Indian crop/input/unit vocabulary |
| `safety_and_uncertainty` | unsafe, no caveat | cites constraints, routes to an extension officer |

Score 0–4 per dimension. **Two independent scorers are recommended**; the
`automatic_proxy` field notes where an automated metric correlates with a
dimension, as a triage aid only.

---

## 5. Protocol

```bash
# 1. baseline BEFORE training, on the test split
uv run lfm-finetune baseline -c configs/agriculture/lfm25_1.2b.yaml

# 2. train

# 3. compare, same split, same decoding
uv run lfm-finetune evaluate -c configs/agriculture/lfm25_1.2b.yaml
```

Decoding must be **deterministic** (`do_sample: false`, greedy) for the A/B
comparison to mean anything. Sampling noise at n=50 swamps the effect sizes you
are looking for.

Outputs land in `evaluations/<run>/`:

```text
comparison.json          # everything below, machine-readable
comparison.md            # same, for humans
base_generations.json
tuned_generations.json
```

### Judging the result

1. **Disjointness** — must be `true`. If not, stop.
2. **Held-out loss** — improved is necessary, not sufficient.
3. **Regressions** — read every entry. A `stopped asking for information`
   regression is more important than a token-F1 gain.
4. **Before/after examples** — read at least ten. Does the tuned model ask the
   right clarifying question?
5. **Rubric sheet** — have an agronomist score base and tuned blind. This is the
   only step that answers "is it better at agriculture?".

### Reference run on the synthetic fixture

Not a domain result. It demonstrates the report works:

```text
held-out loss      base 1.731086  ->  tuned 1.373086  (-20.7 %)
token F1 vs ref    0.391          ->  0.800
regressions        0
eval/train overlap 0 of 5 prompts

example  prompt   How many vowels are in the word 'lantern'?
         BASE     "To determine the number of vowels in the word 'lantern', we need"
         TUNED    "2"
```

Three training steps on synthetic arithmetic. The pipeline works; agriculture is
untested.

---

## 6. Interpreting a loss decrease

| Pattern | Likely cause | Action |
|---|---|---|
| loss ↓, behaviour unchanged | learned surface style only | data may be too easy or too small |
| loss ↓, output much terser | length collapse | lower LR, check `mean_length_words` |
| loss ↓, `verbatim_reference_match_rate` ↑ | memorisation | more data, fewer epochs, lower rank |
| loss ↓, `overconfidence_rate` ↑ | harmful | system prompt and data disagree |
| loss ↓, `asks_for_information_rate` ↓ | **regression** | answers are not following the contract |
| eval loss ↑ while train ↓ | overfitting | early stopping, fewer epochs, lower rank |
| eval loss flat from step 1 | no signal | dataset too small or mismatched |

---

## 7. Scaling the evaluation

| n | What it supports |
|---|---|
| 20 | smoke; catches format regressions |
| 50 | behavioural shift on proxies |
| 200 | a defensible proxy comparison |
| 500+ | rubric-based human scoring becomes practical |

Beyond ~200 examples, budget most of the time on human scoring rather than more
automatic generations — the automatic metrics saturate well before the judgement
does.