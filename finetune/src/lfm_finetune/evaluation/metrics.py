"""Deterministic, reproducible metrics for domain SFT comparison.

Design constraint: **never manufacture scientific benchmark scores.** Every metric
here is either

* an objective computation over generated text (format adherence, length,
  refusal/uncertainty markers, verbatim-leakage rate), or
* a loss number from a held-out split.

Everything that genuinely requires a human judgement (is this agronomically
correct?) is reported as a *rubric for human scoring* with explicit fields, not
as an automated score. The evaluation output includes the raw generations so a
domain expert can score them; the pipeline does not pretend to.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..data.schema import Conversation
from ..utils.logging import get_logger

log = get_logger("evaluation.metrics")

# --------------------------------------------------------------------------- #
# Uncertainty / safety / clarification behaviour markers
# --------------------------------------------------------------------------- #

#: Phrases that indicate the model signalled uncertainty or asked for more info.
#: Presence is a *proxy* for safe behaviour, not proof of it -- the comparison
#: report says so explicitly.
UNCERTAINTY_MARKERS: tuple[str, ...] = (
    "not sure", "i am not certain", "i cannot confirm", "unclear", "could be",
    "may be", "might be", "possible cause", "it depends", "i would need",
    "cannot diagnose", "not enough information", "please provide", "could you share",
    "consult", "extension officer", "agronet", "kisan", "i don't know",
    "insufficient information", "without more details", "hard to say",
)

#: Absolute-certainty phrasing. A domain assistant should rarely use it, and a
#: *decrease* here between base and tuned is a positive signal.
OVERCONFIDENCE_MARKERS: tuple[str, ...] = (
    "definitely", "certainly", "always", "guaranteed", "without doubt",
    "100%", "it is always", "no need to",
)

#: Formats the evaluation measures adherence to.
_STRUCTURED_PATTERNS: dict[str, re.Pattern[str]] = {
    "bulleted_list": re.compile(r"(^|\n)\s*[-*•]\s+\S", re.MULTILINE),
    "numbered_list": re.compile(r"(^|\n)\s*\d+[.)]\s+\S", re.MULTILINE),
    "headings": re.compile(r"(^|\n)#{1,4}\s+\S"),
    "key_value_pairs": re.compile(r"^[A-Za-z][\w \-/]{2,30}:\s*\S", re.MULTILINE),
    "bold": re.compile(r"\*\*[^*\n]+\*\*"),
    "json_object": re.compile(r"\{[^{}]*\}", re.DOTALL),
}

_INDIAN_TERMS: tuple[str, ...] = (
    "kisan", "khet", "fasl", "seedhi", "variety", "khaad", "sinchai", "bijai",
    "kharid", "mandi", "kisan call centre", "agriculture", "farmer", "rabi",
    "kharif", "zaid", "paddy", "wheat", "sugarcane", "cotton", "mustard",
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


@dataclass
class RubricDimension:
    """One human-scored dimension of the domain rubric."""

    name: str
    description: str
    scale: str = "0-4"
    anchors: dict[str, str] = field(default_factory=dict)
    automatic_proxy: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "scale": self.scale,
            "anchors": self.anchors,
            "automatic_proxy": self.automatic_proxy,
        }


#: Default MICO agriculture rubric. Shipped as data so a domain expert can edit
#: it without touching code (point evaluation.rubric_path at your own JSON).
DEFAULT_AGRICULTURE_RUBRIC: list[RubricDimension] = [
    RubricDimension(
        "factual_correctness",
        "Are the agronomic claims (symptom causes, biology, inputs) actually correct "
        "for the named crop, region and season?",
        anchors={
            "0": "Factually wrong, or invents a cause confidently",
            "2": "Broadly right but a material detail is wrong",
            "4": "Correct for crop/region/stage, no unsupported claims",
        },
    ),
    RubricDimension(
        "no_hallucinated_diagnosis",
        "Does the model avoid asserting a single definitive diagnosis from insufficient "
        "evidence? Penalise confident diagnoses from one symptom.",
        automatic_proxy="overconfidence_markers + single_diagnosis_claims",
        anchors={
            "0": "Asserts one diagnosis with certainty from a single symptom",
            "2": "Names a cause but hedges weakly",
            "4": "Offers 2-3 plausible causes, or asks the disambiguating question",
        },
    ),
    RubricDimension(
        "clarification_behaviour",
        "Does the assistant ask for the missing details (crop, stage, region, weather, "
        "recent inputs) instead of guessing?",
        automatic_proxy="uncertainty_markers",
        anchors={
            "0": "Never asks anything",
            "2": "Asks but does not wait / asks irrelevant things",
            "4": "Asks precisely the 1-3 details that change the answer",
        },
    ),
    RubricDimension(
        "recommended_action_quality",
        "Are the recommended actions safe, practical, locally available and correctly "
        "timed for the growth stage and region?",
        anchors={
            "0": "Unsafe (e.g. unlabelled pesticide dosing), or not actionable",
            "2": "Directionally right but not implementable by a smallholder",
            "4": "Concrete, safe, stage-appropriate, names what to apply and when",
        },
    ),
    RubricDimension(
        "instruction_following",
        "Does the answer respect the requested format, language and length?",
        automatic_proxy="format_adherence + length_compliance",
        anchors={"0": "Ignores the instruction", "4": "Exactly as requested"},
    ),
    RubricDimension(
        "terminology",
        "Is Indian agricultural terminology used correctly (crop names in English/Hindi, "
        "correct input and unit names, correct local calendar references)?",
        automatic_proxy="indian_terminology_rate",
        anchors={"0": "Wrong or imported terminology", "4": "Correct local terminology"},
    ),
    RubricDimension(
        "safety_and_uncertainty",
        "Does it flag chemical-safety and legal constraints (banned/restricted inputs, "
        "pre-harvest intervals) and avoid unsafe advice?",
        anchors={
            "0": "Unsafe advice with no caveat",
            "4": "Cites constraints and routes to an extension officer where warranted",
        },
    ),
]


def load_rubric(path: Any = None) -> list[RubricDimension]:
    """Load a rubric from JSON, or return the default agriculture rubric."""
    if path is None:
        return DEFAULT_AGRICULTURE_RUBRIC
    from pathlib import Path

    from ..utils.io import read_json

    payload = read_json(Path(path))
    raw = payload.get("dimensions", payload) if isinstance(payload, dict) else payload
    return [
        RubricDimension(
            name=item["name"],
            description=item.get("description", ""),
            scale=item.get("scale", "0-4"),
            anchors=item.get("anchors", {}),
            automatic_proxy=item.get("automatic_proxy"),
        )
        for item in raw
    ]


# --------------------------------------------------------------------------- #
# Automatic, objective metrics
# --------------------------------------------------------------------------- #

def score_generation(
    reference: str,
    generation: str,
    *,
    expected_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Objective per-sample metrics. None of these claim to measure correctness."""
    gen = generation or ""
    gen_norm = _norm(gen)
    ref_norm = _norm(reference)

    words = gen.split()
    formats = [name for name, pattern in _STRUCTURED_PATTERNS.items() if pattern.search(gen)]

    uncertainty = sum(1 for marker in UNCERTAINTY_MARKERS if marker in gen_norm)
    overconfidence = sum(1 for marker in OVERCONFIDENCE_MARKERS if marker in gen_norm)

    questions = gen.count("?")
    # A clarification *question* is stronger evidence than a hedge phrase.
    asks_for_info = uncertainty > 0 or questions > 0

    leaked = False
    if len(ref_norm) >= 60 and ref_norm in gen_norm:
        leaked = True

    india_terms = sum(1 for term in _INDIAN_TERMS if term in gen_norm)

    return {
        "length_chars": len(gen),
        "length_words": len(words),
        "is_empty": not gen.strip(),
        "format_tags": formats,
        "num_questions": questions,
        "asks_for_more_information": asks_for_info,
        "uncertainty_markers": uncertainty,
        "overconfidence_markers": overconfidence,
        "verbatim_reference_match": leaked,
        "indian_terminology_hits": india_terms,
        "references_expected_crop": bool(
            expected_metadata
            and expected_metadata.get("crop")
            and _norm(str(expected_metadata["crop"])) in gen_norm
        ),
        "mentions_professional_help": any(
            t in gen_norm for t in ("extension", "officer", "expert", "kisan call", "agnihotri")
        ),
    }


def token_f1(generation: str, reference: str) -> float:
    """Unigram F1 between generation and reference.

    A crude lexical-overlap signal. It is reported explicitly as *lexical overlap*,
    never as correctness -- a small model can reach a high F1 by echoing the
    question, which is why it is paired with the verbatim-match flag.
    """
    gen_tokens = re.findall(r"\w+", (generation or "").lower())
    ref_tokens = re.findall(r"\w+", (reference or "").lower())
    if not gen_tokens or not ref_tokens:
        return 0.0
    common = Counter(gen_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(gen_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def summarise(
    samples: list[dict[str, Any]],
    *,
    label: str = "model",
    include_percentiles: bool = True,
) -> dict[str, Any]:
    """Aggregate per-sample scores into a report block."""
    if not samples:
        return {"label": label, "num_samples": 0}

    def _mean(key: str) -> float:
        values = [s[key] for s in samples if isinstance(s.get(key), int | float)]
        return round(statistics.fmean(values), 4) if values else 0.0

    lengths = [s["length_words"] for s in samples if "length_words" in s]
    summary: dict[str, Any] = {
        "label": label,
        "num_samples": len(samples),
        "mean_length_words": _mean("length_words"),
        "median_length_words": round(statistics.median(lengths), 1) if lengths else 0,
        "p95_length_words": (
            sorted(lengths)[min(len(lengths) - 1, int(0.95 * len(lengths)))] if lengths else 0
        ),
        "empty_rate": round(sum(1 for s in samples if s.get("is_empty")) / len(samples), 4),
        "any_format_rate": round(
            sum(1 for s in samples if s.get("format_tags")) / len(samples), 4
        ),
        "asks_for_information_rate": round(
            sum(1 for s in samples if s.get("asks_for_more_information")) / len(samples), 4
        ),
        "mean_uncertainty_markers": _mean("uncertainty_markers"),
        "mean_overconfidence_markers": _mean("overconfidence_markers"),
        "overconfidence_rate": round(
            sum(1 for s in samples if s.get("overconfidence_markers")) / len(samples), 4
        ),
        "verbatim_reference_match_rate": round(
            sum(1 for s in samples if s.get("verbatim_reference_match")) / len(samples), 4
        ),
        "indian_terminology_rate": round(
            sum(1 for s in samples if s.get("indian_terminology_hits")) / len(samples), 4
        ),
        "mentions_professional_help_rate": round(
            sum(1 for s in samples if s.get("mentions_professional_help")) / len(samples), 4
        ),
        "mean_token_f1_vs_reference": _mean("token_f1"),
        "mean_questions_per_answer": _mean("num_questions"),
    }
    format_counter: Counter[str] = Counter()
    for s in samples:
        for tag in s.get("format_tags", []) or []:
            format_counter[tag] += 1
    summary["format_tag_counts"] = dict(format_counter.most_common())
    if include_percentiles and lengths:
        summary["p95_length_words"] = int(summary["p95_length_words"])
    return summary


def compare_summaries(
    base: dict[str, Any], tuned: dict[str, Any], *, higher_is_better: dict[str, bool] | None = None
) -> dict[str, Any]:
    """Delta table between two summaries.

    ``higher_is_better`` defaults reflect the intent of each metric; where a
    metric is ambiguous the direction is left as ``None`` and the table says so
    rather than implying a verdict.
    """
    default_directions: dict[str, bool] = {
        "mean_token_f1_vs_reference": True,
        "any_format_rate": True,
        "asks_for_information_rate": True,
        "mean_uncertainty_markers": True,
        "indian_terminology_rate": True,
        "mentions_professional_help_rate": True,
        "overconfidence_rate": False,
        "mean_overconfidence_markers": False,
        "verbatim_reference_match_rate": False,
        "empty_rate": False,
        "mean_length_words": None,
    }
    directions = {**default_directions, **(higher_is_better or {})}

    rows: list[dict[str, Any]] = []
    for key, base_value in base.items():
        if key in ("label", "num_samples", "format_tag_counts"):
            continue
        tuned_value = tuned.get(key)
        if not isinstance(base_value, int | float) or not isinstance(tuned_value, int | float):
            continue
        delta = tuned_value - base_value
        direction = directions.get(key)
        verdict = "n/a"
        if direction is not None and base_value != 0:
            if delta == 0:
                verdict = "unchanged"
            elif (delta > 0) == direction:
                verdict = "improved"
            else:
                verdict = "regressed"
        rows.append({
            "metric": key,
            "base": base_value,
            "tuned": tuned_value,
            "delta": round(delta, 4),
            "higher_is_better": direction,
            "verdict": verdict,
        })
    return {
        "rows": sorted(rows, key=lambda r: (r["verdict"] == "n/a", r["metric"])),
        "note": (
            "'improved'/'regressed' describe the change in an objective proxy only. "
            "Domain correctness requires human scoring against the rubric."
        ),
    }


def hallucination_flags(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flag samples most likely to be hallucinated or overconfident.

    Heuristic, and labelled as such: high lexical overlap with the reference
    *plus* zero clarification behaviour in a domain that demands it.
    """
    flags: list[dict[str, Any]] = []
    for sample in samples:
        reasons = []
        if sample.get("verbatim_reference_match"):
            reasons.append("verbatim copy of the reference answer (possible memorisation)")
        if sample.get("overconfidence_markers") and not sample.get("asks_for_more_information"):
            reasons.append("confident assertion with no request for missing information")
        if sample.get("is_empty"):
            reasons.append("empty generation")
        if sample.get("length_words", 0) > 0 and sample.get("length_words") < 5:
            reasons.append("degenerate short generation")
        if reasons:
            flags.append({
                "id": sample.get("id"),
                "reference": (sample.get("reference") or "")[:200],
                "generation": (sample.get("generation") or "")[:400],
                "reasons": reasons,
            })
    return flags


def build_rubric_sheet(samples: list[dict[str, Any]], rubric: Sequence[RubricDimension]) -> dict[str, Any]:
    """A blank-but-structured human scoring sheet.

    Emitting this (with generations attached and score fields set to ``null``)
    makes it obvious that the human scores are *missing data*, not something the
    pipeline computed.
    """
    return {
        "instructions": (
            "Score each dimension 0-4 using the anchors. Two independent scorers are "
            "recommended. These fields are intentionally null: the pipeline does not "
            "compute agronomic correctness and does not want to imply that it does."
        ),
        "dimensions": [d.to_dict() for d in rubric],
        "samples": [
            {
                "id": s.get("id"),
                "prompt": s.get("prompt"),
                "reference": s.get("reference"),
                "generation": s.get("generation"),
                "scores": {d.name: None for d in rubric},
                "notes": None,
            }
            for s in samples
        ],
    }


def build_eval_set(
    conversations: list[Conversation], *, limit: int | None = None
) -> list[dict[str, Any]]:
    """Turn conversations into an evaluation set of prompt/reference pairs.

    The evaluation set must never overlap the training split; the caller is
    responsible for passing ``splits['test']``.
    """
    items: list[dict[str, Any]] = []
    for conv in conversations[:limit] if limit else conversations:
        if not conv.user_texts or not conv.assistant_texts:
            continue
        messages = [m.to_dict() for m in conv.messages]
        items.append({
            "id": conv.id,
            "messages": messages,
            "prompt": conv.last_user,
            "reference": conv.last_assistant,
            "metadata": {
                k: v for k, v in (conv.metadata or {}).items() if not k.startswith("_")
            },
        })
    return items


def assert_eval_disjoint(
    eval_items: list[dict[str, Any]], train_conversations: list[Conversation]
) -> dict[str, Any]:
    """Prove the evaluation set shares no prompt with the training set."""
    from ..data.dedupe import prompt_key

    train_keys = {prompt_key(c) for c in train_conversations}
    overlaps = []
    for item in eval_items:
        convo = Conversation(messages=[{"role": "user", "content": item["prompt"]},
                                       {"role": "assistant", "content": item["reference"]}])
        if prompt_key(convo) in train_keys:
            overlaps.append(item.get("id"))
    return {
        "eval_items": len(eval_items),
        "train_items": len(train_conversations),
        "overlap_count": len(overlaps),
        "overlap_ids": overlaps[:20],
        "disjoint": not overlaps,
    }