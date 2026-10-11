"""Synthetic dataset generation for smoke tests and CI.

**These are pipeline fixtures, not agricultural knowledge.** Every record is
synthetically generated from a small combinatorial template with deliberately
*non*-agricultural content (arithmetic, string manipulation, format conversion,
list handling). They exist to exercise the schema, the loss mask, the trainer and
the checkpoint round-trip without a GPU and without inventing crop facts.

The one place agriculture-flavoured text appears is
:func:`placeholder_agriculture_records`, which is explicitly marked
``synthetic: true`` and must never be trained on. It demonstrates the *metadata*
schema only.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..utils.hashing import short_hash
from ..utils.io import write_json, write_jsonl
from ..utils.logging import get_logger

log = get_logger("data.synthetic")

SYSTEM_PROMPT = (
    "You are a careful reasoning assistant. Answer the question directly, show your "
    "working briefly, and state any assumption you had to make."
)

_TEMPLATES: list[tuple[str, Any]] = [
    ("What is {a} + {b}?", lambda r: f"{r['a'] + r['b']}"),
    ("What is {a} * {b}?", lambda r: f"{r['a'] * r['b']}"),
    ("Subtract {b} from {a}.", lambda r: f"{r['a'] - r['b']}"),
    (
        "How many vowels are in the word '{word}'?",
        lambda r: f"'{r['word']}' contains {sum(c in 'aeiou' for c in r['word'].lower())} vowels.",
    ),
    (
        "Capitalise the phrase '{phrase}'.",
        lambda r: r["phrase"].capitalize(),
    ),
    (
        "Reverse the string '{word}'.",
        lambda r: f"The reversed string is '{r['word'][::-1]}'.",
    ),
    (
        "Sort this list of numbers ascending: {numbers}",
        lambda r: f"Sorted ascending: {sorted(r['numbers'])}.",
    ),
    (
        "Is {a} greater than {b}? Answer yes or no.",
        lambda r: f"{'Yes' if r['a'] > r['b'] else 'No'}, {r['a']} is "
                  f"{'greater' if r['a'] > r['b'] else 'not greater'} than {r['b']}.",
    ),
    (
        "Convert {a} seconds into minutes.",
        lambda r: f"{r['a']} seconds is {r['a'] / 60:.2f} minutes.",
    ),
    (
        "How many characters are in the sentence '{phrase}'?",
        lambda r: f"The sentence has {len(r['phrase'])} characters.",
    ),
    (
        "Give the first {k} even numbers starting from {a}.",
        lambda r: f"The numbers are: {[a for a in range(r['a'], r['a'] + 2 * r['k'], 2)]}.",
    ),
    (
        "What is the average of {numbers}?",
        lambda r: f"The average is {sum(r['numbers']) / len(r['numbers']):.2f}.",
    ),
]

_WORDS = ("alpha", "bravo", "delta", "echo", "falcon", "granite", "harbor", "indigo",
          "juniper", "kettle", "lantern", "meadow", "nimbus", "orchard", "pebble")


def _fill(template: str, rng: random.Random) -> str:
    return template.format(
        a=rng.randint(2, 90),
        b=rng.randint(2, 90),
        k=rng.randint(2, 6),
        word=rng.choice(_WORDS),
        phrase=" ".join(rng.sample(_WORDS, 3)),
        numbers=[rng.randint(1, 50) for _ in range(rng.randint(3, 6))],
    )


def generate_synthetic(
    num_examples: int = 32,
    *,
    seed: int = 1234,
    system_prompt: str | None = SYSTEM_PROMPT,
    include_multi_turn: bool = True,
    include_edge_cases: bool = True,
) -> list[dict[str, Any]]:
    """Generate a deterministic, self-consistent conversational SFT fixture.

    The answers are computed, not invented, so the dataset is exactly learnable.
    That makes it a valid loss check: a well-trained adapter should reach a low
    training loss on these examples.
    """
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []

    for i in range(num_examples):
        template, fn = _TEMPLATES[i % len(_TEMPLATES)]
        slots = {
            "a": rng.randint(2, 90),
            "b": rng.randint(2, 90),
            "k": rng.randint(2, 6),
            "word": rng.choice(_WORDS),
            "phrase": " ".join(rng.sample(_WORDS, 3)),
            "numbers": [rng.randint(1, 50) for _ in range(rng.randint(3, 5))],
        }
        question = template.format(**slots)
        answer = str(fn(slots))

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if include_multi_turn and i % 5 == 4:
            # Two-turn dialogue so the loss mask is exercised on a non-final
            # assistant turn as well as the final one.
            first_q = f"Name a word starting with '{slots['word'][0].upper()}'."
            first_a = f"'{slots['word'].capitalize()}' works."
            messages.append({"role": "user", "content": first_q})
            messages.append({"role": "assistant", "content": first_a})

        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})

        records.append({
            "id": f"synthetic-{i:05d}",
            "messages": messages,
            "metadata": {
                "category": "arithmetic" if any(t in question for t in "+*-") else "string_ops",
                "language": "en",
                "synthetic": True,
                "template_index": i % len(_TEMPLATES),
            },
        })

    if include_edge_cases:
        records.extend(_edge_cases(rng))
    log.info("Generated %d synthetic examples (seed=%d)", len(records), seed)
    return records


def _edge_cases(rng: random.Random) -> list[dict[str, Any]]:
    """Deliberate corner cases so unit tests have stable, known inputs."""
    return [
        {
            "id": "synthetic-edge-multiturn-3",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2 + 2 = 4."},
                {"role": "user", "content": "And what is 10-3?"},
                {"role": "assistant", "content": "10 - 3 = 7."},
            ],
            "metadata": {"category": "arithmetic", "language": "en", "synthetic": True},
        },
        {
            "id": "synthetic-edge-unicode",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Reverse 'café-नमस्ते'."},
                {"role": "assistant", "content": "The reversed string is 'ेस्तमन-éfac'."},
            ],
            "metadata": {"category": "string_ops", "language": "en", "synthetic": True},
        },
        {
            "id": "synthetic-edge-long",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Sort this list of numbers ascending: " + str(list(range(60)))},
                {"role": "assistant",
                 "content": "Sorted ascending: " + str(list(range(60)))
                            + ". The sequence is already monotonic, so the sorted order is unchanged."},
            ],
            "metadata": {"category": "list_ops", "language": "en", "synthetic": True},
        },
    ]


def write_synthetic_dataset(
    destination: Path,
    *,
    num_examples: int = 32,
    seed: int = 1234,
) -> dict[str, Any]:
    """Write a synthetic raw dataset plus a provenance manifest.

    The manifest declares ``license: CC0-1.0`` and ``synthetic: true`` so that a
    stray training run on this fixture is immediately identifiable in the run
    manifest.
    """
    from ..data.io import build_manifest

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    records = generate_synthetic(num_examples, seed=seed)

    write_jsonl(destination / "synthetic.jsonl", records)
    manifest = build_manifest(
        name="synthetic-pipeline-fixture",
        version="1.0.0",
        source="generated by lfm_finetune.data.synthetic",
        license="CC0-1.0",
        created=datetime.now(UTC).date().isoformat(),
        examples=len(records),
        content_hash=short_hash(str(len(records)), 12),
        categories=["arithmetic", "string_ops", "list_ops"],
        language=["en"],
        annotation_method="programmatically generated (answers computed, not authored)",
        provenance_notes=(
            "SYNTHETIC TEST FIXTURE. Not real-world data and deliberately contains no "
            "agricultural knowledge. Do not use for training a production model."
        ),
        extra={"synthetic": True},
    )
    write_json(destination / "manifest.json", manifest)
    log.info("Wrote %d synthetic records to %s", len(records), destination / "synthetic.jsonl")
    return manifest


def placeholder_agriculture_records() -> list[dict[str, Any]]:
    """Schema demonstration records for the MICO domain.

    These are STRUCTURAL placeholders: the text deliberately avoids asserting
    agronomic facts (no disease names, no treatments, no dosages). They show how
    ``metadata`` is populated and how a domain ``system_prompt`` is attached.
    A real dataset must replace these with sourced content -- see docs/DATASET.md.
    """
    from ..config.schema import ExperimentConfig  # noqa: F401  (documentation aid)

    return [
        {
            "id": "placeholder-0001",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an agricultural advisory assistant for smallholder farmers "
                        "in India. Give practical, safe advice. Ask for missing details "
                        "instead of guessing. State uncertainty explicitly."
                    ),
                },
                {
                    "role": "user",
                    "content": "[PLACEHOLDER] Farmer describes a crop symptom and asks what to do.",
                },
                {
                    "role": "assistant",
                    "content": (
                        "[PLACEHOLDER] A real answer must (1) ask which crop, growth stage, "
                        "region and recent weather, (2) list 2-3 plausible causes rather "
                        "than asserting one, (3) give a safe next inspection step, and "
                        "(4) advise contacting a local extension officer before spraying."
                    ),
                },
            ],
            "metadata": {
                "domain": "agriculture",
                "category": "clarification_questions",
                "crop": "<crop>",
                "plant": "<plant>",
                "disease": "<unknown>",
                "pest": "<unknown>",
                "nutrient_issue": "<unknown>",
                "growth_stage": "<stage>",
                "symptom": ["<symptom>"],
                "region": "IN",
                "climate": "<climate>",
                "soil": "<soil>",
                "irrigation": "<irrigation>",
                "treatment": "<none-yet>",
                "confidence": None,
                "source": "<citation-required>",
                "license": "<set-me>",
                "language": "en",
                "category_id": "placeholder_0001",
                "synthetic": True,
                "placeholder": True,
            },
        }
    ]