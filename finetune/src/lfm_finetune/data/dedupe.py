"""Deduplication, near-duplicate detection and train/val leakage control.

Three distinct hazards, three distinct mechanisms:

``exact``
    Byte-identical (post-normalisation) conversation payloads. Cheap, exact,
    backed by a SHA-256 set.

``near``
    Paraphrases and template variants. Uses character 5-gram Jaccard similarity
    against an inverted index, which is O(candidates) rather than O(n^2) --
    important because domain datasets are small but template duplication is
    heavy.

``leakage``
    The same *question* appearing in more than one split. Splitting before
    deduplication would put a duplicate pair on opposite sides of the boundary
    and inflate validation metrics. The pipeline therefore deduplicates globally
    *before* splitting, then re-verifies split disjointness afterwards.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ..utils.logging import get_logger
from .schema import Conversation

log = get_logger("data.dedupe")

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def _canonical_text(text: str) -> str:
    return _WS_RE.sub(" ", _PUNCT_RE.sub(" ", text.lower())).strip()


def exact_key(conv: Conversation) -> str:
    """Hash of the full normalised conversation."""
    return hashlib.sha256(conv.fingerprint_payload().encode("utf-8")).hexdigest()


def prompt_key(conv: Conversation) -> str:
    """Hash of the *prompt* only (all turns except the final assistant reply).

    Two records with identical prompts but different answers are not duplicates
    -- they are a legitimately hard test item, or a labelling disagreement. This
    is reported separately so a human can adjudicate.
    """
    parts = [f"{m.role}:{_canonical_text(m.content)}" for m in conv.messages[:-1]]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def answer_key(conv: Conversation) -> str:
    return hashlib.sha256(
        _canonical_text(conv.messages[-1].content).encode("utf-8")
    ).hexdigest()


# --------------------------------------------------------------------------- #
# Near-duplicate detection
# --------------------------------------------------------------------------- #

def char_shingles(text: str, k: int = 5) -> set[int]:
    """Set of hashed character 5-grams over the canonical text.

    Character shingles (not words) make this robust to the spelling variation,
    inflection and code-switching typical of farmer-collected agricultural text.
    """
    canonical = _canonical_text(text)
    if len(canonical) <= k:
        return {hash(canonical)} if canonical else set()
    return {
        int.from_bytes(hashlib.blake2b(canonical[i : i + k].encode("utf-8"), digest_size=4).digest(), "big")
        for i in range(len(canonical) - k + 1)
    }


def conversation_text(conv: Conversation, *, include_system: bool = False) -> str:
    """Text used for near-duplicate similarity.

    The system message is excluded **by default**, and that is not cosmetic.

    Every record in a domain corpus shares the same long system prompt (the MICO
    behaviour contract is ~350 characters). Including it contributes ~350 identical
    character shingles to *every* pair, which inflates Jaccard similarity across
    the whole corpus and makes the threshold far stricter than configured. In
    practice that silently deletes legitimate examples: templated-but-distinct
    Q&A pairs get reported as near-duplicates of each other purely because they
    share boilerplate.

    What we actually want to detect is a repeated *question* or a repeated
    *answer*, so the system message is not part of the comparison. ``prompt_key``
    and ``answer_key`` remain the precise tools for those.
    """
    return " ".join(
        m.content for m in conv.messages
        if include_system or m.role != "system"
    )


@dataclass
class NearDuplicateIndex:
    """Inverted shingle index for O(candidates) near-duplicate queries."""

    k: int = 5
    threshold: float = 0.90
    _index: dict[int, set[int]] = field(default_factory=dict)
    _shingles: list[set[int]] = field(default_factory=list)
    _payloads: list[str] = field(default_factory=list)
    comparisons: int = 0

    def _jaccard(self, a: set[int], b: set[int]) -> float:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        return inter / (len(a) + len(b) - inter) if inter else 0.0

    def query(self, text: str) -> tuple[bool, str | None]:
        """Return ``(is_near_duplicate, matched_text)``.

        Candidates are pre-filtered by shingle overlap: if fewer than
        ``threshold * len(query)`` shingles are shared, Jaccard cannot reach the
        threshold, so the pair is skipped without a full comparison.
        """
        shingles = char_shingles(text, self.k)
        if not shingles:
            return False, None

        counts: dict[int, int] = {}
        for shingle in shingles:
            for doc_id in self._index.get(shingle, ()):
                counts[doc_id] = counts.get(doc_id, 0) + 1

        min_shared = int(self.threshold * len(shingles))
        best: tuple[float, int] | None = None
        for doc_id, shared in counts.items():
            if shared < min_shared:
                continue
            self.comparisons += 1
            score = self._jaccard(shingles, self._shingles[doc_id])
            if score >= self.threshold and (best is None or score > best[0]):
                best = (score, doc_id)

        if best is not None:
            return True, self._payloads[best[1]]
        self.add(text)
        return False, None

    def add(self, text: str) -> None:
        shingles = char_shingles(text, self.k)
        if not shingles:
            return
        doc_id = len(self._shingles)
        self._shingles.append(shingles)
        self._payloads.append(text[:400])
        for shingle in shingles:
            self._index.setdefault(shingle, set()).add(doc_id)

    def stats(self) -> dict[str, int]:
        return {
            "documents": len(self._shingles),
            "distinct_shingles": len(self._index),
            "pair_comparisons": self.comparisons,
        }


# --------------------------------------------------------------------------- #
# Split-level leakage audit
# --------------------------------------------------------------------------- #

def find_cross_split_overlap(
    splits: dict[str, list[Conversation]],
    *,
    keys: Iterable[str] = ("exact", "prompt", "answer"),
) -> dict[str, list[dict[str, Any]]]:
    """Audit split disjointness. Returns per-boundary offending pairs.

    ``exact`` overlaps are hard errors. ``prompt`` and ``answer`` overlaps are
    reported but not fatal: a shared *answer* string across splits is common and
    usually harmless (e.g. "Yes, water twice a week"), whereas a shared *prompt*
    is a genuine leak.
    """
    findings: dict[str, list[dict[str, Any]]] = {}
    split_names = list(splits)
    key_fns = {
        "exact": exact_key,
        "prompt": prompt_key,
        "answer": answer_key,
    }

    for i, a in enumerate(split_names):
        for b in split_names[i + 1 :]:
            boundary = f"{a}|{b}"
            for key in keys:
                fn = key_fns[key]
                index_a: dict[str, Conversation] = {}
                for conv in splits[a]:
                    index_a.setdefault(fn(conv), conv)
                hits = []
                for conv in splits[b]:
                    k = fn(conv)
                    if k in index_a:
                        other = index_a[k]
                        hits.append({
                            "key_type": key,
                            "key": k[:16],
                            "left_id": other.id,
                            "right_id": conv.id,
                            "excerpt": other.messages[-1].content[:160],
                        })
                        if key == "exact":
                            break
                if hits:
                    findings.setdefault(boundary, []).extend(hits)
    return findings


def summarise_overlap(findings: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Condense :func:`find_cross_split_overlap` into a report block."""
    by_type: dict[str, int] = {}
    for hits in findings.values():
        for hit in hits:
            by_type[hit["key_type"]] = by_type.get(hit["key_type"], 0) + 1
    return {
        "has_leakage": bool(findings),
        "counts_by_key_type": by_type,
        "boundaries": {k: v[:20] for k, v in findings.items()},
        "hard_leakage": by_type.get("exact", 0) > 0,
    }