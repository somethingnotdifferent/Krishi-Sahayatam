"""Deterministic train/validation/test splitting.

The default strategy is **hash-based** assignment: each example is placed by
``BLAKE2b(split_seed || exact_key) -> uniform float``. This has one property that
matters more than speed -- the assignment is *content-addressed*. Appending new
examples to the corpus does not move existing examples between splits, so a
dataset can grow across iterations without silently invalidating a previously
recorded validation score. Sequential splitting lacks this property and is
offered only as an escape hatch.

Guarantees enforced here:
  * ratios sum correctly (the remainder is the test split);
  * an empty train or validation split is a hard error, never a silent 0-row run;
  * the split seed and per-example split decisions are recorded, so an exact
    split can be reconstructed from the manifest alone.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..utils.errors import DatasetLeakageError
from ..utils.logging import get_logger
from .schema import Conversation

log = get_logger("data.split")

SPLIT_NAMES = ("train", "validation", "test")


def _unit_float(seed: int, key: str) -> float:
    """Deterministic uniform value in [0, 1) for (seed, key)."""
    payload = f"{seed}\x00{key}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    (value,) = struct.unpack(">Q", digest)
    return value / 2**64


@dataclass
class SplitPlan:
    """Resolved split boundaries plus the assignment function."""

    train_ratio: float
    validation_ratio: float
    test_ratio: float
    seed: int
    strategy: str
    assign: Callable[[str], str]

    @property
    def boundaries(self) -> dict[str, float]:
        return {
            "train": self.train_ratio,
            "validation": self.train_ratio + self.validation_ratio,
            "test": 1.0,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "seed": self.seed,
            "train_ratio": round(self.train_ratio, 6),
            "validation_ratio": round(self.validation_ratio, 6),
            "test_ratio": round(self.test_ratio, 6),
            "boundaries": {k: round(v, 6) for k, v in self.boundaries.items()},
        }


def build_plan(
    *,
    train_ratio: float,
    validation_ratio: float,
    seed: int,
    strategy: str = "hashed",
) -> SplitPlan:
    if not (0 < train_ratio <= 1):
        raise ValueError(f"train_ratio must be in (0, 1], got {train_ratio}")
    if validation_ratio < 0:
        raise ValueError(f"validation_ratio must be >= 0, got {validation_ratio}")
    total = train_ratio + validation_ratio
    if total > 1.0:
        raise ValueError(
            f"train_ratio + validation_ratio = {total:.3f} exceeds 1.0 "
            "(the test split would be negative)"
        )
    test_ratio = 1.0 - total

    if strategy == "hashed":
        def assign(key: str) -> str:
            u = _unit_float(seed, key)
            if u < train_ratio:
                return "train"
            if u < train_ratio + validation_ratio:
                return "validation"
            return "test"
    elif strategy == "sequential":
        def assign(_key: str) -> str:
            raise RuntimeError("sequential strategy assigns by index, not by key")
    else:
        raise ValueError(f"Unknown split strategy {strategy!r}; expected hashed or sequential")

    return SplitPlan(
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        seed=seed,
        strategy=strategy,
        assign=assign,
    )


def split_conversations(
    conversations: list[Conversation],
    plan: SplitPlan,
    *,
    exact_key_fn: Callable[[Conversation], str],
    require_non_empty: bool = True,
) -> dict[str, list[Conversation]]:
    """Partition conversations into train/validation/test.

    Args:
        require_non_empty: Raise when train or validation ends up empty. This is
            the difference between a helpful failure and a training run that
            "succeeds" on zero validation examples.

    Returns:
        ``{"train": [...], "validation": [...], "test": [...]}``
    """
    from .dedupe import exact_key as _exact_key

    key_fn = exact_key_fn or _exact_key
    buckets: dict[str, list[Conversation]] = {name: [] for name in SPLIT_NAMES}

    if plan.strategy == "sequential":
        boundaries = plan.boundaries
        n = len(conversations)
        n_train = int(n * plan.train_ratio)
        n_val = int(n * plan.validation_ratio)
        for idx, conv in enumerate(conversations):
            if idx < n_train:
                buckets["train"].append(conv)
            elif idx < n_train + n_val:
                buckets["validation"].append(conv)
            else:
                buckets["test"].append(conv)
        del boundaries
    else:
        for conv in conversations:
            buckets[plan.assign(key_fn(conv))].append(conv)

    for name in SPLIT_NAMES:
        log.info("  %-10s %d examples (%.1f%%)", name, len(buckets[name]),
                 100 * len(buckets[name]) / max(1, len(conversations)))

    if require_non_empty:
        problems = [name for name in ("train", "validation") if not buckets[name]]
        if problems:
            raise DatasetLeakageError(
                f"Split produced empty {' and '.join(problems)} set(s) from "
                f"{len(conversations)} examples at ratios "
                f"train={plan.train_ratio}/val={plan.validation_ratio}.",
                hint="Lower validation_ratio, add more examples, or change "
                "dataset.splitting.seed. For a smoke test set "
                "dataset.splitting.train_ratio=0.8, validation_ratio=0.2.",
            )
    return buckets


def verify_no_cross_split_duplicates(
    splits: dict[str, list[Conversation]],
) -> dict[str, Any]:
    """Hard-fail on exact duplicates spanning split boundaries.

    Raises:
        DatasetLeakageError: naming the offending boundary and examples.
    """
    from .dedupe import find_cross_split_overlap, summarise_overlap

    findings = find_cross_split_overlap(splits, keys=("exact",))
    summary = summarise_overlap(findings)
    if summary["hard_leakage"]:
        details = "; ".join(
            f"{boundary}: {len(hits)} shared example(s)"
            for boundary, hits in findings.items()
        )
        raise DatasetLeakageError(
            f"Exact duplicate conversations span split boundaries ({details}).",
            hint="Enable dataset.quality.deduplicate (default true) and re-run "
            "`lfm-finetune prepare`. Global deduplication runs before splitting "
            "precisely to prevent this.",
        )
    return summary


def subsample(
    conversations: list[Conversation], limit: int | None, *, seed: int
) -> list[Conversation]:
    """Deterministic subsample preserving original order.

    Uses a seeded shuffle then re-sorts by original index, so the result is a
    stable subset of the input order regardless of Python's hash randomisation.
    """
    if limit is None or limit >= len(conversations):
        return conversations
    import random

    rng = random.Random(seed)
    indices = list(range(len(conversations)))
    rng.shuffle(indices)
    chosen = sorted(indices[:limit])
    log.info("Subsampled %d of %d examples (seed=%d)", limit, len(conversations), seed)
    return [conversations[i] for i in chosen]