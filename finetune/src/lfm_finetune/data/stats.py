"""Token-level dataset statistics.

Statistics are computed with the **actual checkpoint tokenizer**, because token
counts -- not character counts -- are what determine truncation, memory and
training cost. Character heuristics are recorded alongside for cheap triage.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

from ..utils.logging import get_logger
from .schema import Conversation, DatasetStats

log = get_logger("data.stats")

#: Domain metadata keys surfaced in the statistics report.
CATEGORY_KEYS = ("category", "task", "intent", "split_category")
CROP_KEYS = ("crop", "plant", "crop_name")
LANGUAGE_KEYS = ("language", "lang")


def _percentile(values: Sequence[int | float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    # Linear interpolation (numpy 'linear' method) -- matches what practitioners
    # expect from `np.percentile`, and is deterministic without numpy.
    rank = (len(ordered) - 1) * pct / 100.0
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return float(ordered[low] * (1 - weight) + ordered[high] * weight)


def count_conversation_tokens(
    conv: Conversation,
    *,
    tokenizer: Any,
    max_seq_length: int | None = None,
    render: Callable[[Conversation], str] | None = None,
) -> int:
    """Exact rendered-token count for one conversation.

    Uses the checkpoint chat template when available (via ``render``), so the
    number reported matches what training will actually consume -- including
    template overhead, which is easy to forget and can be 5-10% of a short
    example.
    """
    if render is not None:
        text = render(conv)
    elif getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            [m.to_dict() for m in conv.messages], tokenize=False
        )
    else:
        text = "\n".join(f"{m.role}: {m.content}" for m in conv.messages)

    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if max_seq_length is not None and len(ids) > max_seq_length:
        # Count the truncated length so the stats reflect reality.
        return int(max_seq_length)
    return len(ids)


def compute_stats(
    conversations: list[Conversation],
    *,
    tokenizer: Any = None,
    max_seq_length: int | None = None,
    render: Callable[[Conversation], str] | None = None,
    compute_tokens: bool = True,
    label: str = "all",
) -> DatasetStats:
    """Build a :class:`DatasetStats` for a set of conversations.

    When ``compute_tokens`` is False (no tokenizer available, e.g. in unit tests
    for pure-Python logic) token fields are zero and character metrics carry the
    information.
    """
    stats = DatasetStats(num_examples=len(conversations))
    if not conversations:
        return stats

    roles: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    crops: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    turns_histogram: Counter[int] = Counter()
    token_lengths: list[int] = []
    user_chars: list[int] = []
    assistant_chars: list[int] = []

    for conv in conversations:
        stats.num_messages += len(conv.messages)
        turns_histogram[len(conv.messages)] += 1
        roles.update(m.role for m in conv.messages)
        user_chars.extend(len(u) for u in conv.user_texts)
        assistant_chars.extend(len(a) for a in conv.assistant_texts)
        stats.total_chars += sum(len(m.content) for m in conv.messages)

        meta = conv.metadata or {}
        for keys, counter in ((CATEGORY_KEYS, categories), (CROP_KEYS, crops), (LANGUAGE_KEYS, languages)):
            for key in keys:
                if meta.get(key):
                    counter[str(meta[key])] += 1
                    break

        if compute_tokens and tokenizer is not None:
            token_lengths.append(
                count_conversation_tokens(
                    conv, tokenizer=tokenizer, max_seq_length=max_seq_length, render=render
                )
            )

    stats.roles = dict(roles)
    stats.categories = dict(categories)
    stats.crops = dict(crops)
    stats.languages = dict(languages)
    stats.turns_histogram = {str(k): v for k, v in turns_histogram.items()}

    if user_chars:
        stats.mean_user_chars = statistics.fmean(user_chars)
    if assistant_chars:
        stats.mean_assistant_chars = statistics.fmean(assistant_chars)

    if token_lengths:
        stats.total_tokens = sum(token_lengths)
        stats.mean_tokens = statistics.fmean(token_lengths)
        stats.median_tokens = statistics.median(token_lengths)
        stats.p95_tokens = _percentile(token_lengths, 95)
        stats.min_tokens = min(token_lengths)
        stats.max_tokens = max(token_lengths)
        over = sum(1 for t in token_lengths if max_seq_length and t >= max_seq_length)
        if over:
            log.warning(
                "[%s] %d/%d examples (%.1f%%) reach the %d-token cap and will be truncated.",
                label, over, len(token_lengths), 100 * over / len(token_lengths), max_seq_length or 0,
            )

    log.info(
        "[%s] %d examples | tokens total=%d mean=%.1f median=%.0f p95=%.0f max=%d",
        label, stats.num_examples, stats.total_tokens, stats.mean_tokens,
        stats.median_tokens, stats.p95_tokens, stats.max_tokens,
    )
    return stats


def multi_split_stats(
    splits: dict[str, list[Conversation]],
    **kwargs: Any,
) -> dict[str, Any]:
    """Per-split statistics plus totals and the train/validation token ratio."""
    out: dict[str, Any] = {"per_split": {}, "totals": {}}
    all_convs: list[Conversation] = []
    for name, convs in splits.items():
        stats = compute_stats(convs, label=name, **kwargs)
        out["per_split"][name] = stats.to_dict()
        all_convs.extend(convs)

    totals = compute_stats(all_convs, label="all", **kwargs)
    out["totals"] = totals.to_dict()

    train_tokens = out["per_split"].get("train", {}).get("total_tokens", 0)
    val_tokens = out["per_split"].get("validation", {}).get("total_tokens", 0)
    out["train_validation_token_ratio"] = (
        round(train_tokens / val_tokens, 2) if val_tokens else None
    )
    if val_tokens and train_tokens / max(1, val_tokens) > 100:
        log.warning(
            "train/validation token ratio is %.0f:1 -- validation loss will be noisy. "
            "Consider raising dataset.splitting.validation_ratio.",
            train_tokens / val_tokens,
        )
    return out


def dataset_warning(stats: DatasetStats, *, max_seq_length: int, min_examples: int = 200) -> list[str]:
    """Advisory warnings for a suspiciously small or skewed dataset."""
    warnings: list[str] = []
    n = stats.num_examples
    if n < min_examples:
        warnings.append(
            f"Only {n} examples. Domain SFT from a base checkpoint is data-hungry; "
            f"Liquid's guidance is 500-5000 task examples. Results at this scale are "
            f"anecdotal, and {max(1, n)} examples cannot support {3} epochs of LoRA "
            f"without memorisation. Treat this run as a pipeline smoke test."
        )
    if n and stats.mean_tokens < 64:
        warnings.append(
            f"Mean sequence length is only {stats.mean_tokens:.0f} tokens. Very short "
            "examples give the adapter very little signal per step; consider enriching "
            "the reasoning in your answers."
        )
    if n and stats.mean_assistant_chars < 120:
        warnings.append(
            f"Mean assistant answer is {stats.mean_assistant_chars:.0f} characters. Very "
            "short answers tend to teach terseness rather than reasoning capability."
        )
    if n and stats.max_tokens >= max_seq_length:
        warnings.append(
            f"At least one example hits max_seq_length={max_seq_length} and will be "
            "truncated. If those are long conversations, raise the cap (and check VRAM) "
            "or split the conversations."
        )
    return warnings