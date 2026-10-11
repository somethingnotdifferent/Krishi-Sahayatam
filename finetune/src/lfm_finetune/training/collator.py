"""Dataset -> torch tensors, with pre-computed assistant-only labels.

We deliberately pre-tokenise in Python rather than relying on TRL's
``dataset_text_field``/``assistant_only_loss`` machinery. Reasons:

* ``assistant_only_loss`` in TRL requires ``{% generation %}`` markers in the
  chat template, which the LFM2.5 **base** checkpoints do not provide (see
  :mod:`lfm_finetune.training.tokenization`).
* Pre-computing labels once means truncation happens in exactly one place, the
  reported token statistics are the tokens actually trained on, and the loss mask
  can be inspected before training starts.
* It keeps the collator trivial and therefore auditable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..data.schema import Conversation
from ..utils.errors import TokenizerError
from ..utils.logging import get_logger
from .tokenization import IGNORE_INDEX, ChatFormatter, FormattedExample

log = get_logger("training.collator")


@dataclass
class TokenizedDataset:
    """Pre-tokenised examples plus provenance for the run manifest."""

    examples: list[FormattedExample]
    num_dropped: int = 0
    dropped_reasons: dict[str, int] | None = None
    tokenizer_fingerprint: str | None = None
    mean_length: float = 0.0
    mean_loss_coverage: float = 0.0

    def __len__(self) -> int:
        return len(self.examples)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "num_examples": len(self.examples),
            "num_dropped": self.num_dropped,
            "dropped_reasons": self.dropped_reasons or {},
            "mean_length": round(self.mean_length, 2),
            "mean_loss_coverage": round(self.mean_loss_coverage, 4),
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
        }


def tokenize_conversations(
    conversations: Sequence[Conversation],
    formatter: ChatFormatter,
    *,
    assistant_only_loss: bool = True,
    max_length: int | None = None,
    split: str = "train",
    fail_on_error: bool = False,
) -> TokenizedDataset:
    """Tokenise and mask a list of conversations.

    Args:
        fail_on_error: Raise on the first unformattable example instead of
            dropping it. Used by ``validate``/``inspect-samples`` so problems
            surface loudly; ``train`` drops-with-report by default so one bad
            row does not waste a whole run.

    Returns:
        :class:`TokenizedDataset`. Dropped examples are counted by reason.
    """
    from collections import Counter

    from .tokenization import template_fingerprint

    examples: list[FormattedExample] = []
    dropped: Counter[str] = Counter()

    for conv in conversations:
        try:
            examples.append(
                formatter.format_conversation(
                    conv,
                    assistant_only_loss=assistant_only_loss,
                    max_length=max_length,
                )
            )
        except TokenizerError as exc:
            reason = str(exc).split(".")[0][:120]
            dropped[reason] += 1
            log.warning("[%s] dropping %s: %s", split, conv.id, exc.message)
            if fail_on_error:
                raise

    if not examples:
        raise TokenizerError(
            f"Every one of {len(conversations)} {split} examples failed to format.",
            hint="Run `lfm-finetune inspect-samples --config ... --fail-on-error` to see the "
            "first concrete failure. Common causes: no assistant turn, or "
            "tokenization.max_seq_length smaller than the prompt.",
        )

    mean_len = sum(e.length for e in examples) / len(examples)
    coverage = sum(e.loss_coverage for e in examples) / len(examples)
    truncated = sum(1 for e in examples if e.truncated)
    if truncated:
        log.warning("[%s] %d/%d examples were truncated to max_length", split, truncated, len(examples))
    log.info(
        "[%s] tokenised %d examples | mean_len=%.1f | mean loss coverage=%.1f%%",
        split, len(examples), mean_len, coverage * 100,
    )
    if assistant_only_loss and coverage > 0.95:
        log.warning(
            "[%s] %.1f%% of tokens contribute to the loss. Assistant-only masking may not "
            "be working -- the prompt is being trained on.", split, coverage * 100
        )

    return TokenizedDataset(
        examples=examples,
        num_dropped=len(conversations) - len(examples),
        dropped_reasons=dict(dropped),
        tokenizer_fingerprint=template_fingerprint(formatter.tokenizer),
        mean_length=mean_len,
        mean_loss_coverage=coverage,
    )


class SFTCollator:
    """Batches pre-tokenised examples into padded tensors.

    Accepts either a :class:`FormattedExample` or the plain mapping produced by
    :class:`TorchDatasetAdapter`, so it can be unit-tested without constructing
    the intermediate dataclass.

    Padding side is right for training: with labels masked at pad positions,
    left padding would still work for a causal LM but right padding keeps
    positions contiguous from the model's point of view, which is what
    pre-training used.
    """

    def __init__(self, pad_token_id: int, *, pad_to_multiple_of: int = 8) -> None:
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    @staticmethod
    def _as_dict(feature: Any) -> dict[str, list[int]]:
        if isinstance(feature, dict):
            return feature
        return {
            "input_ids": list(feature.input_ids),
            "attention_mask": list(feature.attention_mask),
            "labels": list(feature.labels),
        }

    def __call__(self, features: list[Any]) -> dict[str, Any]:
        import torch

        if not features:
            raise TokenizerError("SFTCollator received an empty batch")

        rows = [self._as_dict(f) for f in features]
        for row in rows:
            if not (len(row["input_ids"]) == len(row["labels"]) == len(row["attention_mask"])):
                raise TokenizerError(
                    "Mismatched input_ids/attention_mask/labels lengths "
                    f"({len(row['input_ids'])}/{len(row['attention_mask'])}/{len(row['labels'])}).",
                    hint="This indicates a bug in the formatter, not in your data.",
                )

        max_len = max(len(r["input_ids"]) for r in rows)
        if self.pad_to_multiple_of > 1:
            multiple = self.pad_to_multiple_of
            max_len = ((max_len + multiple - 1) // multiple) * multiple

        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []
        for r in rows:
            pad = max_len - len(r["input_ids"])
            input_ids.append(list(r["input_ids"]) + [self.pad_token_id] * pad)
            attention_mask.append(list(r["attention_mask"]) + [0] * pad)
            labels.append(list(r["labels"]) + [IGNORE_INDEX] * pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class TorchDatasetAdapter:
    """Adapts :class:`FormattedExample` for ``torch.utils.data.Dataset``.

    Yields **plain dicts of int lists**, not the dataclass, because TRL's
    ``SFTTrainer`` probes the first element with ``"prompt" in sample`` while
    deciding between prompt-completion and conversational mode. A dataclass is
    not iterable and that probe raises ``TypeError``. A standard
    ``datasets``-style mapping is also what HF's Trainer expects for
    ``skip_prepare_dataset`` datasets.
    """

    def __init__(self, examples: list[FormattedExample]) -> None:
        self.examples = examples
        self._rows = [
            {
                "input_ids": list(e.input_ids),
                "attention_mask": list(e.attention_mask),
                "labels": list(e.labels),
            }
            for e in examples
        ]

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self._rows[index]

    def __iter__(self):  # type: ignore[override]
        return iter(self._rows)