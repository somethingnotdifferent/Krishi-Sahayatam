"""Chat-template rendering and assistant-only loss masking.

This module is the correctness core of an SFT pipeline. Two things must be true:

1. **Training formatting == inference formatting, character for character.**
   The template renders ``<|im_start|>assistant\\n`` as the generation prompt, so a
   training target must be exactly the continuation the model will be asked to
   produce.

2. **Only assistant tokens contribute to the loss.**
   Training on the prompt teaches the model to reproduce user questions, which
   wastes capacity and, for a domain assistant, actively degrades it.

Verified facts about the LFM2.5 chat template
---------------------------------------------
Measured against the actual checkpoints with transformers 5.3.0 (see
``scripts/inspect_samples.py`` / ``lfm-finetune inspect-template``):

* Every LFM2.5 checkpoint's template emits ChatML-style control tokens:
  ``<|im_start|>`` (id 6) before each turn and ``<|im_end|>`` (id 7) after it,
  with the role name and content separated by a newline. The rendered text for
  a system/user/assistant conversation is::

      <|startoftext|><|im_start|>system\\nSYS<|im_end|>\\n
      <|im_start|>user\\nUSER<|im_end|>\\n
      <|im_start|>assistant\\nASST<|im_end|>\\n

* ``eos_token_id`` **is** 7 (``<|im_end|>``) for the 65,536-vocab checkpoints
  (1.2B, 350M, 230M) and 124900 for the 128k-vocab ones (2.6B, 8B-A1B), and the
  loaded tokenizer resolves ``eos_token`` to that marker. Beware:
  ``special_tokens_map.json`` in these repos registers ids 6 and 7 with an empty
  ``content`` field, so reading the raw JSON makes the terminator look absent.
  What governs generation is the id, not the raw JSON.

* The rendered text therefore already ends with ``<|im_end|>`` followed by a
  literal newline. :meth:`ChatFormatter._strip_trailing_newlines` removes that
  trailing newline so the supervised target ends *exactly* at the end-of-turn
  token, which is what inference stops on.

* ``add_special_tokens=False`` is mandatory when tokenising the rendered text:
  the template emits ``{{ bos_token }}`` itself, and the tokenizer's own
  post-processor would prepend a **second** BOS. Measured: the default call
  produces ``[1, 1, 24131, ...]``.

Why prefix-length masking instead of ``return_assistant_tokens_mask``
--------------------------------------------------------------------
The obvious approach is to ask the tokenizer for an assistant mask. It requires
``{% generation %}`` blocks in the chat template, and **LFM2.5-1.2B-Base and
LFM2.5-350M-Base do not have them**:

===========================  ==================  ==========================
checkpoint                   ``{% generation %}``  assistant mask result
===========================  ==================  ==========================
LiquidAI/LFM2.5-1.2B-Base    no                 **0 tokens masked**
LiquidAI/LFM2.5-350M-Base    no                 **0 tokens masked**
LiquidAI/LFM2.5-230M-Base    yes                7 tokens, answer covered
LiquidAI/LFM2.5-2.6B-Base    yes                7 tokens, answer covered
===========================  ==================  ==========================

``apply_chat_template(..., return_assistant_tokens_mask=True)`` on the primary
target checkpoint therefore returns an all-zero mask: it would train on 100% of
tokens while appearing to have masked the prompt. TRL's
``assistant_only_loss=True`` inherits the same failure.

Instead the boundary is computed directly and *asserted*:

* ``prompt_ids`` = template(messages[:-1], add_generation_prompt=True)
* ``full_ids``   = template(messages,     add_generation_prompt=False)

If the template ever stopped being prefix-consistent, training would silently
supervise the wrong span, so the prefix property is verified rather than assumed.
The same mask is used for every checkpoint size, so behaviour does not change if
you swap 1.2B for 2.6B.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..config.schema import TokenizationConfig
from ..data.schema import Conversation
from ..utils.errors import TokenizerError
from ..utils.logging import get_logger

log = get_logger("training.tokenization")

IGNORE_INDEX = -100


@dataclass
class FormattedExample:
    """One tokenised, label-masked training example."""

    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    num_prompt_tokens: int
    num_loss_tokens: int
    truncated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.input_ids)

    @property
    def loss_coverage(self) -> float:
        """Fraction of tokens that contribute to the loss. Should be < 1.0."""
        return self.num_loss_tokens / max(1, self.length)


class ChatFormatter:
    """Renders conversations through the checkpoint's own chat template.

    Args:
        tokenizer: The checkpoint tokenizer (with its chat template loaded).
        config: ``tokenization`` section of the experiment config.
    """

    def __init__(self, tokenizer: Any, config: TokenizationConfig) -> None:
        if tokenizer is None:
            raise TokenizerError("Cannot build a ChatFormatter without a tokenizer.")
        if not getattr(tokenizer, "chat_template", None):
            raise TokenizerError(
                "Tokenizer has no chat template; cannot format conversations.",
                hint="Use a checkpoint that ships chat_template.jinja, or set "
                "tokenization.chat_template to a .jinja file.",
            )
        self.tokenizer = tokenizer
        self.config = config
        self.max_seq_length = int(config.max_seq_length)
        self.append_eos = bool(config.append_eos_token)
        self.truncation_strategy = config.truncation_strategy
        self.assistant_only_loss = True  # set by the trainer from training.assistant_only_loss
        self._validate_special_tokens()

    # ------------------------------------------------------------------ #

    def _validate_special_tokens(self) -> None:
        tok = self.tokenizer
        if tok.eos_token_id is None:
            raise TokenizerError(
                "Tokenizer eos_token_id is None; cannot append an end-of-turn token.",
                hint="Every LFM2.5 checkpoint defines it (id 7 for the 65,536-vocab base "
                "models). Check tokenizer_config.json for your checkpoint.",
            )
        if tok.pad_token_id is None:
            log.warning(
                "Tokenizer has no pad_token_id; falling back to eos for padding. "
                "This is safe because pad positions are masked to -100."
            )

    @property
    def pad_token_id(self) -> int:
        tok = self.tokenizer
        return tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    @property
    def eos_token_id(self) -> int:
        return int(self.tokenizer.eos_token_id)

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def render(self, messages: Sequence[dict[str, Any]], *, add_generation_prompt: bool) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception as exc:
            raise TokenizerError(
                f"apply_chat_template failed (add_generation_prompt={add_generation_prompt}): {exc}",
                hint="The dataset may contain a message shape the template cannot render "
                "(e.g. content=None). Check the rejected records in "
                "data/validated/rejected.jsonl.",
            ) from exc

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        """Tokenise rendered template text.

        ``add_special_tokens=False`` is essential and non-obvious: the LFM2.5
        template already emits ``{{ bos_token }}`` itself. Letting the tokenizer
        prepend a BOS as well would produce a doubled BOS and silently shift the
        model's input distribution away from pre-training.
        """
        ids = self.tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"]
        if isinstance(ids[0], list):  # batched call safety
            ids = ids[0]
        return list(ids)

    # ------------------------------------------------------------------ #
    # Formatting
    # ------------------------------------------------------------------ #

    def format_conversation(
        self,
        conversation: Conversation,
        *,
        assistant_only_loss: bool | None = None,
        max_length: int | None = None,
    ) -> FormattedExample:
        """Render, tokenise and mask one conversation.

        Raises:
            TokenizerError: if the template is not prefix-consistent, or if the
                resulting example has no trainable (assistant) tokens.
        """
        use_assistant_only = self.assistant_only_loss if assistant_only_loss is None else assistant_only_loss
        limit = max_length or self.max_seq_length
        messages = [m.to_dict() for m in conversation.messages]

        full_text = self.render(messages, add_generation_prompt=False)
        full_ids = self.encode(full_text)

        prompt_len = 0
        if use_assistant_only and len(messages) > 1:
            prompt_text = self.render(messages[:-1], add_generation_prompt=True)
            prompt_ids = self.encode(prompt_text)
            prompt_len = len(prompt_ids)
            # The safety property that makes prefix masking trustworthy.
            if full_ids[:prompt_len] != prompt_ids[:prompt_len]:
                raise TokenizerError(
                    "Chat template is not prefix-consistent: rendering the conversation "
                    "without the final assistant turn does not reproduce the beginning of "
                    "the full rendering. Assistant-only loss masking would then train on "
                    "the wrong token span.",
                    hint="This usually indicates a chat template that depends on later "
                    "messages. Try setting tokenization.chat_template to the canonical "
                    "LFM2.5 template, or set training.assistant_only_loss=false to mask by "
                    "position instead.",
                )
        else:
            # Even without assistant-only loss we exclude the BOS token from the
            # loss; there is nothing to predict there.
            prompt_len = 1 if full_ids else 0

        if self.append_eos:
            # Order matters: strip the template's trailing newline FIRST, then
            # append the end-of-turn token only if it is not already the last
            # token. The LFM2.5 template already emits `<|im_end|>` after the
            # final content, so after stripping we normally already have it.
            full_ids, prompt_len = self._strip_trailing_newlines(full_ids, prompt_len)
            if not full_ids or full_ids[-1] != self.eos_token_id:
                full_ids = full_ids + [self.eos_token_id]

        truncated = False
        if limit and len(full_ids) > limit:
            full_ids, prompt_len = self._truncate(full_ids, prompt_len, limit)
            truncated = True

        labels = self._build_labels(full_ids, prompt_len, assistant_only=use_assistant_only)
        num_loss_tokens = sum(1 for label in labels if label != IGNORE_INDEX)
        if num_loss_tokens == 0:
            raise TokenizerError(
                "Formatted example contains zero tokens contributing to the loss.",
                hint=f"The conversation's last turn is a role the loss does not cover "
                f"(roles: {[m['role'] for m in messages]}). A trainable SFT record must end "
                "with an assistant turn. Also check training.assistant_only_loss and "
                "tokenization.max_seq_length.",
            )

        return FormattedExample(
            input_ids=full_ids,
            attention_mask=[1] * len(full_ids),
            labels=labels,
            num_prompt_tokens=prompt_len,
            num_loss_tokens=num_loss_tokens,
            truncated=truncated,
            metadata={"id": conversation.id, **(conversation.metadata or {})},
        )

    def _build_labels(
        self, input_ids: list[int], prompt_len: int, *, assistant_only: bool
    ) -> list[int]:
        """Shift-safe labels: token *t* is predicted from tokens ``< t``.

        We deliberately do not shift here. HF models perform the internal shift
        in the loss function, so ``labels[i] == input_ids[i]`` is the correct
        convention; masking position ``i`` removes the prediction of token ``i``
        from the loss.
        """
        if not assistant_only:
            labels = list(input_ids)
            labels[0] = IGNORE_INDEX  # nothing to predict before the first token
            return labels
        labels = [IGNORE_INDEX] * prompt_len + list(input_ids[prompt_len:])
        return labels

    def _strip_trailing_newlines(
        self, ids: list[int], prompt_len: int
    ) -> tuple[list[int], int]:
        """Drop the newline that follows the template's end-of-turn token.

        The rendered conversation ends with ``<|im_end|>\\n``. At inference the
        prompt ends at ``<|im_start|>assistant\\n`` and generation stops the
        moment the model emits ``<|im_end|>``, so the literal newline after it is
        never produced. Leaving it in the training target would teach

            answer, end-of-turn, newline

        i.e. the model would emit the stop token and then be trained to keep
        going. After stripping, the target is exactly

            answer, end-of-turn

        which is what inference asks for.

        Only whitespace is removed, and never below ``prompt_len``, so the prompt
        span and its loss mask are untouched. ``<|im_end|>`` decodes to a
        non-whitespace string and therefore stops the scan, which is what keeps
        a genuinely-empty final answer from being truncated away.
        """
        end = len(ids)
        while end > prompt_len:
            piece = self.tokenizer.decode([ids[end - 1]])
            if piece.strip():
                break
            end -= 1
        if end == len(ids):
            return ids, prompt_len
        return ids[:end], min(prompt_len, end)

    def _truncate(
        self, ids: list[int], prompt_len: int, limit: int
    ) -> tuple[list[int], int]:
        """Truncate while guaranteeing that supervised tokens survive.

        The invariant this enforces: after truncation the example always
        contains at least one token inside the loss span. An earlier version used
        a plain head+tail split and, whenever the prompt alone exceeded
        ``max_seq_length``, produced an example whose every position was masked --
        so a long-prompt record trained on nothing while looking perfectly normal
        in the dataset statistics.

        Budget split for ``limit`` tokens:

        * ``reserve``  -- the tail of the supervised span, so the answer survives.
          At most ``limit // 2`` and at most the answer's own length.
        * ``head``     -- the opening of the conversation (system prompt + first
          user turn), half of what is left.
        * ``tail``     -- the end of the prompt, immediately before the answer,
          which is what keeps the generation prompt adjacent to the answer.

        ``keep_head_tail`` uses this. ``truncate_tail`` keeps the naive
        behaviour and warns, because it can genuinely delete the answer.
        """
        strategy = self.truncation_strategy
        if strategy == "error":
            raise TokenizerError(
                f"Example of {len(ids)} tokens exceeds tokenization.max_seq_length={limit} "
                "and truncation_strategy='error'.",
                hint="Raise tokenization.max_seq_length (watch VRAM), shorten the record, "
                "or switch truncation_strategy to keep_head_tail.",
            )

        answer_len = len(ids) - prompt_len
        if strategy == "truncate_tail":
            log.warning(
                "Truncating a %d-token example to %d from the tail; the final assistant "
                "turn may be cut and contribute no loss signal.",
                len(ids), limit,
            )
            return ids[:limit], min(prompt_len, limit)

        # keep_head_tail (and the safe default for any future strategy)
        reserve = max(1, min(answer_len, limit // 2))
        remaining = limit - reserve
        head = min(prompt_len, remaining // 2)
        tail = remaining - head
        new_prompt_len = head + tail
        new_ids = ids[:head] + ids[len(ids) - tail - reserve :]
        return new_ids, new_prompt_len

    # ------------------------------------------------------------------ #
    # Inference-side helpers
    # ------------------------------------------------------------------ #

    def build_prompt_ids(
        self, messages: Sequence[dict[str, Any]], *, add_generation_prompt: bool = True
    ) -> list[int]:
        """Token ids to feed ``model.generate`` -- the exact training prefix."""
        ids = self.encode(
            self.render(messages, add_generation_prompt=add_generation_prompt)
        )
        limit = self.max_seq_length
        if limit and len(ids) > limit:
            ids = ids[-limit:]
        return ids

    def generation_kwargs(self) -> dict[str, Any]:
        """Decode defaults that match how training built the targets.

        Only ``eos_token_id`` is returned. ``pad_token_id`` is deliberately
        omitted: the generation loop already supplies it from the tokenizer, and
        for a PEFT-wrapped model that reaches
        ``PeftModelForCausalLM.generate(pad_token_id=...)`` twice, raising
        ``got multiple values for keyword argument 'pad_token_id'``.
        """
        return {"eos_token_id": self.eos_token_id}

    def strip_generated(self, token_ids: Sequence[int]) -> str:
        """Decode generated ids, dropping the (invisible) end-of-turn token."""
        ids = list(token_ids)
        while ids and ids[-1] == self.eos_token_id:
            ids.pop()
        return self.tokenizer.decode(ids, skip_special_tokens=True).strip()


# --------------------------------------------------------------------------- #
# Human-readable inspection
# --------------------------------------------------------------------------- #

def format_example_for_display(
    example: FormattedExample, tokenizer: Any, *, max_tokens: int = 400
) -> str:
    """Render a tokenised example with the loss span visibly marked.

    Shown before every training run so an operator can confirm that only the
    assistant tokens are supervised -- the single most common silent SFT bug.
    """
    lines = [
        "=" * 88,
        f"example id={example.metadata.get('id', '<none>')} "
        f"len={example.length} prompt_tokens={example.num_prompt_tokens} "
        f"loss_tokens={example.num_loss_tokens} "
        f"coverage={example.loss_coverage:.1%}"
        + ("  [TRUNCATED]" if example.truncated else ""),
        "-" * 88,
    ]
    for i, (token_id, label) in enumerate(zip(example.input_ids, example.labels, strict=True)):
        if i >= max_tokens:
            lines.append(f"... [{len(example.input_ids) - max_tokens} more tokens]")
            break
        piece = tokenizer.decode([token_id])
        supervised = label != IGNORE_INDEX
        marker = "" if supervised else " "
        lines.append(f"{i:5d} id={token_id:<7d} {marker}{piece!r}")
    lines.append("=" * 88)
    lines.append("Blank 'marker' column = token does NOT contribute to the loss.")
    return "\n".join(lines)


def verify_template_consistency(
    tokenizer: Any, formatter: ChatFormatter, conversations: list[Conversation]
) -> dict[str, Any]:
    """Prove training and inference use the same rendering path.

    Checks, per example:
      * the prompt-only render is a strict prefix of the full render;
      * ``build_prompt_ids(messages[:-1])`` equals the training prompt span;
      * masking covers every user/system token and no assistant token.
    """
    if not conversations:
        return {"checked": 0, "ok": True, "failures": []}

    failures: list[dict[str, Any]] = []
    coverages: list[float] = []
    for conv in conversations:
        try:
            example = formatter.format_conversation(conv)
        except TokenizerError as exc:
            failures.append({"id": conv.id, "error": str(exc)})
            continue

        messages = [m.to_dict() for m in conv.messages]
        prompt_ids = formatter.build_prompt_ids(messages[:-1], add_generation_prompt=True)
        if example.input_ids[: len(prompt_ids)] != prompt_ids[: len(prompt_ids)]:
            failures.append({
                "id": conv.id,
                "error": "inference prompt ids are not a prefix of the training ids",
            })

        # Every token up to prompt_len must be masked; every token after must not be.
        for idx, label in enumerate(example.labels[: example.num_prompt_tokens]):
            if label != IGNORE_INDEX:
                failures.append({"id": conv.id, "error": f"prompt token {idx} is not masked"})
                break
        supervised_after = [
            idx for idx, label in enumerate(example.labels[example.num_prompt_tokens :],
                                            start=example.num_prompt_tokens)
            if label == IGNORE_INDEX
        ]
        if supervised_after:
            failures.append({
                "id": conv.id,
                "error": f"{len(supervised_after)} assistant-span token(s) are masked, "
                         f"first at index {supervised_after[0]}",
            })
        coverages.append(example.loss_coverage)

    return {
        "checked": len(conversations),
        "ok": not failures,
        "failures": failures[:10],
        "mean_loss_coverage": round(sum(coverages) / len(coverages), 4) if coverages else None,
    }


def template_fingerprint(tokenizer: Any) -> str:
    """Hash of the active chat template, recorded in the run manifest."""
    from ..utils.hashing import hash_obj

    return hash_obj({"template": getattr(tokenizer, "chat_template", None)})[:16]