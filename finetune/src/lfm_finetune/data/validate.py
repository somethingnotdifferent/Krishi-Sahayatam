"""Record-level validation and domain quality gates.

Two layers run over the raw stream:

1. **Structural validation** -- can this record become a :class:`Conversation` at
   all? (format detection, role validity, message ordering)
2. **Quality gates** -- is it a *useful* SFT example? (emptiness, length bounds,
   placeholder text, prompt leakage, duplicate answers)

Both layers record a :class:`Rejection` for every dropped row. Nothing is ever
dropped silently: the counts land in ``report.json`` and the examples are kept
in a quarantine file so a human can inspect what was thrown away.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from ..utils.logging import get_logger
from .schema import Conversation, Rejection

log = get_logger("data.validate")

# C0/C1 control characters, excluding tab (\x09), newline (\x0a) and
# carriage return (\x0d) which we normalise rather than delete.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")
_ZERO_WIDTH_RE = re.compile(r"[-‏⁠﻿]")
_TRAILING_WS_RE = re.compile(r"[ \t]+(\n|$)")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_TAG_RE = re.compile(r"<[^>]{1,200}>")
_HTML_ENTITIES = {"&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&#39;": "'", "&nbsp;": " "}

DEFAULT_PLACEHOLDER_PATTERNS = (
    r"\bTODO\b",
    r"\bFIXME\b",
    r"\bXXX+\b",
    r"\bLorem ipsum\b",
    r"\bplaceholder\b",
    r"\bTBD\b",
    r"<\|.*?\|>",
    r"\[insert .*?\]",
    r"\bN/?A\s*$",
)


class TextNormalizer:
    """Unicode/text hygiene applied before any downstream comparison.

    Normalising *before* deduplication is essential: otherwise two copies of the
    same example differing only in trailing whitespace survive as separate rows
    and split across train/validation.
    """

    def __init__(
        self,
        *,
        normalize_whitespace: bool = True,
        strip_control_characters: bool = True,
        strip_html: bool = True,
        unicode_form: str = "NFC",
    ) -> None:
        self.normalize_whitespace = normalize_whitespace
        self.strip_control_characters = strip_control_characters
        self.strip_html = strip_html
        self.unicode_form = unicode_form

    def __call__(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        if self.strip_control_characters:
            text = _SURROGATE_RE.sub("", text)
            text = _CONTROL_RE.sub("", text)
            text = _ZERO_WIDTH_RE.sub("", text)
        if self.unicode_form:
            # NFC before any other work: canonically equivalent sequences
            # ("é" as one codepoint vs "e"+combining accent) would otherwise
            # defeat duplicate detection.
            text = unicodedata.normalize(self.unicode_form, text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        if self.strip_html:
            text = _TAG_RE.sub(" ", text)
            for entity, char in _HTML_ENTITIES.items():
                text = text.replace(entity, char)
        if self.normalize_whitespace:
            text = _MULTI_SPACE_RE.sub(" ", text)
            text = _TRAILING_WS_RE.sub(r"\1", text)
            text = _MULTI_NEWLINE_RE.sub("\n\n", text)
        return text.strip()


@dataclass
class QualityConfig:
    """Runtime form of ``dataset.quality`` (decoupled from the pydantic config)."""

    min_user_chars: int = 2
    min_assistant_chars: int = 8
    max_user_chars: int = 8000
    max_assistant_chars: int = 16000
    min_messages: int = 2
    max_turns: int = 32
    drop_empty_messages: bool = True
    normalize_whitespace: bool = True
    strip_control_characters: bool = True
    strip_html: bool = True
    deduplicate: bool = True
    near_duplicate: bool = True
    near_duplicate_threshold: float = 0.90
    remove_prompt_leakage: bool = True
    placeholder_patterns: list[str] = field(
        default_factory=lambda: list(DEFAULT_PLACEHOLDER_PATTERNS)
    )

    @classmethod
    def from_config(cls, cfg: Any) -> QualityConfig:
        q = cfg.dataset.quality
        return cls(
            min_user_chars=q.min_user_chars,
            min_assistant_chars=q.min_assistant_chars,
            max_user_chars=q.max_user_chars,
            max_assistant_chars=q.max_assistant_chars,
            min_messages=q.min_messages,
            max_turns=q.max_turns,
            drop_empty_messages=q.drop_empty_messages,
            normalize_whitespace=q.normalize_whitespace,
            strip_control_characters=q.strip_control_characters,
            strip_html=q.strip_html,
            deduplicate=q.deduplicate,
            near_duplicate=q.near_duplicate,
            near_duplicate_threshold=q.near_duplicate_threshold,
            remove_prompt_leakage=q.remove_prompt_leakage,
            placeholder_patterns=list(q.placeholder_patterns) or list(DEFAULT_PLACEHOLDER_PATTERNS),
        )


@dataclass
class ValidationOutcome:
    """Result of validating/normalising one raw record."""

    conversation: Conversation | None
    modified: bool = False
    rejection: Rejection | None = None


def _has_placeholder(text: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        try:
            if re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE):
                return pattern
        except re.error as exc:
            raise ValueError(
                f"Invalid dataset.quality.placeholder_patterns regex {pattern!r}: {exc}"
            ) from exc
    return None


def _role_sequence_ok(roles: list[str]) -> bool:
    """Require >=1 user and >=1 assistant, with an assistant after the first user."""
    if "user" not in roles or "assistant" not in roles:
        return False
    return roles.index("assistant") > roles.index("user")


def validate_conversation(
    conv: Conversation,
    *,
    index: int,
    source: str | None,
    quality: QualityConfig,
    normalizer: TextNormalizer,
) -> ValidationOutcome:
    """Apply structural + quality gates to an already-converted conversation.

    Returns:
        :class:`ValidationOutcome`. On rejection, ``conversation`` is ``None``
        and ``rejection`` explains why.
    """
    original = conv.fingerprint_payload()
    modified = False

    # -- text hygiene ---------------------------------------------------- #
    if quality.normalize_whitespace or quality.strip_control_characters or quality.strip_html:
        new_messages = []
        for m in conv.messages:
            cleaned = normalizer(m.content)
            if cleaned != m.content:
                modified = True
            new_messages.append({"role": m.role, "content": cleaned, **(
                {"tool_calls": m.tool_calls} if getattr(m, "tool_calls", None) else {}
            )})
        conv = Conversation(
            id=conv.id,
            messages=[{"role": m["role"], "content": m["content"], **{
                k: v for k, v in m.items() if k not in ("role", "content")
            }} for m in new_messages],
            metadata=conv.metadata,
        )

    # -- empty content --------------------------------------------------- #
    if quality.drop_empty_messages:
        kept = [m for m in conv.messages if m.content.strip()]
        if len(kept) != len(conv.messages):
            modified = True
            if not kept:
                return ValidationOutcome(
                    None,
                    rejection=Rejection(index, "all_messages_empty",
                                        "every message became empty after normalisation", source),
                )
            conv = Conversation(id=conv.id, messages=kept, metadata=conv.metadata)

    messages = conv.messages
    roles = [m.role for m in messages]

    # -- role / structure ------------------------------------------------ #
    unknown = [r for r in roles if r not in {"system", "user", "assistant", "tool"}]
    if unknown:
        return ValidationOutcome(None, rejection=Rejection(
            index, "invalid_role", f"unknown role(s) {sorted(set(unknown))}", source))
    if not _role_sequence_ok(roles):
        return ValidationOutcome(None, rejection=Rejection(
            index, "missing_turn",
            "record needs at least one user turn followed by an assistant turn "
            f"(roles found: {roles})", source))
    if len(messages) > quality.max_turns:
        return ValidationOutcome(None, rejection=Rejection(
            index, "too_many_turns",
            f"{len(messages)} messages exceeds max_turns={quality.max_turns}", source))
    if len(messages) < quality.min_messages:
        return ValidationOutcome(None, rejection=Rejection(
            index, "too_few_messages",
            f"{len(messages)} messages is below min_messages={quality.min_messages}", source))

    # Multiple leading system prompts are a common templating bug: only the
    # first one has any effect in the LFM chat template, the rest are silently
    # rendered as literal "system\n..." user turns.
    system_positions = [i for i, r in enumerate(roles) if r == "system"]
    if len(system_positions) > 1:
        return ValidationOutcome(None, rejection=Rejection(
            index, "multiple_system_prompts",
            f"{len(system_positions)} system messages; the LFM2.5 chat template only "
            "honours messages[0] as the system prompt", source,
            excerpt=str(messages[system_positions[1]].content)[:120]))

    # -- length gates ---------------------------------------------------- #
    for user in conv.user_texts:
        if len(user.strip()) < quality.min_user_chars:
            return ValidationOutcome(None, rejection=Rejection(
                index, "empty_user_message",
                f"user message has {len(user.strip())} chars, below min_user_chars="
                f"{quality.min_user_chars}", source, excerpt=user))
    for answer in conv.assistant_texts:
        stripped = answer.strip()
        if len(stripped) < quality.min_assistant_chars:
            return ValidationOutcome(None, rejection=Rejection(
                index, "empty_assistant_response",
                f"assistant message has {len(stripped)} chars, below min_assistant_chars="
                f"{quality.min_assistant_chars}. An empty target teaches the model to stop "
                "immediately.", source, excerpt=answer))
        if len(answer) > quality.max_assistant_chars:
            return ValidationOutcome(None, rejection=Rejection(
                index, "assistant_response_too_long",
                f"assistant message is {len(answer)} chars, above max_assistant_chars="
                f"{quality.max_assistant_chars}", source, excerpt=answer[:120]))
    for user in conv.user_texts:
        if len(user) > quality.max_user_chars:
            log.warning(
                "Row %d: user message is %d chars (above max_user_chars=%d); it will be "
                "truncated by the tokeniser. Consider splitting the sample.",
                index, len(user), quality.max_user_chars,
            )

    # -- placeholder / unfinished text ----------------------------------- #
    for message in messages:
        hit = _has_placeholder(message.content, quality.placeholder_patterns)
        if hit:
            return ValidationOutcome(None, rejection=Rejection(
                index, "placeholder_text",
                f"matched placeholder pattern {hit!r}", source,
                excerpt=message.content))

    # -- prompt leakage --------------------------------------------------- #
    if quality.remove_prompt_leakage:
        leak = _find_prompt_leakage(conv)
        if leak is not None:
            return ValidationOutcome(None, rejection=Rejection(
                index, "prompt_leakage", leak, source))

    # -- duplicate answers within one conversation ----------------------- #
    if len(set(a.strip() for a in conv.assistant_texts)) < len(conv.assistant_texts):
        log.debug("Row %d: repeated assistant answer inside one conversation", index)

    if modified and original != conv.fingerprint_payload():
        return ValidationOutcome(conv, modified=True, rejection=None)
    return ValidationOutcome(conv, modified=modified, rejection=None)


def _find_prompt_leakage(conv: Conversation) -> str | None:
    """Detect an answer that is (nearly) verbatim present in the prompt.

    Two signals:
      * the whole normalised answer appears inside a user turn;
      * a long contiguous shingle from the answer also appears in the prompt.

    The second catches templated datasets where the assistant turn repeats the
    question verbatim before answering -- a pattern that strongly correlates
    with the model learning to parrot rather than to reason.
    """
    norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()  # noqa: E731

    for answer in conv.assistant_texts:
        answer_norm = norm(answer)
        if len(answer_norm) < 40:
            continue
        prompts_norm = [norm(u) for u in conv.user_texts]
        joined = " || ".join(prompts_norm)
        if answer_norm in joined:
            return "assistant answer is contained verbatim in the user prompt"

        # Shingle overlap: 12-word windows.
        words = answer_norm.split()
        if len(words) < 24:
            continue
        shingle = " ".join(words[:12])
        if shingle in joined:
            # Require substantial overlap to avoid false positives on boilerplate.
            matches = sum(
                1
                for start in range(0, len(words) - 12, 12)
                if " ".join(words[start : start + 12]) in joined
            )
            if matches >= 2:
                return (
                    f"assistant answer shares {matches} repeated 12-word spans with the "
                    "user prompt (templated/parroting data)"
                )
    return None


def structural_reject_reason(raw: dict[str, Any], exc: Exception) -> str:
    """Classify a conversion failure into a stable machine-readable reason."""
    message = str(exc)
    if "role" in message and "Unrecognised" in message:
        return "invalid_role"
    if "empty" in message.lower():
        return "all_messages_empty"
    if "format" in message.lower() or "Could not determine" in message:
        return "unrecognised_format"
    if "missing" in message.lower():
        return "missing_fields"
    return "conversion_failed"