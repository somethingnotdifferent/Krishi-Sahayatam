"""Canonical dataset schema.

One conversation record, plus optional domain metadata kept **out of** the prompt.

::

    {
      "id": "agri-0001",                  # optional, stable; used for hashing/leakage
      "messages": [
        {"role": "system",    "content": "..."},
        {"role": "user",      "content": "..."},
        {"role": "assistant", "content": "..."}
      ],
      "metadata": {                        # optional, opaque to the model
        "domain": "agriculture",
        "category": "diagnosis",
        "crop": "tomato",
        "disease": "early_blight",
        "region": "IN-KR",
        "confidence": 0.8,
        "source": "extension-officer-notes",
        "license": "CC-BY-4.0"
      }
    }

Design decision: metadata is preserved alongside the text but **never**
interpolated into the prompt by default. Structured tags are for analysis,
filtering and eval slicing. Putting them in the prompt is an explicit,
per-dataset opt-in (``dataset.inject_metadata``) so that the same engine can
serve a diagnosis-only model and a metadata-aware model without code changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Role = Literal["system", "user", "assistant", "tool"]

#: Roles that must appear, and in this relative order, for a trainable SFT record.
REQUIRED_ROLE_SEQUENCE: Final[tuple[str, ...]] = ("user", "assistant")

#: Every role the LFM2.5 chat template knows how to render.
KNOWN_ROLES: Final[frozenset[str]] = frozenset({"system", "user", "assistant", "tool"})


class Message(BaseModel):
    """A single chat turn."""

    model_config = ConfigDict(extra="allow")  # keep vendor fields such as `tool_calls`

    role: Role
    content: str

    @field_validator("content", mode="before")
    @classmethod
    def _stringify(cls, v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, str):
            return v
        # Multimodal content blocks collapse to their text; LFM2.5 text-only SFT
        # cannot consume images anyway.
        if isinstance(v, list):
            parts = []
            for block in v:
                if isinstance(block, dict):
                    parts.append(str(block.get("text", "")))
                else:
                    parts.append(str(block))
            return "".join(parts)
        return str(v)

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content}


class Conversation(BaseModel):
    """A validated, trainable conversational example."""

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    messages: list[Message] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("messages")
    @classmethod
    def _non_empty_content_somewhere(cls, v: list[Message]) -> list[Message]:
        if not any(m.content.strip() for m in v):
            raise ValueError("conversation has no non-empty message content")
        return v

    # -- convenience accessors ------------------------------------------- #

    @property
    def system(self) -> str | None:
        for m in self.messages:
            if m.role == "system":
                return m.content
        return None

    @property
    def turns(self) -> list[tuple[str, str]]:
        return [(m.role, m.content) for m in self.messages]

    @property
    def user_texts(self) -> list[str]:
        return [m.content for m in self.messages if m.role == "user"]

    @property
    def assistant_texts(self) -> list[str]:
        return [m.content for m in self.messages if m.role == "assistant"]

    @property
    def last_user(self) -> str:
        users = self.user_texts
        return users[-1] if users else ""

    @property
    def last_assistant(self) -> str:
        assistants = self.assistant_texts
        return assistants[-1] if assistants else ""

    def to_record(self) -> dict[str, Any]:
        """Serialisable form written to processed/split files."""
        out: dict[str, Any] = {"messages": [m.to_dict() for m in self.messages]}
        if self.id:
            out["id"] = self.id
        if self.metadata:
            out["metadata"] = self.metadata
        return out

    def fingerprint_payload(self) -> str:
        """Canonical string used for exact-duplicate detection and split hashing."""
        from ..utils.hashing import canonical_json

        return canonical_json([{"role": m.role, "content": m.content} for m in self.messages])


@dataclass
class Rejection:
    """Why a raw record was not accepted. Never silently discarded."""

    index: int
    reason: str
    detail: str
    source: str | None = None
    excerpt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "reason": self.reason,
            "detail": self.detail,
            "source": self.source,
            "excerpt": self.excerpt[:200],
        }


@dataclass
class DatasetStats:
    """Aggregates emitted by the dataset pipeline and stored in the run manifest."""

    num_examples: int = 0
    num_messages: int = 0
    total_tokens: int = 0
    mean_tokens: float = 0.0
    median_tokens: float = 0.0
    p95_tokens: float = 0.0
    min_tokens: int = 0
    max_tokens: int = 0
    total_chars: int = 0
    mean_user_chars: float = 0.0
    mean_assistant_chars: float = 0.0
    roles: dict[str, int] = field(default_factory=dict)
    categories: dict[str, int] = field(default_factory=dict)
    crops: dict[str, int] = field(default_factory=dict)
    languages: dict[str, int] = field(default_factory=dict)
    turns_histogram: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_examples": self.num_examples,
            "num_messages": self.num_messages,
            "total_tokens": self.total_tokens,
            "mean_tokens": round(self.mean_tokens, 2),
            "median_tokens": round(self.median_tokens, 2),
            "p95_tokens": round(self.p95_tokens, 2),
            "min_tokens": self.min_tokens,
            "max_tokens": self.max_tokens,
            "total_chars": self.total_chars,
            "mean_user_chars": round(self.mean_user_chars, 2),
            "mean_assistant_chars": round(self.mean_assistant_chars, 2),
            "roles": dict(sorted(self.roles.items())),
            "categories": dict(sorted(self.categories.items())),
            "crops": dict(sorted(self.crops.items())),
            "languages": dict(sorted(self.languages.items())),
            "turns_histogram": dict(sorted(self.turns_histogram.items(), key=lambda kv: int(kv[0]))),
        }


#: Aggregated counters written to the machine-readable preparation report.
COUNT_KEYS: Final[tuple[str, ...]] = (
    "raw_rows",
    "accepted",
    "rejected",
    "modified",
    "deduplicated_exact",
    "deduplicated_near",
    "dropped_prompt_leakage",
    "split_train",
    "split_validation",
    "split_test",
    "after_quality_filter",
)