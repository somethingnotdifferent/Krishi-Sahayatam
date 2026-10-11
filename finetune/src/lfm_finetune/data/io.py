"""Dataset readers/writers and format conversion to the canonical schema.

Supported raw formats
---------------------
``conversational``
    ``{"messages": [{"role": ..., "content": ...}, ...]}``
    Also accepts OpenAI-style ``{"role", "content"}`` at top level, and
    ``{"conversations": [...]}`` (ShareGPT / FastChat shape).

``instruction``
    ``{"instruction", "input", "output"}``  (Alpaca)
    ``{"prompt", "completion"}``           (self-instruct)
    ``{"question", "answer"}``             (closed QA)
    ``{"input", "output"}``                (minimal pair)

``auto``
    Sniff each record independently and normalise. Mixed corpora are common;
    rejecting a whole file because 3% of rows use a different shape is hostile.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..utils.errors import DatasetError
from ..utils.logging import get_logger
from .schema import Conversation

log = get_logger("data.io")

SUPPORTED_SUFFIXES = (".jsonl", ".json", ".jsonlines")

#: Files that live alongside raw data but are metadata, not records. Reading a
#: manifest as if it were a single example produces a bogus
#: "unrecognised_format" rejection and an off-by-one in the row count.
NON_DATA_FILENAMES = frozenset({"manifest.json", "manifest.yaml", "manifest.yml",
                                "schema.json", "readme.json", "_meta.json"})


# --------------------------------------------------------------------------- #
# Format detection / conversion
# --------------------------------------------------------------------------- #

_MESSAGES_KEYS = ("messages", "conversations", "conversation", "chat", "turns")


def detect_format(record: dict[str, Any]) -> str | None:
    """Return ``'conversational'``, ``'instruction'`` or ``None``."""
    for key in _MESSAGES_KEYS:
        value = record.get(key)
        if isinstance(value, list) and value:
            if isinstance(value[0], dict) and "role" in value[0]:
                return "conversational"
            if isinstance(value[0], dict) and "from" in value[0]:
                return "conversational"
            if isinstance(value[0], dict) and {"value", "from"} <= set(value[0]):
                return "conversational"
    if "role" in record and "content" in record:
        return "conversational"
    instruction_keys = {"instruction", "input", "output", "prompt", "completion",
                        "question", "answer", "query", "response", "text"}
    if instruction_keys & set(record):
        return "instruction"
    return None


_ROLE_ALIASES = {
    "human": "user",
    "usr": "user",
    "prompter": "user",
    "user": "user",
    "gpt": "assistant",
    "bot": "assistant",
    "chatbot": "assistant",
    "assistant": "assistant",
    "model": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "tool",
    "observation": "tool",
    "function_call": "tool",
}


def normalise_role(role: str) -> str | None:
    """Map a vendor role name to a canonical role, or ``None`` if unusable."""
    return _ROLE_ALIASES.get(str(role).strip().lower())


def _first_text(record: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, list):
            # e.g. Delta/ShareGPT {"value": [...]}
            flat = "".join(str(v) for v in value)
            if flat.strip():
                return flat
    return None


def convert_record(
    record: dict[str, Any],
    *,
    fmt: str = "auto",
    system_prompt: str | None = None,
    inject_metadata: bool = False,
) -> Conversation:
    """Normalise one raw record into a :class:`Conversation`.

    Raises:
        DatasetError: with a specific reason when the shape is unrecognised.
    """
    detected = detect_format(record) if fmt == "auto" else fmt

    if detected == "conversational":
        conv = _from_conversational(record, system_prompt=system_prompt)
    elif detected == "instruction":
        conv = _from_instruction(record, system_prompt=system_prompt)
    else:
        raise DatasetError(
            "Could not determine record format. Expected a 'messages' list, or one of "
            "instruction/input/output, prompt/completion, question/answer, input/output.",
            hint=f"Record keys present: {sorted(record)[:12]}. Add the missing field to your "
            "raw file, or set dataset.format explicitly.",
        )

    if inject_metadata and conv.metadata:
        conv = _inject_metadata(conv)
    return conv


def _from_conversational(record: dict[str, Any], *, system_prompt: str | None) -> Conversation:
    raw_messages: list[Any] = []
    for key in _MESSAGES_KEYS:
        value = record.get(key)
        if isinstance(value, list) and value:
            raw_messages = value
            break
    if not raw_messages and "role" in record and "content" in record:
        # A bare {"role": ..., "content": ...} record: a single turn.
        raw_messages = [{"role": record["role"], "content": record["content"]}]

    messages: list[dict[str, Any]] = []
    for item in raw_messages:
        if not isinstance(item, dict):
            raise DatasetError(f"Message entries must be objects, got {type(item).__name__}")
        raw_role = item.get("role") or item.get("from")
        role = normalise_role(raw_role) if raw_role is not None else None
        if role is None:
            raise DatasetError(
                f"Unrecognised message role {raw_role!r}",
                hint=f"Supported roles: {sorted(_ROLE_ALIASES)}. Known aliases include "
                "human->user, gpt->assistant.",
            )
        content = item.get("content")
        if content is None:
            content = _first_text(item, ("value", "text"))
        entry: dict[str, Any] = {"role": role, "content": content if content is not None else ""}
        # Preserve native tool-call structure when present; the LFM2.5 template
        # renders it and training on it teaches the model's native tool format.
        if item.get("tool_calls"):
            entry["tool_calls"] = item["tool_calls"]
        if item.get("name"):
            entry["name"] = item["name"]
        messages.append(entry)

    if not messages:
        raise DatasetError("'messages' list is empty")

    if system_prompt and messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": system_prompt})

    metadata = {
        k: v for k, v in record.items() if k not in set(_MESSAGES_KEYS) | {"id", "metadata"}
    }
    metadata.update(record.get("metadata") or {})
    return Conversation(id=record.get("id"), messages=messages, metadata=metadata)


def _from_instruction(record: dict[str, Any], *, system_prompt: str | None) -> Conversation:
    instruction = _first_text(record, ("instruction", "prompt", "question", "query", "input"))
    output = _first_text(record, ("output", "completion", "answer", "response", "target"))
    extra_input = _first_text(record, ("input", "details", "context"))

    if instruction is None or output is None:
        raise DatasetError(
            "Instruction record is missing a prompt or a response field.",
            hint="Expected pairs among "
            "(instruction|input|output), (prompt|completion), (question|answer).",
        )

    # Alpaca-style: `input` is a *continuation* of `instruction`, not the prompt.
    if "instruction" in record and extra_input and extra_input != instruction:
        user_text = f"{instruction}\n\n{extra_input}"
    else:
        user_text = instruction

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_text})
    messages.append({"role": "assistant", "content": output})

    reserved = {"instruction", "input", "output", "prompt", "completion",
                "question", "answer", "query", "response", "target", "text",
                "context", "details", "id"}
    metadata = {k: v for k, v in record.items() if k not in reserved}
    metadata.update(record.get("metadata") or {})
    return Conversation(id=record.get("id"), messages=messages, metadata=metadata)


_METADATA_INJECT_ORDER = ("crop", "plant", "disease", "pest", "growth_stage",
                          "nutrient_issue", "region", "climate", "soil",
                          "irrigation", "season")


def _inject_metadata(conv: Conversation) -> Conversation:
    """Append structured tags to the *first user turn* (never the system prompt).

    Opt-in behaviour: a model trained this way learns to read tags; a model
    trained without them must not be evaluated with them, or the A/B comparison
    is invalid.
    """
    tags = [f"{k}: {conv.metadata[k]}" for k in _METADATA_INJECT_ORDER if conv.metadata.get(k)]
    if not tags:
        return conv
    block = "[context] " + "; ".join(tags)
    messages = [m.model_copy() for m in conv.messages]
    for m in messages:
        if m.role == "user":
            m.content = f"{block}\n{m.content}"
            break
    return Conversation(id=conv.id, messages=messages, metadata=conv.metadata)


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #

def iter_raw_records(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(index, record)`` from a ``.jsonl`` or ``.json`` file or directory.

    Directories are walked in sorted order for determinism. Files whose suffix is
    not recognised are skipped with a warning rather than crashing the run.
    """
    path = Path(path)
    if not path.exists():
        raise DatasetError(
            f"Dataset path does not exist: {path}",
            hint="Check dataset.path in your config, or run `lfm-finetune prepare` on a "
            "file that exists.",
        )

    files = sorted(path.rglob("*")) if path.is_dir() else [path]
    data_files = [
        f
        for f in files
        if f.is_file()
        and f.suffix.lower() in SUPPORTED_SUFFIXES
        and f.name.lower() not in NON_DATA_FILENAMES
    ]
    skipped = [
        f.name for f in files
        if f.is_file()
        and f.suffix.lower() in SUPPORTED_SUFFIXES
        and f.name.lower() in NON_DATA_FILENAMES
    ]
    if skipped:
        log.info("Skipping metadata file(s) alongside raw data: %s", ", ".join(sorted(set(skipped))))
    if not data_files:
        raise DatasetError(
            f"No dataset files found under {path} (looked for {', '.join(SUPPORTED_SUFFIXES)})",
            hint="Place a .jsonl file in data/raw/, or point dataset.path at one.",
        )
    if path.is_dir() and len(data_files) > 1:
        log.info("Reading %d files from directory %s", len(data_files), path)

    index = 0
    for file in data_files:
        if file.suffix.lower() == ".jsonl":
            with file.open("r", encoding="utf-8") as handle:
                for lineno, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line or line.startswith("//"):
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise DatasetError(
                            f"Malformed JSON in {file.name} line {lineno}: {exc.msg}",
                            hint="Repair the raw file, or split it into per-record files.",
                        ) from exc
                    if not isinstance(record, dict):
                        raise DatasetError(
                            f"{file.name} line {lineno}: expected a JSON object, "
                            f"got {type(record).__name__}"
                        )
                    yield index, record
                    index += 1
        else:
            with file.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            records = payload if isinstance(payload, list) else payload.get("data", [payload])
            for record in records:
                yield index, record
                index += 1


def count_raw_records(path: Path) -> int:
    return sum(1 for _ in iter_raw_records(Path(path)))


# --------------------------------------------------------------------------- #
# Provenance manifest
# --------------------------------------------------------------------------- #

def build_manifest(
    *,
    name: str,
    version: str,
    source: str,
    license: str,
    created: str,
    examples: int,
    content_hash: str,
    categories: list[str],
    language: list[str],
    annotation_method: str,
    provenance_notes: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the dataset provenance manifest.

    Training refuses to start on a dataset whose ``license`` is ``unknown``;
    see :func:`validate_manifest`.
    """
    return {
        "name": name,
        "version": version,
        "source": source,
        "license": license,
        "created": created,
        "num_examples": examples,
        "content_hash": content_hash,
        "categories": categories,
        "language": language,
        "annotation_method": annotation_method,
        "provenance_notes": provenance_notes,
        **(extra or {}),
    }


UNKNOWN_LICENSES = {"unknown", "unspecified", "", "none", "tbd"}


def validate_manifest(manifest: dict[str, Any], *, source_file: Path) -> list[str]:
    """Return a list of provenance problems. Empty list == acceptable.

    The pipeline *warns* rather than hard-fails on provenance, because blocking a
    run on a missing field would be unhelpful; but it is loud and it lands in the
    run manifest.
    """
    problems: list[str] = []
    license_value = str(manifest.get("license", "")).strip().lower()
    if license_value in UNKNOWN_LICENSES:
        problems.append(
            f"dataset manifest {source_file.name} declares license="
            f"{manifest.get('license')!r}. Training on data of unknown provenance is not "
            "acceptable; set a real SPDX identifier or a clear 'internal-use-only'."
        )
    if not manifest.get("source"):
        problems.append(f"dataset manifest {source_file.name} has no 'source' field")
    if not manifest.get("created"):
        problems.append(f"dataset manifest {source_file.name} has no 'created' date")
    if manifest.get("num_examples") is None:
        problems.append(f"dataset manifest {source_file.name} has no 'num_examples'")
    return problems


def load_manifest(path: Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    if p.suffix.lower() == ".json":
        with p.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    from ..utils.io import read_yaml

    return read_yaml(p) or {}