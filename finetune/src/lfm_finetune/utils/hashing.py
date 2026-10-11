"""Content hashing for datasets, configs and artifacts.

Hashes are recorded in every run manifest so that a finished run can be tied to
an exact dataset state. Dataset hashes are *content* hashes over the canonical
serialised records (not mtimes), so re-running the pipeline on an unchanged
dataset yields an identical hash.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import yaml

_CHUNK = 1024 * 1024


def sha256_file(path: Path | str) -> str:
    """Stream a file through SHA-256 (handles multi-GB shards)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no insignificant whitespace, UTF-8 safe."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def hash_obj(obj: Any) -> str:
    """Content hash of any JSON-serialisable object."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def hash_records(records: Iterable[Any]) -> str:
    """Order-sensitive content hash of a record stream.

    Order matters deliberately: two datasets with the same records in a different
    order produce a different training trajectory and must not share a hash.
    """
    digest = hashlib.sha256()
    for record in records:
        digest.update(canonical_json(record).encode("utf-8"))
        digest.update(b"\x1e")  # record separator, avoids concatenation ambiguity
    return digest.hexdigest()


def short_hash(value: str, length: int = 12) -> str:
    """Human-friendly prefix of a hash, used in run directory names."""
    return value[:length]


def hash_yaml(path: Path | str) -> str:
    """Semantic hash of a YAML file (comments/whitespace/ordering ignored)."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return hash_obj(yaml.safe_load(handle) or {})


def hash_tree(root: Path | str, patterns: Iterable[str] = ("**/*",)) -> dict[str, str]:
    """Map of ``relative path -> sha256`` for every file matching ``patterns``."""
    base = Path(root)
    out: dict[str, str] = {}
    if not base.exists():
        return out
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(base.glob(pattern)):
            if path.is_file() and path not in seen:
                seen.add(path)
                out[str(path.relative_to(base))] = sha256_file(path)
    return out


def hash_directory(root: Path | str, exclude: Iterable[str] = ()) -> str:
    """Single stable hash for a directory's contents (sorted paths, exclusions)."""
    excluded = set(exclude)
    entries = {
        rel: digest
        for rel, digest in hash_tree(root).items()
        if not any(rel.startswith(ex) for ex in excluded)
    }
    return hash_obj(entries)


def iter_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    """Yield records from a JSONL file, reporting the offending line on error."""
    from .errors import DatasetError

    with Path(path).open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(
                    f"Malformed JSON in {Path(path).name} at line {lineno}: {exc.msg}",
                    hint="Fix the raw file, or run `lfm-finetune validate --repair-jsonl` "
                    "to quarantine unparseable lines instead of crashing.",
                ) from exc