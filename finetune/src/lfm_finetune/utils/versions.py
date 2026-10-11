"""Environment/dependency version capture for run manifests."""

from __future__ import annotations

import importlib
import importlib.metadata as md
import platform
import sys
from typing import Any

#: Packages whose exact versions materially change numerics.
TRACKED = (
    "torch",
    "transformers",
    "trl",
    "peft",
    "accelerate",
    "datasets",
    "safetensors",
    "tokenizers",
    "numpy",
    "huggingface-hub",
    "pydantic",
    "PyYAML",
    "bitsandbytes",  # optional; absent is fine
)


def _version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        try:
            mod = importlib.import_module(name.replace("-", "_"))
            return getattr(mod, "__version__", "unknown")
        except Exception:
            return None


def package_versions() -> dict[str, str | None]:
    """Exact versions of every package that can affect results."""
    versions = {name: _version(name) for name in TRACKED}
    extras = {}
    for name in ("bitsandbytes", "flash_attn", "wandb", "deepspeed"):
        module = name.replace("-", "_")
        try:
            mod = importlib.import_module(module)
            extras[name] = getattr(mod, "__version__", "present")
        except Exception:
            extras[name] = None
    versions.update(extras)
    return versions


def runtime_info() -> dict[str, Any]:
    """Python/OS/runtime facts for the manifest."""
    return {
        "python_version": sys.version.split()[0],
        "python_full": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "prefix": sys.prefix,
    }


def git_state(repo_root: str | None = None) -> dict[str, Any]:
    """Current commit, branch and dirty-flag. Never raises."""
    import subprocess
    from pathlib import Path

    root = Path(repo_root) if repo_root else Path.cwd()

    def _run(args: list[str]) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=str(root), capture_output=True, text=True, timeout=10, check=False
            )
        except (subprocess.TimeoutExpired, OSError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    status = _run(["status", "--porcelain"])
    return {
        "commit": _run(["rev-parse", "HEAD"]),
        "branch": _run(["rev-parse", "--abbrev-ref", "HEAD"]),
        "remote": _run(["config", "--get", "remote.origin.url"]),
        "dirty": bool(status) if status is not None else None,
        "dirty_file_count": len(status.splitlines()) if status is not None else None,
    }


def environment_snapshot(repo_root: str | None = None, *, include_hardware: bool = False) -> dict[str, Any]:
    """Full reproducibility snapshot written into every run directory."""
    snapshot: dict[str, Any] = {
        "runtime": runtime_info(),
        "packages": package_versions(),
        "git": git_state(repo_root),
    }
    if include_hardware:
        from .hardware import audit

        snapshot["hardware"] = audit().to_dict()
    return snapshot