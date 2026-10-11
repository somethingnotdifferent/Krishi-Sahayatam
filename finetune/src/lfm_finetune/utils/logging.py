"""Structured logging + per-run log files."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Final

_LOG_FORMAT: Final = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"
_DATE_FORMAT: Final = "%Y-%m-%dT%H:%M:%S%z"

_CONFIGURED = False


class _ColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\1;41m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        if os.environ.get("NO_COLOR"):
            return text
        color = self.COLORS.get(record.levelname)
        return f"{color}{text}{self.RESET}" if color else text


def setup_logging(
    level: str = "INFO",
    *,
    log_file: Path | str | None = None,
    console: bool = True,
    force: bool = False,
) -> None:
    """Configure the root logger exactly once.

    Args:
        level: ``DEBUG`` / ``INFO`` / ``WARNING`` / ``ERROR``.
        log_file: When given, a DEBUG-level copy is also written to this file.
        force: Re-configure even if already configured (used by the CLI).
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    if console:
        stream = logging.StreamHandler(sys.stderr)
        stream.setLevel(getattr(logging, level.upper(), logging.INFO))
        stream.setFormatter(_ColorFormatter(_LOG_FORMAT, _DATE_FORMAT))
        root.addHandler(stream)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        root.addHandler(file_handler)

    # Third-party noise reduction. Elevate any of these with LOG_DEBUG_LIBS=1.
    noisy = os.environ.get("LOG_DEBUG_LIBS", "0") == "1"
    for name in ("httpx", "httpcore", "urllib3", "filelock", "datasets", "transformers.modeling_utils"):
        logging.getLogger(name).setLevel(logging.DEBUG if noisy else logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger. ``__name__``-style dotted names are welcome."""
    if not name.startswith("lfm_finetune"):
        name = f"lfm_finetune.{name}"
    return logging.getLogger(name)