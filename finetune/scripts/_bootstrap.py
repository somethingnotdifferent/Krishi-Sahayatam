"""Shared bootstrap for the standalone ``scripts/*.py`` entry points.

These scripts exist so the pipeline is runnable *without* installing the
package (``python scripts/train.py ...``). The canonical interface is the
installed console script ``lfm-finetune``; every wrapper here delegates to the
same implementation, so there is exactly one code path and no drift.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def load_dotenv_if_present() -> None:
    """Load the repo-root .env if python-dotenv is available. Never fails."""
    root = Path(__file__).resolve().parent.parent
    env_file = root / ".env"
    if not env_file.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_file, override=False)
    except ImportError:
        pass
