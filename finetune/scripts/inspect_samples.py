#!/usr/bin/env python3
"""Stage 3: show the chat template, formatted examples, and exactly which tokens carry loss.

Equivalent to:  uv run lfm-finetune inspect-template
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import load_dotenv_if_present  # noqa: E402

load_dotenv_if_present()

from lfm_finetune.cli.main import main  # noqa: E402

if __name__ == "__main__":
    argv = sys.argv[1:]
    # Allow `python scripts/<name>.py -c cfg.yaml` to omit the subcommand.
    if not argv or argv[0].startswith("-"):
        argv = ["inspect-template", *argv]
    sys.exit(main(argv))
