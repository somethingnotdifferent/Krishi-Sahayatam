"""Unified CLI package. See :mod:`lfm_finetune.cli.main` for the implementation."""

from .main import build_parser
from .main import main as cli_main

__all__ = ["build_parser", "cli_main"]