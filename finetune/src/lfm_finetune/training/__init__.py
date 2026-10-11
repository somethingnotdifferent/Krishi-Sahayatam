"""Training: tokenization with assistant-only masking, SFTTrainer wiring, metadata."""

from .args import build_sft_config, describe_training_plan
from .collator import SFTCollator, TokenizedDataset, tokenize_conversations
from .metadata import (
    METADATA_FILENAME,
    METRICS_FILENAME,
    RunDirectory,
    local_metric_summary,
    resolve_reporting,
    utc_now,
)
from .tokenization import (
    IGNORE_INDEX,
    ChatFormatter,
    FormattedExample,
    format_example_for_display,
    template_fingerprint,
    verify_template_consistency,
)
from .trainer import TrainingResult, preflight, train

__all__ = [
    "IGNORE_INDEX",
    "METADATA_FILENAME",
    "METRICS_FILENAME",
    "ChatFormatter",
    "FormattedExample",
    "RunDirectory",
    "SFTCollator",
    "TokenizedDataset",
    "TrainingResult",
    "build_sft_config",
    "describe_training_plan",
    "format_example_for_display",
    "local_metric_summary",
    "preflight",
    "resolve_reporting",
    "template_fingerprint",
    "tokenize_conversations",
    "train",
    "utc_now",
    "verify_template_consistency",
]