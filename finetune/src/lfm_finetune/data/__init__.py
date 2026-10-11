"""Dataset ingestion, validation, cleaning, splitting and statistics."""

from .io import build_manifest, convert_record, iter_raw_records, load_manifest, validate_manifest
from .pipeline import PreparationResult, load_splits, prepare_dataset, quality_flag_summary
from .schema import COUNT_KEYS, Conversation, DatasetStats, Message, Rejection
from .split import build_plan, split_conversations, subsample, verify_no_cross_split_duplicates
from .stats import compute_stats, dataset_warning, multi_split_stats
from .synthetic import generate_synthetic, placeholder_agriculture_records, write_synthetic_dataset
from .validate import QualityConfig, TextNormalizer, validate_conversation

__all__ = [
    "COUNT_KEYS",
    "Conversation",
    "DatasetStats",
    "Message",
    "PreparationResult",
    "QualityConfig",
    "Rejection",
    "TextNormalizer",
    "build_manifest",
    "build_plan",
    "compute_stats",
    "convert_record",
    "dataset_warning",
    "generate_synthetic",
    "iter_raw_records",
    "load_manifest",
    "load_splits",
    "multi_split_stats",
    "placeholder_agriculture_records",
    "prepare_dataset",
    "quality_flag_summary",
    "split_conversations",
    "subsample",
    "validate_conversation",
    "validate_manifest",
    "verify_no_cross_split_duplicates",
    "write_synthetic_dataset",
]