"""Evaluation: objective proxies, rubric sheets, base-vs-tuned comparison."""

from .metrics import (
    DEFAULT_AGRICULTURE_RUBRIC,
    RubricDimension,
    build_eval_set,
    build_rubric_sheet,
    compare_summaries,
    hallucination_flags,
    load_rubric,
    score_generation,
    summarise,
    token_f1,
)
from .runner import (
    compare_variants,
    evaluate_model,
    run_baseline_evaluation,
    run_comparison_evaluation,
)

__all__ = [
    "DEFAULT_AGRICULTURE_RUBRIC",
    "RubricDimension",
    "build_eval_set",
    "build_rubric_sheet",
    "compare_summaries",
    "compare_variants",
    "evaluate_model",
    "hallucination_flags",
    "load_rubric",
    "run_baseline_evaluation",
    "run_comparison_evaluation",
    "score_generation",
    "summarise",
    "token_f1",
]