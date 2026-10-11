"""Evaluation runner: base vs fine-tuned comparison with loss and generation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..config.schema import ExperimentConfig
from ..data.schema import Conversation
from ..inference.generator import (
    GenerationConfig,
    GenerationRun,
    Generator,
    run_generation_sweep,
)
from ..training.tokenization import ChatFormatter
from ..training.trainer import tokenize_conversations  # noqa: F401  (re-export convenience)
from ..utils.io import ensure_dir, write_json
from ..utils.logging import get_logger
from ..utils.seeding import set_seed
from .metrics import (
    assert_eval_disjoint,
    build_eval_set,
    build_rubric_sheet,
    compare_summaries,
    hallucination_flags,
    load_rubric,
    score_generation,
    summarise,
    token_f1,
)

log = get_logger("evaluation.runner")


def evaluate_model(
    config: ExperimentConfig,
    model: Any,
    tokenizer: Any,
    *,
    eval_conversations: list[Conversation],
    label: str,
    device: str = "cpu",
    gen_config: GenerationConfig | None = None,
) -> tuple[GenerationRun, dict[str, Any], float | None]:
    """Evaluate one model variant: held-out loss + generated outputs + proxies.

    Returns:
        ``(generation_run, objective_summary, eval_loss_or_None)``
    """
    from ..model.loader import LoadedModel  # noqa: F401
    from ..training.collator import tokenize_conversations as _tokenize

    gen_config = gen_config or GenerationConfig.from_experiment_config(config)
    formatter = ChatFormatter(tokenizer, config.tokenization)

    # -- held-out loss ---------------------------------------------------- #
    eval_loss: float | None = None
    if eval_conversations:
        tokens = _tokenize(
            eval_conversations, formatter,
            assistant_only_loss=config.training.assistant_only_loss, split=f"{label}/eval",
        )
        eval_loss = _compute_eval_loss(model, tokens, formatter, device=device)

    # -- generation ------------------------------------------------------- #
    eval_items = build_eval_set(eval_conversations, limit=config.evaluation.num_samples)
    generator = Generator(model, tokenizer, formatter, device=device)
    run = run_generation_sweep(generator, eval_items, gen_config, label=label)

    for sample in run.samples:
        sample.update(
            score_generation(
                sample.get("reference", ""),
                sample.get("generation", ""),
                expected_metadata=sample.get("metadata"),
            )
        )
        sample["token_f1"] = round(
            token_f1(sample.get("generation", ""), sample.get("reference", "")), 4
        )

    summary = summarise(run.samples, label=label)
    summary["eval_loss"] = round(eval_loss, 6) if eval_loss is not None else None
    return run, summary, eval_loss


def _compute_eval_loss(
    model: Any, tokens: Any, formatter: ChatFormatter, *, device: str
) -> float | None:
    """Mean causal-LM loss over the held-out split, assistant tokens only."""
    import torch

    from ..training.collator import SFTCollator

    if not len(tokens):
        return None
    collator = SFTCollator(pad_token_id=formatter.pad_token_id)
    model.eval()
    total_loss = 0.0
    total_batches = 0
    batch_size = 8
    with torch.no_grad():
        for start in range(0, len(tokens), batch_size):
            batch = collator(tokens.examples[start : start + batch_size])
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            if outputs.loss is not None and torch.isfinite(outputs.loss):
                total_loss += float(outputs.loss)
                total_batches += 1
    if total_batches == 0:
        log.warning("Eval loss produced no finite values (all batches NaN/inf).")
        return None
    return total_loss / total_batches


def compare_variants(
    config: ExperimentConfig,
    *,
    base_run: GenerationRun | None,
    tuned_run: GenerationRun | None,
    base_summary: dict[str, Any] | None,
    tuned_summary: dict[str, Any] | None,
    eval_conversations: list[Conversation],
    train_conversations: list[Conversation],
    output_dir: Path,
) -> Path:
    """Write the before/after comparison report. Never overwrites silently."""
    ensure_dir(output_dir)
    rubric = load_rubric(config.evaluation.rubric_path)

    comparison: dict[str, Any] | None = None
    if base_summary and tuned_summary:
        comparison = compare_summaries(base_summary, tuned_summary)

    disjoint = assert_eval_disjoint(build_eval_set(eval_conversations), train_conversations)
    if not disjoint["disjoint"]:
        log.error(
            "EVALUATION LEAKAGE: %d evaluation prompts also appear in the training split "
            "(ids: %s). The comparison below would be invalid.",
            disjoint["overlap_count"], disjoint["overlap_ids"][:5],
        )

    loss_block: dict[str, Any] = {"note": "Lower is better."}
    for label, summary in (("base", base_summary), ("tuned", tuned_summary)):
        if summary and summary.get("eval_loss") is not None:
            loss_block[label] = summary["eval_loss"]
    if "base" in loss_block and "tuned" in loss_block:
        delta = loss_block["tuned"] - loss_block["base"]
        loss_block["delta"] = round(delta, 6)
        loss_block["relative_change_pct"] = round(
            100 * delta / loss_block["base"], 3
        ) if loss_block["base"] else None
        loss_block["improved"] = delta < 0
        loss_block["caution"] = (
            "Lower held-out loss indicates a better fit to this domain's text "
            "distribution. It does NOT by itself demonstrate improved agricultural "
            "correctness, instruction following or safety behaviour."
        )

    before_after: list[dict[str, Any]] = []
    if base_run and tuned_run:
        tuned_by_id = {s.get("id"): s for s in tuned_run.samples}
        for base_sample in base_run.samples:
            tuned_sample = tuned_by_id.get(base_sample.get("id"))
            if not tuned_sample:
                continue
            before_after.append({
                "id": base_sample.get("id"),
                "prompt": base_sample.get("prompt"),
                "reference": base_sample.get("reference"),
                "base": base_sample.get("generation"),
                "tuned": tuned_sample.get("generation"),
                "base_overconfidence": base_sample.get("overconfidence_markers"),
                "tuned_overconfidence": tuned_sample.get("overconfidence_markers"),
                "base_asks_for_info": base_sample.get("asks_for_more_information"),
                "tuned_asks_for_info": tuned_sample.get("asks_for_more_information"),
            })

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_name": config.run.name,
        "model": {"id": config.model.name, "revision": config.model.revision},
        "generation": (base_run or tuned_run).generation_config if (base_run or tuned_run) else {},
        "evaluation_set": {
            "num_items": len(eval_conversations),
            "disjointness_vs_train": disjoint,
        },
        "held_out_loss": loss_block,
        "objective_proxies": {
            "base": base_summary,
            "tuned": tuned_summary,
            "comparison": comparison,
        },
        "hallucination_flags": {
            "base": hallucination_flags(base_run.samples) if base_run else [],
            "tuned": hallucination_flags(tuned_run.samples) if tuned_run else [],
        },
        "before_after_examples": before_after[:50],
        "regressions": _detect_regressions(base_run, tuned_run),
        "human_rubric_sheet": build_rubric_sheet(
            (tuned_run or base_run).samples if (tuned_run or base_run) else [], rubric
        ),
        "interpretation_guard": (
            "This report compares objective proxies only. Agronomic correctness, safety and "
            "helpfulness require a human to score human_rubric_sheet. A decrease in "
            "held-out loss alone is NOT evidence of improved domain capability."
        ),
    }

    path = write_json(output_dir / "comparison.json", report)
    _write_markdown(report, output_dir / "comparison.md")
    log.info("Comparison report written to %s", path)
    return path


def _detect_regressions(
    base_run: GenerationRun | None, tuned_run: GenerationRun | None
) -> list[dict[str, Any]]:
    """Flag samples where the tuned model objectively got worse."""
    if not base_run or not tuned_run:
        return []
    tuned_by_id = {s.get("id"): s for s in tuned_run.samples}
    regressions: list[dict[str, Any]] = []
    for base in base_run.samples:
        tuned = tuned_by_id.get(base.get("id"))
        if not tuned:
            continue
        reasons = []
        if base.get("token_f1", 0) - tuned.get("token_f1", 0) > 0.25:
            reasons.append(
                f"lexical overlap dropped {base.get('token_f1')} -> {tuned.get('token_f1')}"
            )
        if base.get("asks_for_more_information") and not tuned.get("asks_for_more_information"):
            reasons.append("stopped asking for missing information")
        if tuned.get("overconfidence_markers", 0) > base.get("overconfidence_markers", 0):
            reasons.append("increased absolute-certainty phrasing")
        if base.get("format_tags") and not tuned.get("format_tags"):
            reasons.append("lost requested output structure")
        if tuned.get("is_empty") and not base.get("is_empty"):
            reasons.append("produced an empty response where the base model did not")
        if reasons:
            regressions.append({
                "id": base.get("id"),
                "prompt": base.get("prompt"),
                "base": base.get("generation"),
                "tuned": tuned.get("generation"),
                "reasons": reasons,
            })
    return regressions


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    """Human-readable companion to comparison.json."""
    lines: list[str] = [
        f"# Evaluation comparison -- {report['run_name']}",
        "",
        f"Model: `{report['model']['id']}` @ `{report['model']['revision']}`",
        "",
        "## Interpretation guard",
        "",
        report["interpretation_guard"],
        "",
        "## Held-out loss",
        "",
        "| variant | eval loss |",
        "|---|---|",
    ]
    loss = report["held_out_loss"]
    for key in ("base", "tuned"):
        if key in loss:
            lines.append(f"| {key} | {loss[key]:.6f} |")
    if "delta" in loss:
        verdict = "improved" if loss.get("improved") else "regressed"
        lines.append(f"| delta | {loss['delta']:+.6f} ({verdict}) |")

    lines += ["", "## Objective proxies", "", "| metric | base | tuned | delta | verdict |", "|---|---|---|---|---|"]
    comparison = report["objective_proxies"].get("comparison")
    if comparison:
        for row in comparison["rows"]:
            lines.append(
                f"| {row['metric']} | {row['base']} | {row['tuned']} | {row['delta']:+.4f} | {row['verdict']} |"
            )
    else:
        lines.append("| (no comparison available -- run baseline and tuned eval) | | | | |")

    lines += ["", "## Before / after examples", ""]
    for ex in report["before_after_examples"][:5]:
        lines += [
            f"### `{ex['id']}`",
            "",
            f"**Prompt**: {ex['prompt']}",
            "",
            f"**Reference**: {ex['reference']}",
            "",
            f"**Base**: {ex['base']}",
            "",
            f"**Tuned**: {ex['tuned']}",
            "",
        ]

    if report["regressions"]:
        lines += ["## Regressions", ""]
        for reg in report["regressions"][:10]:
            lines.append(f"- `{reg['id']}`: {'; '.join(reg['regressions'])}"
                         if "regressions" in reg else f"- `{reg['id']}`: {'; '.join(reg['reasons'])}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Markdown summary written to %s", path)


def run_baseline_evaluation(
    config: ExperimentConfig,
    *,
    output_dir: Path | None = None,
    split: str = "test",
) -> dict[str, Any]:
    """Evaluate the **base** checkpoint on the held-out split, before training."""
    from ..data.pipeline import load_splits
    from ..model.loader import load_model

    out = Path(output_dir or config.checkpointing.dir).parent / "evaluations" / f"{config.run.name}_baseline"
    ensure_dir(out)

    splits = load_splits(Path(config.dataset.processed_dir or Path("data") / "splits"))
    eval_conversations = splits.get(split, [])
    if not eval_conversations:
        from ..utils.errors import EvaluationError

        raise EvaluationError(
            f"No '{split}' split available for baseline evaluation.",
            hint="Run `lfm-finetune prepare --config ...`; the test split is created only if "
            "train_ratio + validation_ratio < 1.0.",
        )

    log.info("Loading base model %s for baseline evaluation", config.model.name)
    loaded = load_model(config, for_training=False)
    set_seed(config.evaluation.seed)

    run, summary, eval_loss = evaluate_model(
        config, loaded.model, loaded.tokenizer,
        eval_conversations=eval_conversations,
        label="base",
        device=loaded.device,
    )

    result = {
        "schema_version": "1.0",
        "label": "base",
        "model": {"id": loaded.model_id, "revision": loaded.revision},
        "split": split,
        "summary": summary,
        "generations": run.to_dict(),
    }
    path = write_json(out / "baseline.json", result)
    log.info("Baseline evaluation saved to %s", path)
    log.info("Baseline summary: %s", _summary_line(summary))
    return result


def _summary_line(summary: dict[str, Any]) -> str:
    parts = [f"{k}={v}" for k, v in summary.items() if isinstance(v, int | float)]
    return ", ".join(parts)


def run_comparison_evaluation(
    config: ExperimentConfig,
    adapter_path: Path | str,
    *,
    output_dir: Path | None = None,
    split: str = "test",
) -> dict[str, Any]:
    """Evaluate base + tuned on the same held-out split and write the comparison."""
    from ..data.pipeline import load_splits
    from ..model.loader import load_model

    out = Path(output_dir or Path("evaluations") / config.run.name)
    ensure_dir(out)

    splits = load_splits(Path(config.dataset.processed_dir or Path("data") / "splits"))
    eval_conversations = splits.get(split, [])
    if not eval_conversations:
        from ..utils.errors import EvaluationError

        raise EvaluationError(
            f"No '{split}' split to evaluate against.",
            hint="Create the test split with `lfm-finetune prepare` (leave room in the ratios).",
        )

    gen_config = GenerationConfig.from_experiment_config(config)
    log.info("Decoding strategy: %s", gen_config.describe())

    # -- base ------------------------------------------------------------- #
    log.info("Evaluating BASE model %s", config.model.name)
    base_loaded = load_model(config, for_training=False)
    set_seed(config.evaluation.seed)
    base_run, base_summary, _ = evaluate_model(
        config, base_loaded.model, base_loaded.tokenizer,
        eval_conversations=eval_conversations, label="base", device=base_loaded.device,
        gen_config=gen_config,
    )
    del base_loaded
    _free_memory()

    # -- tuned ------------------------------------------------------------ #
    log.info("Evaluating TUNED model (adapter %s)", adapter_path)
    tuned_loaded = load_model(config, for_training=False)
    from peft import PeftModel

    try:
        tuned_model = PeftModel.from_pretrained(tuned_loaded.model, str(adapter_path))
    except Exception as exc:
        from ..utils.errors import CheckpointError

        raise CheckpointError(
            f"Could not load adapter {adapter_path}: {exc}",
            hint="Confirm the adapter was trained for this base model id/revision.",
        ) from exc
    tuned_loaded.model = tuned_model
    set_seed(config.evaluation.seed)
    tuned_run, tuned_summary, _ = evaluate_model(
        config, tuned_model, tuned_loaded.tokenizer,
        eval_conversations=eval_conversations, label="tuned", device=tuned_loaded.device,
        gen_config=gen_config,
    )

    write_json(out / "base_generations.json", base_run.to_dict())
    write_json(out / "tuned_generations.json", tuned_run.to_dict())

    path = compare_variants(
        config,
        base_run=base_run, tuned_run=tuned_run,
        base_summary=base_summary, tuned_summary=tuned_summary,
        eval_conversations=eval_conversations,
        train_conversations=splits.get("train", []),
        output_dir=out,
    )
    return {"comparison_path": str(path), "base": base_summary, "tuned": tuned_summary}


def _free_memory() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()