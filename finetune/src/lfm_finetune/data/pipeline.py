"""End-to-end dataset preparation pipeline.

Order of operations (this order is load-bearing):

1. read raw records
2. convert to canonical :class:`Conversation` (format sniffing)
3. text normalisation
4. structural + quality validation (record every rejection)
5. **global** exact deduplication
6. **global** near-duplicate detection
7. deterministic split (content-hashed)
8. cross-split leakage audit (hard fail on exact duplicates)
9. token-level statistics + advisory warnings
10. write validated/, splits/, and a machine-readable report

Deduplication happens *before* splitting on purpose. Deduplicating after
splitting would still leave a duplicate pair straddling the boundary, which is
the single most common cause of a suspiciously good validation curve.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config.schema import ExperimentConfig
from ..utils.hashing import hash_records, short_hash
from ..utils.io import ensure_dir, write_json, write_jsonl
from ..utils.logging import get_logger
from .dedupe import (
    NearDuplicateIndex,
    answer_key,
    conversation_text,
    exact_key,
    find_cross_split_overlap,
    prompt_key,
    summarise_overlap,
)
from .io import convert_record, iter_raw_records, load_manifest, validate_manifest
from .schema import COUNT_KEYS, Conversation, Rejection
from .split import build_plan, split_conversations, subsample, verify_no_cross_split_duplicates
from .stats import compute_stats, dataset_warning, multi_split_stats
from .validate import QualityConfig, TextNormalizer, structural_reject_reason, validate_conversation

log = get_logger("data.pipeline")


@dataclass
class PreparationResult:
    """Everything `prepare` produced, for the CLI to report and the trainer to consume."""

    splits: dict[str, list[Conversation]] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    rejections: list[dict[str, Any]] = field(default_factory=list)
    rejection_summary: dict[str, int] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    dataset_hash: str = ""
    split_hash: dict[str, str] = field(default_factory=dict)
    report_path: Path | None = None
    splits_dir: Path | None = None
    manifest: dict[str, Any] = field(default_factory=dict)
    near_dedupe_stats: dict[str, int] = field(default_factory=dict)

    @property
    def num_train(self) -> int:
        return len(self.splits.get("train", []))


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def load_and_clean(
    config: ExperimentConfig,
    *,
    raw_path: Path | None = None,
    limit: int | None = None,
) -> tuple[list[Conversation], dict[str, int], list[dict[str, Any]], dict[str, int], dict[str, int], dict[str, int]]:
    """Stages 1-6: read, convert, normalise, validate, deduplicate.

    Returns:
        ``(conversations, counts, rejection_records, rejection_summary,
          seen_keys, dup_stats)``
    """
    source = raw_path or Path(config.dataset.path)  # type: ignore[arg-type]
    quality = QualityConfig.from_config(config)
    normalizer = TextNormalizer(
        normalize_whitespace=quality.normalize_whitespace,
        strip_control_characters=quality.strip_control_characters,
        strip_html=quality.strip_html,
    )

    counts: Counter[str] = Counter()
    rejections: list[dict[str, Any]] = []
    reason_counter: Counter[str] = Counter()
    accepted: list[Conversation] = []
    seen_exact: set[str] = set()
    seen_prompt: dict[str, Conversation] = {}
    answer_counter: Counter[str] = Counter()
    dup_stats: Counter[str] = Counter()
    near_index = NearDuplicateIndex(threshold=quality_near_threshold(config))

    log.info("Reading raw records from %s", source)
    for index, raw in iter_raw_records(source):
        counts["raw_rows"] += 1
        if limit is not None and counts["accepted"] + counts["rejected"] >= limit:
            log.warning("Reached --limit=%d; stopping ingestion", limit)
            break

        record_id = str(raw.get("id")) if raw.get("id") is not None else None
        try:
            conv = convert_record(
                raw,
                fmt=config.dataset.format,
                system_prompt=config.dataset.system_prompt,
            )
        except Exception as exc:
            counts["rejected"] += 1
            reason = structural_reject_reason(raw, exc)
            reason_counter[reason] += 1
            rejections.append(
                Rejection(index, reason, str(exc), str(record_id or source.name)).to_dict()
            )
            continue

        outcome = validate_conversation(
            conv, index=index, source=record_id or source.name,
            quality=quality, normalizer=normalizer,
        )
        if outcome.rejection is not None:
            counts["rejected"] += 1
            reason_counter[outcome.rejection.reason] += 1
            rejections.append(outcome.rejection.to_dict())
            continue
        conv = outcome.conversation
        assert conv is not None

        if outcome.modified:
            counts["modified"] += 1

        # -- exact duplicate ------------------------------------------- #
        ekey = exact_key(conv)
        if quality.deduplicate and ekey in seen_exact:
            counts["deduplicated_exact"] += 1
            dup_stats["exact"] += 1
            reason_counter["duplicate_exact"] += 1
            rejections.append(
                Rejection(index, "duplicate_exact", "byte-identical conversation already present",
                          record_id, conv.messages[-1].content).to_dict()
            )
            continue
        seen_exact.add(ekey)

        # -- shared prompt, different answer --------------------------- #
        # Kept (it may be a legitimately hard item) but surfaced for adjudication.
        pkey = prompt_key(conv)
        if pkey in seen_prompt:
            dup_stats["shared_prompt"] += 1
            conv.metadata.setdefault("_quality_flags", [])
            if "shared_prompt" not in conv.metadata["_quality_flags"]:
                conv.metadata["_quality_flags"].append("shared_prompt")
        else:
            seen_prompt[pkey] = conv

        # -- identical answer to many different prompts ----------------- #
        akey = answer_key(conv)
        answer_counter[akey] += 1
        if answer_counter[akey] > 3:
            conv.metadata.setdefault("_quality_flags", [])
            if "repeated_answer" not in conv.metadata["_quality_flags"]:
                conv.metadata["_quality_flags"].append("repeated_answer")

        # -- near duplicate -------------------------------------------- #
        if quality.near_duplicate:
            is_near, matched = near_index.query(conversation_text(conv))
            if is_near:
                counts["deduplicated_near"] += 1
                dup_stats["near"] += 1
                reason_counter["duplicate_near"] += 1
                rejections.append(
                    Rejection(index, "duplicate_near",
                              f"near-duplicate (Jaccard >= {near_index.threshold}) of: "
                              f"{(matched or '')[:120]}", record_id,
                              conv.messages[-1].content).to_dict()
                )
                continue

        if conv.id is None:
            conv.id = f"{source.stem}-{index:06d}"
        accepted.append(conv)
        counts["accepted"] += 1

    # A Counter omits keys it was never incremented for, so a perfectly clean
    # dataset would return no "rejected" key at all and every caller doing
    # counts["rejected"] would raise KeyError. Return a complete mapping.
    full_counts = {key: int(counts.get(key, 0)) for key in COUNT_KEYS}
    return (
        accepted,
        full_counts,
        rejections,
        dict(reason_counter),
        {k: v for k, v in dup_stats.items()},
        near_index.stats(),
    )


def quality_near_threshold(config: ExperimentConfig) -> float:
    return float(config.dataset.quality.near_duplicate_threshold)


def prepare_dataset(
    config: ExperimentConfig,
    *,
    output_root: Path | None = None,
    raw_path: Path | None = None,
    limit: int | None = None,
    tokenizer: Any = None,
    run_tokenizer_stats: bool = True,
) -> PreparationResult:
    """Run the full preparation pipeline and persist its outputs.

    Args:
        output_root: Where ``validated/`` and ``splits/`` are written. Defaults
            to the config's project layout.
        tokenizer: Used only for token statistics. Passing the real checkpoint
            tokenizer makes the reported token counts authoritative.

    Raises:
        DatasetLeakageError: on cross-split duplicates or an empty split.
    """
    root = Path(output_root) if output_root else Path.cwd()
    validated_dir = ensure_dir(root / "data" / "validated")
    splits_dir = ensure_dir(root / "data" / "splits")

    started = _now()
    (
        conversations,
        counts,
        rejections,
        rejection_summary,
        dup_stats,
        near_stats,
    ) = load_and_clean(config, raw_path=raw_path, limit=limit)

    if not conversations:
        raise __import__(
            "lfm_finetune.utils.errors", fromlist=["DatasetError"]
        ).DatasetError(
            f"All {counts['raw_rows']} raw rows were rejected; there is nothing to train on.",
            hint="Inspect data/validated/report.json -> rejections[] for the specific "
            "reasons, then fix the raw data or relax dataset.quality thresholds.",
        )

    plan = build_plan(
        train_ratio=config.dataset.splitting.train_ratio,
        validation_ratio=config.dataset.splitting.validation_ratio,
        seed=config.dataset.splitting.seed,
        strategy=config.dataset.splitting.strategy,
    )
    log.info("Splitting %d examples with strategy=%s seed=%d",
             len(conversations), plan.strategy, plan.seed)

    splits = split_conversations(conversations, plan, exact_key_fn=exact_key)
    leakage = verify_no_cross_split_duplicates(splits)

    soft_findings = find_cross_split_overlap(splits, keys=("prompt", "answer"))
    soft_summary = summarise_overlap(soft_findings)
    if soft_summary["has_leakage"]:
        log.warning(
            "Soft leakage across splits: %s. Shared *prompts* across splits inflate "
            "validation; shared short answers are usually benign.",
            soft_summary["counts_by_key_type"],
        )

    if config.dataset.max_train_examples:
        splits["train"] = subsample(
            splits["train"], config.dataset.max_train_examples,
            seed=config.training.seed,
        )

    # -- write outputs ---------------------------------------------------- #
    for name, convs in splits.items():
        write_jsonl(splits_dir / f"{name}.jsonl", [c.to_record() for c in convs])
    write_jsonl(validated_dir / "accepted.jsonl", [c.to_record() for c in conversations])
    write_jsonl(validated_dir / "rejected.jsonl", rejections)

    render = None
    if tokenizer is not None:
        def render(conv: Conversation) -> str:  # noqa: ANN202
            return tokenizer.apply_chat_template(
                [m.to_dict() for m in conv.messages], tokenize=False
            )

    stats = multi_split_stats(
        splits,
        tokenizer=tokenizer,
        max_seq_length=config.tokenization.max_seq_length,
        render=render,
        compute_tokens=run_tokenizer_stats and tokenizer is not None,
    )
    train_stats = compute_stats(
        splits["train"],
        tokenizer=tokenizer,
        max_seq_length=config.tokenization.max_seq_length,
        render=render,
        compute_tokens=run_tokenizer_stats and tokenizer is not None,
        label="train",
    )
    warnings = dataset_warning(
        train_stats,
        max_seq_length=config.tokenization.max_seq_length,
        min_examples=200,
    )
    if config.training.num_train_epochs > 1 and train_stats.num_examples < 200:
        warnings.append(
            f"{config.training.num_train_epochs} epochs over {train_stats.num_examples} "
            "training examples will memorise the set. Reduce epochs or collect more data."
        )
    for w in warnings:
        log.warning("%s", w)

    dataset_hash = hash_records([c.to_record() for c in conversations])
    split_hash = {name: hash_records([c.to_record() for c in convs]) for name, convs in splits.items()}

    # -- provenance manifest ---------------------------------------------- #
    # Accept either extension: synthetic fixtures write JSON, hand-authored
    # corpora usually prefer YAML. Search the dataset directory itself when
    # dataset.path points at a directory, otherwise the file's parent.
    manifest: dict[str, Any] = {}
    manifest_source: Path | None = None
    if raw_path:
        raw_p = Path(raw_path)
        search_dir = raw_p if raw_p.is_dir() else raw_p.parent
    elif config.dataset.path:
        cfg_p = Path(config.dataset.path)
        search_dir = cfg_p if cfg_p.is_dir() else cfg_p.parent
    else:
        search_dir = root / "data" / "raw"
    for candidate_name in ("manifest.yaml", "manifest.yml", "manifest.json"):
        candidate = search_dir / candidate_name
        if candidate.exists():
            manifest = load_manifest(candidate)
            manifest_source = candidate
            break
    if manifest_source is None:
        log.warning(
            "No dataset manifest found in %s. Provenance (source, license, creation "
            "date) cannot be recorded, and this run's manifest will be incomplete.",
            search_dir,
        )
        manifest_path = search_dir / "manifest.yaml"
    else:
        manifest_path = manifest_source
        log.info("Loaded dataset manifest %s", manifest_source)
    manifest.setdefault("name", config.run.domain or "dataset")
    manifest.setdefault("version", "unversioned")
    manifest.setdefault("source", str(raw_path or config.dataset.path))
    manifest.setdefault("license", "unknown")
    manifest.setdefault("created", started)
    manifest.setdefault("annotation_method", "unknown")
    manifest["num_examples"] = len(conversations)
    manifest["content_hash"] = dataset_hash
    manifest["prepared_at"] = started
    manifest["manifest_path"] = str(manifest_path)
    manifest_problems = validate_manifest(manifest, source_file=manifest_path)
    for problem in manifest_problems:
        log.warning("PROVENANCE: %s", problem)

    counts_out = {key: int(counts.get(key, 0)) for key in COUNT_KEYS}
    for name, convs in splits.items():
        counts_out[f"split_{name}"] = len(convs)

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at": started,
        "config_fingerprint": config.fingerprint(),
        "source": str(raw_path or config.dataset.path),
        "counts": counts_out,
        "rejection_summary": dict(sorted(rejection_summary.items(), key=lambda kv: -kv[1])),
        "duplicate_summary": dup_stats,
        "near_dedupe_index": near_stats,
        "split_plan": plan.to_dict(),
        "leakage": {"hard": leakage, "soft": soft_summary},
        "statistics": stats,
        "warnings": warnings,
        "dataset_hash": dataset_hash,
        "dataset_hash_short": short_hash(dataset_hash),
        "split_hashes": split_hash,
        "manifest": manifest,
        "provenance_problems": manifest_problems,
        "provenance_ok": not manifest_problems,
        "outputs": {
            "splits_dir": str(splits_dir),
            "accepted": str(validated_dir / "accepted.jsonl"),
            "rejected": str(validated_dir / "rejected.jsonl"),
            "report": str(validated_dir / "report.json"),
        },
        "quality_config": config.dataset.quality.model_dump(mode="json"),
    }
    report_path = write_json(validated_dir / "report.json", report)

    log.info(
        "Prepared dataset: %d accepted / %d rejected / %d deduped -> "
        "train=%d val=%d test=%d (hash %s)",
        counts_out["accepted"], counts_out["rejected"],
        counts_out["deduplicated_exact"] + counts_out["deduplicated_near"],
        counts_out["split_train"], counts_out["split_validation"], counts_out["split_test"],
        short_hash(dataset_hash),
    )

    return PreparationResult(
        splits=splits,
        counts=counts_out,
        rejections=rejections,
        rejection_summary=report["rejection_summary"],
        stats=stats,
        warnings=warnings,
        dataset_hash=dataset_hash,
        split_hash=split_hash,
        report_path=report_path,
        splits_dir=splits_dir,
        manifest=manifest,
        near_dedupe_stats=near_stats,
    )


def load_splits(splits_dir: Path) -> dict[str, list[Conversation]]:
    """Read previously-prepared splits back into :class:`Conversation` objects."""
    from .io import iter_raw_records

    splits_dir = Path(splits_dir)
    out: dict[str, list[Conversation]] = {}
    for name in ("train", "validation", "test"):
        path = splits_dir / f"{name}.jsonl"
        if not path.exists():
            continue
        conversations: list[Conversation] = []
        for _, raw in iter_raw_records(path):
            conv = convert_record(raw, fmt="conversational")
            if conv.id is None:
                conv.id = raw.get("id")
            conversations.append(conv)
        out[name] = conversations
    return out


def quality_flag_summary(splits: dict[str, list[Conversation]]) -> dict[str, int]:
    """Aggregate the ``_quality_flags`` written during preparation."""
    counter: Counter[str] = Counter()
    for convs in splits.values():
        for conv in convs:
            for flag in conv.metadata.get("_quality_flags", []) or []:
                counter[flag] += 1
    return dict(counter)