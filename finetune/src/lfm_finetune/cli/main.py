"""Unified command-line interface.

::

    lfm-finetune <command> [options]

Commands follow the staged workflow in docs/ARCHITECTURE.md, and every stage can
be run independently -- that is the point of the design. A failed run should
never require re-running earlier stages.

    stage 0  env / hw      doctor, inspect-hardware
    stage 1  model         inspect-model, inspect-samples
    stage 2  dataset       validate, prepare, split-stats
    stage 3  tokenizer     inspect-template
    stage 4  lora          inspect-lora
    stage 5  smoke         train --smoke-test
    stage 6  pilot         train --preset pilot
    stage 7  full          train
    stage 8  evaluation    baseline, evaluate
    stage 9  export        export-adapter, merge-adapter
             serving       inference
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ..utils.errors import ConfigError, PipelineError
from ..utils.io import ensure_dir, write_json
from ..utils.logging import get_logger, setup_logging
from ..utils.seeding import ensure_deterministic_cuda_algos, set_seed

log = get_logger("cli")

BANNER = "lfm-finetune -- LoRA/PEFT SFT for Liquid AI LFM2 / LFM2.5"


# --------------------------------------------------------------------------- #
# Argument plumbing
# --------------------------------------------------------------------------- #

def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", "-c", required=True, help="Path to the experiment YAML.")
    parser.add_argument(
        "--set", "-s", action="append", default=[], metavar="KEY.PATH=VALUE",
        help="Override any config value, e.g. -s training.learning_rate=1e-4",
    )


def _add_common_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", "-o", default=None, help="Override the output directory.")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="Print the machine-readable result to stdout.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lfm-finetune",
        description=BANNER,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--version", action="store_true", help="Print version information.")
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging.")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # -- stage 0 ---------------------------------------------------------- #
    doctor = sub.add_parser("doctor", help="Verify the environment: imports, versions, devices.")
    doctor.set_defaults(func=cmd_doctor)

    hardware = sub.add_parser("inspect-hardware", help="Full hardware audit (GPU/VRAM/CUDA/RAM/disk).")
    _add_common_output(hardware)
    hardware.add_argument("--recommend", action="store_true",
                          help="Also rank LFM checkpoints for this host.")
    hardware.set_defaults(func=cmd_inspect_hardware)

    # -- stage 1 ---------------------------------------------------------- #
    inspect_model = sub.add_parser(
        "inspect-model", help="Report architecture, parameter counts and PEFT-relevant modules.")
    _add_config_args(inspect_model)
    _add_common_output(inspect_model)
    inspect_model.add_argument("--discover-lora", action="store_true", default=True,
                               help="Also derive the LoRA target set (default).")
    inspect_model.add_argument("--no-discover-lora", dest="discover_lora", action="store_false")
    inspect_model.set_defaults(func=cmd_inspect_model)

    # -- stage 2 ---------------------------------------------------------- #
    validate = sub.add_parser("validate", help="Validate a dataset against the canonical schema.")
    _add_config_args(validate)
    _add_common_output(validate)
    validate.add_argument("--path", default=None, help="Override dataset.path.")
    validate.add_argument("--limit", type=int, default=None, help="Only check the first N rows.")
    validate.add_argument("--fail-on-reject", action="store_true",
                          help="Exit non-zero if any row is rejected.")
    validate.set_defaults(func=cmd_validate)

    prepare = sub.add_parser("prepare", help="Clean, deduplicate, split and write the dataset.")
    _add_config_args(prepare)
    _add_common_output(prepare)
    prepare.add_argument("--path", default=None, help="Override dataset.path.")
    prepare.add_argument("--limit", type=int, default=None, help="Ingest at most N raw rows.")
    prepare.add_argument("--no-token-stats", action="store_true",
                         help="Skip token-level statistics (no tokenizer needed).")
    prepare.set_defaults(func=cmd_prepare)

    stats = sub.add_parser("split-stats", help="Print statistics for the prepared splits.")
    _add_config_args(stats)
    _add_common_output(stats)
    stats.set_defaults(func=cmd_split_stats)

    # -- stage 3 ---------------------------------------------------------- #
    template = sub.add_parser("inspect-template",
                             help="Show the chat template and token-level loss masking.")
    _add_config_args(template)
    _add_common_output(template)
    template.add_argument("--samples", type=int, default=3)
    template.add_argument("--check-consistency", action="store_true", default=True)
    template.set_defaults(func=cmd_inspect_template)

    # -- stage 4 ---------------------------------------------------------- #
    lora = sub.add_parser("inspect-lora", help="Attach LoRA and audit trainable parameters.")
    _add_config_args(lora)
    _add_common_output(lora)
    lora.add_argument("--skip-vram", action="store_true")
    lora.set_defaults(func=cmd_inspect_lora)

    # -- stage 5-7 -------------------------------------------------------- #
    train = sub.add_parser("train", help="Run LoRA SFT (or resume from a checkpoint).")
    _add_config_args(train)
    _add_common_output(train)
    train.add_argument("--smoke-test", action="store_true",
                       help="Tiny end-to-end run that validates the whole pipeline.")
    train.add_argument("--pilot", action="store_true",
                       help="Short pilot run on a subset (stage 6).")
    train.add_argument("--resume", default=None,
                       help="Checkpoint dir to resume from, or 'auto' for the latest.")
    train.add_argument("--dry-run", action="store_true",
                       help="Preflight + LoRA audit + sample inspection, but no training.")
    train.add_argument("--limit-steps", type=int, default=None, help="Override training.max_steps.")
    train.set_defaults(func=cmd_train)

    # -- stage 8 ---------------------------------------------------------- #
    baseline = sub.add_parser("baseline", help="Evaluate the base model on the held-out split.")
    _add_config_args(baseline)
    _add_common_output(baseline)
    baseline.add_argument("--split", default="test", choices=["train", "validation", "test"])
    baseline.add_argument("--num-samples", type=int, default=None)
    baseline.set_defaults(func=cmd_baseline)

    evaluate = sub.add_parser("evaluate", help="Compare base vs fine-tuned on the held-out split.")
    _add_config_args(evaluate)
    _add_common_output(evaluate)
    evaluate.add_argument("--adapter", required=False, default=None, help="Adapter directory.")
    evaluate.add_argument("--split", default="test", choices=["train", "validation", "test"])
    evaluate.add_argument("--run-dir", default=None,
                          help="Run directory to read the adapter from (default: newest).")
    evaluate.set_defaults(func=cmd_evaluate)

    # -- stage 9 ---------------------------------------------------------- #
    inference = sub.add_parser("inference", help="Chat with base+adapter or a merged model.")
    _add_config_args(inference)
    _add_common_output(inference)
    inference.add_argument("--adapter", default=None, help="Adapter directory.")
    inference.add_argument("--merged", default=None, help="Merged model directory.")
    inference.add_argument("--run-dir", default=None, help="Resolve the adapter from a run dir.")
    inference.add_argument("--prompt", "-p", action="append", default=[],
                           help="One-shot prompt; repeatable. Omit for interactive mode.")
    inference.add_argument("--system", default=None, help="Override the system prompt.")
    inference.add_argument("--max-new-tokens", type=int, default=None)
    inference.add_argument("--temperature", type=float, default=None)
    inference.add_argument("--save", default=None, help="Write generations to this JSONL.")
    inference.add_argument("--no-sample", action="store_true",
                           help="Force greedy decoding regardless of config.")
    inference.set_defaults(func=cmd_inference)

    export = sub.add_parser("export-adapter", help="Export and verify the LoRA adapter.")
    _add_config_args(export)
    _add_common_output(export)
    export.add_argument("--adapter", default=None)
    export.add_argument("--run-dir", default=None)
    export.add_argument("--destination", default=None)
    export.add_argument("--no-verify", action="store_true")
    export.set_defaults(func=cmd_export_adapter)

    merge = sub.add_parser("merge-adapter", help="Merge LoRA into the base weights.")
    _add_config_args(merge)
    _add_common_output(merge)
    merge.add_argument("--adapter", default=None)
    merge.add_argument("--run-dir", default=None)
    merge.add_argument("--destination", default=None)
    merge.add_argument("--float32", action="store_true", help="Merge in fp32 instead of bf16.")
    merge.add_argument("--no-verify", action="store_true")
    merge.set_defaults(func=cmd_merge_adapter)

    # -- utils ------------------------------------------------------------ #
    show = sub.add_parser("show-config", help="Print the fully-resolved configuration.")
    _add_config_args(show)
    show.add_argument("--json", dest="as_json", action="store_true")
    show.set_defaults(func=cmd_show_config)

    recommend = sub.add_parser("recommend-model", help="Rank LFM checkpoints for this host.")
    recommend.add_argument("--seq-length", type=int, default=1024)
    recommend.set_defaults(func=cmd_recommend_model)

    init = sub.add_parser("init-synthetic", help="Write a synthetic smoke-test dataset.")
    init.add_argument("--destination", default="data/raw/synthetic")
    init.add_argument("--examples", type=int, default=32)
    init.add_argument("--seed", type=int, default=1234)
    init.set_defaults(func=cmd_init_synthetic)

    preflight = sub.add_parser(
        "preflight",
        help="One-command GO/NO-GO readiness check before training.",
    )
    _add_config_args(preflight)
    _add_common_output(preflight)
    preflight.add_argument("--deep", action="store_true",
                           help="Also load the checkpoint and attach LoRA (stage 4).")
    preflight.add_argument("--no-strict", dest="strict", action="store_false", default=True,
                           help="Always exit 0, even with blocking problems.")
    preflight.set_defaults(func=cmd_preflight)

    stages = sub.add_parser("stages", help="List the staged validation workflow.")
    stages.set_defaults(func=cmd_stages)

    return parser


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _load(args: argparse.Namespace):
    from ..config.loader import describe, load_config, parse_set_args

    config, provenance = load_config(args.config, overrides=parse_set_args(args.set))
    return config, provenance, describe


def _emit(args: argparse.Namespace, payload: dict[str, Any], text: str) -> None:
    """Print either human text or JSON, honouring --output/--json."""
    if getattr(args, "output", None):
        path = write_json(args.output, payload)
        log.info("Wrote %s", path)
    if getattr(args, "as_json", False) or getattr(args, "output", None):
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(text)


def _resolve_adapter(args: argparse.Namespace, config: Any) -> Path:
    """Find the adapter directory from --adapter, --run-dir, or the newest run."""
    explicit = getattr(args, "adapter", None)
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise ConfigError(
                f"Adapter directory not found: {path}",
                hint="Pass --adapter <run_dir>/adapter or --run-dir <run_dir>.",
            )
        return path

    run_dir = getattr(args, "run_dir", None)
    search_root = Path(run_dir) if run_dir else Path(config.checkpointing.dir)
    if not search_root.exists():
        raise ConfigError(
            f"No run directory at {search_root}.",
            hint="Train first (`lfm-finetune train --config ...`) or pass --adapter.",
        )

    candidates: list[Path] = []
    if run_dir:
        inner = search_root / "adapter"
        if inner.exists():
            candidates.append(inner)
    else:
        candidates = [d / "adapter" for d in sorted(search_root.iterdir())
                      if d.is_dir() and (d / "adapter").exists()]
    if not candidates:
        raise ConfigError(
            f"No adapter/ directory found under {search_root}.",
            hint="A completed run writes <run_dir>/adapter. Check "
            f"`ls {search_root}`.",
        )
    latest = candidates[-1]
    log.info("Using adapter %s", latest)
    return latest


def _print_header(text: str) -> None:
    print("=" * 88)
    print(text)
    print("=" * 88)


# --------------------------------------------------------------------------- #
# Stage 0
# --------------------------------------------------------------------------- #

def cmd_doctor(args: argparse.Namespace) -> int:
    """Import check + version table. Fails fast on any missing dependency."""
    _print_header("Environment check")
    required = {
        "torch": "torch",
        "transformers": "transformers",
        "trl": "trl",
        "peft": "peft",
        "accelerate": "accelerate",
        "datasets": "datasets",
        "safetensors": "safetensors",
        "yaml": "PyYAML",
        "pydantic": "pydantic",
    }
    results: dict[str, Any] = {"imports": {}, "ok": True}

    import importlib

    for module, package in required.items():
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, "__version__", None)
            from ..utils.versions import _version

            version = version or _version(package)
            results["imports"][module] = {"ok": True, "version": version}
            print(f"  OK      {module:14s} {version}")
        except ImportError as exc:
            results["imports"][module] = {"ok": False, "error": str(exc)}
            results["ok"] = False
            print(f"  MISSING {module:14s} {exc}")

    try:
        import torch

        results["torch_cuda_available"] = torch.cuda.is_available()
        results["torch_version"] = torch.__version__
        results["cuda_version"] = torch.version.cuda
        print(f"\n  torch.cuda.is_available() = {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                major, minor = torch.cuda.get_device_capability(i)
                print(f"  GPU[{i}] {props.name} {props.total_memory / 1024**3:.1f} GiB "
                      f"cc {major}.{minor} bf16={major >= 8}")
        else:
            print("  -> CPU only. Use a smoke config (configs/smoke.yaml) for validation runs.")
    except ImportError:
        pass

    try:
        from transformers import Lfm2ForCausalLM  # noqa: F401

        results["lfm2_supported"] = True
        print("  OK      LFM2/LFM2.5 supported by this transformers build")
    except ImportError:
        results["lfm2_supported"] = False
        results["ok"] = False
        print("  MISSING LFM2 support -- transformers is too old (<4.57 required)")

    from ..utils.versions import environment_snapshot

    results["environment"] = environment_snapshot()
    _print_header("")
    print(json.dumps({"ok": results["ok"]}, indent=2))
    return 0 if results["ok"] else 1


def cmd_inspect_hardware(args: argparse.Namespace) -> int:
    from ..utils.hardware import audit
    from ..utils.versions import environment_snapshot

    report = audit()
    lines = list(report.summary_lines())
    payload: dict[str, Any] = report.to_dict()

    if args.recommend:
        from ..model.loader import recommend_model

        payload["recommendations"] = recommend_model(report, seq_length=1024)
        lines.append("")
        lines.append("Checkpoint recommendations for this host:")
        for rec in payload["recommendations"]:
            lines.append(
                f"  {rec['model_id']:32s} {rec['params'] / 1e9:.2f}B  "
                f"peak~{rec['estimated_peak_gb']:.1f} GiB  "
                f"{'FEASIBLE' if rec['feasible_on_this_host'] else 'NOT FEASIBLE'}  "
                f"[{rec['edge_class']}]"
            )
            for t in rec["tradeoffs"]:
                lines.append(f"      - {t}")

    _emit(args, payload, "\n".join(lines))
    del environment_snapshot
    return 0


# --------------------------------------------------------------------------- #
# Stage 1
# --------------------------------------------------------------------------- #

def cmd_inspect_model(args: argparse.Namespace) -> int:
    from ..model.loader import load_model

    config, _, _ = _load(args)
    loaded = load_model(config, for_training=False)

    payload = {
        "model": loaded.model_id,
        "revision": loaded.revision,
        "summary": loaded.summary(),
        "inventory": loaded.inventory.to_dict(),
    }

    lines = [loaded.summary()]

    inv = loaded.inventory
    lines.append("")
    lines.append("PEFT-relevant Linear modules (suffix -> instance count):")
    for suffix, count in sorted(inv.linear_suffixes.items()):
        lines.append(f"  {suffix:34s} {count:>4d}")
    lines.append("")
    lines.append(f"attention blocks   {len(inv.attention_modules)}")
    lines.append(f"convolution blocks {len(inv.conv_modules)}")
    lines.append(f"norm modules       {len(inv.norm_modules)}")
    lines.append(f"embeddings         {inv.embedding_modules}")
    lines.append(f"output head        {inv.output_modules}")
    lines.append(f"tied embeddings    {inv.tie_embedding}")

    if args.discover_lora:
        from ..model.discovery import describe_lora_plan, discover_lora_targets

        targets, inventory, diagnostics = discover_lora_targets(
            loaded.model, model_id=loaded.model_id
        )
        payload["lora_discovery"] = diagnostics
        lines.append("")
        lines.append("Discovered LoRA target modules:")
        lines.append(describe_lora_plan(inventory, targets, diagnostics))

    _emit(args, payload, "\n".join(lines))
    return 0


# --------------------------------------------------------------------------- #
# Stage 2
# --------------------------------------------------------------------------- #

def cmd_validate(args: argparse.Namespace) -> int:
    """Schema + quality validation without writing splits."""
    from ..data.pipeline import load_and_clean
    from ..utils.errors import DatasetError

    config, _, _ = _load(args)
    raw_path = Path(args.path) if args.path else None

    conversations, counts, rejections, summary, dup_stats, near = load_and_clean(
        config, raw_path=raw_path, limit=args.limit
    )

    accepted = counts.get("accepted", 0)
    rejected = counts.get("rejected", 0)
    total = accepted + rejected
    rate = accepted / total if total else 0.0

    lines = [
        f"rows read          {total}",
        f"accepted           {accepted} ({rate:.1%})",
        f"rejected           {rejected} ({1 - rate:.1%})",
        f"modified           {counts.get('modified', 0)}",
        f"deduped exact      {counts.get('deduplicated_exact', 0)}",
        f"deduped near       {counts.get('deduplicated_near', 0)}",
        "",
        "Rejection reasons:",
    ]
    for reason, n in sorted(summary.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {reason:28s} {n:>6d}")

    payload = {
        "counts": counts,
        "rejection_summary": summary,
        "duplicate_summary": dup_stats,
        "near_dedupe_index": near,
        "acceptance_rate": round(rate, 4),
        "rejections_sample": rejections[:25],
    }

    if accepted == 0:
        lines.append("")
        lines.append("FATAL: zero valid examples.")
        lines.append("First rejections:")
        for r in rejections[:10]:
            lines.append(f"  row {r['index']}: [{r['reason']}] {r['detail'][:120]}")
        _emit(args, payload, "\n".join(lines))
        raise DatasetError(
            f"All {total} rows were rejected.",
            hint="Inspect the rejection reasons above and fix data/raw/<file>.jsonl.",
        )

    lines.append("")
    lines.append(f"OK: {accepted} usable examples.")
    _emit(args, payload, "\n".join(lines))
    return 1 if (args.fail_on_reject and rejected) else 0


def cmd_prepare(args: argparse.Namespace) -> int:
    from ..data.pipeline import prepare_dataset

    config, _, _ = _load(args)
    tokenizer = None
    if not args.no_token_stats:
        try:
            from ..model.loader import load_tokenizer

            tokenizer, _ = load_tokenizer(config.model)
        except Exception as exc:
            log.warning(
                "Could not load the tokenizer (%s); continuing without token statistics.", exc
            )

    raw_path = Path(args.path) if args.path else None
    result = prepare_dataset(
        config, raw_path=raw_path, limit=args.limit, tokenizer=tokenizer
    )

    payload = {
        "counts": result.counts,
        "rejection_summary": result.rejection_summary,
        "dataset_hash": result.dataset_hash,
        "split_hashes": result.split_hash,
        "statistics": result.stats,
        "warnings": result.warnings,
        "report_path": str(result.report_path),
        "splits_dir": str(result.splits_dir),
        "manifest": result.manifest,
        "provenance_problems": result.manifest.get("_provenance", None) or [],
    }

    lines = [
        f"accepted           {result.counts['accepted']}",
        f"rejected           {result.counts['rejected']}",
        f"deduplicated       {result.counts['deduplicated_exact']} exact, "
        f"{result.counts['deduplicated_near']} near",
        f"split train        {result.counts['split_train']}",
        f"split validation   {result.counts['split_validation']}",
        f"split test         {result.counts['split_test']}",
        f"dataset hash       {result.dataset_hash[:16]}",
        "",
        f"report             {result.report_path}",
        f"splits             {result.splits_dir}",
    ]
    if result.warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"  ! {w}" for w in result.warnings)
    _emit(args, payload, "\n".join(lines))
    return 0


def cmd_split_stats(args: argparse.Namespace) -> int:
    from ..data.pipeline import load_splits

    config, _, _ = _load(args)
    directory = Path(config.dataset.processed_dir or Path("data") / "splits")
    splits = load_splits(directory)

    payload = {
        name: {"count": len(convs)} for name, convs in splits.items()
    }
    lines = [f"splits directory   {directory}"]
    for name, convs in splits.items():
        lines.append(f"  {name:12s} {len(convs):>6d} examples")
    _emit(args, payload, "\n".join(lines))
    return 0


# --------------------------------------------------------------------------- #
# Stage 3
# --------------------------------------------------------------------------- #

def cmd_inspect_template(args: argparse.Namespace) -> int:
    from ..data.pipeline import load_splits
    from ..model.loader import load_tokenizer
    from ..training.collator import tokenize_conversations
    from ..training.tokenization import (
        ChatFormatter,
        format_example_for_display,
        template_fingerprint,
        verify_template_consistency,
    )

    config, _, _ = _load(args)
    tokenizer, source = load_tokenizer(
        config.model,
        chat_template_override=Path(config.tokenization.chat_template)
        if config.tokenization.chat_template
        else None,
    )
    formatter = ChatFormatter(tokenizer, config.tokenization)
    formatter.assistant_only_loss = config.training.assistant_only_loss

    lines = [
        f"chat template      {source}",
        f"fingerprint        {template_fingerprint(tokenizer)}",
        f"bos / eos / pad    {tokenizer.bos_token} ({tokenizer.bos_token_id}) / "
        f"{tokenizer.eos_token!r} ({tokenizer.eos_token_id}) / "
        f"{tokenizer.pad_token} ({tokenizer.pad_token_id})",
        f"max_seq_length     {config.tokenization.max_seq_length}",
        f"append_eos_token   {config.tokenization.append_eos_token}",
        f"loss masking       "
        f"{'assistant-only' if config.training.assistant_only_loss else 'full-sequence'}",
        "",
        "--- raw chat template ---",
        (tokenizer.chat_template or "")[:2000],
        "--- end template ---",
        "",
        "--- render probe ---",
        repr(formatter.render(
            [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"},
             {"role": "assistant", "content": "ASST"}], add_generation_prompt=False
        )),
        "generation prompt:",
        repr(formatter.render(
            [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}],
            add_generation_prompt=True,
        )),
    ]

    payload: dict[str, Any] = {
        "chat_template_source": source,
        "fingerprint": template_fingerprint(tokenizer),
        "special_tokens": {
            "bos": [tokenizer.bos_token, tokenizer.bos_token_id],
            "eos": [tokenizer.eos_token, tokenizer.eos_token_id],
            "pad": [tokenizer.pad_token, tokenizer.pad_token_id],
        },
    }

    directory = Path(config.dataset.processed_dir or Path("data") / "splits")
    if directory.exists():
        splits = load_splits(directory)
        sample_pool = list(splits.get("train", []))[: max(args.samples * 4, 8)]
        if sample_pool:
            tokens = tokenize_conversations(
                sample_pool[: args.samples], formatter,
                assistant_only_loss=config.training.assistant_only_loss, split="train",
            )
            lines.append("")
            lines.append(f"--- {args.samples} fully formatted examples "
                         f"(blank marker = no loss) ---")
            for example in tokens.examples:
                lines.append(format_example_for_display(example, tokenizer))
            payload["tokenization"] = tokens.to_manifest()
            payload["consistency"] = verify_template_consistency(
                tokenizer, formatter, sample_pool
            )

    _emit(args, payload, "\n".join(lines))
    return 0


# --------------------------------------------------------------------------- #
# Stage 4
# --------------------------------------------------------------------------- #

def cmd_inspect_lora(args: argparse.Namespace) -> int:
    from ..model.discovery import describe_lora_plan
    from ..model.loader import load_model
    from ..model.lora import attach_lora

    config, _, _ = _load(args)
    loaded = load_model(config, for_training=False)

    model, audit_result, targets, inventory, diagnostics = attach_lora(
        loaded.model, config.lora, model_id=loaded.model_id
    )
    del model

    payload = {
        "audit": audit_result.to_dict(),
        "resolved_target_modules": targets,
        "diagnostics": diagnostics,
        "inventory": inventory.to_dict(),
    }
    lines = [
        "LoRA configuration:",
        f"  r={config.lora.r} alpha={config.lora.alpha} dropout={config.lora.dropout} "
        f"bias={config.lora.bias}",
        f"  effective scaling = alpha/r = {config.lora.effective_scaling:g}",
        f"  use_rslora={config.lora.use_rslora} use_dora={config.lora.use_dora}",
        "",
        "Parameter audit:",
        "  " + audit_result.summary().replace("\n", "\n  "),
        "",
        describe_lora_plan(inventory, targets, diagnostics),
    ]
    for warning in audit_result.warnings:
        lines.append(f"  ! {warning}")
    _emit(args, payload, "\n".join(lines))
    del args
    return 0


# --------------------------------------------------------------------------- #
# Stage 5-7
# --------------------------------------------------------------------------- #

def cmd_train(args: argparse.Namespace) -> int:
    from ..training.trainer import train as run_training

    config, _, describe = _load(args)
    if args.smoke_test:
        config.training.smoke_test = True
    if args.pilot:
        config.training.preset = "pilot"
    if args.limit_steps is not None:
        config.training.max_steps = args.limit_steps

    ensure_deterministic_cuda_algos(strict=config.training.deterministic_algorithms)
    set_seed(config.training.seed)

    output_dir = Path(args.output) if args.output else None

    print()
    print(describe(config))
    print()

    def on_step(step: int, logs: dict[str, Any]) -> None:
        rendered = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                            for k, v in logs.items())
        print(f"  step {step:>5d}  {rendered}")

    result = run_training(
        config,
        output_dir=output_dir,
        resume_from=args.resume,
        smoke_test=args.smoke_test,
        dry_run=args.dry_run,
        on_step=on_step,
    )

    payload = {
        "run_dir": str(result.run_dir),
        "adapter": str(result.adapter_path) if result.adapter_path else None,
        "metrics": result.metrics,
        "duration_seconds": round(result.duration_seconds, 2),
        "trainable_params": result.trainable_params,
        "total_params": result.total_params,
        "verified": result.verified,
        "resumed_from": str(result.resumed_from) if result.resumed_from else None,
        "metadata": str(result.metadata_path),
    }
    print()
    print(result.summary())
    if args.as_json or args.output:
        print(json.dumps(payload, indent=2, default=str))
    return 0


# --------------------------------------------------------------------------- #
# Stage 8
# --------------------------------------------------------------------------- #

def cmd_baseline(args: argparse.Namespace) -> int:
    from ..evaluation.runner import run_baseline_evaluation

    config, _, _ = _load(args)
    if args.num_samples:
        config.evaluation.num_samples = args.num_samples

    output_dir = Path(args.output) if args.output else None
    result = run_baseline_evaluation(config, output_dir=output_dir, split=args.split)

    summary = result["summary"]
    lines = [
        f"split              {args.split}",
        f"samples            {summary['num_samples']}",
        f"eval loss          {summary.get('eval_loss')}",
        f"mean length        {summary.get('mean_length_words')} words",
        f"asks for info      {summary.get('asks_for_information_rate')}",
        f"overconfidence     {summary.get('overconfidence_rate')}",
        f"token F1 vs ref    {summary.get('mean_token_f1_vs_reference')}",
    ]
    _emit(args, {"baseline": result}, "\n".join(lines))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from ..evaluation.runner import run_comparison_evaluation

    config, _, _ = _load(args)
    adapter = _resolve_adapter(args, config)
    output_dir = Path(args.output) if args.output else Path("evaluations") / config.run.name

    result = run_comparison_evaluation(
        config, adapter, output_dir=output_dir, split=args.split
    )
    payload = {
        "comparison_report": result["comparison_path"],
        "base": result["base"],
        "tuned": result["tuned"],
    }
    comparison = result["base"].get("eval_loss"), result["tuned"].get("eval_loss")
    lines = [
        f"comparison report  {result['comparison_path']}",
        "",
        "Held-out loss (lower is better):",
        f"  base   {comparison[0]}",
        f"  tuned  {comparison[1]}",
        "",
        "Objective proxy deltas:",
    ]
    tuned_summary, base_summary = result["tuned"], result["base"]
    for key in (
        "asks_for_information_rate",
        "overconfidence_rate",
        "any_format_rate",
        "mean_token_f1_vs_reference",
        "verbatim_reference_match_rate",
    ):
        if key in base_summary and key in tuned_summary:
            lines.append(
                f"  {key:34s} {base_summary[key]:>8} -> {tuned_summary[key]:>8} "
                f"({tuned_summary[key] - base_summary[key]:+.4f})"
            )
    lines += [
        "",
        "NOTE: these are objective proxies. Agricultural correctness requires a human to "
        "score the rubric sheet in the report.",
    ]
    _emit(args, payload, "\n".join(lines))
    return 0


# --------------------------------------------------------------------------- #
# Stage 9 / serving
# --------------------------------------------------------------------------- #

def cmd_inference(args: argparse.Namespace) -> int:
    from ..inference.cli_loop import run_batch, run_interactive
    from ..inference.generator import GenerationConfig

    config, _, _ = _load(args)

    adapter = None
    merged = None
    if args.merged:
        merged = args.merged
    elif args.adapter or args.run_dir:
        adapter = str(_resolve_adapter(args, config))

    gen_config = GenerationConfig.from_experiment_config(config)
    if args.max_new_tokens:
        gen_config.max_new_tokens = args.max_new_tokens
    if args.temperature is not None:
        gen_config.temperature = args.temperature
        gen_config.do_sample = args.temperature > 0
    if args.no_sample:
        gen_config.do_sample = False
        gen_config.num_beams = 1
    config.evaluation.__dict__.update(gen_config.__dict__)

    from ..inference.cli_loop import InferenceEngine

    engine = InferenceEngine.load(config, adapter_path=adapter, merged_path=merged)

    if args.prompt:
        machine_readable = bool(args.as_json or args.output)
        records = run_batch(
            engine, args.prompt,
            system_prompt=args.system,
            output_path=Path(args.save) if args.save else None,
            quiet=machine_readable,
        )
        if machine_readable:
            # Nothing else may write to stdout in this mode.
            print(json.dumps({"generations": records}, indent=2, ensure_ascii=False))
        return 0
    return run_interactive(engine, system_prompt=args.system)


def cmd_export_adapter(args: argparse.Namespace) -> int:
    from ..export.adapter import export_adapter

    config, _, _ = _load(args)
    adapter = _resolve_adapter(args, config)
    destination = (
        Path(args.destination)
        if args.destination
        else ensure_dir(Path(config.export.adapters_dir) / config.run.name)
    )

    result = export_adapter(
        adapter, destination,
        base_model_id=config.model.name,
        base_revision=config.model.revision,
        verify=not args.no_verify,
    )
    _emit(args, {"export": {"path": str(result.path), "verified": result.verified,
                            "details": result.details}}, result.summary())
    return 0


def cmd_merge_adapter(args: argparse.Namespace) -> int:
    from ..export.adapter import merge_adapter

    config, _, _ = _load(args)
    adapter = _resolve_adapter(args, config)
    destination = (
        Path(args.destination)
        if args.destination
        else ensure_dir(Path(config.export.merged_dir) / config.run.name)
    )

    result = merge_adapter(
        adapter, destination,
        base_model_id=config.model.name,
        base_revision=config.model.revision,
        config=config,
        merge_in_bf16=not args.float32,
        safe_serialization=config.export.safe_serialization,
        max_shard_size=config.export.max_shard_size,
        save_tokenizer=config.export.export_tokenizer,
        verify=not args.no_verify,
    )
    _emit(args, {"merge": {"path": str(result.path), "verified": result.verified,
                           "details": result.details}}, result.summary())
    return 0


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

def cmd_preflight(args: argparse.Namespace) -> int:
    from ..preflight import cmd_preflight as _run

    return _run(args)


def cmd_show_config(args: argparse.Namespace) -> int:
    config, provenance, describe = _load(args)
    payload = {"config": config.to_dict(), "provenance": provenance}
    if args.as_json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        _print_header("Resolved configuration")
        print(describe(config))
        _print_header("Provenance")
        print(f"file           {provenance['config_path']}")
        print(f"extends chain  {' -> '.join(provenance['extends_chain'])}")
        print(f"fingerprint    {provenance['fingerprint']}")
    return 0


def cmd_recommend_model(args: argparse.Namespace) -> int:
    from ..model.loader import recommend_model
    from ..utils.hardware import audit

    hardware = audit()
    recommendations = recommend_model(hardware, seq_length=args.seq_length)

    print(_summary_of_hardware(hardware))
    print()
    print(f"Checkpoint ranking for seq_length={args.seq_length}:")
    for rec in recommendations:
        status = "FEASIBLE" if rec["feasible_on_this_host"] else "NOT FEASIBLE"
        print(f"  {rec['model_id']:32s} {rec['params'] / 1e9:5.2f}B  "
              f"peak~{rec['estimated_peak_gb']:5.2f} GiB vs {rec['budget_gb']:5.2f} GiB {rec['target']:12s} {status}")
        print(f"      class: {rec['edge_class']}")
        print(f"      use:   {rec['use_for']}")
        for t in rec["tradeoffs"]:
            print(f"      note:  {t}")
    return 0


def _summary_of_hardware(hardware: Any) -> str:
    return "\n".join(hardware.summary_lines())


def cmd_init_synthetic(args: argparse.Namespace) -> int:
    from ..data.synthetic import write_synthetic_dataset

    manifest = write_synthetic_dataset(
        Path(args.destination), num_examples=args.examples, seed=args.seed
    )
    print(json.dumps(manifest, indent=2))
    return 0


def cmd_stages(args: argparse.Namespace) -> int:
    stages = [
        ("0", "doctor", "Environment: imports, versions, CUDA", "lfm-finetune doctor"),
        ("0", "inspect-hardware", "GPU/VRAM/CUDA/RAM/disk audit", "lfm-finetune inspect-hardware --recommend"),
        ("1", "inspect-model", "Load model, list PEFT-relevant modules", "lfm-finetune inspect-model -c configs/...yaml"),
        ("2", "validate", "Schema + quality gates on raw data", "lfm-finetune validate -c configs/...yaml"),
        ("2", "prepare", "Clean, dedupe, split, statistics, report", "lfm-finetune prepare -c configs/...yaml"),
        ("3", "inspect-template", "Chat template + loss mask verification", "lfm-finetune inspect-template -c configs/...yaml"),
        ("4", "inspect-lora", "Attach LoRA, audit trainable params", "lfm-finetune inspect-lora -c configs/...yaml"),
        ("5", "train --smoke-test", "Tiny end-to-end training run", "lfm-finetune train -c configs/smoke.yaml --smoke-test"),
        ("6", "train --pilot", "Short pilot on a subset", "lfm-finetune train -c configs/...yaml --pilot"),
        ("7", "train", "Full SFT", "lfm-finetune train -c configs/...yaml"),
        ("8", "baseline", "Base model on the held-out split", "lfm-finetune baseline -c configs/...yaml"),
        ("8", "evaluate", "Base vs tuned comparison report", "lfm-finetune evaluate -c configs/...yaml"),
        ("9", "export-adapter", "Export + verify the adapter", "lfm-finetune export-adapter -c configs/...yaml"),
        ("9", "merge-adapter", "Merge into standalone weights", "lfm-finetune merge-adapter -c configs/...yaml"),
        ("--", "inference", "Chat with base+adapter", "lfm-finetune inference -c configs/...yaml"),
    ]
    print(f"{'stage':<6} {'command':<20} purpose")
    print("-" * 88)
    for stage, command, purpose, _example in stages:
        print(f"{stage:<6} {command:<20} {purpose}")
    print("-" * 88)
    print("Example commands:")
    for _, _, _, example in stages:
        print(f"  {example}")
    return 0


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    ensure_deterministic_cuda_algos()
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "version", False):
        from importlib.metadata import version

        from ..utils.versions import package_versions

        versions = package_versions()
        print(BANNER)
        print(json.dumps({"lfm-sft": version("lfm-sft"), **versions}, indent=2))
        return 0

    setup_logging("DEBUG" if args.verbose else "INFO", force=True)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0

    try:
        return int(args.func(args) or 0)
    except PipelineError as exc:
        log.error("%s", exc.render())
        return exc.exit_code
    except KeyboardInterrupt:
        log.error("Interrupted.")
        return 130
    except Exception as exc:  # noqa: BLE001
        log.exception("Unhandled error: %s", exc)
        log.error(
            "This is an unexpected failure. Re-run with -v for the full traceback. "
            "If it is a compatibility problem, verify the pinned versions with "
            "`lfm-finetune doctor`."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())