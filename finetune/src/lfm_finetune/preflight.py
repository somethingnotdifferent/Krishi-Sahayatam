"""One-command pre-training readiness check.

Runs every cheap gate the pipeline enforces, in the order a run would hit them,
and prints a GO / NO-GO checklist. Exits non-zero if any required gate fails, so
it can be used as a CI step.

    uv run lfm-finetune preflight -c configs/agriculture/lfm25_1.2b_6gb.yaml

Checks, cheapest first:

    1. environment      imports, versions, CUDA availability
    2. config           resolves, validates, fingerprint
    3. hardware         dtype supported, VRAM/RAM estimate
    4. dataset          readable, schema-valid, provenance declared
    5. dataset quality  rejection reasons, split sizes, leakage, statistics
    6. splits           present on disk (stage: after `prepare`)
    7. template         chat template renders, prefix-consistent
    8. lora             target modules discovered and attached  [--deep]

Stage 8 loads the real checkpoint, so it is skipped unless ``--deep`` is passed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config.loader import load_config, parse_set_args
from .utils.errors import PipelineError
from .utils.hardware import audit, estimate_training_vram_bytes
from .utils.io import read_json, write_json
from .utils.logging import get_logger
from .utils.versions import package_versions

log = get_logger("preflight")

OK, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""
    remedy: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def blocking(self) -> bool:
        return self.status == FAIL

    def render(self) -> str:
        icon = {OK: "PASS", FAIL: "FAIL", WARN: "WARN", SKIP: "SKIP"}[self.status]
        lines = [f"  [{icon}] {self.name}"]
        if self.detail:
            lines.append(f"         {self.detail}")
        if self.remedy and self.status in (FAIL, WARN):
            lines.append(f"         -> {self.remedy}")
        return "\n".join(lines)


def run_preflight(config_path: str, overrides: dict[str, Any] | None,
                  *, deep: bool = False, strict: bool = True) -> tuple[list[Check], dict[str, Any]]:
    checks: list[Check] = []
    summary: dict[str, Any] = {}

    # ---- 1. environment ---------------------------------------------- #
    missing = []
    for module, package in (("torch", "torch"), ("transformers", "transformers"),
                            ("trl", "trl"), ("peft", "peft"), ("accelerate", "accelerate"),
                            ("safetensors", "safetensors")):
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    versions = package_versions()
    if missing:
        checks.append(Check("environment / imports", FAIL,
                            f"missing: {missing}", "run `uv sync` inside finetune/"))
    else:
        checks.append(Check("environment / imports", OK,
                            f"torch {versions['torch']} | transformers {versions['transformers']}"
                            f" | trl {versions['trl']} | peft {versions['peft']}"))

    try:
        import transformers

        has_lfm2 = hasattr(transformers, "Lfm2ForCausalLM")
        checks.append(Check(
            "environment / LFM2 support", OK if has_lfm2 else FAIL,
            "Lfm2ForCausalLM available" if has_lfm2 else "transformers lacks LFM2",
            "transformers >= 4.57 required" if not has_lfm2 else "",
        ))
    except ImportError:
        pass

    # ---- 2. config ----------------------------------------------------- #
    config, provenance = load_config(config_path, overrides=overrides or {})
    summary["config"] = {"path": provenance["config_path"],
                         "fingerprint": provenance["fingerprint"],
                         "extends_chain": provenance["extends_chain"]}
    checks.append(Check("config / resolves + validates", OK,
                        f"{provenance['config_path'].split('/')[-1]} "
                        f"(fingerprint {provenance['fingerprint'][:12]}, "
                        f"{len(provenance['extends_chain'])} file chain)"))

    if not config.lora.enabled:
        checks.append(Check("config / LoRA enabled", WARN,
                            "lora.enabled=false -> this is a FULL fine-tune",
                            "LoRA is strongly recommended: it fits in far less VRAM and "
                            "keeps the base model reusable."))
    else:
        checks.append(Check("config / LoRA enabled", OK,
                            f"r={config.lora.r} alpha={config.lora.alpha} "
                            f"effective scaling={config.lora.effective_scaling:g} "
                            f"target_modules="
                            f"{'auto (discovered)' if config.lora.target_modules == 'auto' else 'explicit'}"))

    eff = (config.training.per_device_train_batch_size
           * config.training.gradient_accumulation_steps)
    checks.append(Check("config / effective batch", OK,
                        f"{config.training.per_device_train_batch_size} x "
                        f"{config.training.gradient_accumulation_steps} = {eff}"))

    # ---- 3. hardware --------------------------------------------------- #
    hardware = audit()
    if hardware.gpus:
        g = hardware.gpus[0]
        checks.append(Check("hardware / GPU", OK,
                            f"{g.name}, {g.total_memory_gb:.1f} GiB VRAM "
                            f"(cc {g.compute_capability_str}), "
                            f"bf16={'yes' if g.supports_bf16 else 'NO'}, "
                            f"{hardware.nvidia_driver_version} driver"))
    else:
        checks.append(Check("hardware / GPU", WARN, "no CUDA device visible",
                            "CPU runs work for validation; training needs a GPU."))

    dtype = config.model.dtype.lower()
    if dtype == "auto":
        resolved = ("bfloat16" if hardware.gpus and hardware.supports("bfloat16")
                    else "float16" if hardware.gpus else "float32")
    else:
        resolved = {"bf16": "bfloat16", "fp16": "float16",
                    "fp32": "float32"}.get(dtype, dtype)

    if hardware.gpus and not hardware.supports(resolved):
        checks.append(Check("hardware / dtype", FAIL,
                            f"model.dtype={config.model.dtype} -> {resolved} unsupported",
                            "pre-Ampere GPUs need float16 or float32"))
    else:
        checks.append(Check("hardware / dtype", OK, f"{config.model.dtype} -> {resolved}"))

    if not hardware.gpus and resolved != "float32":
        checks[-1] = Check("hardware / dtype", WARN,
                           "CPU run will be forced to float32 (bf16/fp16 unsupported on CPU)",
                           "expected; CPU is for validation only")

    # ---- 4 + 5. dataset ------------------------------------------------ #
    dataset_path = Path(config.dataset.path) if config.dataset.path else None
    if dataset_path is None:
        checks.append(Check("dataset / path", WARN, "dataset.path is null",
                            "set dataset.path, or dataset.use_processed=true"))
    elif not dataset_path.exists():
        checks.append(Check("dataset / path", FAIL, f"missing: {dataset_path}",
                            "create data/raw/<name>/ with your corpus and manifest"))
    else:
        checks.append(Check("dataset / path", OK, str(dataset_path)))

        manifest, manifest_path = _find_manifest(dataset_path)
        if manifest:
            licence = str(manifest.get("license", "")).lower()
            ok_licence = licence not in ("unknown", "unspecified", "", "none", "tbd", "<fill in>")
            checks.append(Check(
                "dataset / provenance manifest", OK if ok_licence else FAIL,
                f"{manifest_path} (license={manifest.get('license')!r})" if ok_licence
                else f"{manifest_path}: license is {manifest.get('license')!r}",
                "set a real SPDX id (CC-BY-4.0, ...) or 'internal-use-only'",
            ))
            missing_fields = [f for f in ("source", "created") if not manifest.get(f)]
            if missing_fields:
                checks.append(Check("dataset / manifest fields", WARN,
                                    f"missing: {missing_fields}",
                                    "fill these in; they are permanent provenance"))
        else:
            checks.append(Check("dataset / provenance manifest", FAIL,
                                f"no manifest.yaml/.json next to the data in {dataset_path}",
                                "copy datasets/TEMPLATE_manifest.yaml and fill it in"))

        _dataset_checks(config, checks, summary, dataset_path)

    # ---- 6. prepared splits -------------------------------------------- #
    splits_dir = Path(config.dataset.processed_dir or Path("data") / "splits")
    if splits_dir.exists():
        present = [n for n in ("train", "validation", "test")
                   if (splits_dir / f"{n}.jsonl").exists()]
        if not present:
            checks.append(Check("splits / prepared", FAIL,
                                f"{splits_dir} exists but contains no split files",
                                f"re-run: uv run lfm-finetune prepare -c {config_path}"))
        else:
            checks.append(Check("splits / prepared", OK,
                                f"{', '.join(present)} in {splits_dir}"))
            _stale_split_check(config, dataset_path, splits_dir, checks)
    else:
        checks.append(Check("splits / prepared", SKIP,
                            f"not prepared yet ({splits_dir})",
                            f"run: uv run lfm-finetune prepare -c {config_path}"))

    # ---- 7. chat template ---------------------------------------------- #
    try:
        from .model.loader import load_tokenizer
        from .training.tokenization import ChatFormatter

        tokenizer, source = load_tokenizer(config.model)
        formatter = ChatFormatter(tokenizer, config.tokenization)
        rendered = formatter.render(
            [{"role": "system", "content": "S"}, {"role": "user", "content": "U"},
             {"role": "assistant", "content": "A"}], add_generation_prompt=False)
        prompt = formatter.render([{"role": "system", "content": "S"},
                                   {"role": "user", "content": "U"}],
                                  add_generation_prompt=True)
        prefix_ok = rendered.startswith(prompt)
        checks.append(Check("template / renders + prefix-consistent",
                            OK if prefix_ok else FAIL,
                            f"source={source}, eos_token_id={formatter.eos_token_id}, "
                            f"append_eos={config.tokenization.append_eos_token}",
                            "a non-prefix-consistent template breaks assistant-only masking"
                            if not prefix_ok else ""))

        has_generation_blocks = ("{% generation" in (tokenizer.chat_template or "")
                                 or "{%- generation" in (tokenizer.chat_template or ""))
        checks.append(Check(
            "template / assistant mask strategy",
            OK,
            ("checkpoint provides {% generation %} blocks" if has_generation_blocks
             else "checkpoint has NO {% generation %} blocks -> masking by prefix "
                  "length (this pipeline handles it correctly and asserts the "
                  "prefix property)"),
            "no action needed either way",
        ))

        ctx = int(getattr(tokenizer, "model_max_length", 0) or 0)
        seq = config.tokenization.max_seq_length
        if ctx and ctx < 10**9 and seq > ctx:
            checks.append(Check("tokenization / sequence length", FAIL,
                                f"max_seq_length={seq} > tokenizer model_max_length",
                                f"lower tokenization.max_seq_length to <= {ctx}"))
        elif seq > 32768:
            checks.append(Check("tokenization / sequence length", WARN,
                                f"max_seq_length={seq} exceeds LFM2.5's 32,768 trained context",
                                "512-2048 is typical for domain SFT"))
        else:
            checks.append(Check("tokenization / sequence length", OK, f"{seq} tokens"))
    except PipelineError as exc:
        checks.append(Check("template / tokenizer", FAIL, exc.message, exc.hint or ""))

    # ---- VRAM estimate ---------------------------------------------------- #
    summary["vram_estimate"] = _vram_block(config, hardware, resolved)

    # ---- 8. LoRA (deep only) ---------------------------------------------- #
    if deep:
        try:
            from .model.loader import load_model
            from .model.lora import attach_lora

            loaded = load_model(config, for_training=True)
            _model, audit_result, targets, _inv, diag = attach_lora(
                loaded.model, config.lora, model_id=loaded.model_id)
            checks.append(Check("lora / target modules discovered", OK,
                                f"{len(targets)} module types, "
                                f"{diag['num_target_instances']} instances; "
                                f"matches Liquid reference preset: "
                                f"{diag.get('matches_official_reference')}"))
            checks.append(Check("lora / trainable parameter audit",
                                OK if audit_result.ok else FAIL,
                                f"{audit_result.trainable_params:,} trainable "
                                f"({audit_result.trainable_pct:.3f}% of "
                                f"{audit_result.total_params:,})"))
            summary["lora"] = {"targets": targets, "audit": audit_result.to_dict(),
                               "discovery": diag}
            del _model, loaded
        except PipelineError as exc:
            checks.append(Check("lora / attachment", FAIL, exc.message, exc.hint or ""))
    else:
        checks.append(Check("lora / attachment", SKIP,
                            "not checked (pass --deep to load the checkpoint)",
                            "optional: uv run lfm-finetune inspect-lora -c <config>"))

    return checks, summary


def _normalise_path(value: str) -> str:
    """Absolute, symlink-resolved, trailing-slash-free path string.

    Falls back to the raw string when the path does not exist, so an absolute
    path to a corpus that has since been moved is still compared meaningfully.
    """
    if not value:
        return ""
    try:
        return str(Path(value).expanduser().resolve())
    except OSError:  # pragma: no cover
        return str(value)


def _stale_split_check(config: Any, dataset_path: Path | None, splits_dir: Path,
                       checks: list[Check]) -> None:
    """Catch the most expensive mistake in this pipeline: training one dataset
    while believing you trained another.

    ``data/splits`` is shared across configs by default, so switching
    ``dataset.path`` and forgetting to re-run ``prepare`` leaves the previous
    corpus's splits in place. Training then proceeds happily -- on the wrong data.
    """
    report_path = splits_dir.parent / "validated" / "report.json"
    if not report_path.exists():
        checks.append(Check("splits / provenance", SKIP,
                            "no preparation report to compare against",
                            f"re-run: uv run lfm-finetune prepare -c {config.run.name}.yaml"))
        return
    try:
        report = read_json(report_path)
    except PipelineError:
        return

    # Compare normalised absolute paths. A relative path in the config and an
    # absolute path in the report describe the same directory, and treating that
    # as "stale" would train people to ignore the gate.
    prepared_from = _normalise_path(report.get("source", ""))
    current = _normalise_path(str(dataset_path)) if dataset_path else ""
    if current and prepared_from and prepared_from != current:
        checks.append(Check(
            "splits / provenance", FAIL,
            f"splits were built from {prepared_from!r} but this config points at "
            f"{current!r}",
            "re-run `prepare -c <this config>` -- otherwise you will train on the "
            "previous dataset while believing you trained this one",
        ))
        return

    prepared_fp = report.get("config_fingerprint")
    checks.append(Check("splits / provenance", OK,
                        f"prepared from {report.get('source') or prepared_from} "
                        f"(dataset hash {(report.get('dataset_hash') or '')[:12]})"
                        + (f", config fingerprint {prepared_fp[:12]}" if prepared_fp else "")))


def _find_manifest(dataset_path: Path) -> tuple[dict[str, Any] | None, Path | None]:
    """Locate the provenance manifest sitting beside the raw corpus."""
    from .data.io import load_manifest

    base = dataset_path if dataset_path.is_dir() else dataset_path.parent
    for name in ("manifest.yaml", "manifest.yml", "manifest.json"):
        candidate = base / name
        if candidate.exists():
            return (load_manifest(candidate) or {}), candidate
    return None, None


def _dataset_checks(config: Any, checks: list[Check], summary: dict[str, Any],
                    dataset_path: Path) -> None:
    from .data.pipeline import load_and_clean

    try:
        conversations, counts, rejections, reasons, dup, _near = load_and_clean(
            config, raw_path=dataset_path)
    except PipelineError as exc:
        checks.append(Check("dataset / readable", FAIL, exc.message, exc.hint or ""))
        return

    accepted = counts.get("accepted", 0)
    rejected = counts.get("rejected", 0)
    total = accepted + rejected
    rate = accepted / total if total else 0.0

    # Acceptance rate is a blocking signal, not a cosmetic one. A corpus where
    # 98 % of rows are unusable is a data-shape bug (wrong key names, missing
    # assistant turn, placeholder text), and training on the surviving 2 % would
    # "succeed" while learning nothing.
    if not accepted:
        status, remedy = FAIL, ("run `prepare` and read data/validated/report.json -> "
                                "rejections[]; every rejected row has a reason and an excerpt")
    elif rate < 0.5:
        status, remedy = FAIL, (
            f"only {rate:.1%} of rows are usable -- that is almost always a data-shape "
            f"bug, not a content problem. Check the top rejection reasons below and the "
            f"examples in data/validated/rejected.jsonl.")
    elif rate < 0.9:
        status, remedy = WARN, ("a few rows were rejected; confirm the reasons are the "
                               "ones you expect rather than a systematic problem")
    else:
        status, remedy = OK, ""

    checks.append(Check("dataset / schema + quality", status,
                        f"{accepted} accepted ({rate:.1%}), {rejected} rejected, "
                        f"{counts.get('modified', 0)} normalised, "
                        f"{counts.get('deduplicated_exact', 0)} exact + "
                        f"{counts.get('deduplicated_near', 0)} near duplicates",
                        remedy))

    if rejected:
        top = sorted(reasons.items(), key=lambda kv: -kv[1])[:3]
        checks.append(Check("dataset / top rejection reasons", WARN,
                            ", ".join(f"{k}={v}" for k, v in top),
                            "each rejected row is kept in data/validated/rejected.jsonl "
                            "with a reason and an excerpt"))

    if accepted < 500:
        checks.append(Check("dataset / size", WARN,
                            f"{accepted} examples",
                            "Liquid's guidance is 500-5000 task examples. Below ~200 the "
                            "model memorises rather than generalises."))
    else:
        checks.append(Check("dataset / size", OK, f"{accepted} examples"))

    no_provenance = sum(
        1 for c in conversations
        if not (c.metadata or {}).get("source") or not (c.metadata or {}).get("license")
    )
    if no_provenance:
        checks.append(Check("dataset / per-record provenance", WARN,
                            f"{no_provenance}/{accepted} records lack metadata.source "
                            f"or metadata.license",
                            "every record should carry its own source and licence"))

    summary["dataset"] = {"accepted": accepted, "rejected": rejected,
                          "reasons": reasons, "duplicates": dup}


def _vram_block(config: Any, hardware: Any, resolved_dtype: str) -> dict[str, Any]:
    params = {"LiquidAI/LFM2.5-1.2B-Base": 1_200_000_000,
              "LiquidAI/LFM2.5-350M-Base": 354_483_968,
              "LiquidAI/LFM2.5-230M-Base": 230_000_000,
              "LiquidAI/LFM2.5-2.6B-Base": 2_600_000_000,
              "LiquidAI/LFM2.5-8B-A1B-Base": 8_000_000_000}.get(config.model.name)
    if params is None:
        return {"note": "unknown model; run `recommend-model` for a size-based estimate"}
    lora = int(params * (0.0169 if config.lora.r == 16 else 0.0085 * config.lora.r / 16))
    est = estimate_training_vram_bytes(
        param_count=params, dtype=resolved_dtype,
        batch_size=config.training.per_device_train_batch_size,
        grad_accum=config.training.gradient_accumulation_steps,
        max_seq_length=config.tokenization.max_seq_length,
        gradient_checkpointing=(
            config.model.gradient_checkpointing
            if config.training.gradient_checkpointing is None
            else config.training.gradient_checkpointing),
        trainable_params=lora,
    )
    est["model"] = config.model.name
    est["peak_gib"] = round(est["peak_bytes"] / 1024**3, 2)
    est["target"] = "vram" if hardware.gpus else "system_ram"
    est["budget_gib"] = round(
        (hardware.total_vram_gb if hardware.gpus
         else hardware.system_ram_available_bytes / 1024**3), 2)
    return est


# --------------------------------------------------------------------------- #
# CLI command
# --------------------------------------------------------------------------- #

def cmd_preflight(args: Any) -> int:

    from .utils.seeding import ensure_deterministic_cuda_algos

    ensure_deterministic_cuda_algos()

    checks, summary = run_preflight(
        args.config,
        parse_set_args(args.set),
        deep=args.deep,
    )

    print()
    print("=" * 88)
    print("PRE-TRAINING READINESS CHECK")
    print("=" * 88)
    for check in checks:
        print(check.render())

    vram = summary.get("vram_estimate") or {}
    if "peak_gib" in vram:
        fits = vram["peak_gib"] <= vram["budget_gib"] * 0.85
        status = OK if fits else WARN
        checks.append(Check(
            "hardware / memory estimate", status,
            f"estimated peak {vram['peak_gib']} GiB vs {vram['budget_gib']} GiB "
            f"{vram['target']} available",
            "if training OOMs: lower per_device_train_batch_size and raise "
            "gradient_accumulation_steps (keeps the effective batch), or lower "
            "tokenization.max_seq_length",
        ))
        print(checks[-1].render())

    failures = [c for c in checks if c.blocking]
    warnings = [c for c in checks if c.status == WARN]
    skipped = [c for c in checks if c.status == SKIP]

    print()
    print("=" * 88)
    if failures:
        print(f"RESULT: NO-GO -- {len(failures)} blocking problem(s). Fix these first:")
        for c in failures:
            print(f"  * {c.name}: {c.remedy or c.detail}")
        print("=" * 88)
    else:
        print(f"RESULT: GO -- no blocking problems. ({len(warnings)} warning(s), "
              f"{len(skipped)} skipped)")
        print()
        print("Next:")
        if any(c.name == "splits / prepared" for c in skipped):
            print(f"  uv run lfm-finetune prepare  -c {args.config}")
        print(f"  uv run lfm-finetune train    -c {args.config} --pilot")
        print(f"  uv run lfm-finetune train    -c {args.config}")
    print("=" * 88)

    if args.output:
        write_json(args.output, {
            "ready": not failures,
            "checks": [{"name": c.name, "status": c.status, "detail": c.detail} for c in checks],
            **summary,
        })
        print(f"\nreport written to {args.output}")

    return 1 if (failures and args.strict) else 0


