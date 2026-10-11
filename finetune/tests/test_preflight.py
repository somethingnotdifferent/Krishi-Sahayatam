"""Preflight readiness gates.

These matter because ``preflight`` is the thing an operator is told to trust
before spending money on GPU time. Every gate that says NO-GO must be there for
a real failure mode, and must not fire on a healthy corpus.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lfm_finetune.preflight import run_preflight
from lfm_finetune.utils.io import write_json, write_jsonl

REPO = Path(__file__).resolve().parent.parent


def statuses(checks) -> dict[str, str]:
    return {c.name: c.status for c in checks}


def blocking(checks) -> list[str]:
    return [c.name for c in checks if c.blocking]


def run(config: str, **kw):
    checks, summary = run_preflight(str(REPO / config), overrides={}, **kw)
    return checks, summary


@pytest.fixture
def healthy(tmp_path: Path):
    """A dataset that should produce GO on every data-related gate."""
    from lfm_finetune.data.synthetic import write_synthetic_dataset

    root = tmp_path / "raw"
    write_synthetic_dataset(root, num_examples=40, seed=3)
    return root


def _config_for(tmp_path: Path, dataset: Path, name="pf.yaml",
                splits_dir: Path | None = None) -> Path:
    """A minimal config pointing at `dataset`, written to tmp_path.

    `splits_dir` defaults to an isolated directory so tests never pick up the
    repository's real data/splits or data/validated/report.json.
    """
    splits_dir = splits_dir or (tmp_path / "splits")
    config = f"""run: {{name: {name}}}
model: {{name: LiquidAI/LFM2.5-350M-Base, local_files_only: true}}
dataset:
  path: {dataset}
  processed_dir: {splits_dir}
  use_processed: false
  quality:
    min_assistant_chars: 1
    near_duplicate: false
    remove_prompt_leakage: false
tokenization: {{max_seq_length: 256}}
lora: {{enabled: true, r: 8, alpha: 16, target_modules: auto}}
training: {{preset: smoke, seed: 1}}
"""
    path = tmp_path / name
    path.write_text(config)
    return path


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #

def test_healthy_dataset_is_go(healthy, tmp_path) -> None:
    checks, summary = run_preflight(str(_config_for(tmp_path, healthy)), overrides={})
    st = statuses(checks)
    assert not blocking(checks), [c.render() for c in checks if c.blocking]
    assert st["dataset / schema + quality"] == "PASS"
    assert st["dataset / provenance manifest"] == "PASS"
    assert st["config / LoRA enabled"] == "PASS"
    assert "dataset" in summary


def test_every_gate_has_a_readable_name(healthy, tmp_path) -> None:
    checks, _ = run_preflight(str(_config_for(tmp_path, healthy)), overrides={})
    for c in checks:
        assert c.name and c.status in ("PASS", "FAIL", "WARN", "SKIP")
        assert c.render()


# --------------------------------------------------------------------------- #
# Blocking gates
# --------------------------------------------------------------------------- #

def test_missing_dataset_path_blocks(healthy, tmp_path) -> None:
    checks, _ = run_preflight(
        str(_config_for(tmp_path, healthy.parent / "nope")), overrides={})
    assert "dataset / path" in blocking(checks)
    st = statuses(checks)
    assert st["dataset / path"] == "FAIL"


def test_missing_manifest_blocks(healthy, tmp_path) -> None:
    (healthy / "manifest.json").unlink()
    checks, _ = run_preflight(str(_config_for(tmp_path, healthy)), overrides={})
    assert "dataset / provenance manifest" in blocking(checks)


def test_unknown_license_blocks(healthy, tmp_path) -> None:
    write_json(healthy / "manifest.json", {
        "name": "x", "version": "1", "source": "somewhere",
        "license": "unknown", "created": "2026-01-01", "num_examples": 40,
    })
    checks, _ = run_preflight(str(_config_for(tmp_path, healthy)), overrides={})
    assert "dataset / provenance manifest" in blocking(checks)


def test_internal_use_only_license_passes(healthy, tmp_path) -> None:
    write_json(healthy / "manifest.json", {
        "name": "x", "version": "1", "source": "farmer interviews",
        "license": "internal-use-only", "created": "2026-01-01", "num_examples": 40,
    })
    checks, _ = run_preflight(str(_config_for(tmp_path, healthy)), overrides={})
    assert "dataset / provenance manifest" not in blocking(checks)


def test_low_acceptance_rate_blocks(healthy, tmp_path) -> None:
    """The important one: 98 % of rows unusable is a data-shape bug, and training
    on the surviving 2 % would 'succeed' while learning nothing."""
    # Force near-total rejection by making answers too short for the gate.
    config_path = _config_for(tmp_path, healthy, name="strict.yaml")
    config_path.write_text(config_path.read_text().replace(
        "min_assistant_chars: 1", "min_assistant_chars: 100000"))

    checks, _ = run_preflight(str(config_path), overrides={})
    assert "dataset / schema + quality" in blocking(checks)
    detail = next(c.detail for c in checks if c.name == "dataset / schema + quality")
    assert "rejected" in detail


def test_all_rows_rejected_blocks(healthy, tmp_path) -> None:
    write_jsonl(healthy / "synthetic.jsonl", [{"nonsense": 1}, {"also": "bad"}])
    checks, _ = run_preflight(str(_config_for(tmp_path, healthy)), overrides={})
    assert "dataset / schema + quality" in blocking(checks)


def test_small_dataset_warns_but_does_not_block(healthy, tmp_path) -> None:
    """Size is advisory. A small corpus is a real risk, not a broken pipeline."""
    checks, _ = run_preflight(str(_config_for(tmp_path, healthy)), overrides={})
    assert statuses(checks)["dataset / size"] == "WARN"
    assert "dataset / size" not in blocking(checks)


# --------------------------------------------------------------------------- #
# Split provenance -- training the wrong corpus
# --------------------------------------------------------------------------- #

def test_stale_splits_block(tmp_path) -> None:
    """Switching dataset.path without re-running `prepare` must be caught.

    data/splits is shared across configs, so this is the most expensive silent
    mistake available: training happily on the previous corpus.
    """
    from lfm_finetune.data.synthetic import write_synthetic_dataset

    corpus_a = tmp_path / "a"
    corpus_b = tmp_path / "b"
    write_synthetic_dataset(corpus_a, num_examples=20, seed=1)
    write_synthetic_dataset(corpus_b, num_examples=20, seed=2)

    splits = tmp_path / "splits"
    # Simulate: splits were prepared from corpus A.
    report = {
        "source": str(corpus_a),
        "dataset_hash": "abc123",
        "config_fingerprint": "deadbeef",
    }
    (tmp_path / "validated").mkdir()
    write_json(tmp_path / "validated" / "report.json", report)
    splits.mkdir(exist_ok=True)
    for name in ("train", "validation", "test"):
        write_jsonl(splits / f"{name}.jsonl", [{"messages": []}])

    config_path = tmp_path / "stale.yaml"
    config_path.write_text(f"""run: {{name: stale}}
model: {{name: LiquidAI/LFM2.5-350M-Base, local_files_only: true}}
dataset:
  path: {corpus_b}
  processed_dir: {splits}
  use_processed: true
  quality: {{min_assistant_chars: 1, near_duplicate: false, remove_prompt_leakage: false}}
tokenization: {{max_seq_length: 256}}
lora: {{enabled: true, r: 8, alpha: 16, target_modules: auto}}
training: {{preset: smoke, seed: 1}}
""")

    checks, _ = run_preflight(str(config_path), overrides={})
    assert "splits / provenance" in blocking(checks)


def test_matching_splits_pass(healthy, tmp_path) -> None:
    """The provenance gate must not fire when the splits really do match."""
    splits = tmp_path / "splits"
    splits.mkdir()
    for name in ("train", "validation", "test"):
        write_jsonl(splits / f"{name}.jsonl", [{"messages": []}])
    (tmp_path / "validated").mkdir()
    write_json(tmp_path / "validated" / "report.json",
               {"source": str(healthy), "dataset_hash": "abc", "config_fingerprint": "fp"})

    config_path = _config_for(tmp_path, healthy, name="match.yaml")
    config_path.write_text(config_path.read_text().replace(
        "use_processed: false", f"use_processed: true\n  processed_dir: {splits}"))

    checks, _ = run_preflight(str(config_path), overrides={})
    assert "splits / provenance" not in blocking(checks)


# --------------------------------------------------------------------------- #
# Config / hardware gates
# --------------------------------------------------------------------------- #

def test_invalid_config_fails_fast(healthy, tmp_path) -> None:
    from lfm_finetune.utils.errors import ConfigError

    bad = tmp_path / "bad.yaml"
    bad.write_text("run: {name: x}\nmodel: {name: m}\ndataset: {path: d}\n"
                   "lora: {r: 999, alpha: 1, target_modules: auto}\n")
    with pytest.raises(ConfigError):
        run_preflight(str(bad), overrides={})


def test_full_finetune_warns(healthy, tmp_path) -> None:
    path = _config_for(tmp_path, healthy, name="full.yaml")
    path.write_text(path.read_text().replace(
        "lora: {enabled: true",
        "lora: {enabled: false, target_modules: auto, r: 8, alpha: 16"))
    checks, _ = run_preflight(str(path), overrides={})
    assert statuses(checks)["config / LoRA enabled"] == "WARN"


def test_vram_estimate_is_reported(healthy, tmp_path) -> None:
    _, summary = run_preflight(str(_config_for(tmp_path, healthy)), overrides={})
    vram = summary["vram_estimate"]
    assert vram["model"] == "LiquidAI/LFM2.5-350M-Base"
    assert vram["peak_gib"] > 0
    assert vram["target"] in ("vram", "system_ram")


def test_6gb_config_is_in_repo_and_valid() -> None:
    """The laptop config referenced in the docs must keep working."""
    checks, summary = run_preflight(
        str(REPO / "configs/agriculture/lfm25_1.2b_6gb.yaml"), overrides={})
    st = {c.name: c.status for c in checks}
    assert st["config / LoRA enabled"] == "PASS"
    t = next(c for c in checks if c.name == "config / effective batch")
    assert "= 16" in t.detail, "effective batch must stay 16 to match the 12GB config"
    assert summary["vram_estimate"]["peak_gib"] < 6.0