"""Shared pytest fixtures.

Design constraint: the default test suite must run on a **CPU-only machine with
no GPU and small RAM**. Tests that need to download a real checkpoint are marked
``slow`` and are skipped unless a checkpoint is already cached or
``LFM_RUN_SLOW=1`` is set.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("NO_COLOR", "1")

#: The tiny checkpoint used by every test that touches a real model. Overridable
#: so CI can point at a different (still small) LFM2.5 checkpoint.
SMOKE_MODEL = os.environ.get("LFM_TEST_MODEL", "LiquidAI/LFM2.5-350M-Base")

RUN_SLOW = os.environ.get("LFM_RUN_SLOW", "1") not in ("0", "false", "False")


def _model_is_cached(model_id: str) -> bool:
    """True when the checkpoint is already in the local HF cache."""
    cache = Path(
        os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    ) / "hub"
    if not cache.exists():
        return False
    folder = "models--" + model_id.replace("/", "--")
    target = cache / folder / "snapshots"
    if not target.exists():
        return False
    return any((snap / "config.json").exists() for snap in target.iterdir())


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: downloads/loads a real LFM checkpoint")


@pytest.fixture(scope="session")
def smoke_model_id() -> str:
    return SMOKE_MODEL


@pytest.fixture(scope="session")
def requires_model() -> None:
    """Skip a test unless a real checkpoint can be used."""
    if not RUN_SLOW:
        pytest.skip("set LFM_RUN_SLOW=1 to run tests that need a real checkpoint")
    if not _model_is_cached(SMOKE_MODEL):
        pytest.skip(
            f"{SMOKE_MODEL} is not in the local HF cache; "
            f"run `uv run lfm-finetune inspect-model -c configs/smoke.yaml` first, "
            f"or set LFM_TEST_MODEL to a cached checkpoint"
        )


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def synthetic_raw(tmp_path: Path) -> Path:
    """A tiny raw dataset with known-good and known-bad rows."""
    from lfm_finetune.data.synthetic import write_synthetic_dataset

    write_synthetic_dataset(tmp_path / "raw", num_examples=24, seed=7)
    return tmp_path


@pytest.fixture
def base_config_dict() -> dict:
    """A minimal valid config mapping, loadable without touching the filesystem."""
    return {
        "run": {"name": "test-run", "domain": "test"},
        "model": {"name": SMOKE_MODEL},
        "dataset": {"path": "data/raw/synthetic", "use_processed": False},
        "tokenization": {"max_seq_length": 256},
        "lora": {"enabled": True, "r": 4, "alpha": 8, "target_modules": "auto"},
        "training": {"preset": "smoke", "seed": 1},
    }