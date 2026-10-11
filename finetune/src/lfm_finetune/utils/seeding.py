"""Deterministic seeding for Python / NumPy / PyTorch (+ CUDA, when present)."""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass
from typing import Any

from .errors import PipelineError
from .logging import get_logger

log = get_logger("utils.seeding")

# cuBLAS requires this to be set *before* the first CUDA context for deterministic
# GEMMs. We set it at import time of the training entrypoint, not here, because
# importing torch is expensive -- see `ensure_deterministic_cuda_algos`.
_ALGORITHMS = {
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}


@dataclass(frozen=True)
class SeedState:
    """Everything needed to reproduce the random streams of one run."""

    seed: int
    python_hash_seed: int | None
    torch_seed: int
    numpy_seed: int
    cuda_seeds: list[int] = None  # type: ignore[assignment]
    deterministic_algorithms: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ensure_deterministic_cuda_algos(strict: bool = False) -> None:
    """Pin cuBLAS/cuDNN for reproducibility.

    Must run before any CUDA tensor is created, hence the explicit call site in
    the CLI entrypoint rather than lazy behaviour inside :func:`set_seed`.
    """
    for key, value in _ALGORITHMS.items():
        if key.startswith("CUBLAS"):
            os.environ.setdefault(key, value)
        else:
            try:
                import torch

                setattr(torch.backends.cudnn, key, value)
            except Exception:  # pragma: no cover - CPU-only environments
                pass
    if strict:
        try:
            import torch

            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # pragma: no cover
            log.warning("Could not enable torch.use_deterministic_algorithms")


def set_seed(seed: int, *, deterministic_algorithms: bool = False) -> SeedState:
    """Seed every RNG the pipeline can reach and return the resulting state.

    Args:
        seed: Master seed. Dataset split seeds are derived from it (see
            :func:`derive_seed`) so a single number reproduces the whole run.
        deterministic_algorithms: Also request
            ``torch.use_deterministic_algorithms(warn_only=True)``. This can
            raise for ops lacking a deterministic CUDA kernel; ``warn_only``
            keeps training alive instead of crashing at step 1.

    Returns:
        A :class:`SeedState` snapshot suitable for the run manifest.
    """
    if seed < 0:
        raise PipelineError(f"seed must be non-negative, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    numpy_seed = seed
    try:
        import numpy as np

        np.random.seed(numpy_seed)
    except ImportError:  # pragma: no cover
        pass

    cuda_seeds: list[int] = []
    try:
        import torch

        # torch.manual_seed returns a Generator object, not an int. Assigning it
        # to torch_seed and then using it in a %d format string raises, so the
        # seed value itself is what gets recorded.
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            cuda_seeds = [torch.cuda.initial_seed()]
    except ImportError:  # pragma: no cover
        pass

    if deterministic_algorithms:
        ensure_deterministic_cuda_algos(strict=True)

    state = SeedState(
        seed=seed,
        python_hash_seed=int(os.environ.get("PYTHONHASHSEED", seed)),
        torch_seed=seed,
        numpy_seed=numpy_seed,
        cuda_seeds=cuda_seeds,
        deterministic_algorithms=deterministic_algorithms,
    )
    log.info(
        "Seeded RNGs: python=%d numpy=%d torch=%d cuda=%s deterministic_algorithms=%s",
        seed,
        numpy_seed,
        seed,
        cuda_seeds or "n/a",
        deterministic_algorithms,
    )
    return state


def derive_seed(master_seed: int, *namespace: str) -> int:
    """Derive a stable child seed from ``master_seed`` plus a string namespace.

    Uses BLAKE2b rather than :func:`hash` because Python's string hashing is
    randomised per process (PYTHONHASHSEED), which would silently break split
    reproducibility across runs.
    """
    import hashlib

    payload = f"{master_seed}::" + "::".join(namespace).encode()
    digest = hashlib.blake2b(payload, digest_size=4).digest()
    return int.from_bytes(digest, "big")