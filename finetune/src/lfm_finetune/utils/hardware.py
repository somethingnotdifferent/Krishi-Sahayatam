"""Hardware discovery, dtype capability probing and VRAM feasibility checks.

This module is the single source of truth for "can this machine run that run?".
It is used by ``scripts/inspect_hardware.py``, by the CLI's preflight stage, and
by the fail-fast guards in :mod:`lfm_finetune.model.loader`.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .errors import HardwareError
from .logging import get_logger

log = get_logger("utils.hardware")

#: Bytes of VRAM a LoRA SFT run needs *beyond* the resident weights. Measured
#: empirically for LFM2.5 at short sequence lengths; deliberately generous
#: because activation memory scales with ``batch * seq_len * hidden * layers``.
#: See docs/TROUBLESHOOTING.md for the derivation.
ACTIVATION_VRAM_FUDGE: dict[int, float] = {
    # max_seq_length -> multiplier on (batch_size * seq_len) for activation budget
    512: 2.0e9,
    1024: 4.0e9,
    2048: 8.0e9,
    4096: 1.6e10,
    8192: 3.2e10,
}


@dataclass
class GPUInfo:
    index: int
    name: str
    total_memory_bytes: int
    free_memory_bytes: int
    capability: tuple[int, int]
    supports_bf16: bool
    supports_fp16: bool
    supports_tf32: bool
    driver_version: str | None = None
    compute_capability_str: str = ""

    @property
    def total_memory_gb(self) -> float:
        return self.total_memory_bytes / 1024**3

    @property
    def free_memory_gb(self) -> float:
        return self.free_memory_bytes / 1024**3

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["total_memory_gb"] = round(self.total_memory_gb, 2)
        data["free_memory_gb"] = round(self.free_memory_gb, 2)
        data["capability"] = list(self.capability)
        return data


@dataclass
class HardwareReport:
    platform: str
    machine: str
    processor: str
    python_version: str
    python_executable: str
    cpu_count_logical: int
    cpu_count_physical: int | None
    cpu_model: str
    system_ram_bytes: int
    system_ram_available_bytes: int
    disk_total_bytes: int
    disk_free_bytes: int
    disk_mount: str
    torch_version: str | None
    torch_cuda_available: bool
    torch_cuda_version: str | None
    torch_cudnn_version: str | None
    nvidia_driver_version: str | None
    gpus: list[GPUInfo] = field(default_factory=list)
    hf_cache_dir: str = ""
    hf_cache_free_bytes: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def system_ram_gb(self) -> float:
        return self.system_ram_bytes / 1024**3

    @property
    def disk_free_gb(self) -> float:
        return self.disk_free_bytes / 1024**3

    @property
    def total_vram_gb(self) -> float:
        return sum(g.total_memory_bytes for g in self.gpus) / 1024**3

    def supports(self, dtype: str) -> bool:
        """Whether ``dtype`` is usable for *training* on this machine."""
        if not self.gpus:
            return dtype in ("float32", "fp32")
        if dtype in ("bfloat16", "bf16"):
            return all(g.supports_bf16 for g in self.gpus)
        if dtype in ("float16", "fp16"):
            return all(g.supports_fp16 for g in self.gpus)
        return True

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["system_ram_gb"] = round(self.system_ram_gb, 2)
        data["disk_free_gb"] = round(self.disk_free_gb, 2)
        data["total_vram_gb"] = round(self.total_vram_gb, 2)
        data["gpus"] = [g.to_dict() for g in self.gpus]
        return data

    def summary_lines(self) -> list[str]:
        lines = [
            f"platform            {self.platform} ({self.machine})",
            f"python              {self.python_version}  [{Path(self.python_executable).name}]",
            f"cpu                 {self.cpu_model}",
            f"cpu cores           {self.cpu_count_physical or self.cpu_count_logical} physical "
            f"/ {self.cpu_count_logical} logical",
            f"system RAM          {self.system_ram_gb:.1f} GiB total, "
            f"{self.system_ram_available_bytes / 1024**3:.1f} GiB available",
            f"disk                {self.disk_free_gb:.1f} GiB free on {self.disk_mount}",
            f"torch               {self.torch_version or 'NOT INSTALLED'}",
        ]
        if self.gpus:
            for g in self.gpus:
                lines.append(
                    f"GPU[{g.index}]             {g.name}, {g.total_memory_gb:.1f} GiB VRAM "
                    f"({g.free_memory_gb:.1f} GiB free), cc {g.compute_capability_str}"
                )
            lines.append(f"cuda runtime        {self.torch_cuda_version} "
                         f"(driver {self.nvidia_driver_version}, cudnn {self.torch_cudnn_version})")
            lines.append(
                "bf16 support        "
                + ("yes" if all(g.supports_bf16 for g in self.gpus) else "NO -- use float16/float32")
            )
            lines.append(
                "fp16 support        " + ("yes" if all(g.supports_fp16 for g in self.gpus) else "NO")
            )
        else:
            lines.append("GPU                 NONE -- CUDA is not available to torch")
            lines.append("dtype support       float32 only (bf16/fp16 require a CUDA device)")
        if self.hf_cache_free_bytes:
            lines.append(
                f"HF cache            {self.hf_cache_dir} "
                f"({self.hf_cache_free_bytes / 1024**3:.1f} GiB free)"
            )
        for w in self.warnings:
            lines.append(f"WARNING             {w}")
        return lines


def _run(cmd: list[str]) -> str | None:
    """Best-effort shell probe; returns None if the tool is absent."""
    exe = shutil.which(cmd[0])
    if exe is None:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=False)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _meminfo(key: str) -> int:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith(key):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _disk_for(path: str) -> tuple[int, int]:
    usage = shutil.disk_usage(path)
    return usage.total, usage.free


def _nvidia_driver() -> str | None:
    out = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    return out.splitlines()[0].strip() if out else None


def probe_gpus() -> tuple[list[GPUInfo], str | None, str | None, str | None]:
    """Enumerate CUDA devices via torch (not nvidia-smi) and probe dtype support.

    Returns ``(gpus, torch_cuda_version, cudnn_version, driver_version)``.
    """
    try:
        import torch
    except ImportError:
        return [], None, None, None

    cudnn = None
    try:
        if torch.backends.cudnn.is_available():
            cudnn = str(torch.backends.cudnn.version())
    except Exception:  # pragma: no cover
        pass

    if not torch.cuda.is_available():
        return [], None, cudnn, None

    gpus: list[GPUInfo] = []
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        major, minor = torch.cuda.get_device_capability(idx)
        free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
        gpus.append(
            GPUInfo(
                index=idx,
                name=props.name,
                total_memory_bytes=total_bytes,
                free_memory_bytes=free_bytes,
                capability=(major, minor),
                # Ampere (8.0) and newer have native bf16. Checking via the
                # capability matrix is more reliable than torch.cuda.is_bf16_supported()
                # on multi-GPU nodes with heterogeneous cards.
                supports_bf16=major >= 8,
                supports_fp16=major >= 6,
                supports_tf32=major >= 8,
                compute_capability_str=f"{major}.{minor}",
            )
        )
    return gpus, torch.version.cuda, cudnn, _nvidia_driver()


def _physical_cores() -> int | None:
    out = _run(["lscpu", "-p=Core,Socket"])
    if out is None:
        return None
    cores: set[tuple[str, str]] = set()
    for line in out.splitlines():
        if line.startswith("#") or "," not in line:
            continue
        socket, core = line.split(",")[:2]
        cores.add((socket, core))
    return len(cores) or None


def audit(max_download_gb: float = 2.0) -> HardwareReport:
    """Collect a full hardware report.

    ``max_download_gb`` is the largest checkpoint we expect to fetch; if free
    disk cannot hold the weights *and* a training working copy, a warning is
    attached rather than an immediate crash (the download cache may be
    pre-populated).
    """
    gpus, cuda_ver, cudnn_ver, driver = probe_gpus()
    try:
        import torch

        torch_version = torch.__version__
    except ImportError:
        torch_version = None

    hf_cache = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")

    disk_total, disk_free = _disk_for(str(Path.cwd().anchor or "/"))
    hf_free = _disk_for(hf_cache)[1] if Path(hf_cache).exists() else disk_free

    report = HardwareReport(
        platform=f"{platform.system()} {platform.release()}",
        machine=platform.machine(),
        processor=platform.processor(),
        python_version=platform.python_version(),
        python_executable=sys.executable,
        cpu_count_logical=os.cpu_count() or 1,
        cpu_count_physical=_physical_cores(),
        cpu_model=_cpu_model(),
        system_ram_bytes=_meminfo("MemTotal:") or (1 << 31),
        system_ram_available_bytes=_meminfo("MemAvailable:"),
        disk_total_bytes=disk_total,
        disk_free_bytes=disk_free,
        disk_mount=str(Path.cwd().anchor or "/"),
        torch_version=torch_version,
        torch_cuda_available=bool(gpus),
        torch_cuda_version=cuda_ver,
        torch_cudnn_version=cudnn_ver,
        nvidia_driver_version=driver,
        gpus=gpus,
        hf_cache_dir=hf_cache,
        hf_cache_free_bytes=hf_free,
    )

    if not gpus:
        report.warnings.append(
            "No CUDA device. Full GPU SFT cannot run here; use a smoke config on CPU "
            "or move the run to a CUDA host."
        )
    if report.disk_free_bytes < max_download_gb * 1024**3:
        report.warnings.append(
            f"Only {report.disk_free_gb:.1f} GiB free on {report.disk_mount}; "
            f"a {max_download_gb:.0f} GiB checkpoint plus working copies may not fit."
        )
    if report.system_ram_available_bytes < 4 * 1024**3 and not gpus:
        report.warnings.append(
            f"Only {report.system_ram_available_bytes / 1024**3:.1f} GiB RAM available; "
            "CPU training is limited to the smallest checkpoints (230M-350M)."
        )
    return report


# --------------------------------------------------------------------------- #
# VRAM feasibility
# --------------------------------------------------------------------------- #

def estimate_training_vram_bytes(
    *,
    param_count: int,
    dtype: str,
    batch_size: int,
    grad_accum: int,
    max_seq_length: int,
    gradient_checkpointing: bool,
    trainable_params: int,
    world_size: int = 1,
) -> dict[str, int]:
    """Estimate peak training memory for LoRA SFT.

    Components:

    * **Weights** -- ``param_count * bytes_per_element``, resident for the whole run.
    * **Gradients** -- only for trainable (LoRA) parameters.
    * **Optimizer state** -- AdamW keeps 2 fp32 moments per trainable parameter,
      plus (in the default HF setup) no fp32 master copy for LoRA because the
      adapter itself is stored in fp32.
    * **Activations** -- the dominant and most seq-length-sensitive term. If
      gradient checkpointing is on, only layer *inputs* are stored, which cuts it
      roughly proportional to depth; otherwise every layer's intermediates are
      kept.
    * **Logits** -- ``vocab_size`` dominates activation memory for LFM2.5
      (65,536 vocab at 1.2B). fp32 log-softmax over the full sequence is
      frequently the single largest term; it scales with ``batch * seq``.

    This is an estimate used for *fail-fast gating and reporting*, not for
    capacity planning. Treat the output as an order-of-magnitude check.
    """
    dtype_bytes = {"bfloat16": 2, "bf16": 2, "float16": 2, "fp16": 2, "float32": 4, "fp32": 4}[
        dtype.lower()
    ]
    weights = param_count * dtype_bytes
    grads = trainable_params * dtype_bytes
    optimizer = trainable_params * 4 * 2  # AdamW m and v in fp32

    # Activation term: hidden_size is approximated from LFM2.5 block_dim=2048.
    hidden = 2048
    token_budget = batch_size * max_seq_length
    if gradient_checkpointing:
        activations = int(token_budget * hidden * dtype_bytes * 2)  # ~2 stored tensors/layer-block
    else:
        activations = int(token_budget * hidden * dtype_bytes * 12)  # ~12 per layer-equivalent
    vocab_logits = int(token_budget * 65536 * 4)  # fp32 upcast for CE loss

    peak = weights + grads + optimizer + activations + vocab_logits
    return {
        "weights_bytes": weights,
        "gradients_bytes": grads,
        "optimizer_bytes": optimizer,
        "activations_bytes": activations,
        "logits_bytes": vocab_logits,
        "peak_bytes": peak,
        "per_device_peak_bytes": peak // max(1, world_size),
    }


def _activation_budget(max_seq_length: int) -> float:
    """Nearest tabulated activation budget for ``max_seq_length``."""
    keys = sorted(ACTIVATION_VRAM_FUDGE)
    for key in keys:
        if key >= max_seq_length:
            return ACTIVATION_VRAM_FUDGE[key]
    return ACTIVATION_VRAM_FUDGE[keys[-1]] * (max_seq_length / keys[-1])


def assert_vram_sufficient(
    report: HardwareReport,
    *,
    param_count: int,
    dtype: str,
    batch_size: int,
    grad_accum: int,
    max_seq_length: int,
    trainable_params: int,
    gradient_checkpointing: bool = False,
    world_size: int = 1,
    headroom_fraction: float = 0.10,
) -> dict[str, Any]:
    """Fail fast when the requested run cannot fit, otherwise return the plan.

    Raises:
        HardwareError: when GPU training is requested with no CUDA device, when
            the dtype is unsupported, or when the estimated peak exceeds
            ``(1 - headroom_fraction)`` of usable VRAM.
    """
    estimate = estimate_training_vram_bytes(
        param_count=param_count,
        dtype=dtype,
        batch_size=batch_size,
        grad_accum=grad_accum,
        max_seq_length=max_seq_length,
        gradient_checkpointing=gradient_checkpointing,
        trainable_params=trainable_params,
        world_size=world_size,
    )

    if not report.gpus:
        raise HardwareError(
            "GPU training requested but torch reports no CUDA device.",
            hint="Set training.device=cpu and training.dtype=float32 for a CPU smoke run, "
            "or run on a CUDA host. See configs/smoke.yaml for a working CPU example.",
        )

    if not report.supports(dtype):
        raise HardwareError(
            f"dtype '{dtype}' is not supported by the detected GPU(s) "
            f"(compute capability {[list(g.capability) for g in report.gpus]}).",
            hint="bf16 needs compute capability >= 8.0 (Ampere+). Use float16 or float32.",
        )

    usable = sum(g.free_memory_bytes for g in report.gpus) * (1.0 - headroom_fraction)
    peak = estimate["peak_bytes"]
    if peak > usable:
        need_gb, have_gb = peak / 1024**3, usable / 1024**3
        suggestions = []
        if gradient_checkpointing:
            suggestions.append("gradient_checkpointing is already on")
        else:
            suggestions.append("set training.gradient_checkpointing=true")
        if batch_size > 1:
            suggestions.append(
                f"lower training.per_device_train_batch_size (currently {batch_size}; "
                "gradient accumulation preserves the effective batch size)"
            )
        if max_seq_length > 512:
            suggestions.append(f"lower tokenization.max_seq_length (currently {max_seq_length})")
        if dtype.lower() in ("float32", "fp32"):
            suggestions.append("use bfloat16 instead of float32")
        raise HardwareError(
            f"Estimated peak training memory {need_gb:.1f} GiB exceeds usable VRAM "
            f"{have_gb:.1f} GiB (free {report.total_vram_gb - sum(g.total_memory_bytes - g.free_memory_bytes for g in report.gpus) / 1024**3:.1f} GiB across "
            f"{len(report.gpus)} GPU(s)).",
            hint="Reduce memory: " + "; ".join(suggestions) + ".",
        )

    estimate["usable_vram_bytes"] = int(usable)
    log.info(
        "VRAM feasibility OK: estimated peak %.2f GiB of %.2f GiB usable",
        estimate["peak_bytes"] / 1024**3,
        usable / 1024**3,
    )
    return estimate