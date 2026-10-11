"""Run directories and experiment metadata.

Every run gets a unique, self-describing directory:

.. code-block:: text

    checkpoints/agri_lfm25_1.2b_20260101-120000/
    |- resolved_config.yaml     # fully-resolved config; no external YAML needed
    |- run_metadata.json        # env, versions, git, hardware, seeds, hashes
    |- dataset_report.json      # counts, stats, warnings from `prepare`
    |- metrics.jsonl            # streamed training metrics (local, no service)
    |- train.log                # full DEBUG-level log
    |- checkpoint-20/           # Trainer checkpoints (resumable)
    `- adapter/                 # best LoRA adapter (lightweight artifact)

The rule enforced here: **a run directory must be sufficient to reproduce and to
audit the run without consulting anything outside itself.** If a field would
require reading mutable external state later, it is frozen here at write time.
"""

from __future__ import annotations

import json
import os
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..utils.hashing import hash_obj, short_hash
from ..utils.io import ensure_dir, write_json, write_yaml
from ..utils.logging import get_logger
from ..utils.seeding import SeedState
from ..utils.versions import environment_snapshot

log = get_logger("training.metadata")

METADATA_FILENAME = "run_metadata.json"
METRICS_FILENAME = "metrics.jsonl"


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class RunDirectory:
    """Creates and owns a single training run's directory layout."""

    def __init__(
        self,
        root: Path | str,
        run_name: str,
        *,
        checkpointing_dir: Path | str | None = None,
        exact_dir: Path | str | None = None,
    ) -> None:
        """Create a run directory.

        Args:
            root: ``checkpointing.dir`` from the config.
            run_name: Experiment name. A UTC timestamp is appended to it.
            checkpointing_dir: Where the directory is created.
            exact_dir: Use this directory verbatim -- no run name, no timestamp.
                Used when resuming: the run already owns a directory, and nesting a
                second timestamped directory inside it would fragment
                ``metrics.jsonl`` and leave the original ``resolved_config.yaml``
                describing a run it no longer contains.
        """
        if exact_dir is not None:
            self.path = ensure_dir(Path(exact_dir))
            self.run_name = self.path.name
        else:
            self.run_name = f"{run_name}_{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
            base = Path(checkpointing_dir) if checkpointing_dir else Path(root)
            self.path = base / self.run_name
            ensure_dir(self.path)
        self.log_path = self.path / "train.log"
        self.adapter_dir = self.path / "adapter"
        self._metadata: dict[str, Any] = {}

    # -- layout ---------------------------------------------------------- #

    @property
    def config_path(self) -> Path:
        return self.path / "resolved_config.yaml"

    @property
    def metadata_path(self) -> Path:
        return self.path / METADATA_FILENAME

    @property
    def metrics_path(self) -> Path:
        return self.path / METRICS_FILENAME

    @property
    def dataset_report_path(self) -> Path:
        return self.path / "dataset_report.json"

    def checkpoint_dirs(self) -> list[Path]:
        return sorted(self.path.glob("checkpoint-*"),
                      key=lambda p: int(p.name.rsplit("-", 1)[-1])
                      if p.name.rsplit("-", 1)[-1].isdigit() else 0)

    def latest_checkpoint(self) -> Path | None:
        dirs = self.checkpoint_dirs()
        return dirs[-1] if dirs else None

    # -- writes ---------------------------------------------------------- #

    def write_config(self, config: Any) -> Path:
        return write_yaml(self.config_path, config.to_dict())

    def write_dataset_report(self, report: dict[str, Any]) -> Path:
        return write_json(self.dataset_report_path, report)

    def append_metric(self, record: dict[str, Any]) -> None:
        """Stream one metric record as JSONL (crash-safe, tail-able)."""
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"timestamp": utc_now(), **record}, default=str) + "\n")

    def read_metrics(self) -> list[dict[str, Any]]:
        if not self.metrics_path.exists():
            return []
        out = []
        for line in self.metrics_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    # -- metadata -------------------------------------------------------- #

    def build_metadata(
        self,
        *,
        config: Any,
        seed_state: SeedState | None,
        model_info: dict[str, Any],
        dataset_info: dict[str, Any],
        lora_info: dict[str, Any],
        tokenization_info: dict[str, Any],
        hardware: Any = None,
        vram_estimate: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Assemble the full reproducibility record for this run."""
        cfg_dict = config.to_dict()
        metadata: dict[str, Any] = {
            "schema_version": "1.0",
            "run_name": self.run_name,
            "run_id": short_hash(hash_obj([self.run_name, cfg_dict]), 10),
            "created_at": utc_now(),
            "command": {
                "argv": os.sys.argv,
                "reproduce": self.reproduce_command(config),
            },
            "config_fingerprint": config.fingerprint(),
            "run": cfg_dict["run"],
            "model": model_info,
            "dataset": dataset_info,
            "lora": lora_info,
            "tokenization": tokenization_info,
            "training_plan": cfg_dict["training"],
            "reproducibility": cfg_dict["reproducibility"],
            "environment": environment_snapshot(
                include_hardware=bool(cfg_dict["reproducibility"].get("record_hardware", True))
            ),
            "seeds": seed_state.to_dict() if seed_state else None,
            "vram_estimate": vram_estimate,
            "host": {
                "hostname": platform.node(),
                "cpu_count": os.cpu_count(),
            },
        }
        if extra:
            metadata.update(extra)
        self._metadata = metadata
        return metadata

    def write_metadata(self, metadata: dict[str, Any] | None = None) -> Path:
        return write_json(self.metadata_path, metadata or self._metadata)

    def update_metadata(self, **updates: Any) -> dict[str, Any]:
        """Merge new facts (final loss, duration, ...) into the metadata."""
        current: dict[str, Any] = {}
        if self.metadata_path.exists():
            try:
                current = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                current = {}
        current.update(updates)
        current["updated_at"] = utc_now()
        write_json(self.metadata_path, current)
        self._metadata = current
        return current

    def reproduce_command(self, config: Any) -> str:
        """The exact command that reproduces this run."""
        return (
            f"uv run lfm-finetune train --config {config.run.name}.yaml "
            f"--set reproducibility.seed={config.training.seed} "
            f"# see {self.metadata_path} for the fully-resolved config"
        )

    def summary(self) -> str:
        return "\n".join(
            [
                f"run directory        {self.path}",
                f"resolved config      {self.config_path}",
                f"metadata             {self.metadata_path}",
                f"metrics              {self.metrics_path}",
                f"log                  {self.log_path}",
                f"adapter (best)       {self.adapter_dir}",
            ]
        )


# --------------------------------------------------------------------------- #
# Optional Weights & Biases integration
# --------------------------------------------------------------------------- #

def resolve_reporting(report_to: list[str], run_name: str) -> tuple[list[str], dict[str, Any]]:
    """Resolve which integrations to enable, never failing without a service.

    ``WANDB_API_KEY`` absent -> W&B is silently disabled with an INFO message.
    Local JSONL logging is always on, so the pipeline never *requires* an
    external service to execute.
    """
    resolved: list[str] = []
    extra: dict[str, Any] = {}
    for target in report_to:
        name = target.strip().lower()
        if name in ("none", ""):
            continue
        if name == "wandb":
            if not os.environ.get("WANDB_API_KEY"):
                log.info(
                    "report_to includes 'wandb' but WANDB_API_KEY is unset; skipping W&B. "
                    "Local metrics.jsonl logging remains active."
                )
                continue
            try:
                import wandb  # noqa: F401
            except ImportError:
                log.info("wandb is not installed; skipping W&B reporting.")
                continue
            project = os.environ.get("WANDB_PROJECT", "lfm-sft")
            resolved.append("wandb")
            extra["wandb"] = {
                "project": project,
                "entity": os.environ.get("WANDB_ENTITY"),
                "name": run_name,
                "mode": os.environ.get("WANDB_MODE", "offline"),
            }
        elif name in ("tensorboard", "mlflow", "comet", "clearml", "trackio"):
            try:
                __import__(name)
                resolved.append(name)
            except ImportError:
                log.info("Integration %r requested but not installed; skipping.", name)
        else:
            log.warning("Unknown report_to target %r; ignoring.", name)
    return resolved, extra


def local_metric_summary(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract the headline numbers from a metrics stream."""
    train_losses = [m["loss"] for m in metrics if "loss" in m]
    eval_losses = [m["eval_loss"] for m in metrics if "eval_loss" in m]
    learning_rates = [m["learning_rate"] for m in metrics if "learning_rate" in m]
    return {
        "final_train_loss": train_losses[-1] if train_losses else None,
        "min_train_loss": min(train_losses) if train_losses else None,
        "final_eval_loss": eval_losses[-1] if eval_losses else None,
        "best_eval_loss": min(eval_losses) if eval_losses else None,
        "num_logged_steps": len(metrics),
        "max_learning_rate": max(learning_rates) if learning_rates else None,
        "final_learning_rate": learning_rates[-1] if learning_rates else None,
    }