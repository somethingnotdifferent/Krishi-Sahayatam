"""Typed error hierarchy.

Every failure the pipeline can detect is raised as a :class:`PipelineError`
subclass carrying (a) a human sentence describing what went wrong and (b) an
actionable remediation hint. Nothing in this codebase is allowed to fail with a
bare ``RuntimeError`` -- a training run that dies after 40 minutes must tell the
operator exactly what to change.
"""

from __future__ import annotations


class PipelineError(RuntimeError):
    """Base class for all recoverable, user-actionable pipeline failures."""

    exit_code: int = 1

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        self.message = message
        self.hint = hint
        super().__init__(self.render())

    def render(self) -> str:
        body = f"{self.__class__.__name__}: {self.message}"
        if self.hint:
            body += f"\n\n  Hint: {self.hint}"
        return body


class ConfigError(PipelineError):
    """Malformed, missing or semantically invalid configuration."""

    exit_code = 2


class EnvironmentError_(PipelineError):
    """Interpreter/package/device level problem (missing torch, no CUDA, ...)."""

    exit_code = 3


class HardwareError(PipelineError):
    """The machine cannot satisfy the requested run (VRAM, dtype, device)."""

    exit_code = 3


class ModelError(PipelineError):
    """Model/tokenizer could not be loaded, or is an unsupported architecture."""

    exit_code = 4


class TokenizerError(PipelineError):
    """Tokenizer or chat template is missing or unusable."""

    exit_code = 4


class DatasetError(PipelineError):
    """Raw data could not be read, parsed or conforms to the schema."""

    exit_code = 5


class DatasetLeakageError(DatasetError):
    """Train/validation/test overlap detected."""


class LoRAError(PipelineError):
    """LoRA target modules missing, unattached, or trainable-parameter audit failed."""

    exit_code = 6


class TrainingError(PipelineError):
    """Training could not start or crashed."""

    exit_code = 7


class CheckpointError(PipelineError):
    """Checkpoint / adapter missing, incomplete, or reload failed."""

    exit_code = 8


class ExportError(PipelineError):
    """Adapter export or merge failed."""

    exit_code = 9


class EvaluationError(PipelineError):
    """Evaluation could not run."""

    exit_code = 10