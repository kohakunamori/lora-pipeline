from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping, Sequence


class ConceptType(StrEnum):
    CHARACTER = "character"
    STYLE = "style"


class StepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    SKIPPED = "skipped"
    FAILED = "failed"


STEP_NAMES = (
    "inspect",
    "dedup",
    "identity",
    "caption",
    "review",
    "prepare",
    "preflight",
    "train",
    "evaluate",
)

OPTIONAL_STEPS = frozenset({"dedup", "identity", "caption", "review", "evaluate"})


@dataclass(frozen=True)
class ProjectRef:
    name: str
    path: Path


@dataclass(frozen=True)
class BaseModel:
    id: str
    name: str
    path: Path
    family: str
    prediction_type: str
    sha256: str | None
    enabled: bool
    generation_defaults: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepResult:
    status: StepStatus = StepStatus.DONE
    input_hash: str | None = None
    output_manifest: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedConfig:
    hardware: Mapping[str, Any]
    concept: Mapping[str, Any]
    training: Mapping[str, Any]
    merged: Mapping[str, Any]


class PipelineError(RuntimeError):
    """Expected user-facing pipeline failure."""


class ConfigurationError(PipelineError):
    """Invalid or unsafe configuration."""


class StateError(PipelineError):
    """Invalid project state transition."""


class ExternalCommandError(PipelineError):
    def __init__(self, message: str, *, exit_code: int, log_path: Path):
        super().__init__(message)
        self.exit_code = exit_code
        self.log_path = log_path


class OptionalBackendUnavailable(PipelineError):
    """An explicitly requested optional model/backend is not available."""


@dataclass(frozen=True)
class TrainingRequest:
    project_dir: Path
    run_dir: Path
    base: BaseModel
    config: ResolvedConfig
    optimizer_steps: int
    command_line: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class TrainingResult:
    run_id: str
    run_dir: Path
    checkpoints: Sequence[Path]
    accounting: Mapping[str, Any]
    metrics: Mapping[str, Any] = field(default_factory=dict)
    dry_run: bool = False


@dataclass(frozen=True)
class GenerationCase:
    checkpoint: Path
    checkpoint_label: str
    strength: float
    prompt_id: str
    prompt: str
    negative_prompt: str
    seed: int
    contains_trigger: bool


@dataclass(frozen=True)
class GeneratedImage:
    case: GenerationCase
    path: Path
