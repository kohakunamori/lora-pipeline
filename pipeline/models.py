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
    INTERRUPTED = "interrupted"
    DONE = "done"
    SKIPPED = "skipped"
    FAILED = "failed"


# Persisted only so old Projects and explicit legacy CLI/API calls keep working
# while DatasetWorkspace owns curation going forward.
PROJECT_UTILITY_STEPS = (
    "inspect",
    "dedup",
    "identity",
    "caption",
    "review",
)

# The actual Project training lifecycle. Caption generation/normalization is an
# input transform of `prepare` (future name: materialize), rather than a separate
# state-machine stage. Results are also kept outside this lifecycle.
PROJECT_RUN_STEPS = (
    "prepare",
    "preflight",
    "train",
)

# Result operations are repeatable derivatives of a completed run and belong to
# the Results area, not Project step navigation.
PROJECT_RESULT_STEPS = (
    "evaluate",
)

# User-facing Project steps. Dataset curation and Results operations are both
# intentionally excluded from the Project step menu.
STEP_NAMES = PROJECT_RUN_STEPS

# Full persisted/dispatch namespace, including legacy compatibility records and
# repeatable Results operations.
ALL_STEP_NAMES = PROJECT_UTILITY_STEPS + PROJECT_RUN_STEPS + PROJECT_RESULT_STEPS
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
    target_images_seen: int
    resume_state: Path | None = None
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
    case_id: str
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
