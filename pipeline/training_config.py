from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Mapping

from .config import load_base_registry, read_yaml, repository_root, stable_hash, write_yaml_atomic
from .dataset_workspace import DatasetWorkspace, create_project_from_dataset
from .models import PipelineError, StateError
from .state import ProjectState, utc_now


_CONFIG_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_STRATEGIES = {"quality", "fast", "cached"}
_CONCEPT_TYPES = {"character", "style"}
_CAPTION_MODES = {
    "auto",
    "generate",
    "existing_passthrough",
    "existing_taglist_clean",
    "hybrid",
    "skip",
}


def default_workflow(concept_type: str) -> dict[str, Any]:
    return {
        "run_dedup": True,
        "exclude_exact_duplicates": False,
        "run_identity": concept_type == "character",
        "caption_mode": "auto",
        "allow_trigger_only": False,
        "run_review": True,
        # Evaluation belongs to the Results area in the four-part UI.
        "run_screening_evaluation": False,
    }


class TrainingConfig:
    """A mutable, reusable recipe for creating immutable training runs.

    A TrainingConfig never owns a dataset snapshot. Dataset and config snapshots
    are frozen together only when a run workspace is created.
    """

    def __init__(self, path: Path, payload: dict[str, Any]):
        self.path = path
        self.payload = payload
        self._normalize()

    @classmethod
    def create(
        cls,
        name: str,
        *,
        concept_type: str,
        base: str,
        trigger: str,
        strategy: str = "quality",
        images_seen: int = 1000,
        hardware: str = "v100_16gb",
        overrides: Mapping[str, Any] | None = None,
        workflow: Mapping[str, Any] | None = None,
        evaluation: Mapping[str, Any] | None = None,
        root: Path | None = None,
    ) -> "TrainingConfig":
        path = training_config_path(name, root=root)
        if path.exists():
            raise StateError(f"Training config already exists: {name}")
        registry = load_base_registry(root or repository_root())
        if base not in registry or not registry[base].enabled:
            raise PipelineError(f"Base model is not registered and enabled: {base}")
        now = utc_now()
        payload = {
            "schema_version": 1,
            "config": {
                "name": name,
                "concept_type": concept_type,
                "base": base,
                "trigger": trigger.strip(),
                "strategy": strategy,
                "images_seen": int(images_seen),
                "hardware": hardware,
                "overrides": copy.deepcopy(dict(overrides or {})),
                "workflow": {
                    **default_workflow(concept_type),
                    **copy.deepcopy(dict(workflow or {})),
                },
                "evaluation": copy.deepcopy(dict(evaluation or {})),
                "created_at": now,
                "updated_at": now,
            },
        }
        config = cls(path, payload)
        config.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        config.save()
        return config

    @classmethod
    def load(cls, name: str, *, root: Path | None = None) -> "TrainingConfig":
        path = training_config_path(name, root=root)
        if not path.is_file():
            raise StateError(f"Training config does not exist: {name}")
        return cls(path, read_yaml(path))

    @property
    def data(self) -> dict[str, Any]:
        return self.payload["config"]

    @property
    def name(self) -> str:
        return str(self.data["name"])

    @property
    def concept_type(self) -> str:
        return str(self.data["concept_type"])

    @property
    def base(self) -> str:
        return str(self.data["base"])

    @property
    def trigger(self) -> str:
        return str(self.data["trigger"])

    @property
    def strategy(self) -> str:
        return str(self.data["strategy"])

    @property
    def images_seen(self) -> int:
        return int(self.data["images_seen"])

    @property
    def workflow(self) -> dict[str, Any]:
        return self.data["workflow"]

    @property
    def overrides(self) -> dict[str, Any]:
        return self.data["overrides"]

    @property
    def evaluation(self) -> dict[str, Any]:
        return self.data["evaluation"]

    def _normalize(self) -> None:
        self.payload.setdefault("schema_version", 1)
        data = self.payload.setdefault("config", {})
        concept = str(data.get("concept_type") or "character")
        data.setdefault("hardware", "v100_16gb")
        data.setdefault("strategy", "quality")
        data.setdefault("images_seen", 1000)
        data.setdefault("overrides", {})
        data.setdefault("evaluation", {})
        workflow = data.setdefault("workflow", {})
        for key, value in default_workflow(concept).items():
            workflow.setdefault(key, value)
        data.setdefault("created_at", utc_now())
        data.setdefault("updated_at", data["created_at"])

    def validate(self, *, require_enabled_base: bool = False, root: Path | None = None) -> None:
        data = self.data
        name = str(data.get("name") or "")
        if not _CONFIG_NAME.fullmatch(name) or name in {".", ".."}:
            raise StateError("Training config name must be 1-64 letters, numbers, '.', '_' or '-'")
        if str(data.get("concept_type")) not in _CONCEPT_TYPES:
            raise PipelineError("Training config concept type must be character or style")
        if str(data.get("strategy")) not in _STRATEGIES:
            raise PipelineError("Training strategy must be quality, fast, or cached")
        trigger = str(data.get("trigger") or "").strip()
        if not trigger or "," in trigger:
            raise PipelineError("Training config trigger must be non-empty and cannot contain a comma")
        if int(data.get("images_seen", 0)) < 1:
            raise PipelineError("Training config images_seen must be at least 1")
        caption_mode = str(data.get("workflow", {}).get("caption_mode", "auto"))
        if caption_mode not in _CAPTION_MODES:
            raise PipelineError(f"Unsupported training config caption mode: {caption_mode}")
        if require_enabled_base:
            registry = load_base_registry(root or repository_root())
            base = str(data.get("base") or "")
            if base not in registry or not registry[base].enabled:
                raise PipelineError(f"Base model is not registered and enabled: {base}")

    def save(self) -> None:
        self.validate()
        self.data["updated_at"] = utc_now()
        write_yaml_atomic(self.path, self.payload)

    def snapshot(self) -> dict[str, Any]:
        data = copy.deepcopy(self.data)
        data.pop("created_at", None)
        data.pop("updated_at", None)
        basis = {
            "schema_version": 1,
            "name": self.name,
            "concept_type": self.concept_type,
            "base": self.base,
            "trigger": self.trigger,
            "strategy": self.strategy,
            "images_seen": self.images_seen,
            "hardware": str(self.data.get("hardware", "v100_16gb")),
            "overrides": copy.deepcopy(self.overrides),
            "workflow": copy.deepcopy(self.workflow),
            "evaluation": copy.deepcopy(self.evaluation),
        }
        return {
            **basis,
            "snapshot_hash": stable_hash(basis),
            "snapshot_created_at": utc_now(),
            "source": str(self.path),
        }


def training_configs_root(root: Path | None = None) -> Path:
    return (root or repository_root()) / "training-configs"


def training_config_path(name: str, *, root: Path | None = None) -> Path:
    if not _CONFIG_NAME.fullmatch(name) or name in {".", ".."}:
        raise StateError("Training config name must be 1-64 letters, numbers, '.', '_' or '-'")
    return training_configs_root(root) / f"{name}.yaml"


def list_training_configs(*, root: Path | None = None) -> list[TrainingConfig]:
    base = training_configs_root(root)
    if not base.is_dir():
        return []
    result: list[TrainingConfig] = []
    for path in sorted(base.glob("*.yaml"), key=lambda item: item.name.casefold()):
        if path.name.startswith("."):
            continue
        result.append(TrainingConfig(path, read_yaml(path)))
    return result


def create_project_from_training_config(
    workspace: DatasetWorkspace,
    config: TrainingConfig,
    *,
    project_name: str,
    root: Path | None = None,
) -> ProjectState:
    """Freeze Dataset + TrainingConfig into an internal Project run workspace."""

    config.validate(require_enabled_base=True, root=root)
    if workspace.concept_type != config.concept_type:
        raise PipelineError(
            f"Dataset type {workspace.concept_type!r} is incompatible with training config type {config.concept_type!r}"
        )
    snapshot = config.snapshot()
    state = create_project_from_dataset(
        workspace,
        name=project_name,
        base=config.base,
        trigger=config.trigger,
        strategy=config.strategy,
        images_seen=config.images_seen,
        hardware=str(config.data.get("hardware", "v100_16gb")),
        root=root,
    )
    project = state.payload["project"]
    project["workspace_role"] = "training_run"
    project["training_config_snapshot"] = snapshot
    project["overrides"] = copy.deepcopy(config.overrides)
    project["evaluation"] = copy.deepcopy(config.evaluation)

    workflow = copy.deepcopy(config.workflow)
    if workflow.get("caption_mode") == "auto":
        dataset_snapshot = project["dataset_snapshot"]
        workflow["caption_mode"] = (
            "existing_taglist_clean"
            if dataset_snapshot.get("caption_count") == dataset_snapshot.get("image_count")
            else "generate"
        )
    # The Results area owns evaluation in the four-part UI.
    workflow["run_screening_evaluation"] = False
    if state.concept_type != "character":
        workflow["run_identity"] = False
    project["interactive_preferences"] = workflow
    project["training_identity"] = {
        "dataset": workspace.name,
        "dataset_snapshot_hash": project["dataset_snapshot"]["snapshot_hash"],
        "config": config.name,
        "config_snapshot_hash": snapshot["snapshot_hash"],
    }
    state.save()
    return state


def make_training_workspace_name(dataset_name: str, config_name: str, *, timestamp: str) -> str:
    def clean(value: str) -> str:
        value = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._") or "item"
        return value[:14]

    value = f"run-{clean(dataset_name)}-{clean(config_name)}-{timestamp}"
    return value[:64].rstrip("-._")
