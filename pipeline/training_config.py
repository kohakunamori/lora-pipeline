from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import (
    deep_merge,
    load_base_registry,
    read_yaml,
    repository_root,
    stable_hash,
    write_yaml_atomic,
)
from .dataset_workspace import DatasetWorkspace, create_project_from_dataset
from .models import PipelineError, StateError
from .state import ProjectState, utc_now
from .trigger_policy import TRIGGER_STRATEGIES, resolve_trigger_policy


_CONFIG_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_STRATEGIES = {"quality", "fast", "cached"}
_CONCEPT_TYPES = {"character", "style"}
_TRAINING_TARGETS = {"character", "character_outfit", "style"}
_TARGET_CONCEPT_TYPES = {
    "character": "character",
    "character_outfit": "character",
    "style": "style",
}
_CAPTION_MODES = {
    "auto",
    "generate",
    "existing_passthrough",
    "existing_taglist_clean",
    "hybrid",
    "skip",
}
_CHARACTER_OUTFIT_EVALUATION = {
    "screening_prompts": ["portrait", "full body", "different expression"],
    "prompts": [
        "portrait",
        "upper body",
        "full body",
        "different expression",
        "dynamic pose",
        "complex background",
        "indoor",
        "outdoor",
        "day",
        "night",
    ],
}


def default_workflow(concept_type: str) -> dict[str, Any]:
    del concept_type
    return {
        # Legacy curation utilities remain callable for old workspaces, but the
        # normal contract assumes input identity is already correct. Training only
        # needs materialization/captioning, preflight and train.
        "run_dedup": False,
        "exclude_exact_duplicates": False,
        "run_identity": False,
        "caption_mode": "auto",
        "allow_trigger_only": False,
        "run_review": False,
        "run_screening_evaluation": False,
    }


def training_target_concept(target_type: str) -> str:
    try:
        return _TARGET_CONCEPT_TYPES[target_type]
    except KeyError as exc:
        raise PipelineError(
            "Training target must be character, character_outfit, or style"
        ) from exc


def parse_anchor_tags(value: str | Sequence[str]) -> list[str]:
    raw_values = [value] if isinstance(value, str) else list(value)
    result: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        for part in re.split(r"[,\n]+", str(raw)):
            tag = re.sub(r"\s+", " ", part.strip())
            normalized = _normalize_prompt_tag(tag)
            if not tag or normalized in seen:
                continue
            seen.add(normalized)
            result.append(tag)
    return result


def prompt_contains_trigger(prompt: str, trigger: str) -> bool:
    trigger_normalized = _normalize_prompt_tag(trigger)
    if not trigger_normalized:
        return False
    parts = [
        _normalize_prompt_tag(part)
        for part in re.split(r"[,;\n]+", str(prompt))
        if part.strip()
    ]
    return trigger_normalized in parts


def _normalize_prompt_tag(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).replace("_", " ").strip()).casefold()


class TrainingConfig:
    """A mutable, reusable recipe for creating immutable training runs."""

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
        target_type: str | None = None,
        anchor_tags: Sequence[str] = (),
        trigger_strategy: str = "explicit",
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
        resolved_target = str(target_type or concept_type)
        anchors = parse_anchor_tags(anchor_tags)
        policy = resolve_trigger_policy(
            trigger,
            strategy=trigger_strategy,
            anchors=anchors,
        )
        payload = {
            "schema_version": 1,
            "config": {
                "name": name,
                "concept_type": concept_type,
                "target_type": resolved_target,
                "base": base,
                "trigger": policy.trigger,
                "trigger_strategy": policy.strategy,
                "trigger_requested": policy.requested,
                "anchor_tags": anchors,
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
        path = training_config_path(name, root=root) / Path("")
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
    def target_type(self) -> str:
        return str(self.data["target_type"])

    @property
    def base(self) -> str:
        return str(self.data["base"])

    @property
    def trigger(self) -> str:
        return str(self.data["trigger"])

    @property
    def trigger_strategy(self) -> str:
        return str(self.data.get("trigger_strategy", "explicit"))

    @property
    def anchor_tags(self) -> list[str]:
        return list(self.data.get("anchor_tags", []))

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

    @property
    def trigger_policy(self) -> dict[str, object]:
        return resolve_trigger_policy(
            str(self.data.get("trigger_requested") or self.trigger),
            strategy=self.trigger_strategy,
            anchors=self.anchor_tags,
        ).as_dict()

    def runtime_overrides(self) -> dict[str, Any]:
        target_defaults: dict[str, Any] = {}
        if self.target_type == "character_outfit":
            target_defaults = {
                "caption": {"anchor_tags": self.anchor_tags},
                "evaluation": copy.deepcopy(_CHARACTER_OUTFIT_EVALUATION),
            }
        return deep_merge(target_defaults, self.overrides)

    def effective_evaluation(self) -> dict[str, Any]:
        result = copy.deepcopy(self.evaluation)
        if self.target_type == "character_outfit":
            subject = str(result.get("subject_prompt") or "1girl")
            subject_tags = parse_anchor_tags([subject, *self.anchor_tags])
            result["subject_prompt"] = ", ".join(subject_tags)
        return result

    def _normalize(self) -> None:
        self.payload.setdefault("schema_version", 1)
        data = self.payload.setdefault("config", {})
        concept = str(data.get("concept_type") or "character")
        data.setdefault("target_type", concept)
        data["anchor_tags"] = parse_anchor_tags(data.get("anchor_tags", []))
        data.setdefault("trigger_strategy", "explicit")
        data.setdefault("trigger_requested", data.get("trigger", ""))
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
        concept_type = str(data.get("concept_type"))
        if concept_type not in _CONCEPT_TYPES:
            raise PipelineError("Training config concept type must be character or style")
        target_type = str(data.get("target_type") or concept_type)
        if target_type not in _TRAINING_TARGETS:
            raise PipelineError("Training target must be character, character_outfit, or style")
        if training_target_concept(target_type) != concept_type:
            raise PipelineError(
                f"Training target {target_type!r} is incompatible with concept type {concept_type!r}"
            )
        if str(data.get("strategy")) not in _STRATEGIES:
            raise PipelineError("Training strategy must be quality, fast, or cached")
        trigger = str(data.get("trigger") or "").strip()
        if not trigger or "," in trigger:
            raise PipelineError("Training config trigger must be non-empty and cannot contain a comma")
        trigger_strategy = str(data.get("trigger_strategy") or "explicit")
        if trigger_strategy not in TRIGGER_STRATEGIES:
            raise PipelineError(
                "Trigger strategy must be one of: " + ", ".join(TRIGGER_STRATEGIES)
            )

        anchors = parse_anchor_tags(data.get("anchor_tags", []))
        if target_type == "character_outfit" and not anchors:
            raise PipelineError(
                "Character outfit training requires at least one character anchor tag"
            )
        if target_type != "character_outfit" and anchors:
            raise PipelineError("Anchor tags are only valid for character_outfit training")
        if trigger_strategy == "multi_anchor" and target_type != "character_outfit":
            raise PipelineError("multi_anchor trigger strategy is only valid for character_outfit training")
        resolve_trigger_policy(
            str(data.get("trigger_requested") or trigger),
            strategy=trigger_strategy,
            anchors=anchors,
        )
        trigger_normalized = _normalize_prompt_tag(trigger)
        if any(_normalize_prompt_tag(anchor) == trigger_normalized for anchor in anchors):
            raise PipelineError("Character anchor tags must not repeat the LoRA trigger")

        subject_prompt = str(data.get("evaluation", {}).get("subject_prompt") or "").strip()
        if subject_prompt and prompt_contains_trigger(subject_prompt, trigger):
            raise PipelineError(
                "Evaluation subject prompt must not contain the LoRA trigger; "
                "trigger-on/off evaluation adds it automatically"
            )
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
        basis = {
            "schema_version": 1,
            "name": self.name,
            "concept_type": self.concept_type,
            "target_type": self.target_type,
            "base": self.base,
            "trigger": self.trigger,
            "trigger_strategy": self.trigger_strategy,
            "trigger_policy": self.trigger_policy,
            "anchor_tags": self.anchor_tags,
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
    project["training_target_type"] = config.target_type
    project["trigger_strategy"] = config.trigger_strategy
    project["trigger_policy"] = config.trigger_policy
    project["caption_anchor_tags"] = config.anchor_tags
    project["overrides"] = config.runtime_overrides()
    project["evaluation"] = config.effective_evaluation()

    workflow = copy.deepcopy(config.workflow)
    if workflow.get("caption_mode") == "auto":
        dataset_snapshot = project["dataset_snapshot"]
        workflow["caption_mode"] = (
            "existing_taglist_clean"
            if dataset_snapshot.get("caption_count") == dataset_snapshot.get("image_count")
            else "generate"
        )
    workflow["run_screening_evaluation"] = False
    workflow["run_identity"] = False
    workflow["run_dedup"] = False
    workflow["run_review"] = False
    project["interactive_preferences"] = workflow
    project["training_identity"] = {
        "dataset": workspace.name,
        "dataset_snapshot_hash": project["dataset_snapshot"]["snapshot_hash"],
        "config": config.name,
        "config_snapshot_hash": snapshot["snapshot_hash"],
        "target_type": config.target_type,
    }
    state.save()
    return state


def make_training_workspace_name(dataset_name: str, config_name: str, *, timestamp: str) -> str:
    def clean(value: str) -> str:
        value = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._") or "item"
        return value[:14]

    value = f"run-{clean(dataset_name)}-{clean(config_name)}-{timestamp}"
    return value[:64].rstrip("-._")
