from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from .config import load_base_registry, resolve_profiles, sha256_file, stable_hash
from .dataset.image_info import discover_images
from .models import PROJECT_RUN_STEPS, STEP_ALIASES, PipelineError
from .prepared import load_current_generation

if TYPE_CHECKING:
    from .state import ProjectState


FINGERPRINT_VERSION = 9
TRAINING_PROFILE_KEYS = (
    "precision",
    "attention",
    "memory",
    "resolution",
    "caption",
    "training",
    "checkpoints",
    "data_loader",
    "storage",
    "limits",
)

STEP_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "materialize": (),
    "preflight": ("materialize",),
    "train": ("preflight", "materialize"),
}


def downstream_steps(name: str) -> tuple[str, ...]:
    canonical = STEP_ALIASES.get(name, name)
    if canonical not in PROJECT_RUN_STEPS:
        raise PipelineError(f"Unknown Project step: {name}")
    reverse: dict[str, set[str]] = {step: set() for step in PROJECT_RUN_STEPS}
    for step, dependencies in STEP_DEPENDENCIES.items():
        for dependency in dependencies:
            reverse[dependency].add(step)
    discovered: list[str] = []
    queue: deque[str] = deque(sorted(reverse[canonical], key=PROJECT_RUN_STEPS.index))
    seen: set[str] = set()
    while queue:
        step = queue.popleft()
        if step in seen:
            continue
        seen.add(step)
        discovered.append(step)
        queue.extend(sorted(reverse[step], key=PROJECT_RUN_STEPS.index))
    return tuple(sorted(discovered, key=PROJECT_RUN_STEPS.index))


def compute_step_signature(
    state: "ProjectState",
    name: str,
    *,
    options: Mapping[str, Any] | None = None,
) -> str:
    """Hash the effective inputs that decide whether a Project step is reusable."""

    canonical = STEP_ALIASES.get(name, name)
    if canonical not in PROJECT_RUN_STEPS:
        raise PipelineError(f"Unknown Project step: {name}")
    options = dict(options or {})
    project = state.payload["project"]
    payload: dict[str, Any] = {
        "fingerprint_version": FINGERPRINT_VERSION,
        "step": canonical,
        "options": options,
    }

    if canonical == "materialize":
        profiles = _profiles(state)
        payload.update(
            {
                "concept_type": project.get("type"),
                "training_target_type": project.get("training_target_type", project.get("type")),
                "trigger": project.get("trigger"),
                "trigger_policy": project.get("trigger_policy", {}),
                "caption_anchor_tags": project.get("caption_anchor_tags", []),
                "raw_images": _raw_images(state),
                "raw_captions": _raw_captions(state),
                "exclusions": _hash_optional(state.project_dir / "review" / "exclusions.yaml"),
                "caption_mode": _effective_caption_mode(project, options.get("caption_mode")),
                "caption_profile": profiles.merged.get("caption", {}),
                "tagger_profile": profiles.concept.get("tagger", {}),
                "token_budget": profiles.merged.get("caption", {}).get(
                    "max_token_length",
                    profiles.hardware.get("caption", {}).get("default_max_token_length", 75),
                ),
                "allow_trigger_only": (
                    bool(options["allow_trigger_only"])
                    if options.get("allow_trigger_only") is not None
                    else bool(project.get("allow_trigger_only", False))
                ),
            }
        )
    else:
        profiles = _profiles(state)
        payload.update(
            {
                "prepared": _prepared_fingerprint(state),
                "base": _base_fingerprint(state),
                "profiles": _training_profile_slice(profiles.merged),
                "profile_ids": {
                    "hardware": profiles.hardware.get("id"),
                    "concept": profiles.concept.get("id"),
                    "training": profiles.training.get("id"),
                },
                "budget": project.get("budget", {}),
            }
        )
    return stable_hash(payload)


def _effective_caption_mode(project: Mapping[str, Any], requested: Any) -> str:
    mode = str(requested or project.get("caption_mode") or "skip")
    if mode != "auto":
        return mode
    snapshot = project.get("dataset_snapshot", {})
    if isinstance(snapshot, Mapping):
        image_count = int(snapshot.get("image_count", 0) or 0)
        caption_count = int(snapshot.get("caption_count", 0) or 0)
        if image_count > 0 and caption_count == image_count:
            return "existing_taglist_clean"
    return "generate"


def _training_profile_slice(merged: Mapping[str, Any]) -> dict[str, Any]:
    return {key: merged[key] for key in TRAINING_PROFILE_KEYS if key in merged}


def _profiles(state: "ProjectState"):
    project = state.payload["project"]
    return resolve_profiles(
        str(project.get("hardware", "v100_16gb")),
        str(project["type"]),
        str(project.get("strategy", "quality")),
        project_overrides=project.get("overrides", {}),
    )


def _image_fingerprints(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": image.relative_to(root).as_posix(),
            "bytes": image.stat().st_size,
            "sha256": sha256_file(image),
        }
        for image in discover_images(root)
    ]


def _raw_images(state: "ProjectState") -> list[dict[str, Any]]:
    return _image_fingerprints(state.project_dir / "raw")


def _raw_captions(state: "ProjectState") -> list[dict[str, Any]]:
    raw = state.project_dir / "raw"
    records: list[dict[str, Any]] = []
    for image in discover_images(raw):
        caption = image.with_suffix(".txt")
        if caption.is_file():
            records.append(
                {
                    "path": caption.relative_to(raw).as_posix(),
                    "bytes": caption.stat().st_size,
                    "sha256": sha256_file(caption),
                }
            )
    return records


def _prepared_fingerprint(state: "ProjectState") -> Mapping[str, Any]:
    generation = load_current_generation(state.project_dir)
    return {
        "generation_id": generation.generation_id,
        "manifest": _hash_optional(generation.manifest_path),
    }


def _base_fingerprint(state: "ProjectState") -> Mapping[str, Any]:
    base_id = str(state.payload["project"]["base"])
    registry = load_base_registry()
    if base_id not in registry:
        return {"id": base_id, "missing": True}
    base = registry[base_id]
    stat: dict[str, Any] | None = None
    if base.path.is_file():
        info = base.path.stat()
        stat = {
            "bytes": info.st_size,
            "mtime_ns": info.st_mtime_ns,
            "inode": info.st_ino,
            "device": info.st_dev,
        }
    return {
        "id": base.id,
        "path": str(base.path),
        "registered_sha256": base.sha256,
        "stat": stat,
        "generation_defaults": dict(base.generation_defaults),
    }


def _hash_optional(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
