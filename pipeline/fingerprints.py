from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING

from .config import load_base_registry, resolve_profiles, sha256_file, stable_hash
from .dataset.image_info import discover_images
from .models import PipelineError, STEP_NAMES
from .prepared import load_current_generation

if TYPE_CHECKING:
    from .state import ProjectState


FINGERPRINT_VERSION = 2

STEP_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "inspect": (),
    "dedup": ("inspect",),
    "identity": ("inspect",),
    "caption": ("inspect",),
    "review": ("dedup", "identity", "caption"),
    "prepare": ("inspect", "review", "caption"),
    "preflight": ("prepare",),
    "train": ("preflight", "prepare"),
    "evaluate": ("train",),
}


def downstream_steps(name: str) -> tuple[str, ...]:
    if name not in STEP_NAMES:
        raise PipelineError(f"Unknown step: {name}")
    reverse: dict[str, set[str]] = {step: set() for step in STEP_NAMES}
    for step, dependencies in STEP_DEPENDENCIES.items():
        for dependency in dependencies:
            reverse[dependency].add(step)
    discovered: list[str] = []
    queue: deque[str] = deque(sorted(reverse[name], key=STEP_NAMES.index))
    seen: set[str] = set()
    while queue:
        step = queue.popleft()
        if step in seen:
            continue
        seen.add(step)
        discovered.append(step)
        queue.extend(sorted(reverse[step], key=STEP_NAMES.index))
    return tuple(sorted(discovered, key=STEP_NAMES.index))


def compute_step_signature(
    state: "ProjectState",
    name: str,
    *,
    options: Mapping[str, Any] | None = None,
) -> str:
    """Hash the effective inputs that decide whether a completed step is reusable."""

    if name not in STEP_NAMES:
        raise PipelineError(f"Unknown step: {name}")
    options = dict(options or {})
    project = state.payload["project"]
    payload: dict[str, Any] = {
        "fingerprint_version": FINGERPRINT_VERSION,
        "step": name,
        "project": {
            "type": project.get("type"),
            "base": project.get("base"),
            "trigger": project.get("trigger"),
            "hardware": project.get("hardware"),
            "strategy": project.get("strategy"),
            "overrides": project.get("overrides", {}),
        },
        "options": options,
    }

    if name == "inspect":
        payload["raw_images"] = _raw_images(state)
    elif name == "dedup":
        payload["inspection"] = _step_output_fingerprint(state, "inspect")
    elif name == "identity":
        payload["inspection"] = _step_output_fingerprint(state, "inspect")
        payload["identity_profile"] = _profiles(state).concept.get("identity_check", {})
    elif name == "caption":
        payload["raw_images"] = _raw_images(state)
        payload["raw_captions"] = _raw_captions(state)
        profiles = _profiles(state)
        payload["caption_profile"] = profiles.concept.get("caption", {})
        payload["tagger_profile"] = profiles.concept.get("tagger", {})
        payload["token_budget"] = profiles.merged.get("caption", {}).get(
            "max_token_length",
            profiles.hardware.get("caption", {}).get("default_max_token_length", 75),
        )
    elif name == "review":
        payload["dedup"] = _step_output_fingerprint(state, "dedup")
        payload["identity"] = _step_output_fingerprint(state, "identity")
        payload["caption"] = _step_output_fingerprint(state, "caption")
        payload["existing_exclusions"] = _hash_optional(state.project_dir / "review" / "exclusions.yaml")
    elif name == "prepare":
        payload["raw_images"] = _raw_images(state)
        payload["raw_captions"] = _raw_captions(state)
        payload["caption"] = _step_output_fingerprint(state, "caption")
        payload["exclusions"] = _hash_optional(state.project_dir / "review" / "exclusions.yaml")
        payload["caption_mode"] = project.get("caption_mode")
        payload["allow_trigger_only"] = bool(project.get("allow_trigger_only", False))
    elif name == "preflight":
        payload["prepared"] = _prepared_fingerprint(state)
        payload["base"] = _base_fingerprint(state)
        payload["profiles"] = _profiles(state).merged
        payload["budget"] = project.get("budget", {})
    elif name == "train":
        payload["prepared"] = _prepared_fingerprint(state)
        payload["base"] = _base_fingerprint(state)
        payload["profiles"] = _profiles(state).merged
        payload["budget"] = project.get("budget", {})
    elif name == "evaluate":
        payload["run"] = _latest_run_fingerprint(state)
        profiles = _profiles(state)
        payload["evaluation"] = profiles.merged.get("evaluation", {})
        payload["base"] = _base_fingerprint(state)
    return stable_hash(payload)


def _profiles(state: "ProjectState"):
    project = state.payload["project"]
    return resolve_profiles(
        str(project.get("hardware", "v100_16gb")),
        str(project["type"]),
        str(project.get("strategy", "quality")),
        project_overrides=project.get("overrides", {}),
    )


def _raw_images(state: "ProjectState") -> list[dict[str, Any]]:
    raw = state.project_dir / "raw"
    return [
        {
            "path": image.relative_to(raw).as_posix(),
            "bytes": image.stat().st_size,
            "sha256": sha256_file(image),
        }
        for image in discover_images(raw)
    ]


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


def _step_output_fingerprint(state: "ProjectState", name: str) -> Mapping[str, Any]:
    record = state.step(name)
    manifest = record.get("output_manifest")
    return {
        "status": record.get("status"),
        "input_hash": record.get("input_hash"),
        "manifest": _hash_optional(Path(manifest)) if manifest else None,
    }


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
        stat = {"bytes": info.st_size, "mtime_ns": info.st_mtime_ns, "inode": info.st_ino}
    return {
        "id": base.id,
        "path": str(base.path),
        "registered_sha256": base.sha256,
        "stat": stat,
        "generation_defaults": dict(base.generation_defaults),
    }


def _latest_run_fingerprint(state: "ProjectState") -> Mapping[str, Any]:
    for record in reversed(state.payload.get("runs", [])):
        if record.get("status") in {"trained", "evaluated", "promoted"}:
            checkpoints = []
            for value in record.get("checkpoints", []):
                path = Path(value)
                checkpoints.append(
                    {
                        "path": str(path),
                        "bytes": path.stat().st_size if path.is_file() else None,
                        "sha256": sha256_file(path) if path.is_file() else None,
                    }
                )
            return {
                "id": record.get("id"),
                "accounting": record.get("accounting", {}),
                "checkpoints": checkpoints,
            }
    return {"missing": True}


def _hash_optional(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
