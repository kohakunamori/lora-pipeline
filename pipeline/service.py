from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, Callable

from .config import load_base_registry, repository_root, write_json_atomic
from .dataset.image_info import discover_images, inspect_dataset
from .fingerprints import compute_step_signature
from .materialization import run as materialize
from .models import (
    PROJECT_RUN_STEPS,
    STEP_ALIASES,
    PipelineError,
    StateError,
    StepResult,
    StepStatus,
)
from .state import ProjectState, execute_step, project_lock
from .steps import preflight, train
from .trainer.base import TrainerBackend


def project_path(name: str, *, root: Path | None = None) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", name) or name in {".", ".."}:
        raise StateError("Project name must be 1-64 letters, numbers, '.', '_' or '-'")
    return (root or repository_root()) / "projects" / name


def create_project(
    *,
    name: str,
    concept_type: str,
    base: str,
    trigger: str,
    strategy: str,
    dataset: Path,
    images_seen: int = 1000,
    optimizer_steps: int | None = None,
    hardware: str = "v100_16gb",
    root: Path | None = None,
) -> ProjectState:
    root = root or repository_root()
    if concept_type not in {"character", "style"}:
        raise PipelineError("Concept type must be 'character' or 'style'")
    if strategy not in {"quality", "fast", "cached"}:
        raise PipelineError("Training strategy must be quality, fast, or cached")
    if not trigger.strip() or "," in trigger:
        raise PipelineError("Trigger must be non-empty and cannot contain a comma")
    if optimizer_steps is not None:
        if images_seen != 1000:
            raise PipelineError("Use either images_seen or the legacy optimizer_steps alias")
        images_seen = optimizer_steps
    if images_seen < 1:
        raise PipelineError("images_seen must be at least 1")
    registry = load_base_registry(root)
    if base not in registry or not registry[base].enabled:
        raise PipelineError(f"Base model is not registered and enabled: {base}")
    dataset = dataset.expanduser().resolve()
    if not dataset.is_dir():
        raise PipelineError(f"Dataset directory does not exist: {dataset}")
    images = discover_images(dataset)
    if not images:
        raise PipelineError(f"No supported images were found under {dataset}")

    destination = project_path(name, root=root)
    state = ProjectState.create(
        destination,
        name=name,
        concept_type=concept_type,
        base=base,
        trigger=trigger.strip(),
        strategy=strategy,
        hardware=hardware,
        raw_source=str(dataset),
    )
    try:
        for image in images:
            relative = image.relative_to(dataset)
            target = destination / "raw" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, target)
            sidecar = image.with_suffix(".txt")
            if sidecar.is_file():
                shutil.copy2(sidecar, target.with_suffix(".txt"))
    except BaseException as exc:
        state.payload["project"]["import_error"] = f"{type(exc).__name__}: {exc}"
        state.save()
        raise

    # Dataset inspection is frozen evidence attached to the raw snapshot. It is
    # not a Project lifecycle step.
    inspection = inspect_dataset(destination / "raw")
    write_json_atomic(destination / "dataset-manifest.json", inspection)
    state.payload["project"].update(
        {
            "budget": {"unit": "images_seen", "value": images_seen},
            "imported_images": len(images),
        }
    )
    state.save()
    return state


def load_project(name: str, *, root: Path | None = None) -> ProjectState:
    path = project_path(name, root=root)
    if not (path / "project.yaml").is_file():
        raise StateError(f"Project does not exist: {name}")
    return ProjectState.load(path)


def run_single_step(
    state: ProjectState,
    name: str,
    *,
    force: bool = False,
    break_lock: bool = False,
    dry_run: bool = False,
    verbose: int = 0,
    caption_mode: str | None = None,
    allow_trigger_only: bool | None = None,
    images_seen: int | None = None,
    optimizer_steps: int | None = None,
    resume_run: str | None = None,
    trainer_backend: TrainerBackend | None = None,
) -> StepResult:
    canonical = STEP_ALIASES.get(name, name)
    if canonical not in PROJECT_RUN_STEPS:
        raise StateError(f"Unknown Project step: {name}")
    options = _step_options(
        canonical,
        caption_mode=caption_mode,
        allow_trigger_only=allow_trigger_only,
        images_seen=images_seen,
        optimizer_steps=optimizer_steps,
        resume_run=resume_run,
    )

    if dry_run and canonical != "train":
        fresh = ProjectState.load(state.project_dir)
        fingerprint = compute_step_signature(fresh, canonical, options=options)
        return StepResult(
            details={
                "dry_run": True,
                "would_run": canonical,
                "input_fingerprint": fingerprint,
                "currently_reusable": fresh.step(canonical).get("input_hash") == fingerprint
                and fresh.status(canonical) in {StepStatus.DONE, StepStatus.SKIPPED},
            }
        )

    with project_lock(state.project_dir, break_lock=break_lock):
        state = ProjectState.load(state.project_dir)
        fingerprint = compute_step_signature(state, canonical, options=options)
        if canonical == "train" and dry_run:
            result, _ = train.run(
                state,
                backend=trainer_backend,
                images_seen=images_seen,
                optimizer_steps=optimizer_steps,
                resume_run=resume_run,
                dry_run=True,
                verbose=verbose,
            )
            return result

        handler: Callable[[], StepResult]
        if canonical == "materialize":
            handler = lambda: materialize(
                state,
                allow_trigger_only=allow_trigger_only,
                caption_mode=caption_mode,
            )
        elif canonical == "preflight":
            handler = lambda: preflight.run(state)
        else:
            handler = lambda: train.run(
                state,
                backend=trainer_backend,
                images_seen=images_seen,
                optimizer_steps=optimizer_steps,
                resume_run=resume_run,
                verbose=verbose,
            )[0]
        return execute_step(state, canonical, handler, input_hash=fingerprint, force=force)


def run_remaining(
    state: ProjectState,
    *,
    skip_preflight: bool = False,
    force: bool = False,
    break_lock: bool = False,
    dry_run: bool = False,
    verbose: int = 0,
    caption_mode: str = "generate",
    allow_trigger_only: bool | None = None,
    images_seen: int | None = None,
    resume_run: str | None = None,
    on_step: Callable[[str], None] | None = None,
    trainer_backend: TrainerBackend | None = None,
) -> list[tuple[str, StepResult]]:
    results: list[tuple[str, StepResult]] = []
    can_break = break_lock
    for name in PROJECT_RUN_STEPS:
        state = ProjectState.load(state.project_dir)
        if on_step is not None:
            on_step(name)
        if dry_run and name == "preflight" and skip_preflight:
            result = StepResult(
                status=StepStatus.SKIPPED,
                details={
                    "dry_run": True,
                    "would_skip": "preflight",
                    "reason": "expert --skip-preflight override",
                    "warning": True,
                },
            )
        elif name == "preflight" and skip_preflight:
            result = skip_preflight_step(state, break_lock=can_break)
        else:
            result = run_single_step(
                state,
                name,
                force=force,
                break_lock=can_break,
                dry_run=dry_run,
                verbose=verbose,
                caption_mode=caption_mode if name == "materialize" else None,
                allow_trigger_only=allow_trigger_only if name == "materialize" else None,
                images_seen=images_seen,
                resume_run=resume_run if name == "train" else None,
                trainer_backend=trainer_backend,
            )
        can_break = False
        if not result.details.get("reused"):
            results.append((name, result))
        if dry_run and not result.details.get("reused"):
            break
    return results


def skip_preflight_step(state: ProjectState, *, break_lock: bool = False) -> StepResult:
    with project_lock(state.project_dir, break_lock=break_lock):
        state = ProjectState.load(state.project_dir)
        fingerprint = compute_step_signature(state, "preflight", options={"skip": True})
        before = state.step("preflight").get("input_hash")
        state.skip_preflight("expert --skip-preflight override", input_hash=fingerprint)
        result = _record_result(state, "preflight")
        return StepResult(
            status=result.status,
            input_hash=result.input_hash,
            output_manifest=result.output_manifest,
            details={"reused": before == fingerprint, **dict(result.details)},
        )


def _record_result(state: ProjectState, name: str) -> StepResult:
    record = state.step(name)
    return StepResult(
        status=StepStatus(record["status"]),
        input_hash=record.get("input_hash"),
        output_manifest=record.get("output_manifest"),
        details=dict(record.get("details", {})),
    )


def _step_options(name: str, **values: Any) -> dict[str, Any]:
    if name == "materialize":
        return {
            "allow_trigger_only": values.get("allow_trigger_only"),
            "caption_mode": values.get("caption_mode"),
        }
    if name == "train":
        return {
            "images_seen_override": values.get("images_seen"),
            "optimizer_steps_override": values.get("optimizer_steps"),
            "resume_run": values.get("resume_run"),
        }
    return {}
