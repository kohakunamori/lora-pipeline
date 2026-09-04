from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, Callable

from .config import load_base_registry, repository_root, write_json_atomic
from .dataset.image_info import discover_images, inspect_dataset
from .evaluation.generation import GenerationBackend
from .fingerprints import compute_step_signature
from .models import (
    OPTIONAL_STEPS,
    PROJECT_RUN_STEPS,
    STEP_NAMES,
    PipelineError,
    StateError,
    StepResult,
    StepStatus,
)
from .state import ProjectState, execute_step, project_lock
from .steps import caption, dedup, evaluate, identity, inspect, preflight, prepare, review, train
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

    # Inspection is a dataset import invariant, not a training-run stage. Freeze
    # it immediately beside the immutable Project raw snapshot so Preflight and
    # legacy readers still consume the same manifest without replaying `inspect`
    # during every Project run.
    inspection = inspect_dataset(destination / "raw")
    inspection_path = destination / "dataset-manifest.json"
    write_json_atomic(inspection_path, inspection)
    state.payload["steps"]["inspect"] = {
        "status": StepStatus.SKIPPED.value,
        "attempts": 0,
        "reason": "inspection frozen when the Project raw snapshot was created",
        "permanent": True,
        "input_hash": str(inspection["input_hash"]),
        "output_manifest": str(inspection_path),
        "details": dict(inspection["summary"]),
    }
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
    caption_mode: str = "generate",
    exclude_exact: bool = False,
    exclusions: list[str] | None = None,
    allow_trigger_only: bool | None = None,
    images_seen: int | None = None,
    optimizer_steps: int | None = None,
    resume_run: str | None = None,
    evaluation_stage: str = "screening",
    evaluation_run: str | None = None,
    evaluation_checkpoints: list[str] | None = None,
    trainer_backend: TrainerBackend | None = None,
    generation_backend: GenerationBackend | None = None,
) -> StepResult:
    if name not in STEP_NAMES:
        raise StateError(f"Unknown step: {name}")
    options = _step_options(
        name,
        caption_mode=caption_mode,
        exclude_exact=exclude_exact,
        exclusions=exclusions,
        allow_trigger_only=allow_trigger_only,
        images_seen=images_seen,
        optimizer_steps=optimizer_steps,
        resume_run=resume_run,
        evaluation_stage=evaluation_stage,
        evaluation_run=evaluation_run,
        evaluation_checkpoints=evaluation_checkpoints,
    )
    if dry_run and name != "train":
        fresh = ProjectState.load(state.project_dir)
        fingerprint = compute_step_signature(fresh, name, options=options)
        return StepResult(
            details={
                "dry_run": True,
                "would_run": name,
                "input_fingerprint": fingerprint,
                "currently_reusable": fresh.step(name).get("input_hash") == fingerprint
                and fresh.status(name) in {StepStatus.DONE, StepStatus.SKIPPED},
            }
        )

    with project_lock(state.project_dir, break_lock=break_lock):
        state = ProjectState.load(state.project_dir)
        fingerprint = compute_step_signature(state, name, options=options)
        if name == "caption" and caption_mode == "skip":
            state.payload["project"]["caption_mode"] = "skip"
            state.skip(name, "caption explicitly skipped", input_hash=fingerprint)
            return _record_result(state, name)
        if name == "train" and dry_run:
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
        if name == "inspect":
            handler = lambda: inspect.run(state)
        elif name == "dedup":
            handler = lambda: dedup.run(state, exclude_exact=exclude_exact)
        elif name == "identity":
            handler = lambda: identity.run(state)
        elif name == "caption":
            handler = lambda: caption.run(state, mode=caption_mode)
        elif name == "review":
            handler = lambda: review.run(state, exclude=exclusions)
        elif name == "prepare":
            handler = lambda: prepare.run(state, allow_trigger_only=allow_trigger_only)
        elif name == "preflight":
            handler = lambda: preflight.run(state)
        elif name == "train":
            handler = lambda: train.run(
                state,
                backend=trainer_backend,
                images_seen=images_seen,
                optimizer_steps=optimizer_steps,
                resume_run=resume_run,
                verbose=verbose,
            )[0]
        else:
            handler = lambda: evaluate.run(
                state,
                backend=generation_backend,
                verbose=verbose,
                stage=evaluation_stage,
                run_id=evaluation_run,
                checkpoint_names=evaluation_checkpoints,
            )
        return execute_step(state, name, handler, input_hash=fingerprint, force=force)


def run_remaining(
    state: ProjectState,
    *,
    skip: set[str] | None = None,
    skip_preflight: bool = False,
    force: bool = False,
    break_lock: bool = False,
    dry_run: bool = False,
    verbose: int = 0,
    caption_mode: str = "generate",
    exclude_exact: bool = False,
    allow_trigger_only: bool | None = None,
    images_seen: int | None = None,
    resume_run: str | None = None,
    on_step: Callable[[str], None] | None = None,
    trainer_backend: TrainerBackend | None = None,
    generation_backend: GenerationBackend | None = None,
) -> list[tuple[str, StepResult]]:
    skip = skip or set()
    invalid = skip - OPTIONAL_STEPS
    if invalid:
        raise PipelineError("These steps cannot be skipped: " + ", ".join(sorted(invalid)))
    results: list[tuple[str, StepResult]] = []
    can_break = break_lock
    for name in PROJECT_RUN_STEPS:
        state = ProjectState.load(state.project_dir)
        if state.step(name).get("permanent") and state.status(name) is StepStatus.SKIPPED:
            continue
        if on_step is not None:
            on_step(name)
        if dry_run and name in skip:
            result = StepResult(
                status=StepStatus.SKIPPED,
                details={
                    "dry_run": True,
                    "would_skip": name,
                    "reason": "explicitly skipped by run command",
                },
            )
        elif dry_run and name == "preflight" and skip_preflight:
            result = StepResult(
                status=StepStatus.SKIPPED,
                details={
                    "dry_run": True,
                    "would_skip": "preflight",
                    "reason": "expert --skip-preflight override",
                    "warning": True,
                },
            )
        elif name in skip:
            result = skip_optional_step(
                state,
                name,
                reason="explicitly skipped by run command",
                break_lock=can_break,
                options=_step_options(
                    name,
                    caption_mode="skip" if name == "caption" else caption_mode,
                    exclude_exact=exclude_exact,
                    allow_trigger_only=allow_trigger_only,
                ),
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
                caption_mode=caption_mode,
                exclude_exact=exclude_exact,
                allow_trigger_only=allow_trigger_only,
                images_seen=images_seen,
                resume_run=resume_run if name == "train" else None,
                trainer_backend=trainer_backend,
                generation_backend=generation_backend,
            )
        can_break = False
        if not result.details.get("reused"):
            results.append((name, result))
        if dry_run and not result.details.get("reused"):
            break
    return results


def skip_optional_step(
    state: ProjectState,
    name: str,
    *,
    reason: str,
    break_lock: bool = False,
    options: dict[str, Any] | None = None,
) -> StepResult:
    with project_lock(state.project_dir, break_lock=break_lock):
        state = ProjectState.load(state.project_dir)
        fingerprint = compute_step_signature(state, name, options={**(options or {}), "skip": True})
        before = state.step(name).get("input_hash")
        if name == "caption":
            state.payload["project"]["caption_mode"] = "skip"
        state.skip(name, reason, input_hash=fingerprint)
        result = _record_result(state, name)
        return StepResult(
            status=result.status,
            input_hash=result.input_hash,
            output_manifest=result.output_manifest,
            details={"reused": before == fingerprint, **dict(result.details)},
        )


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
    if name == "dedup":
        return {"exclude_exact": values.get("exclude_exact", False)}
    if name == "caption":
        return {"mode": values.get("caption_mode", "generate")}
    if name == "review":
        return {"exclusions": sorted(values.get("exclusions") or [])}
    if name == "prepare":
        return {"allow_trigger_only": values.get("allow_trigger_only")}
    if name == "train":
        return {
            "images_seen_override": values.get("images_seen"),
            "optimizer_steps_override": values.get("optimizer_steps"),
            "resume_run": values.get("resume_run"),
        }
    if name == "evaluate":
        return {
            "stage": values.get("evaluation_stage", "screening"),
            "run_id": values.get("evaluation_run"),
            "checkpoints": sorted(values.get("evaluation_checkpoints") or []),
        }
    return {}
