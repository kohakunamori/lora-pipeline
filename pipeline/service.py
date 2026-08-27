from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from .config import load_base_registry, repository_root
from .dataset.image_info import discover_images
from .fingerprints import compute_step_signature
from .models import OPTIONAL_STEPS, STEP_NAMES, PipelineError, StateError, StepResult, StepStatus
from .state import ProjectState, execute_step, project_lock
from .steps import caption, dedup, evaluate, identity, inspect, preflight, prepare, review, train
from .trainer.base import TrainerBackend
from .evaluation.generation import GenerationBackend


def project_path(name: str, *, root: Path | None = None) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", name) or name in {".", ".."}:
        raise StateError("Project name must be 1–64 letters, numbers, '.', '_' or '-'")
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
    """Create a project whose canonical training budget is image exposure.

    ``optimizer_steps`` remains as a compatibility alias for older callers.  Its
    value is interpreted as image exposures, so changing physical batch no
    longer silently changes the amount of training.
    """

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
            caption_source = image.with_suffix(".txt")
            if caption_source.is_file():
                shutil.copy2(caption_source, target.with_suffix(".txt"))
    except BaseException as exc:
        state.payload["project"]["import_error"] = f"{type(exc).__name__}: {exc}"
        state.save()
        raise
    state.payload["project"]["budget"] = {"unit": "images_seen", "value": images_seen}
    state.payload["project"]["imported_images"] = len(images)
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
    trainer_backend: TrainerBackend | None = None,
    generation_backend: GenerationBackend | None = None,
) -> StepResult:
    if name not in STEP_NAMES:
        raise StateError(f"Unknown step: {name}")
    options = _signature_options(
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
    )
    if dry_run and name != "train":
        fresh = ProjectState.load(state.project_dir)
        signature = compute_step_signature(fresh, name, options=options)
        return StepResult(
            details={
                "dry_run": True,
                "would_run": name,
                "project": fresh.name,
                "input_fingerprint": signature,
                "currently_reusable": fresh.step(name).get("input_hash") == signature
                and fresh.status(name) in {StepStatus.DONE, StepStatus.SKIPPED},
            }
        )

    with project_lock(state.project_dir, break_lock=break_lock):
        state = ProjectState.load(state.project_dir)
        signature = compute_step_signature(state, name, options=options)
        if name == "inspect":
            return execute_step(state, name, lambda: inspect.run(state), input_hash=signature, force=force)
        if name == "dedup":
            return execute_step(
                state,
                name,
                lambda: dedup.run(state, exclude_exact=exclude_exact),
                input_hash=signature,
                force=force,
            )
        if name == "identity":
            return execute_step(
                state, name, lambda: identity.run(state), input_hash=signature, force=force
            )
        if name == "caption":
            if caption_mode == "skip":
                state.skip("caption", "caption explicitly skipped", input_hash=signature)
                record = state.step("caption")
                return StepResult(
                    status=StepStatus.SKIPPED,
                    input_hash=signature,
                    output_manifest=record.get("output_manifest"),
                    details=dict(record.get("details", {})),
                )
            return execute_step(
                state,
                name,
                lambda: caption.run(state, mode=caption_mode),
                input_hash=signature,
                force=force,
            )
        if name == "review":
            return execute_step(
                state,
                name,
                lambda: review.run(state, exclude=exclusions),
                input_hash=signature,
                force=force,
            )
        if name == "prepare":
            return execute_step(
                state,
                name,
                lambda: prepare.run(state, allow_trigger_only=allow_trigger_only),
                input_hash=signature,
                force=force,
            )
        if name == "preflight":
            return execute_step(
                state, name, lambda: preflight.run(state), input_hash=signature, force=force
            )
        if name == "train":
            if dry_run:
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

            def handler() -> StepResult:
                step_result, _ = train.run(
                    state,
                    backend=trainer_backend,
                    images_seen=images_seen,
                    optimizer_steps=optimizer_steps,
                    resume_run=resume_run,
                    verbose=verbose,
                )
                return step_result

            return execute_step(state, name, handler, input_hash=signature, force=force)
        if name == "evaluate":
            return execute_step(
                state,
                name,
                lambda: evaluate.run(
                    state,
                    backend=generation_backend,
                    verbose=verbose,
                    stage=evaluation_stage,
                    run_id=evaluation_run,
                ),
                input_hash=signature,
                force=force,
            )
    raise AssertionError(name)


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
    trainer_backend: TrainerBackend | None = None,
    generation_backend: GenerationBackend | None = None,
) -> list[tuple[str, StepResult]]:
    skip = skip or set()
    invalid = skip - OPTIONAL_STEPS
    if invalid:
        raise PipelineError("These steps cannot be skipped: " + ", ".join(sorted(invalid)))
    results: list[tuple[str, StepResult]] = []
    may_break_lock = break_lock
    for name in STEP_NAMES:
        state = ProjectState.load(state.project_dir)
        if state.step(name).get("permanent") and state.status(name) is StepStatus.SKIPPED:
            continue
        if name in skip:
            result = _skip_step(
                state,
                name,
                reason="explicitly skipped by run command",
                break_lock=may_break_lock,
                options=_signature_options(
                    name,
                    caption_mode="skip" if name == "caption" else caption_mode,
                    exclude_exact=exclude_exact,
                    allow_trigger_only=allow_trigger_only,
                ),
            )
            may_break_lock = False
        elif name == "preflight" and skip_preflight:
            result = _skip_preflight(state, break_lock=may_break_lock)
            may_break_lock = False
        else:
            result = run_single_step(
                state,
                name,
                force=force,
                break_lock=may_break_lock,
                dry_run=dry_run,
                verbose=verbose,
                caption_mode=caption_mode,
                exclude_exact=exclude_exact,
                allow_trigger_only=allow_trigger_only,
                images_seen=images_seen,
                trainer_backend=trainer_backend,
                generation_backend=generation_backend,
            )
            may_break_lock = False
        if not result.details.get("reused"):
            results.append((name, result))
        if dry_run and not result.details.get("reused"):
            break
    return results


def _skip_step(
    state: ProjectState,
    name: str,
    *,
    reason: str,
    break_lock: bool,
    options: dict[str, Any],
) -> StepResult:
    with project_lock(state.project_dir, break_lock=break_lock):
        state = ProjectState.load(state.project_dir)
        signature = compute_step_signature(state, name, options={**options, "skip": True})
        before = state.step(name).get("input_hash")
        state.skip(name, reason, input_hash=signature)
        record = state.step(name)
        return StepResult(
            status=StepStatus.SKIPPED,
            input_hash=signature,
            output_manifest=record.get("output_manifest"),
            details={"reused": before == signature, **dict(record.get("details", {}))},
        )


def _skip_preflight(state: ProjectState, *, break_lock: bool) -> StepResult:
    with project_lock(state.project_dir, break_lock=break_lock):
        state = ProjectState.load(state.project_dir)
        signature = compute_step_signature(state, "preflight", options={"skip": True})
        before = state.step("preflight").get("input_hash")
        state.skip_preflight("expert --skip-preflight override", input_hash=signature)
        record = state.step("preflight")
        return StepResult(
            status=StepStatus.SKIPPED,
            input_hash=signature,
            details={"reused": before == signature, **dict(record.get("details", {}))},
        )


def _signature_options(
    name: str,
    *,
    caption_mode: str = "generate",
    exclude_exact: bool = False,
    exclusions: list[str] | None = None,
    allow_trigger_only: bool | None = None,
    images_seen: int | None = None,
    optimizer_steps: int | None = None,
    resume_run: str | None = None,
    evaluation_stage: str = "screening",
    evaluation_run: str | None = None,
) -> dict[str, Any]:
    if name == "dedup":
        return {"exclude_exact": exclude_exact}
    if name == "caption":
        return {"mode": caption_mode}
    if name == "review":
        return {"exclusions": sorted(exclusions or [])}
    if name == "prepare":
        return {"allow_trigger_only": allow_trigger_only}
    if name == "train":
        return {
            "images_seen_override": images_seen,
            "optimizer_steps_override": optimizer_steps,
            "resume_run": resume_run,
        }
    if name == "evaluate":
        return {"stage": evaluation_stage, "run_id": evaluation_run}
    return {}
