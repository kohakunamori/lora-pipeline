from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from .config import load_base_registry, repository_root
from .dataset.image_info import SUPPORTED_EXTENSIONS, discover_images
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
    optimizer_steps: int = 1000,
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
    if optimizer_steps < 1:
        raise PipelineError("optimizer_steps must be at least 1")
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
    state.payload["project"]["budget"] = {"optimizer_steps": optimizer_steps}
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
    dry_run: bool = False,
    verbose: int = 0,
    caption_mode: str = "generate",
    exclude_exact: bool = False,
    exclusions: list[str] | None = None,
    optimizer_steps: int | None = None,
    trainer_backend: TrainerBackend | None = None,
    generation_backend: GenerationBackend | None = None,
) -> StepResult:
    if name not in STEP_NAMES:
        raise StateError(f"Unknown step: {name}")
    if dry_run and name != "train":
        return StepResult(details={"dry_run": True, "would_run": name, "project": state.name})
    with project_lock(state.project_dir, force=force):
        if name == "inspect":
            return execute_step(state, name, lambda: inspect.run(state), force=force)
        if name == "dedup":
            return execute_step(
                state, name, lambda: dedup.run(state, exclude_exact=exclude_exact), force=force
            )
        if name == "identity":
            return execute_step(state, name, lambda: identity.run(state), force=force)
        if name == "caption":
            if caption_mode == "skip":
                state.skip("caption", "caption explicitly skipped")
                return StepResult(status=StepStatus.SKIPPED, details={"reason": "caption explicitly skipped"})
            return execute_step(
                state, name, lambda: caption.run(state, mode=caption_mode), force=force
            )
        if name == "review":
            return execute_step(
                state, name, lambda: review.run(state, exclude=exclusions), force=force
            )
        if name == "prepare":
            return execute_step(state, name, lambda: prepare.run(state), force=force)
        if name == "preflight":
            return execute_step(state, name, lambda: preflight.run(state), force=force)
        if name == "train":
            if dry_run:
                result, _ = train.run(
                    state,
                    backend=trainer_backend,
                    optimizer_steps=optimizer_steps,
                    dry_run=True,
                    verbose=verbose,
                )
                return result
            captured: list[Any] = []

            def handler() -> StepResult:
                step_result, training_result = train.run(
                    state,
                    backend=trainer_backend,
                    optimizer_steps=optimizer_steps,
                    verbose=verbose,
                )
                captured.append(training_result)
                return step_result

            return execute_step(state, name, handler, force=force)
        if name == "evaluate":
            return execute_step(
                state,
                name,
                lambda: evaluate.run(state, backend=generation_backend, verbose=verbose),
                force=force,
            )
    raise AssertionError(name)


def run_remaining(
    state: ProjectState,
    *,
    skip: set[str] | None = None,
    skip_preflight: bool = False,
    force: bool = False,
    dry_run: bool = False,
    verbose: int = 0,
    caption_mode: str = "generate",
    exclude_exact: bool = False,
    trainer_backend: TrainerBackend | None = None,
    generation_backend: GenerationBackend | None = None,
) -> list[tuple[str, StepResult]]:
    skip = skip or set()
    invalid = skip - OPTIONAL_STEPS
    if invalid:
        raise PipelineError("These steps cannot be skipped: " + ", ".join(sorted(invalid)))
    results: list[tuple[str, StepResult]] = []
    for name in STEP_NAMES:
        state = ProjectState.load(state.project_dir)
        if state.status(name) in {StepStatus.DONE, StepStatus.SKIPPED} and not force:
            continue
        if name in skip:
            state.skip(name, "explicitly skipped by run command")
            results.append((name, StepResult(status=StepStatus.SKIPPED, details={"reason": "explicit skip"})))
            continue
        if name == "preflight" and skip_preflight:
            state.skip_preflight("expert --skip-preflight override")
            results.append(
                (name, StepResult(status=StepStatus.SKIPPED, details={"warning": "preflight bypassed"}))
            )
            continue
        result = run_single_step(
            state,
            name,
            force=force,
            dry_run=dry_run,
            verbose=verbose,
            caption_mode=caption_mode,
            exclude_exact=exclude_exact,
            trainer_backend=trainer_backend,
            generation_backend=generation_backend,
        )
        results.append((name, result))
        if dry_run:
            # A dry run is a preview of the next actionable step, not a simulated state machine.
            break
    return results
