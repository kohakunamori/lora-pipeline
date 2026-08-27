from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

from ..config import load_base_registry, resolve_profiles, stable_hash
from ..models import PipelineError, StepResult, TrainingRequest, TrainingResult
from ..state import ProjectState
from ..trainer.base import TrainerBackend
from ..trainer.sd_scripts import SdScriptsTrainer


def create_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def run(
    state: ProjectState,
    *,
    backend: TrainerBackend | None = None,
    optimizer_steps: int | None = None,
    dry_run: bool = False,
    verbose: int = 0,
    command_line: list[str] | None = None,
) -> tuple[StepResult, TrainingResult]:
    project = state.payload["project"]
    registry = load_base_registry()
    base_id = str(project["base"])
    if base_id not in registry:
        raise PipelineError(f"Base model is not registered: {base_id}")
    base = registry[base_id]
    profiles = resolve_profiles(
        str(project.get("hardware", "v100_16gb")),
        str(project["type"]),
        str(project.get("strategy", "quality")),
        project_overrides=project.get("overrides", {}),
    )
    budget = project.setdefault("budget", {})
    steps = int(optimizer_steps or budget.get("optimizer_steps", 1000))
    if steps < 1:
        raise PipelineError("optimizer_steps must be at least 1")
    run_id = create_run_id()
    run_dir = state.project_dir / "runs" / run_id
    suffix = 1
    while run_dir.exists():
        run_dir = state.project_dir / "runs" / f"{run_id}-{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)
    backend = backend or SdScriptsTrainer()
    request = TrainingRequest(
        project_dir=state.project_dir,
        run_dir=run_dir,
        base=base,
        config=profiles,
        optimizer_steps=steps,
        command_line=tuple(command_line or sys.argv),
    )
    run_record = {
        "id": run_dir.name,
        "path": str(run_dir),
        "status": "configuring" if dry_run else "running",
        "started_at": datetime.now(UTC).isoformat(),
        "base": {"id": base.id, "filename": base.path.name, "sha256": base.sha256},
    }
    state.payload.setdefault("runs", []).append(run_record)
    state.save()
    _pipeline_log(run_dir, "train.start", {"optimizer_steps": steps, "dry_run": dry_run})
    try:
        result = backend.train(request, dry_run=dry_run, verbose=verbose)
    except BaseException as exc:
        run_record.update(
            {
                "status": "failed",
                "finished_at": datetime.now(UTC).isoformat(),
                "last_error": f"{type(exc).__name__}: {exc}",
                "log_path": str(getattr(exc, "log_path", run_dir / "logs" / "train.log")),
            }
        )
        state.save()
        _pipeline_log(
            run_dir,
            "train.failed",
            {"error": run_record["last_error"], "log_path": run_record["log_path"]},
        )
        raise
    run_record.update(
        {
            "status": "dry-run" if result.dry_run else "trained",
            "finished_at": datetime.now(UTC).isoformat(),
            "checkpoints": [str(path) for path in result.checkpoints],
            "accounting": dict(result.accounting),
            "metrics": dict(result.metrics),
        }
    )
    state.save()
    _pipeline_log(
        run_dir,
        "train.finished",
        {"status": run_record["status"], "checkpoints": len(result.checkpoints), "metrics": dict(result.metrics)},
    )
    input_hash = stable_hash(
        {
            "base_sha256": base.sha256,
            "config": result.metrics.get("config_hash"),
            "dataset_snapshot_hash": result.accounting.get("dataset_snapshot_hash"),
            "captions_hash": result.accounting.get("captions_hash"),
            "optimizer_steps": steps,
        }
    )
    step_result = StepResult(
        input_hash=input_hash,
        output_manifest=str(run_dir / "config" / "run-metadata.json"),
        details={
            "run_id": result.run_id,
            "run_dir": str(result.run_dir),
            "checkpoints": len(result.checkpoints),
            "accounting": dict(result.accounting),
            "metrics": dict(result.metrics),
            "dry_run": result.dry_run,
        },
    )
    return step_result, result


def _pipeline_log(run_dir: Path, event: str, details: dict[str, object]) -> None:
    import json

    path = run_dir / "logs" / "pipeline.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"at": datetime.now(UTC).isoformat(), "event": event, "details": details}
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n")
