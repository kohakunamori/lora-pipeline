from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from .models import PipelineError
from .web_jobs import (
    create_job,
    list_jobs,
    read_job,
    update_job,
    utc_now,
    write_job,
)


GPU_JOB_KINDS = {
    "train",
    "evaluate",
    "video_prepare",
    "video_finalize",
    "dataset_tag",
    "dataset_analyze",
}
_ACTIVE_STATES = {"queued", "starting", "running"}


def active_gpu_jobs(*, root: Path | None = None) -> list[dict[str, Any]]:
    return [
        job
        for job in list_jobs(root=root, limit=200)
        if job.get("kind") in GPU_JOB_KINDS and job.get("status") in _ACTIVE_STATES
    ]


def spawn_job(kind: str, payload: Mapping[str, Any], *, root: Path | None = None) -> dict[str, Any]:
    root = (root or Path.cwd()).resolve()
    if kind in GPU_JOB_KINDS:
        active = active_gpu_jobs(root=root)
        if active:
            current = active[0]
            raise PipelineError(
                f"GPU is already reserved by web job {current['id']} ({current['kind']})"
            )
    record = create_job(kind, payload, root=root)
    return _spawn_existing(str(record["id"]), root=root)


def resume_job(
    job_id: str,
    *,
    kind: str,
    payload_updates: Mapping[str, Any],
    root: Path | None = None,
) -> dict[str, Any]:
    root = (root or Path.cwd()).resolve()
    if kind in GPU_JOB_KINDS:
        active = [job for job in active_gpu_jobs(root=root) if str(job.get("id")) != job_id]
        if active:
            current = active[0]
            raise PipelineError(
                f"GPU is already reserved by web job {current['id']} ({current['kind']})"
            )
    record = read_job(job_id, root=root)
    merged = dict(record.get("payload") or {})
    merged.update(dict(payload_updates))
    record.update(
        {
            "kind": kind,
            "status": "queued",
            "pid": None,
            "payload": merged,
            "error": None,
            "updated_at": utc_now(),
        }
    )
    write_job(job_id, record, root=root)
    return _spawn_existing(job_id, root=root)


def _spawn_existing(job_id: str, *, root: Path) -> dict[str, Any]:
    record = update_job(
        job_id,
        root=root,
        status="starting",
        pid=None,
        error=None,
        started_at=utc_now(),
    )
    log_path = Path(str(record["log"]))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["LORA_PIPELINE_ROOT"] = str(root)
    try:
        with log_path.open("ab", buffering=0) as log:
            subprocess.Popen(
                [sys.executable, "-m", "pipeline.web_worker_enriched", job_id],
                cwd=root,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    except BaseException as exc:
        return update_job(
            job_id,
            root=root,
            status="failed",
            error=f"Could not start worker: {type(exc).__name__}: {exc}",
            finished_at=utc_now(),
        )
    return read_job(job_id, root=root)
