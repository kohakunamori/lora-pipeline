from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from .config import repository_root, write_json_atomic
from .models import PipelineError


GPU_JOB_KINDS = {"train", "evaluate", "video_prepare", "video_finalize", "dataset_tag"}
_FINAL_STATES = {"completed", "failed", "cancelled", "awaiting_identity"}


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def jobs_root(root: Path | None = None) -> Path:
    path = (root or repository_root()) / "web" / "jobs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def job_path(job_id: str, *, root: Path | None = None) -> Path:
    if not job_id or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789-" for ch in job_id):
        raise PipelineError("Invalid web job id")
    return jobs_root(root) / f"{job_id}.json"


def job_data_dir(job_id: str, *, root: Path | None = None) -> Path:
    path = jobs_root(root) / f"{job_id}.data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_job(job_id: str, *, root: Path | None = None) -> dict[str, Any]:
    path = job_path(job_id, root=root)
    if not path.is_file():
        raise PipelineError(f"Web job does not exist: {job_id}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PipelineError(f"Invalid web job record: {path}")
    return payload


def write_job(job_id: str, payload: Mapping[str, Any], *, root: Path | None = None) -> None:
    write_json_atomic(job_path(job_id, root=root), dict(payload))


def update_job(job_id: str, *, root: Path | None = None, **changes: Any) -> dict[str, Any]:
    payload = read_job(job_id, root=root)
    payload.update(changes)
    payload["updated_at"] = utc_now()
    write_job(job_id, payload, root=root)
    return payload


def create_job(kind: str, payload: Mapping[str, Any], *, root: Path | None = None) -> dict[str, Any]:
    root = (root or repository_root()).resolve()
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    job_id = f"{stamp}-{uuid.uuid4().hex[:8]}"
    record = {
        "schema_version": 1,
        "id": job_id,
        "kind": kind,
        "status": "queued",
        "pid": None,
        "payload": dict(payload),
        "result": {},
        "error": None,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "log": str(jobs_root(root) / f"{job_id}.log"),
    }
    write_job(job_id, record, root=root)
    return record


def spawn_job(kind: str, payload: Mapping[str, Any], *, root: Path | None = None) -> dict[str, Any]:
    root = (root or repository_root()).resolve()
    if kind in GPU_JOB_KINDS:
        active = active_gpu_jobs(root=root)
        if active:
            current = active[0]
            raise PipelineError(
                f"GPU is already reserved by web job {current['id']} ({current['kind']})"
            )
    record = create_job(kind, payload, root=root)
    return _spawn_existing(record["id"], root=root)


def resume_job(job_id: str, *, kind: str, payload_updates: Mapping[str, Any], root: Path | None = None) -> dict[str, Any]:
    root = (root or repository_root()).resolve()
    if kind in GPU_JOB_KINDS:
        active = [job for job in active_gpu_jobs(root=root) if job["id"] != job_id]
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
    record = read_job(job_id, root=root)
    log_path = Path(str(record["log"]))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["LORA_PIPELINE_ROOT"] = str(root)
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "pipeline.web_worker", job_id],
            cwd=root,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return update_job(job_id, root=root, status="running", pid=process.pid, started_at=utc_now())


def list_jobs(*, root: Path | None = None, limit: int = 100) -> list[dict[str, Any]]:
    base = jobs_root(root)
    records: list[dict[str, Any]] = []
    for path in sorted(base.glob("*.json"), key=lambda item: item.name, reverse=True):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(record, dict):
            records.append(_refresh_liveness(record, root=root))
        if len(records) >= limit:
            break
    return records


def active_gpu_jobs(*, root: Path | None = None) -> list[dict[str, Any]]:
    return [
        job
        for job in list_jobs(root=root, limit=200)
        if job.get("kind") in GPU_JOB_KINDS and job.get("status") == "running"
    ]


def jobs_for_project(project: str, *, root: Path | None = None) -> list[dict[str, Any]]:
    return [
        job
        for job in list_jobs(root=root, limit=200)
        if str((job.get("payload") or {}).get("project") or "") == project
    ]


def tail_job_log(job_id: str, *, root: Path | None = None, max_bytes: int = 48_000) -> str:
    record = read_job(job_id, root=root)
    path = Path(str(record.get("log") or ""))
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        size = path.stat().st_size
        if size > max_bytes:
            handle.seek(size - max_bytes)
            handle.readline()
        return handle.read().decode("utf-8", errors="replace")


def _refresh_liveness(record: dict[str, Any], *, root: Path | None = None) -> dict[str, Any]:
    if record.get("status") != "running":
        return record
    pid = record.get("pid")
    if isinstance(pid, int) and pid > 0 and _pid_alive(pid):
        return record
    # A worker normally records its final state itself. If it vanished before doing
    # so, make that failure visible instead of showing an eternal running job.
    record = dict(record)
    record.update(
        {
            "status": "failed",
            "error": record.get("error") or "Worker process exited without recording a final state",
            "updated_at": utc_now(),
        }
    )
    try:
        write_job(str(record["id"]), record, root=root)
    except Exception:
        pass
    return record


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
