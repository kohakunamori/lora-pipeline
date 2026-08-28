from __future__ import annotations

import json
import os
import socket
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .config import repository_root
from .models import PipelineError
from .state import ProjectState, _lock_is_live, _read_lock
from .web_jobs import list_jobs


ACTIVE_JOB_STATES = {"queued", "starting", "running", "cancelling", "awaiting_identity"}
ACTIVE_RUN_STATES = {"configuring", "running"}
_PROCESS_LOCK = threading.RLock()
_LOCAL = threading.local()


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _lock_path(root: Path) -> Path:
    return root / ".pipeline-lifecycle.lock"


def _read_lifecycle_lock(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _lifecycle_lock_live(path: Path) -> bool | None:
    payload = _read_lifecycle_lock(path)
    host = str(payload.get("host") or "")
    pid = payload.get("pid")
    if not host or not isinstance(pid, int):
        return None
    if host != socket.gethostname():
        return None
    return _pid_alive(pid)


@contextmanager
def lifecycle_lock(root: Path | None = None) -> Iterator[None]:
    """Serialize destructive lifecycle operations and training-workspace creation.

    The lock is re-entrant inside one process and also represented by a small file so
    independent CLI/Web processes cannot interleave a deletion with snapshot creation.
    """

    resolved = (root or repository_root()).resolve()
    path = _lock_path(resolved)
    path.parent.mkdir(parents=True, exist_ok=True)

    with _PROCESS_LOCK:
        depth = int(getattr(_LOCAL, "depth", 0))
        if depth:
            _LOCAL.depth = depth + 1
            try:
                yield
            finally:
                _LOCAL.depth -= 1
            return

        descriptor: int | None = None
        for _attempt in range(2):
            try:
                descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                break
            except FileExistsError:
                live = _lifecycle_lock_live(path)
                if live is False:
                    path.unlink(missing_ok=True)
                    continue
                state = "active" if live is True else "unverifiable"
                raise PipelineError(
                    f"Another lifecycle operation has an {state} lock: {path}. Retry after it finishes."
                )
        if descriptor is None:
            raise PipelineError(f"Could not acquire lifecycle lock: {path}")

        token = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
        }
        try:
            os.write(descriptor, (json.dumps(token, sort_keys=True) + "\n").encode("utf-8"))
            os.fsync(descriptor)
            _LOCAL.depth = 1
            yield
        finally:
            _LOCAL.depth = 0
            os.close(descriptor)
            current = _read_lifecycle_lock(path)
            if current.get("pid") == token["pid"] and current.get("host") == token["host"]:
                path.unlink(missing_ok=True)


def _project_states(root: Path) -> list[ProjectState]:
    base = root / "projects"
    if not base.is_dir():
        return []
    states: list[ProjectState] = []
    for path in sorted(base.iterdir(), key=lambda item: item.name.casefold()):
        if not (path / "project.yaml").is_file():
            continue
        try:
            states.append(ProjectState.load(path))
        except Exception:
            continue
    return states


def _project_identity(state: ProjectState) -> tuple[str, str]:
    project = state.payload.get("project", {})
    identity = project.get("training_identity", {}) if isinstance(project, dict) else {}
    dataset = str(
        (identity.get("dataset") if isinstance(identity, dict) else None)
        or project.get("dataset_snapshot", {}).get("dataset")
        or ""
    )
    config = str((identity.get("config") if isinstance(identity, dict) else None) or "")
    return dataset, config


def _project_runtime_markers(state: ProjectState) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    lock_path = state.project_dir / ".pipeline.lock"
    if lock_path.is_file():
        try:
            live = _lock_is_live(_read_lock(lock_path))
        except Exception:
            live = None
        if live is not False:
            blockers.append(
                {
                    "type": "project_lock",
                    "id": state.name,
                    "status": "running" if live is True else "unverifiable",
                    "reason": "project has a live or unverifiable .pipeline.lock",
                }
            )

    pid_path = state.project_dir / "web-worker.pid"
    if pid_path.is_file():
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            pid = 0
        if _pid_alive(pid):
            blockers.append(
                {
                    "type": "web_worker",
                    "id": state.name,
                    "status": "running",
                    "reason": f"legacy web training worker PID {pid} is still alive",
                }
            )

    runs = list(state.payload.get("runs", []))
    if runs:
        latest = runs[-1]
        status = str(latest.get("status") or "")
        if status in ACTIVE_RUN_STATES:
            blockers.append(
                {
                    "type": "training_run",
                    "id": str(latest.get("id") or state.name),
                    "status": status,
                    "reason": "project state still marks the latest run as active",
                }
            )
    return blockers


def _project_pending_dependency(state: ProjectState) -> bool:
    project = state.payload.get("project", {})
    if not isinstance(project, dict) or project.get("workspace_role") != "training_run":
        return False
    runs = list(state.payload.get("runs", []))
    if runs:
        return False
    return state.next_actionable_step() is not None


def deletion_blockers(
    resource_type: str,
    resource_id: str,
    *,
    root: Path | None = None,
) -> list[dict[str, str]]:
    """Return active lifecycle references that make permanent deletion unsafe."""

    resolved = (root or repository_root()).resolve()
    if resource_type not in {"dataset", "training_config", "project", "run"}:
        raise PipelineError(f"Unknown lifecycle resource type: {resource_type}")

    states = _project_states(resolved)
    states_by_name = {state.name: state for state in states}
    run_project = ""
    run_id = ""
    if resource_type == "run":
        if "/" not in resource_id:
            raise PipelineError("Run resource id must be PROJECT/RUN")
        run_project, run_id = resource_id.split("/", 1)
    blockers: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    def add(blocker: dict[str, str]) -> None:
        key = (blocker["type"], blocker["id"], blocker["reason"])
        if key not in seen:
            seen.add(key)
            blockers.append(blocker)

    for job in list_jobs(root=resolved, limit=500):
        status = str(job.get("status") or "")
        if status not in ACTIVE_JOB_STATES:
            continue
        payload = job.get("payload") or {}
        if not isinstance(payload, dict):
            payload = {}
        job_id = str(job.get("id") or "unknown")
        kind = str(job.get("kind") or "job")
        project_name = str(payload.get("project") or "")
        direct_dataset = str(payload.get("dataset") or "")

        matches = False
        if resource_type == "project":
            matches = project_name == resource_id
        elif resource_type == "run":
            matches = (
                project_name == run_project
                and str(payload.get("run_id") or "") in {"", run_id}
            )
        elif resource_type == "dataset" and direct_dataset == resource_id:
            matches = True
        elif project_name and project_name in states_by_name:
            dataset_name, config_name = _project_identity(states_by_name[project_name])
            matches = (
                resource_type == "dataset" and dataset_name == resource_id
            ) or (
                resource_type == "training_config" and config_name == resource_id
            )

        if matches:
            add(
                {
                    "type": "web_job",
                    "id": job_id,
                    "status": status,
                    "reason": f"active {kind} job references this {resource_type}",
                }
            )

    if resource_type == "run":
        state = states_by_name.get(run_project)
        if state is not None:
            for blocker in _project_runtime_markers(state):
                # A live project lock/worker protects every Run. A stale run-status marker
                # only protects the same selected run.
                if blocker["type"] != "training_run" or blocker["id"] == run_id:
                    add(blocker)
        return blockers

    if resource_type == "project":
        state = states_by_name.get(resource_id)
        if state is not None:
            for blocker in _project_runtime_markers(state):
                add(blocker)
        return blockers

    for state in states:
        dataset_name, config_name = _project_identity(state)
        if resource_type == "dataset" and dataset_name != resource_id:
            continue
        if resource_type == "training_config" and config_name != resource_id:
            continue
        runtime = _project_runtime_markers(state)
        for blocker in runtime:
            add(blocker)
        if not runtime and _project_pending_dependency(state):
            add(
                {
                    "type": "training_workspace",
                    "id": state.name,
                    "status": "pending",
                    "reason": "a frozen training workspace is waiting to start",
                }
            )
    return blockers


def assert_deletable(
    resource_type: str,
    resource_id: str,
    *,
    root: Path | None = None,
) -> None:
    blockers = deletion_blockers(resource_type, resource_id, root=root)
    if not blockers:
        return
    preview = "; ".join(
        f"{item['id']} ({item['status']}): {item['reason']}" for item in blockers[:4]
    )
    if len(blockers) > 4:
        preview += f"; plus {len(blockers) - 4} more"
    raise PipelineError(
        f"Cannot delete {resource_type} {resource_id!r} while it has active lifecycle references: {preview}"
    )
