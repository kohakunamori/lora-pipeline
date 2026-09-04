from __future__ import annotations

import json
import os
import socket
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterator

from .config import read_yaml, write_yaml_atomic
from .models import PROJECT_RUN_STEPS, STEP_NAMES, StateError, StepResult, StepStatus


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class ProjectState:
    def __init__(self, path: Path, payload: dict[str, Any]):
        self.path = path
        self.payload = payload
        self._normalize()

    @classmethod
    def load(cls, project_dir: Path) -> "ProjectState":
        path = project_dir / "project.yaml"
        return cls(path, read_yaml(path))

    @classmethod
    def create(
        cls,
        project_dir: Path,
        *,
        name: str,
        concept_type: str,
        base: str,
        trigger: str,
        strategy: str,
        hardware: str = "v100_16gb",
        raw_source: str | None = None,
    ) -> "ProjectState":
        path = project_dir / "project.yaml"
        if path.exists():
            raise StateError(f"Project already exists: {project_dir}")
        now = utc_now()
        payload: dict[str, Any] = {
            "schema_version": 2,
            "project": {
                "name": name,
                "type": concept_type,
                "base": base,
                "trigger": trigger,
                "hardware": hardware,
                "strategy": strategy,
                "raw_source": raw_source,
                "created_at": now,
                "updated_at": now,
                "overrides": {},
                "allow_trigger_only": False,
            },
            "steps": {step: {"status": StepStatus.PENDING.value, "attempts": 0} for step in STEP_NAMES},
            "runs": [],
        }
        if concept_type == "style":
            payload["steps"]["identity"] = {
                "status": StepStatus.SKIPPED.value,
                "attempts": 0,
                "reason": "N/A for style concepts",
                "permanent": True,
                "finished_at": now,
            }
        project_dir.mkdir(parents=True, exist_ok=False)
        for relative in (
            "raw",
            "validation",
            "prepared/generations",
            "review/duplicates",
            "review/outliers",
            "review/captions",
            "cache",
            "runs",
        ):
            (project_dir / relative).mkdir(parents=True, exist_ok=True)
        state = cls(path, payload)
        state.save()
        return state

    @property
    def project_dir(self) -> Path:
        return self.path.parent

    @property
    def name(self) -> str:
        return str(self.payload["project"]["name"])

    @property
    def concept_type(self) -> str:
        return str(self.payload["project"]["type"])

    def _normalize(self) -> None:
        project = self.payload.setdefault("project", {})
        steps = self.payload.setdefault("steps", {})
        self.payload.setdefault("schema_version", 1)
        self.payload.setdefault("runs", [])
        project.setdefault("overrides", {})
        project.setdefault("allow_trigger_only", False)
        budget = project.setdefault("budget", {})
        if "unit" not in budget and "optimizer_steps" in budget:
            budget["unit"] = "legacy_optimizer_steps"
            budget["value"] = int(budget["optimizer_steps"])
        for step in STEP_NAMES:
            steps.setdefault(step, {"status": StepStatus.PENDING.value, "attempts": 0})
            status = steps[step].get("status", StepStatus.PENDING.value)
            try:
                StepStatus(status)
            except ValueError as exc:
                raise StateError(f"Invalid status {status!r} for step {step}") from exc
        if not project.get("name"):
            raise StateError(f"Project name is missing in {self.path}")

    def save(self) -> None:
        self.payload["project"]["updated_at"] = utc_now()
        write_yaml_atomic(self.path, self.payload)

    def step(self, name: str) -> dict[str, Any]:
        if name not in STEP_NAMES:
            raise StateError(f"Unknown step: {name}")
        return self.payload["steps"][name]

    def status(self, name: str) -> StepStatus:
        return StepStatus(self.step(name)["status"])

    def begin(self, name: str, *, input_hash: str, force: bool = False) -> bool:
        if not input_hash:
            raise StateError(f"Step {name} requires a non-empty input fingerprint")
        record = self.step(name)
        current = StepStatus(record["status"])
        if record.get("permanent") and current is StepStatus.SKIPPED:
            return False
        unchanged = record.get("input_hash") == input_hash
        if current in {StepStatus.DONE, StepStatus.SKIPPED} and not force and unchanged:
            return False
        if current in {StepStatus.DONE, StepStatus.SKIPPED} and (force or not unchanged):
            reason = "forced rerun" if force else "input fingerprint changed"
            self.invalidate_downstream(name, reason=reason)
        if current in {StepStatus.RUNNING, StepStatus.INTERRUPTED}:
            record["interrupted_at"] = utc_now()
        record.update(
            {
                "status": StepStatus.RUNNING.value,
                "started_at": utc_now(),
                "attempts": int(record.get("attempts", 0)) + 1,
                "input_hash": input_hash,
            }
        )
        for key in ("last_error", "finished_at", "invalidated_at", "invalidation_reason"):
            record.pop(key, None)
        self.save()
        return True

    def finish(self, name: str, result: StepResult) -> None:
        if result.status not in {StepStatus.DONE, StepStatus.SKIPPED}:
            raise StateError(f"A successful result cannot finish with {result.status}")
        record = self.step(name)
        record.update(
            {
                "status": result.status.value,
                "finished_at": utc_now(),
                "input_hash": result.input_hash or record.get("input_hash"),
                "output_manifest": result.output_manifest,
                "details": dict(result.details),
            }
        )
        self.save()

    def fail(self, name: str, error: BaseException, *, log_path: Path | None = None) -> None:
        record = self.step(name)
        record.update(
            {
                "status": StepStatus.FAILED.value,
                "finished_at": utc_now(),
                "last_error": f"{type(error).__name__}: {error}",
            }
        )
        if log_path is not None:
            record["log_path"] = str(log_path)
        self.save()

    def interrupt(self, name: str, error: BaseException | None = None) -> None:
        record = self.step(name)
        record.update(
            {
                "status": StepStatus.INTERRUPTED.value,
                "interrupted_at": utc_now(),
            }
        )
        if error is not None:
            record["last_error"] = f"{type(error).__name__}: {error}"
        self.save()

    def invalidate_downstream(self, name: str, *, reason: str) -> None:
        from .fingerprints import downstream_steps

        now = utc_now()
        for downstream in downstream_steps(name):
            record = self.step(downstream)
            if record.get("permanent"):
                continue
            record.update(
                {
                    "status": StepStatus.PENDING.value,
                    "invalidated_at": now,
                    "invalidation_reason": f"{name}: {reason}",
                }
            )
            for key in (
                "input_hash",
                "output_manifest",
                "finished_at",
                "started_at",
                "last_error",
                "details",
                "log_path",
            ):
                record.pop(key, None)

    def skip(self, name: str, reason: str, *, input_hash: str) -> None:
        if name not in {"dedup", "identity", "caption", "review", "evaluate"}:
            raise StateError(f"Step {name} is not optional")
        if not self.begin(name, input_hash=input_hash):
            return
        self.finish(
            name,
            StepResult(status=StepStatus.SKIPPED, input_hash=input_hash, details={"reason": reason}),
        )

    def skip_preflight(self, reason: str, *, input_hash: str) -> None:
        """Record the expert-only preflight bypass without making it generally optional."""
        if not self.begin("preflight", input_hash=input_hash):
            return
        self.finish(
            "preflight",
            StepResult(
                status=StepStatus.SKIPPED,
                input_hash=input_hash,
                details={"reason": reason, "warning": True},
            ),
        )

    def next_actionable_step(self) -> str | None:
        for name in PROJECT_RUN_STEPS:
            status = self.status(name)
            if status in {
                StepStatus.PENDING,
                StepStatus.FAILED,
                StepStatus.RUNNING,
                StepStatus.INTERRUPTED,
            }:
                return name
        return None


@contextmanager
def project_lock(project_dir: Path, *, break_lock: bool = False) -> Iterator[None]:
    lock_path = project_dir / ".pipeline.lock"
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    descriptor: int | None = None
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except FileExistsError:
        existing = _read_lock(lock_path)
        live = _lock_is_live(existing)
        if live is True:
            raise StateError(
                "Project is locked by a live process "
                f"(pid={existing.get('pid')} host={existing.get('host')})"
            )
        if not break_lock:
            status = "unverifiable" if live is None else "stale"
            raise StateError(
                f"Project has a {status} lock; inspect it and retry with --break-lock: {existing}"
            )
        lock_path.unlink(missing_ok=True)
        descriptor = os.open(lock_path, flags, 0o600)

    token = uuid.uuid4().hex
    payload = {
        "schema_version": 1,
        "token": token,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "boot_id": _boot_id(),
        "process_start_ticks": _process_start_ticks(os.getpid()),
        "started_at": utc_now(),
    }
    assert descriptor is not None
    try:
        os.write(descriptor, (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"))
        os.fsync(descriptor)
        yield
    finally:
        os.close(descriptor)
        try:
            current = _read_lock(lock_path)
        except StateError:
            current = {}
        if current.get("token") == token:
            lock_path.unlink(missing_ok=True)


def execute_step(
    state: ProjectState,
    name: str,
    handler: Callable[[], StepResult],
    *,
    input_hash: str,
    force: bool = False,
) -> StepResult:
    if not state.begin(name, input_hash=input_hash, force=force):
        record = state.step(name)
        return StepResult(
            status=StepStatus(record["status"]),
            input_hash=record.get("input_hash"),
            output_manifest=record.get("output_manifest"),
            details={"reused": True, **dict(record.get("details", {}))},
        )
    try:
        result = handler()
    except (KeyboardInterrupt, SystemExit) as exc:
        state.interrupt(name, exc)
        raise
    except BaseException as exc:
        state.fail(name, exc, log_path=getattr(exc, "log_path", None))
        raise
    details = dict(result.details)
    if result.input_hash and result.input_hash != input_hash:
        details.setdefault("handler_input_hash", result.input_hash)
    normalized = StepResult(
        status=result.status,
        input_hash=input_hash,
        output_manifest=result.output_manifest,
        details=details,
    )
    state.finish(name, normalized)
    return normalized


def _read_lock(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StateError(f"Project lock is unreadable: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise StateError(f"Project lock is invalid: {path}")
    return payload


def _lock_is_live(payload: dict[str, Any]) -> bool | None:
    if payload.get("host") != socket.gethostname():
        return None
    if payload.get("boot_id") and payload.get("boot_id") != _boot_id():
        return False
    try:
        pid = int(payload["pid"])
    except (KeyError, TypeError, ValueError):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    expected_start = payload.get("process_start_ticks")
    actual_start = _process_start_ticks(pid)
    if expected_start is not None and actual_start is not None:
        return str(expected_start) == str(actual_start)
    return True


def _boot_id() -> str | None:
    path = Path("/proc/sys/kernel/random/boot_id")
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _process_start_ticks(pid: int) -> str | None:
    path = Path("/proc") / str(pid) / "stat"
    try:
        fields = path.read_text(encoding="utf-8").split()
    except OSError:
        return None
    return fields[21] if len(fields) > 21 else None
