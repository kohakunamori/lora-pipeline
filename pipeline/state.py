from __future__ import annotations

import os
import socket
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterator

from .config import read_yaml, write_yaml_atomic
from .models import STEP_NAMES, StateError, StepResult, StepStatus


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
            "schema_version": 1,
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
            },
            "steps": {step: {"status": StepStatus.PENDING.value, "attempts": 0} for step in STEP_NAMES},
            "runs": [],
        }
        if concept_type == "style":
            payload["steps"]["identity"] = {
                "status": StepStatus.SKIPPED.value,
                "attempts": 0,
                "reason": "N/A for style concepts",
                "finished_at": now,
            }
        project_dir.mkdir(parents=True, exist_ok=False)
        for relative in (
            "raw",
            "prepared/images",
            "prepared/captions",
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

    def begin(self, name: str, *, input_hash: str | None = None, force: bool = False) -> bool:
        record = self.step(name)
        current = StepStatus(record["status"])
        if current is StepStatus.DONE and not force and (
            input_hash is None or record.get("input_hash") == input_hash
        ):
            return False
        if current is StepStatus.RUNNING:
            record["interrupted_at"] = utc_now()
        record.update(
            {
                "status": StepStatus.RUNNING.value,
                "started_at": utc_now(),
                "attempts": int(record.get("attempts", 0)) + 1,
                "input_hash": input_hash,
            }
        )
        record.pop("last_error", None)
        record.pop("finished_at", None)
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

    def skip(self, name: str, reason: str) -> None:
        if name not in {"dedup", "identity", "caption", "review", "evaluate"}:
            raise StateError(f"Step {name} is not optional")
        self.finish(
            name,
            StepResult(status=StepStatus.SKIPPED, details={"reason": reason}),
        )

    def skip_preflight(self, reason: str) -> None:
        """Record the expert-only preflight bypass without making it generally optional."""
        self.finish(
            "preflight",
            StepResult(status=StepStatus.SKIPPED, details={"reason": reason, "warning": True}),
        )

    def next_actionable_step(self) -> str | None:
        for name in STEP_NAMES:
            status = self.status(name)
            if status in {StepStatus.PENDING, StepStatus.FAILED, StepStatus.RUNNING}:
                return name
        return None


@contextmanager
def project_lock(project_dir: Path, *, force: bool = False) -> Iterator[None]:
    lock_path = project_dir / ".pipeline.lock"
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except FileExistsError:
        if not force:
            detail = lock_path.read_text(encoding="utf-8", errors="replace").strip()
            raise StateError(f"Project is locked by another process ({detail or 'unknown'})")
        lock_path.unlink(missing_ok=True)
        descriptor = os.open(lock_path, flags, 0o600)
    try:
        payload = f"pid={os.getpid()} host={socket.gethostname()} started={utc_now()}\n"
        os.write(descriptor, payload.encode("utf-8"))
        os.fsync(descriptor)
        yield
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def execute_step(
    state: ProjectState,
    name: str,
    handler: Callable[[], StepResult],
    *,
    input_hash: str | None = None,
    force: bool = False,
) -> StepResult:
    if not state.begin(name, input_hash=input_hash, force=force):
        record = state.step(name)
        return StepResult(
            status=StepStatus.DONE,
            input_hash=record.get("input_hash"),
            output_manifest=record.get("output_manifest"),
            details={"reused": True},
        )
    try:
        result = handler()
    except BaseException as exc:
        state.fail(name, exc, log_path=getattr(exc, "log_path", None))
        raise
    state.finish(name, result)
    return result
