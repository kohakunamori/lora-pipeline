from __future__ import annotations

from pathlib import Path

from pipeline import service
from pipeline.models import STEP_NAMES, StepResult, StepStatus


class FakeState:
    project_dir = Path("/fake/project")

    def step(self, name: str) -> dict[str, object]:
        return {"permanent": False}

    def status(self, name: str) -> StepStatus:
        return StepStatus.PENDING


def test_run_remaining_forwards_resume_and_emits_step_callbacks(monkeypatch) -> None:
    state = FakeState()
    calls: list[tuple[str, object]] = []
    callbacks: list[str] = []
    monkeypatch.setattr(service.ProjectState, "load", lambda path: state)

    def fake_run_single_step(current, name, **kwargs):
        calls.append((name, kwargs.get("resume_run")))
        return StepResult(details={"reused": True})

    monkeypatch.setattr(service, "run_single_step", fake_run_single_step)

    results = service.run_remaining(
        state,
        resume_run="run-interrupted",
        on_step=callbacks.append,
    )

    assert results == []
    assert callbacks == list(STEP_NAMES)
    assert dict(calls)["train"] == "run-interrupted"
    assert all(resume is None for step, resume in calls if step != "train")
