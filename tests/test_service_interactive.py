from __future__ import annotations

from pathlib import Path

from pipeline import service
from pipeline.models import PROJECT_RUN_STEPS, StepResult, StepStatus


class FakeState:
    project_dir = Path("/fake/project")

    def step(self, name: str) -> dict[str, object]:
        return {"permanent": False}

    def status(self, name: str) -> StepStatus:
        return StepStatus.PENDING


def test_run_remaining_forwards_resume_and_emits_only_training_callbacks(monkeypatch) -> None:
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
    assert callbacks == list(PROJECT_RUN_STEPS)
    assert dict(calls)["train"] == "run-interrupted"
    assert all(resume is None for step, resume in calls if step != "train")
    assert {"inspect", "dedup", "identity", "caption", "review", "evaluate"}.isdisjoint(
        dict(calls)
    )


def test_dry_run_preflight_bypass_has_no_side_effect(monkeypatch) -> None:
    state = FakeState()
    monkeypatch.setattr(service.ProjectState, "load", lambda path: state)
    monkeypatch.setattr(
        service,
        "run_single_step",
        lambda current, name, **kwargs: StepResult(details={"reused": True}),
    )
    monkeypatch.setattr(
        service,
        "skip_preflight_step",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not persist")),
    )

    results = service.run_remaining(state, skip_preflight=True, dry_run=True)

    assert len(results) == 1
    step, result = results[0]
    assert step == "preflight"
    assert result.status is StepStatus.SKIPPED
    assert result.details["dry_run"] is True
    assert result.details["would_skip"] == "preflight"
