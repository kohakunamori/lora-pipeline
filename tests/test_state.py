from __future__ import annotations

from pipeline.models import StepResult, StepStatus
from pipeline.state import ProjectState, execute_step


def test_state_transitions_and_resume(tmp_path) -> None:
    state = ProjectState.create(
        tmp_path / "project",
        name="test",
        concept_type="character",
        base="base",
        trigger="zz_test",
        strategy="quality",
    )
    calls = 0

    def handler() -> StepResult:
        nonlocal calls
        calls += 1
        return StepResult(input_hash="abc", output_manifest="manifest.json", details={"count": 3})

    first = execute_step(state, "inspect", handler)
    second = execute_step(state, "inspect", handler)
    assert first.status is StepStatus.DONE
    assert second.details["reused"] is True
    assert calls == 1
    reloaded = ProjectState.load(state.project_dir)
    assert reloaded.status("inspect") is StepStatus.DONE
    assert reloaded.step("inspect")["input_hash"] == "abc"


def test_style_identity_is_n_a(tmp_path) -> None:
    state = ProjectState.create(
        tmp_path / "style",
        name="style",
        concept_type="style",
        base="base",
        trigger="zz_style",
        strategy="quality",
    )
    assert state.status("identity") is StepStatus.SKIPPED
    assert state.step("identity")["reason"] == "N/A for style concepts"


def test_failure_is_persisted_and_retryable(tmp_path) -> None:
    state = ProjectState.create(
        tmp_path / "failed",
        name="failed",
        concept_type="character",
        base="base",
        trigger="zz_failed",
        strategy="quality",
    )
    try:
        execute_step(state, "inspect", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    except RuntimeError:
        pass
    reloaded = ProjectState.load(state.project_dir)
    assert reloaded.status("inspect") is StepStatus.FAILED
    assert "boom" in reloaded.step("inspect")["last_error"]
    execute_step(reloaded, "inspect", lambda: StepResult())
    assert ProjectState.load(state.project_dir).status("inspect") is StepStatus.DONE
