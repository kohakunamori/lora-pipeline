from __future__ import annotations

import json

import pytest

from pipeline.models import StateError, StepResult, StepStatus
from pipeline.state import ProjectState, execute_step, project_lock


def _project(tmp_path, name: str = "test", concept: str = "character") -> ProjectState:
    return ProjectState.create(
        tmp_path / name,
        name=name,
        concept_type=concept,
        base="base",
        trigger=f"zz_{name}",
        strategy="quality",
    )


def test_state_reuses_only_matching_fingerprint(tmp_path) -> None:
    state = _project(tmp_path)
    calls = 0

    def handler() -> StepResult:
        nonlocal calls
        calls += 1
        return StepResult(output_manifest="manifest.json", details={"count": calls})

    first = execute_step(state, "inspect", handler, input_hash="raw-v1")
    second = execute_step(state, "inspect", handler, input_hash="raw-v1")
    third = execute_step(state, "inspect", handler, input_hash="raw-v2")
    assert first.status is StepStatus.DONE
    assert second.details["reused"] is True
    assert third.details["count"] == 2
    assert calls == 2
    reloaded = ProjectState.load(state.project_dir)
    assert reloaded.step("inspect")["input_hash"] == "raw-v2"


def test_changed_caption_utility_invalidates_only_legacy_review(tmp_path) -> None:
    state = _project(tmp_path)
    execute_step(state, "inspect", lambda: StepResult(), input_hash="raw-v1")
    execute_step(state, "caption", lambda: StepResult(), input_hash="caption-v1")
    execute_step(state, "review", lambda: StepResult(), input_hash="review-v1")
    execute_step(state, "prepare", lambda: StepResult(), input_hash="prepare-v1")
    execute_step(state, "preflight", lambda: StepResult(), input_hash="preflight-v1")
    execute_step(state, "train", lambda: StepResult(), input_hash="train-v1")
    execute_step(state, "evaluate", lambda: StepResult(), input_hash="eval-v1")

    execute_step(state, "caption", lambda: StepResult(), input_hash="caption-v2")
    reloaded = ProjectState.load(state.project_dir)
    assert reloaded.status("inspect") is StepStatus.DONE
    assert reloaded.status("caption") is StepStatus.DONE
    assert reloaded.status("review") is StepStatus.PENDING
    assert "caption" in reloaded.step("review")["invalidation_reason"]

    # Materialization fingerprints the raw images/captions and effective caption
    # policy directly, so changing a compatibility caption-step record must not
    # invalidate the independent training lifecycle.
    for name in ("prepare", "preflight", "train", "evaluate"):
        assert reloaded.status(name) is StepStatus.DONE


def test_style_identity_is_permanently_n_a(tmp_path) -> None:
    state = _project(tmp_path, "style", "style")
    assert state.status("identity") is StepStatus.SKIPPED
    assert state.step("identity")["permanent"] is True
    assert state.begin("identity", input_hash="anything", force=True) is False


def test_failure_and_interruption_are_persisted(tmp_path) -> None:
    state = _project(tmp_path, "failed")
    with pytest.raises(RuntimeError, match="boom"):
        execute_step(
            state,
            "inspect",
            lambda: (_ for _ in ()).throw(RuntimeError("boom")),
            input_hash="raw-v1",
        )
    assert ProjectState.load(state.project_dir).status("inspect") is StepStatus.FAILED

    with pytest.raises(KeyboardInterrupt):
        execute_step(
            ProjectState.load(state.project_dir),
            "inspect",
            lambda: (_ for _ in ()).throw(KeyboardInterrupt()),
            input_hash="raw-v1",
        )
    assert ProjectState.load(state.project_dir).status("inspect") is StepStatus.INTERRUPTED


def test_live_lock_cannot_be_overridden(tmp_path) -> None:
    state = _project(tmp_path, "locked")
    with project_lock(state.project_dir):
        payload = json.loads((state.project_dir / ".pipeline.lock").read_text(encoding="utf-8"))
        assert payload["pid"] > 0
        with pytest.raises(StateError, match="live process"):
            with project_lock(state.project_dir, break_lock=True):
                pass


def test_explicit_break_lock_removes_stale_lock(tmp_path) -> None:
    state = _project(tmp_path, "stale")
    lock = state.project_dir / ".pipeline.lock"
    lock.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "token": "stale",
                "pid": 99999999,
                "host": __import__("socket").gethostname(),
                "started_at": "2020-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(StateError, match="stale lock"):
        with project_lock(state.project_dir):
            pass
    with project_lock(state.project_dir, break_lock=True):
        assert lock.is_file()
    assert not lock.exists()
