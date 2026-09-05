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

    first = execute_step(state, "materialize", handler, input_hash="raw-v1")
    second = execute_step(state, "materialize", handler, input_hash="raw-v1")
    third = execute_step(state, "materialize", handler, input_hash="raw-v2")
    assert first.status is StepStatus.DONE
    assert second.details["reused"] is True
    assert third.details["count"] == 2
    assert calls == 2
    reloaded = ProjectState.load(state.project_dir)
    assert reloaded.step("materialize")["input_hash"] == "raw-v2"


def test_changed_materialization_invalidates_preflight_and_train(tmp_path) -> None:
    state = _project(tmp_path)
    execute_step(state, "materialize", lambda: StepResult(), input_hash="materialize-v1")
    execute_step(state, "preflight", lambda: StepResult(), input_hash="preflight-v1")
    execute_step(state, "train", lambda: StepResult(), input_hash="train-v1")

    execute_step(state, "materialize", lambda: StepResult(), input_hash="materialize-v2")
    reloaded = ProjectState.load(state.project_dir)
    assert reloaded.status("materialize") is StepStatus.DONE
    for name in ("preflight", "train"):
        assert reloaded.status(name) is StepStatus.PENDING
        assert "materialize" in reloaded.step(name)["invalidation_reason"]


def test_prepare_alias_targets_materialize(tmp_path) -> None:
    state = _project(tmp_path)
    execute_step(state, "prepare", lambda: StepResult(), input_hash="snapshot-v1")
    reloaded = ProjectState.load(state.project_dir)
    assert reloaded.status("prepare") is StepStatus.DONE
    assert reloaded.step("prepare") is reloaded.step("materialize")
    assert reloaded.step("materialize")["input_hash"] == "snapshot-v1"


def test_legacy_project_steps_are_migrated_to_opaque_history(tmp_path) -> None:
    state = _project(tmp_path, "legacy", "style")
    state.payload["steps"]["identity"] = {
        "status": StepStatus.SKIPPED.value,
        "attempts": 0,
        "permanent": True,
        "reason": "legacy style n/a",
    }
    state.payload["steps"]["evaluate"] = {
        "status": StepStatus.DONE.value,
        "attempts": 1,
        "input_hash": "legacy-eval",
    }
    state.save()

    reloaded = ProjectState.load(state.project_dir)
    assert set(reloaded.payload["steps"]) == {"materialize", "preflight", "train"}
    assert reloaded.payload["legacy_steps"]["identity"]["permanent"] is True
    assert reloaded.payload["legacy_steps"]["evaluate"]["input_hash"] == "legacy-eval"
    with pytest.raises(StateError, match="Unknown Project step"):
        reloaded.status("identity")


def test_failure_and_interruption_are_persisted(tmp_path) -> None:
    state = _project(tmp_path, "failed")
    with pytest.raises(RuntimeError, match="boom"):
        execute_step(
            state,
            "materialize",
            lambda: (_ for _ in ()).throw(RuntimeError("boom")),
            input_hash="raw-v1",
        )
    assert ProjectState.load(state.project_dir).status("materialize") is StepStatus.FAILED

    with pytest.raises(KeyboardInterrupt):
        execute_step(
            ProjectState.load(state.project_dir),
            "materialize",
            lambda: (_ for _ in ()).throw(KeyboardInterrupt()),
            input_hash="raw-v1",
        )
    assert ProjectState.load(state.project_dir).status("materialize") is StepStatus.INTERRUPTED


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
