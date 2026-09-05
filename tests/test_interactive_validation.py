from __future__ import annotations

import shutil
from io import StringIO

import pytest
from PIL import Image
from rich.console import Console

from pipeline.interactive_app import InteractiveWizard
from pipeline.models import PipelineError, StepStatus
from pipeline.state import ProjectState


def _project(tmp_path, monkeypatch) -> ProjectState:
    monkeypatch.setenv("LORA_PIPELINE_ROOT", str(tmp_path))
    (tmp_path / "projects").mkdir()
    state = ProjectState.create(
        tmp_path / "projects" / "demo",
        name="demo",
        concept_type="character",
        base="base",
        trigger="zz_demo",
        strategy="quality",
    )
    Image.new("RGB", (64, 64), "red").save(state.project_dir / "raw" / "train.png")
    for step in ("materialize", "preflight", "train"):
        state.payload["steps"][step].update(
            {"status": "done", "input_hash": f"old-{step}"}
        )
    state.save()
    return state


def _wizard() -> InteractiveWizard:
    return InteractiveWizard(
        console=Console(file=StringIO(), force_terminal=False, width=120)
    )


def test_interactive_validation_import_copies_holdouts_without_invalidating_training(
    tmp_path, monkeypatch
) -> None:
    state = _project(tmp_path, monkeypatch)
    source = tmp_path / "holdout"
    source.mkdir()
    Image.new("RGB", (64, 64), "blue").save(source / "validation.png")
    wizard = _wizard()
    monkeypatch.setattr(wizard, "_ask_text", lambda *args, **kwargs: str(source))
    monkeypatch.setattr(wizard, "_confirm", lambda *args, **kwargs: True)

    wizard.import_validation_images(state.name)

    reloaded = ProjectState.load(state.project_dir)
    assert (state.project_dir / "validation" / "validation.png").is_file()
    assert (state.project_dir / "raw" / "train.png").is_file()
    for step in ("materialize", "preflight", "train"):
        assert reloaded.status(step) is StepStatus.DONE
    assert "evaluate" not in reloaded.payload["steps"]
    assert reloaded.payload["project"]["validation_imports"][-1]["imported_images"] == 1


def test_interactive_validation_import_blocks_exact_training_overlap(
    tmp_path, monkeypatch
) -> None:
    state = _project(tmp_path, monkeypatch)
    source = tmp_path / "holdout"
    source.mkdir()
    shutil.copy2(state.project_dir / "raw" / "train.png", source / "copied.png")
    wizard = _wizard()
    monkeypatch.setattr(wizard, "_ask_text", lambda *args, **kwargs: str(source))
    monkeypatch.setattr(wizard, "_confirm", lambda *args, **kwargs: True)

    with pytest.raises(PipelineError, match="exactly overlap"):
        wizard.import_validation_images(state.name)

    assert not list((state.project_dir / "validation").glob("*.png"))


def test_interactive_budget_change_invalidates_only_training_dependent_steps(
    tmp_path, monkeypatch
) -> None:
    state = _project(tmp_path, monkeypatch)
    state.payload["project"]["budget"] = {"unit": "images_seen", "value": 1000}
    state.save()
    wizard = _wizard()
    menu_answers = iter(["budget", "back"])
    monkeypatch.setattr(wizard, "_menu", lambda *args, **kwargs: next(menu_answers))
    monkeypatch.setattr(wizard, "_ask_positive_int", lambda *args, **kwargs: 2400)

    wizard.project_settings(state.name)

    reloaded = ProjectState.load(state.project_dir)
    assert reloaded.payload["project"]["budget"] == {
        "unit": "images_seen",
        "value": 2400,
    }
    assert reloaded.status("materialize") is StepStatus.DONE
    for step in ("preflight", "train"):
        assert reloaded.status(step) is StepStatus.PENDING
