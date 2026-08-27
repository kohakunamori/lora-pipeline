from __future__ import annotations

from io import StringIO
from pathlib import Path

import yaml
from PIL import Image
from rich.console import Console

from pipeline import i18n
from pipeline.models import STEP_NAMES
from pipeline.state import ProjectState
from pipeline.wizard import Wizard


def _console() -> tuple[Console, StringIO]:
    stream = StringIO()
    return Console(file=stream, force_terminal=False, width=140), stream


def _state(root: Path, *, name: str = "demo", concept: str = "character") -> ProjectState:
    (root / "projects").mkdir(parents=True, exist_ok=True)
    return ProjectState.create(
        root / "projects" / name,
        name=name,
        concept_type=concept,
        base="base",
        trigger=f"zz_{name}",
        strategy="quality",
    )


def test_home_is_a_dashboard_and_can_exit_without_creating_a_project(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LORA_PIPELINE_ROOT", str(tmp_path))
    i18n.set_language("en")
    console, stream = _console()
    wizard = Wizard(console=console)
    monkeypatch.setattr(wizard, "_menu", lambda *args, **kwargs: "quit")

    wizard.home()

    output = stream.getvalue()
    assert "Interactive mode" in output
    assert "No projects yet" in output
    assert not (tmp_path / "projects").exists()


def test_guided_workflow_preferences_are_saved_in_project_state(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LORA_PIPELINE_ROOT", str(tmp_path))
    state = _state(tmp_path)
    console, _ = _console()
    wizard = Wizard(console=console)
    confirms = iter([False, False, True, False, False])
    monkeypatch.setattr(wizard, "_confirm", lambda *args, **kwargs: next(confirms))
    monkeypatch.setattr(
        wizard,
        "_menu",
        lambda *args, **kwargs: "existing_passthrough",
    )

    preferences = wizard.configure_workflow(state.name)

    reloaded = ProjectState.load(state.project_dir)
    assert preferences == reloaded.payload["project"]["interactive_preferences"]
    assert preferences["run_dedup"] is False
    assert preferences["run_identity"] is False
    assert preferences["caption_mode"] == "existing_passthrough"
    assert preferences["allow_trigger_only"] is True
    assert preferences["run_review"] is False
    assert preferences["run_screening_evaluation"] is False


def test_new_project_can_be_created_entirely_through_prompt_wrappers(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LORA_PIPELINE_ROOT", str(tmp_path))
    (tmp_path / "bases").mkdir(parents=True)
    checkpoint = tmp_path / "base.safetensors"
    checkpoint.write_bytes(b"base")
    (tmp_path / "bases" / "registry.yaml").write_text(
        yaml.safe_dump(
            {
                "bases": {
                    "base": {
                        "name": "Base",
                        "path": str(checkpoint),
                        "family": "illustrious_sdxl",
                        "prediction_type": "epsilon",
                        "sha256": None,
                        "enabled": True,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    image = dataset / "sample.png"
    Image.new("RGB", (64, 64), "red").save(image)
    image.with_suffix(".txt").write_text("portrait\n", encoding="utf-8")

    console, _ = _console()
    wizard = Wizard(console=console)
    monkeypatch.setattr(wizard, "_ask_project_name", lambda: "interactive-demo")
    monkeypatch.setattr(wizard, "_ask_dataset", lambda: (dataset, 1, 1))
    monkeypatch.setattr(wizard, "_ask_trigger", lambda name: "zz_interactive_demo")
    menu_answers = iter(["character", "base", "quality"])
    monkeypatch.setattr(wizard, "_menu", lambda *args, **kwargs: next(menu_answers))
    monkeypatch.setattr(wizard, "_ask_positive_int", lambda *args, **kwargs: 1000)
    confirms = iter([True, False, False])
    monkeypatch.setattr(wizard, "_confirm", lambda *args, **kwargs: next(confirms))

    state = wizard.new_project()

    assert state is not None
    assert state.name == "interactive-demo"
    assert (state.project_dir / "raw" / "sample.png").is_file()
    assert (state.project_dir / "raw" / "sample.txt").read_text(encoding="utf-8").strip() == "portrait"


def test_checkpoint_picker_accepts_numbered_multi_selection(tmp_path, monkeypatch) -> None:
    console, _ = _console()
    wizard = Wizard(console=console)
    checkpoints = []
    for index in range(3):
        path = tmp_path / f"candidate-{index}.safetensors"
        path.write_bytes(b"x" * (index + 1))
        checkpoints.append(path)
    monkeypatch.setattr(wizard, "_ask_text", lambda *args, **kwargs: "1, 3")

    selected = wizard._select_checkpoints(
        checkpoints,
        title="Finalists",
        minimum=1,
        maximum=2,
    )

    assert selected == [checkpoints[0], checkpoints[2]]


def test_recommended_action_moves_from_screening_to_full_to_promotion(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LORA_PIPELINE_ROOT", str(tmp_path))
    state = _state(tmp_path)
    for step in STEP_NAMES:
        state.payload["steps"][step]["status"] = "done"
    run = {
        "id": "run-1",
        "path": str(state.project_dir / "runs" / "run-1"),
        "status": "trained",
        "checkpoints": [],
        "evaluation": {},
    }
    state.payload["runs"].append(run)
    state.save()
    wizard = Wizard(console=_console()[0])
    monkeypatch.setattr(wizard, "_successful_runs", lambda current: [run])

    assert wizard._recommended_action(state) == "run screening evaluation"
    run["evaluation"]["screening"] = {}
    assert wizard._recommended_action(state) == "select finalists for full evaluation"
    run["evaluation"]["full"] = {}
    assert wizard._recommended_action(state) == "review sheets and promote a checkpoint"
    run["promotion"] = {"checkpoint": "candidate.safetensors"}
    assert wizard._recommended_action(state).startswith("complete")
