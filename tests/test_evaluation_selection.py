from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml
from PIL import Image

from pipeline.config import repository_root, sha256_file
from pipeline.evaluation.generation import GenerationBackend
from pipeline.models import GeneratedImage, GenerationCase, PipelineError
from pipeline.state import ProjectState
from pipeline.steps import evaluate, prepare


class FailIfCalledGenerator(GenerationBackend):
    def generate(
        self,
        cases: list[GenerationCase],
        *,
        base_path: Path,
        output_dir: Path,
        settings: dict,
        verbose: int = 0,
    ) -> list[GeneratedImage]:
        del cases, base_path, output_dir, settings, verbose
        raise AssertionError("generation must not start for an invalid finalist selection")


def _state_with_checkpoints(tmp_path, monkeypatch, count: int = 3) -> ProjectState:
    source = repository_root()
    root = tmp_path / "repo"
    shutil.copytree(source / "profiles", root / "profiles")
    (root / "bases").mkdir(parents=True)
    (root / "projects").mkdir()
    base = root / "base.safetensors"
    base.write_bytes(b"base")
    (root / "bases" / "registry.yaml").write_text(
        yaml.safe_dump(
            {
                "bases": {
                    "base": {
                        "name": "Base",
                        "path": str(base),
                        "family": "illustrious_sdxl",
                        "prediction_type": "epsilon",
                        "sha256": sha256_file(base),
                        "enabled": True,
                        "generation_defaults": {"sampler": "euler_a", "cfg": 4.5, "steps": 2},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LORA_PIPELINE_ROOT", str(root))
    state = ProjectState.create(
        root / "projects" / "eval",
        name="eval",
        concept_type="character",
        base="base",
        trigger="zz_eval",
        strategy="quality",
    )
    image = state.project_dir / "raw" / "sample.png"
    Image.new("RGB", (64, 64), "red").save(image)
    image.with_suffix(".txt").write_text("zz_eval, portrait\n", encoding="utf-8")
    prepare.run(state)
    run_dir = state.project_dir / "runs" / "run-1"
    checkpoints: list[str] = []
    for index in range(count):
        checkpoint = run_dir / "checkpoints" / f"candidate-{index}.safetensors"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"lora-{index}".encode())
        checkpoints.append(str(checkpoint))
    state.payload["runs"].append(
        {
            "id": "run-1",
            "path": str(run_dir),
            "status": "trained",
            "checkpoints": checkpoints,
            "accounting": {"images_seen": 100},
        }
    )
    state.save()
    return state


def test_full_evaluation_rejects_more_than_two_explicit_finalists(tmp_path, monkeypatch) -> None:
    state = _state_with_checkpoints(tmp_path, monkeypatch, count=3)
    with pytest.raises(PipelineError, match="one or two finalists"):
        evaluate.run(
            state,
            backend=FailIfCalledGenerator(),
            stage="full",
            run_id="run-1",
            checkpoint_names=["candidate-0", "candidate-1", "candidate-2"],
        )


def test_full_evaluation_requires_explicit_finalists_when_many_candidates(
    tmp_path, monkeypatch
) -> None:
    state = _state_with_checkpoints(tmp_path, monkeypatch, count=3)
    with pytest.raises(PipelineError, match="explicit finalists"):
        evaluate.run(
            state,
            backend=FailIfCalledGenerator(),
            stage="full",
            run_id="run-1",
        )
