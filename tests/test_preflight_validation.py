from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml
from PIL import Image

from pipeline.config import repository_root, sha256_file, write_json_atomic
from pipeline.dataset.image_info import inspect_dataset
from pipeline.materialization import run as materialize
from pipeline.models import PipelineError
from pipeline.state import ProjectState
from pipeline.steps import preflight


def _project(tmp_path: Path, monkeypatch) -> ProjectState:
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
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LORA_PIPELINE_ROOT", str(root))
    state = ProjectState.create(
        root / "projects" / "preflight",
        name="preflight",
        concept_type="character",
        base="base",
        trigger="zz_preflight",
        strategy="quality",
    )
    image = state.project_dir / "raw" / "train.png"
    Image.new("RGB", (640, 832), "red").save(image)
    image.with_suffix(".txt").write_text(
        "zz_preflight, portrait, red dress\n", encoding="utf-8"
    )
    write_json_atomic(
        state.project_dir / "dataset-manifest.json",
        inspect_dataset(state.project_dir / "raw"),
    )
    materialize(state)
    return state


def test_preflight_accepts_distinct_valid_holdout(tmp_path, monkeypatch) -> None:
    state = _project(tmp_path, monkeypatch)
    Image.new("RGB", (832, 640), "blue").save(
        state.project_dir / "validation" / "holdout.png"
    )
    result = preflight.run(state, minimum_free_gib=0)
    validation = result.details["checks"]["validation"]
    assert validation["summary"]["valid_images"] == 1
    assert validation["exact_training_overlap"] == []
    assert validation["excluded_from_training"] is True


def test_preflight_blocks_corrupt_holdout(tmp_path, monkeypatch) -> None:
    state = _project(tmp_path, monkeypatch)
    (state.project_dir / "validation" / "corrupt.png").write_bytes(b"not an image")
    with pytest.raises(PipelineError, match="Validation split contains 1 corrupt image"):
        preflight.run(state, minimum_free_gib=0)


def test_preflight_blocks_exact_training_duplicate_in_holdout(tmp_path, monkeypatch) -> None:
    state = _project(tmp_path, monkeypatch)
    shutil.copy2(
        state.project_dir / "raw" / "train.png",
        state.project_dir / "validation" / "leaked.png",
    )
    with pytest.raises(PipelineError, match="exact training duplicate"):
        preflight.run(state, minimum_free_gib=0)
