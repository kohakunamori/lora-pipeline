from __future__ import annotations

from pathlib import Path

import yaml
from PIL import Image

from pipeline.dataset_workspace import DatasetWorkspace, create_project_from_dataset
from pipeline.models import PROJECT_RUN_STEPS
from pipeline.state import ProjectState


def _registry(root: Path) -> None:
    (root / "bases").mkdir(parents=True, exist_ok=True)
    checkpoint = root / "base.safetensors"
    checkpoint.write_bytes(b"base")
    (root / "bases" / "registry.yaml").write_text(
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


def test_new_dataset_project_contains_only_canonical_project_steps(tmp_path: Path) -> None:
    _registry(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    image = source / "a.png"
    Image.new("RGB", (768, 1024), "red").save(image)
    image.with_suffix(".txt").write_text("portrait, smile\n", encoding="utf-8")

    workspace = DatasetWorkspace.create("demo", concept_type="style", root=tmp_path)
    workspace.add_source_from_directory(source, kind="image_directory")
    workspace.analyze_duplicates()

    state = create_project_from_dataset(
        workspace,
        name="run-demo",
        base="base",
        trigger="zz_demo",
        strategy="quality",
        images_seen=1000,
        root=tmp_path,
    )

    assert tuple(state.payload["steps"]) == PROJECT_RUN_STEPS
    assert not state.payload.get("legacy_steps")
    dedup = state.payload["project"]["dataset_curation"]["analyses"]["dedup"]
    assert dedup["fresh"] is True
    assert dedup["frozen"] is True
    assert dedup["frozen_manifest"] == "review/duplicates/manifest.json"
    assert (state.project_dir / dedup["frozen_manifest"]).is_file()

    reloaded = ProjectState.load(state.project_dir)
    assert tuple(reloaded.payload["steps"]) == PROJECT_RUN_STEPS
    assert not reloaded.payload.get("legacy_steps")
