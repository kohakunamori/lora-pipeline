from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image
from safetensors import safe_open
from safetensors.numpy import save_file

from pipeline.activation_caption import apply_activation_group_captions
from pipeline.activation_recipe import (
    activation_recipe_snapshot,
    activation_safetensors_metadata,
    activation_usage_hint,
    load_activation_recipe,
    set_group_images,
    tag_candidates,
    upsert_character_tags_group,
)
from pipeline.config import sha256_file, write_json_atomic
from pipeline.dataset_workspace import DatasetWorkspace
from pipeline.evaluation import promotion
from pipeline.models import PipelineError, StepResult, StepStatus
from pipeline.state import ProjectState


def _image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (768, 1024), "white").save(path)


def _character_workspace(tmp_path: Path) -> DatasetWorkspace:
    source = tmp_path / "source"
    captions = {
        "a.png": "1girl, purple hair, purple eyes, ponytail, white bikini, swimsuit\n",
        "b.png": "1girl, purple hair, purple eyes, ponytail, white bikini, swimsuit\n",
        "c.png": "1girl, purple hair, purple eyes, long hair, school uniform, blazer\n",
        "d.png": "1girl, purple hair, purple eyes, long hair, school uniform, blazer\n",
    }
    for name, caption in captions.items():
        path = source / name
        _image(path)
        path.with_suffix(".txt").write_text(caption, encoding="utf-8")
    workspace = DatasetWorkspace.create("misuzu", concept_type="character", root=tmp_path)
    workspace.add_source_from_directory(source, kind="image_directory", label="cards")
    return workspace


def test_character_groups_use_selected_identity_and_outfit_candidates(tmp_path: Path) -> None:
    workspace = _character_workspace(tmp_path)
    identity = {row["tag"] for row in tag_candidates(workspace, kind="identity")}
    outfit = {row["tag"] for row in tag_candidates(workspace, kind="outfit")}
    assert {"purple hair", "purple eyes", "ponytail", "long hair"} <= identity
    assert {"white bikini", "swimsuit", "school uniform"} <= outfit

    upsert_character_tags_group(
        workspace,
        name="NIC26 Swimsuit",
        group_tag="misuzu_nic26",
        identity_tags=["purple hair", "purple eyes", "ponytail"],
        outfit_tags=["white bikini", "swimsuit"],
    )
    upsert_character_tags_group(
        workspace,
        name="School Uniform",
        group_tag="misuzu_school",
        identity_tags=["purple hair", "purple eyes", "long hair"],
        outfit_tags=["school uniform", "blazer"],
    )
    items = workspace.items(include_disabled=False, include_excluded=False)
    swim = [item.key for item in items if item.relative.name in {"a.png", "b.png"}]
    school = [item.key for item in items if item.relative.name in {"c.png", "d.png"}]
    set_group_images(workspace, "NIC26 Swimsuit", swim)
    set_group_images(workspace, "School Uniform", school)

    snapshot = activation_recipe_snapshot(workspace, trigger="hataya_misuzu")
    assert snapshot["trigger"] == "hataya_misuzu"
    assert len(snapshot["character_tags_groups"]) == 2
    first = snapshot["character_tags_groups"][0]
    assert "id" not in first
    assert first["name"] == "NIC26 Swimsuit"
    assert first["group_tag"] == "misuzu_nic26"
    assert first["tags"] == [
        "purple hair",
        "purple eyes",
        "ponytail",
        "white bikini",
        "swimsuit",
    ]
    assert first["coverage"] == 0.5


def test_group_snapshot_blocks_unassigned_active_images(tmp_path: Path) -> None:
    workspace = _character_workspace(tmp_path)
    upsert_character_tags_group(
        workspace,
        name="NIC26 Swimsuit",
        group_tag="misuzu_nic26",
        identity_tags=["ponytail"],
        outfit_tags=["white bikini"],
    )
    with pytest.raises(PipelineError, match="unassigned"):
        activation_recipe_snapshot(workspace, trigger="hataya_misuzu")


def test_activation_group_tag_is_inserted_after_global_trigger(tmp_path: Path) -> None:
    state = ProjectState.create(
        tmp_path / "project",
        name="group-caption",
        concept_type="character",
        base="base",
        trigger="hataya_misuzu",
        strategy="quality",
    )
    state.payload["project"]["caption_anchor_tags"] = []
    state.payload["project"]["activation_recipe"] = {
        "schema_version": 1,
        "snapshot_hash": "recipe-hash",
        "character_tags_groups": [
            {
                "name": "NIC26 Swimsuit",
                "group_tag": "misuzu_nic26",
                "identity_tags": ["purple hair", "ponytail"],
                "outfit_tags": ["white bikini"],
                "tags": ["purple hair", "ponytail", "white bikini"],
            }
        ],
        "assignments": {"source-001/a.png": "NIC26 Swimsuit"},
    }
    caption = tmp_path / "generated" / "a.txt"
    caption.parent.mkdir(parents=True)
    caption.write_text("hataya_misuzu, purple hair, white bikini\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    write_json_atomic(
        manifest,
        {
            "schema_version": 2,
            "mode": "existing_taglist_clean",
            "records": [
                {
                    "image": "source-001/a.png",
                    "caption": str(caption),
                    "text": "hataya_misuzu, purple hair, white bikini",
                }
            ],
            "summary": {},
            "input_hash": "old",
        },
    )
    result = apply_activation_group_captions(
        state,
        StepResult(
            status=StepStatus.DONE,
            input_hash="old",
            output_manifest=str(manifest),
        ),
    )
    assert result.details["activation_groups"] == {"NIC26 Swimsuit": 1}
    assert caption.read_text(encoding="utf-8").strip().startswith(
        "hataya_misuzu, misuzu_nic26, purple hair"
    )


def test_activation_usage_and_safetensors_metadata_keep_groups() -> None:
    snapshot = {
        "schema_version": 1,
        "trigger": "hataya_misuzu",
        "character_anchors": [],
        "character_tags_groups": [
            {
                "name": "NIC26 Swimsuit",
                "group_tag": "misuzu_nic26",
                "identity_tags": ["purple hair", "ponytail"],
                "outfit_tags": ["white bikini"],
                "tags": ["purple hair", "ponytail", "white bikini"],
                "coverage": 0.5,
            }
        ],
    }
    hint = activation_usage_hint(snapshot)
    assert "Trigger word:\nhataya_misuzu" in hint
    assert "Character Tags Group — NIC26 Swimsuit" in hint
    assert "Group tag:\nmisuzu_nic26" in hint
    metadata = activation_safetensors_metadata(snapshot)
    encoded = json.loads(metadata["lora_pipeline.activation"])
    assert encoded["character_tags_groups"][0]["group_tag"] == "misuzu_nic26"


def test_promotion_embeds_group_recipe_in_best_safetensors(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "repo"
    (root / "bases").mkdir(parents=True)
    base = root / "base.safetensors"
    save_file({"base": np.zeros((1,), dtype=np.float32)}, str(base))
    registry = {
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
    (root / "bases" / "registry.yaml").write_text(yaml.safe_dump(registry), encoding="utf-8")
    monkeypatch.setenv("LORA_PIPELINE_ROOT", str(root))

    state = ProjectState.create(
        root / "projects" / "demo",
        name="demo",
        concept_type="character",
        base="base",
        trigger="hataya_misuzu",
        strategy="quality",
    )
    state.payload["project"]["activation_recipe"] = {
        "schema_version": 1,
        "trigger": "hataya_misuzu",
        "character_anchors": [],
        "character_tags_groups": [
            {
                "name": "NIC26 Swimsuit",
                "group_tag": "misuzu_nic26",
                "identity_tags": ["purple hair", "ponytail"],
                "outfit_tags": ["white bikini"],
                "tags": ["purple hair", "ponytail", "white bikini"],
                "coverage": 1.0,
            }
        ],
        "assignments": {"source/a.png": "NIC26 Swimsuit"},
        "snapshot_hash": "recipe-hash",
    }
    run_dir = state.project_dir / "runs" / "run-1"
    checkpoint = run_dir / "checkpoints" / "candidate.safetensors"
    checkpoint.parent.mkdir(parents=True)
    save_file({"lora": np.ones((2,), dtype=np.float32)}, str(checkpoint), metadata={"ss_test": "kept"})
    state.payload["runs"] = [
        {
            "id": "run-1",
            "path": str(run_dir),
            "status": "trained",
            "checkpoints": [str(checkpoint)],
            "accounting": {},
            "evaluation": {},
        }
    ]
    state.save()

    result = promotion.run(
        state,
        run_id="run-1",
        checkpoint_name=checkpoint.name,
        strength=0.8,
        allow_unreviewed=True,
    )
    best = run_dir / "best.safetensors"
    with safe_open(str(best), framework="np") as handle:
        metadata = dict(handle.metadata() or {})
    assert metadata["ss_test"] == "kept"
    assert metadata["modelspec.trigger_phrase"] == "hataya_misuzu"
    assert "misuzu_nic26" in metadata["modelspec.usage_hint"]
    embedded = json.loads(metadata["lora_pipeline.activation"])
    assert embedded["character_tags_groups"][0]["name"] == "NIC26 Swimsuit"
    assert result["activation"]["character_tags_groups"][0]["group_tag"] == "misuzu_nic26"
