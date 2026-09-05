from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from pipeline.dataset_semantics import load_semantics, set_character_features, set_outfit_features
from pipeline.models import PipelineError, StepResult, StepStatus
from pipeline.target_policy import (
    _apply_character_outfit_semantic_captions,
    attach_target_aware_dataset_semantics_snapshot,
    target_caption_policy,
)


@dataclass(frozen=True)
class _Item:
    key: str


class _Workspace:
    concept_type = "character"
    name = "misuzu-demo"

    def __init__(self, root: Path) -> None:
        self.dataset_dir = root / "dataset"
        self.dataset_dir.mkdir()
        self._captions = {
            "1.png": "1girl, purple hair, purple eyes, white bikini, frilled bikini, smile",
            "2.png": "1girl, purple hair, purple eyes, white bikini, frilled bikini, beach",
            "3.png": "1girl, purple hair, purple eyes, white bikini, frilled bikini, standing",
        }

    def items(self, **_kwargs):
        return [_Item(key) for key in self._captions]

    def caption_text(self, key: str) -> str:
        return self._captions[key]

    def save(self) -> None:
        pass


class _State:
    def __init__(self, root: Path, *, target_type: str = "character_outfit") -> None:
        self.project_dir = root / "project"
        self.project_dir.mkdir()
        self.payload = {
            "project": {
                "type": "character",
                "training_target_type": target_type,
                "trigger": "misuzu_swimsuit",
                "caption_anchor_tags": ["hataya misuzu"],
                "training_identity": {},
                "hardware": "v100_16gb",
                "strategy": "quality",
                "overrides": {},
            }
        }
        self.saved = 0

    def save(self) -> None:
        self.saved += 1


def _make_manifest(state: _State, records: list[tuple[str, str]]) -> Path:
    caption_dir = state.project_dir / "review" / "captions" / "generated"
    caption_dir.mkdir(parents=True)
    manifest_records = []
    for image, text in records:
        destination = caption_dir / f"{Path(image).stem}.txt"
        destination.write_text(text + "\n", encoding="utf-8")
        manifest_records.append(
            {
                "image": image,
                "caption": str(destination),
                "text": text,
                "pruned": [],
                "token_counts": {},
            }
        )
    manifest = {
        "schema_version": 2,
        "mode": "existing_taglist_clean",
        "records": manifest_records,
        "summary": {},
        "input_hash": "old",
    }
    path = state.project_dir / "review" / "captions" / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_character_outfit_policy_keeps_training_trigger_owner(tmp_path: Path) -> None:
    workspace = _Workspace(tmp_path)
    set_character_features(workspace, ["purple hair", "purple eyes"])
    state = _State(tmp_path)

    attach_target_aware_dataset_semantics_snapshot(state, workspace)

    project = state.payload["project"]
    assert project["trigger"] == "misuzu_swimsuit"
    assert project["trigger_source"] == "training_config"
    assert "training_config_trigger" not in project
    assert project["semantic_caption_policy"]["outfit_features"] == "suppress"
    assert project["semantic_caption_policy"]["invariant_outfit_tags"] == "suppress"


def test_character_policy_keeps_training_config_trigger_owner(tmp_path: Path) -> None:
    workspace = _Workspace(tmp_path)
    state = _State(tmp_path, target_type="character")

    attach_target_aware_dataset_semantics_snapshot(state, workspace)

    project = state.payload["project"]
    assert project["trigger"] == "misuzu_swimsuit"
    assert project["trigger_source"] == "training_config"
    assert "training_config_trigger" not in project
    assert project["semantic_caption_policy"] == target_caption_policy("character")
    assert project["semantic_caption_policy"]["character_token"] == "do_not_inject"


def test_outfit_runtime_suppresses_manual_and_invariant_garment_tags(tmp_path: Path) -> None:
    workspace = _Workspace(tmp_path)
    set_character_features(workspace, ["purple hair", "purple eyes"])
    set_outfit_features(workspace, "default", ["frilled bikini"])
    state = _State(tmp_path)
    attach_target_aware_dataset_semantics_snapshot(state, workspace)
    manifest_path = _make_manifest(
        state,
        [
            (
                "1.png",
                "misuzu_demo, misuzu_demo_default, 1girl, purple hair, purple eyes, white bikini, frilled bikini, smile",
            ),
            (
                "2.png",
                "misuzu_demo, misuzu_demo_default, 1girl, purple hair, purple eyes, white bikini, frilled bikini, beach",
            ),
            (
                "3.png",
                "misuzu_demo, misuzu_demo_default, 1girl, purple hair, purple eyes, white bikini, frilled bikini, standing",
            ),
        ],
    )
    result = StepResult(
        status=StepStatus.DONE,
        input_hash="old",
        output_manifest=str(manifest_path),
        details={},
    )

    updated = _apply_character_outfit_semantic_captions(state, result)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert updated.status == StepStatus.DONE
    assert manifest["target_policy"]["inferred_invariant_outfit_features"] == ["white bikini"]
    first = manifest["records"][0]["text"]
    assert first.startswith("misuzu_swimsuit, hataya misuzu, ")
    assert "purple hair" not in first
    assert "purple eyes" not in first
    assert "white bikini" not in first
    assert "frilled bikini" not in first
    assert "misuzu_demo" not in first
    assert "misuzu_demo_default" not in first
    assert "smile" in first
    assert manifest["summary"]["target_feature_suppressions"] >= 4


def test_outfit_runtime_blocks_mixed_semantic_outfits(tmp_path: Path) -> None:
    workspace = _Workspace(tmp_path)
    semantics = load_semantics(workspace, create=True)
    assert semantics is not None
    semantics["outfits"]["stage"] = {
        "label": "Stage",
        "token": "misuzu_demo_stage",
        "features": ["stage costume"],
    }
    semantics["images"]["2.png"] = {"outfit": "stage"}

    state = _State(tmp_path)
    attach_target_aware_dataset_semantics_snapshot(state, workspace)
    state.payload["project"]["dataset_semantics_snapshot"] = {
        **state.payload["project"]["dataset_semantics_snapshot"],
        "outfits": semantics["outfits"],
        "images": semantics["images"],
    }
    manifest_path = _make_manifest(
        state,
        [
            ("1.png", "1girl, white bikini"),
            ("2.png", "1girl, stage costume"),
        ],
    )
    result = StepResult(
        status=StepStatus.DONE,
        input_hash="old",
        output_manifest=str(manifest_path),
        details={},
    )

    with pytest.raises(PipelineError, match="multiple semantic outfits"):
        _apply_character_outfit_semantic_captions(state, result)
