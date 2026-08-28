from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pipeline.dataset_semantics import (
    add_outfit,
    attach_dataset_semantics_snapshot,
    character_feature_candidates,
    load_semantics,
    outfit_feature_candidates,
    outfit_for_image,
    set_character_features,
)


@dataclass(frozen=True)
class _Item:
    key: str


class _Workspace:
    concept_type = "character"
    name = "misuzu-demo"

    def __init__(self, root: Path) -> None:
        self.dataset_dir = root
        self._captions = {
            "src/1.png": "1girl, purple hair, purple eyes, white bikini, frilled bikini",
            "src/2.png": "1girl, purple hair, purple eyes, white bikini, blue ribbon",
            "src/3.png": "1girl, purple hair, purple eyes, school uniform, necktie",
            "src/4.png": "1girl, purple hair, purple eyes, school uniform, pleated skirt",
        }

    def items(self, **_kwargs):
        return [_Item(key) for key in self._captions]

    def caption_text(self, key: str) -> str:
        return self._captions[key]

    def save(self) -> None:
        pass


class _State:
    def __init__(self, root: Path) -> None:
        self.project_dir = root / "project"
        self.project_dir.mkdir()
        self.payload = {
            "project": {
                "type": "character",
                "trigger": "config_trigger",
                "training_identity": {},
            }
        }
        self.saved = False

    def save(self) -> None:
        self.saved = True


def test_character_dataset_gets_default_character_and_default_outfit(tmp_path: Path) -> None:
    workspace = _Workspace(tmp_path)
    semantics = load_semantics(workspace, create=True)
    assert semantics is not None
    assert semantics["character"]["token"] == "misuzu_demo"
    assert semantics["outfits"]["default"]["token"] == "misuzu_demo_default"
    assert all(outfit_for_image(semantics, key) == "default" for key in workspace._captions)


def test_outfit_candidates_reward_tags_specific_to_selected_images(tmp_path: Path) -> None:
    workspace = _Workspace(tmp_path)
    add_outfit(
        workspace,
        "swimsuit",
        label="Swimsuit",
        image_keys=["src/1.png", "src/2.png"],
    )
    semantics = load_semantics(workspace, create=False)
    assert semantics is not None
    rows = outfit_feature_candidates(workspace, semantics, "swimsuit", minimum_coverage=0.5)
    by_tag = {row["tag"]: row for row in rows}
    assert by_tag["white bikini"]["coverage"] == 1.0
    assert by_tag["white bikini"]["other_coverage"] == 0.0
    assert by_tag["white bikini"]["specificity"] == 1.0
    assert by_tag["purple hair"]["specificity"] == 0.0


def test_character_feature_candidates_use_coverage_not_strict_intersection(tmp_path: Path) -> None:
    workspace = _Workspace(tmp_path)
    workspace._captions["src/4.png"] = "1girl, purple hair, school uniform"
    rows = character_feature_candidates(workspace, minimum_coverage=0.7)
    by_tag = {row["tag"]: row for row in rows}
    assert by_tag["purple hair"]["coverage"] == 1.0
    assert by_tag["purple eyes"]["coverage"] == 0.75


def test_semantic_snapshot_owns_runtime_character_trigger(tmp_path: Path) -> None:
    workspace = _Workspace(tmp_path)
    set_character_features(workspace, ["purple hair", "purple eyes"])
    state = _State(tmp_path)
    attach_dataset_semantics_snapshot(state, workspace)
    project = state.payload["project"]
    assert project["trigger"] == "misuzu_demo"
    assert project["training_config_trigger"] == "config_trigger"
    assert project["semantic_caption_policy"]["character_features"] == "suppress"
    assert project["dataset_semantics_snapshot"]["outfits"]["default"]["token"] == "misuzu_demo_default"
    assert state.saved
