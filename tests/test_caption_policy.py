from __future__ import annotations

import json
from pathlib import Path

from pipeline.caption_policy import apply_target_caption_policy, resolve_caption_policy
from pipeline.models import StepResult, StepStatus


class State:
    def __init__(
        self,
        root: Path,
        *,
        target_type: str,
        overrides: dict | None = None,
        anchors: list[str] | None = None,
    ) -> None:
        self.project_dir = root / "project"
        self.project_dir.mkdir(parents=True)
        concept_type = "style" if target_type == "style" else "character"
        self.payload = {
            "project": {
                "type": concept_type,
                "training_target_type": target_type,
                "trigger": "zz_target",
                "caption_anchor_tags": list(anchors or []),
                "hardware": "v100_16gb",
                "strategy": "quality",
                "overrides": overrides or {},
            }
        }


def _result(state: State, texts: list[str]) -> tuple[StepResult, Path]:
    generated = state.project_dir / "review" / "captions" / "generated"
    generated.mkdir(parents=True)
    records = []
    for index, text in enumerate(texts):
        path = generated / f"{index}.txt"
        path.write_text(text + "\n", encoding="utf-8")
        records.append(
            {
                "image": f"{index}.png",
                "caption": str(path),
                "text": text,
                "pruned": [],
                "token_counts": {},
            }
        )
    manifest = state.project_dir / "review" / "captions" / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "mode": "generate",
                "records": records,
                "summary": {},
                "input_hash": "before",
            }
        ),
        encoding="utf-8",
    )
    return (
        StepResult(
            status=StepStatus.DONE,
            input_hash="before",
            output_manifest=str(manifest),
            details={},
        ),
        manifest,
    )


def test_character_balanced_absorbs_only_stable_identity_not_outfit(tmp_path: Path) -> None:
    state = State(tmp_path, target_type="character")
    result, manifest_path = _result(
        state,
        [
            "zz_target, 1girl, purple hair, purple eyes, white bikini, smile",
            "zz_target, 1girl, purple hair, purple eyes, white bikini, beach",
            "zz_target, 1girl, purple hair, purple eyes, white bikini, standing",
        ],
    )

    updated = apply_target_caption_policy(state, result)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    first = manifest["records"][0]["text"]

    assert resolve_caption_policy(state) == "balanced"
    assert "purple hair" not in first
    assert "purple eyes" not in first
    assert "white bikini" in first
    assert "smile" in first
    assert updated.details["caption_policy"] == "balanced"


def test_controllable_character_keeps_identity_tags(tmp_path: Path) -> None:
    state = State(
        tmp_path,
        target_type="character",
        overrides={"caption": {"policy": "controllable"}},
    )
    result, manifest_path = _result(
        state,
        [
            "zz_target, 1girl, purple hair, purple eyes, smile",
            "zz_target, 1girl, purple hair, purple eyes, beach",
            "zz_target, 1girl, purple hair, purple eyes, standing",
        ],
    )

    apply_target_caption_policy(state, result)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert "purple hair" in manifest["records"][0]["text"]
    assert "purple eyes" in manifest["records"][0]["text"]
    assert manifest["target_caption_policy"]["policy"] == "controllable"


def test_character_outfit_balanced_absorbs_stable_character_and_garment(tmp_path: Path) -> None:
    state = State(
        tmp_path,
        target_type="character_outfit",
        anchors=["zz_character"],
    )
    result, manifest_path = _result(
        state,
        [
            "zz_target, zz_character, 1girl, purple hair, white bikini, smile",
            "zz_target, zz_character, 1girl, purple hair, white bikini, beach",
            "zz_target, zz_character, 1girl, purple hair, white bikini, standing",
        ],
    )

    apply_target_caption_policy(state, result)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    first = manifest["records"][0]["text"]

    assert first.startswith("zz_target, zz_character")
    assert "purple hair" not in first
    assert "white bikini" not in first
    assert "smile" in first
    assert manifest["target_caption_policy"]["suppressed_outfit_tags"] == ["white bikini"]


def test_style_is_fixed_content_rich_policy(tmp_path: Path) -> None:
    state = State(tmp_path, target_type="style")
    assert resolve_caption_policy(state) == "content_rich"
