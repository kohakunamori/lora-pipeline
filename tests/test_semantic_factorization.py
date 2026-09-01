from __future__ import annotations

import json
from pathlib import Path

from pipeline.models import StepResult, StepStatus
from pipeline.semantic_factorization import (
    apply_character_semantic_factorization,
    infer_invariant_identity_tags,
    infer_outfit_features_by_group,
)


class _State:
    def __init__(self, root: Path) -> None:
        self.project_dir = root / "project"
        self.project_dir.mkdir()
        self.payload = {
            "project": {
                "type": "character",
                "training_target_type": "character",
                "trigger": "misuzu_demo",
                "caption_anchor_tags": [],
                "dataset_semantics_snapshot": {
                    "snapshot_hash": "semantic-hash",
                    "character": {"token": "misuzu_demo", "features": []},
                    "outfits": {
                        "default": {
                            "token": "misuzu_demo_default",
                            "features": ["frilled bikini"],
                        }
                    },
                    "images": {},
                },
            }
        }


def _manifest(state: _State) -> Path:
    caption_dir = state.project_dir / "review" / "captions" / "generated"
    caption_dir.mkdir(parents=True)
    rows = []
    for index, variable in enumerate(("smile", "beach", "standing"), start=1):
        text = (
            "misuzu_demo, misuzu_demo_default, 1girl, purple hair, purple eyes, "
            f"white bikini, frilled bikini, {variable}"
        )
        destination = caption_dir / f"{index}.txt"
        destination.write_text(text + "\n", encoding="utf-8")
        rows.append(
            {
                "image": f"{index}.png",
                "caption": str(destination),
                "text": text,
                "pruned": [],
                "token_counts": {},
                "semantic_concepts": {
                    "character_token": "misuzu_demo",
                    "outfit": "default",
                    "outfit_token": "misuzu_demo_default",
                },
            }
        )
    path = state.project_dir / "review" / "captions" / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "mode": "existing_taglist_clean",
                "records": rows,
                "summary": {},
                "input_hash": "old",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_identity_inference_is_strict_and_requires_stable_intrinsic_tags() -> None:
    records = [
        {"text": "1girl, purple hair, purple eyes, white bikini, smile"},
        {"text": "1girl, purple hair, purple eyes, white bikini, beach"},
        {"text": "1girl, purple hair, purple eyes, white bikini, standing"},
    ]
    assert infer_invariant_identity_tags(records) == {"purple hair", "purple eyes"}


def test_outfit_inference_requires_group_specificity() -> None:
    records = [
        {"image": "a1.png", "text": "white bikini, frilled bikini, purple hair"},
        {"image": "a2.png", "text": "white bikini, frilled bikini, purple hair"},
        {"image": "a3.png", "text": "white bikini, frilled bikini, purple hair"},
        {"image": "b1.png", "text": "school uniform, purple hair"},
        {"image": "b2.png", "text": "school uniform, purple hair"},
        {"image": "b3.png", "text": "school uniform, purple hair"},
    ]
    bindings = {
        "a1.png": {"outfit": "swimsuit"},
        "a2.png": {"outfit": "swimsuit"},
        "a3.png": {"outfit": "swimsuit"},
        "b1.png": {"outfit": "school"},
        "b2.png": {"outfit": "school"},
        "b3.png": {"outfit": "school"},
    }
    inferred = infer_outfit_features_by_group(records, bindings)
    assert inferred["swimsuit"] == {"white bikini", "frilled bikini"}
    assert inferred["school"] == {"school uniform"}
    assert all("purple hair" not in tags for tags in inferred.values())


def test_character_factorization_moves_identity_and_outfit_features_into_tokens(tmp_path: Path) -> None:
    state = _State(tmp_path)
    manifest_path = _manifest(state)
    result = StepResult(
        status=StepStatus.DONE,
        input_hash="old",
        output_manifest=str(manifest_path),
        details={},
    )

    updated = apply_character_semantic_factorization(state, result)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    first = manifest["records"][0]["text"]

    assert updated.status == StepStatus.DONE
    assert first.startswith("misuzu_demo, misuzu_demo_default, 1girl")
    assert "purple hair" not in first
    assert "purple eyes" not in first
    assert "white bikini" not in first
    assert "frilled bikini" not in first
    assert "smile" in first
    assert manifest["target_policy"]["inferred_identity_features"] == ["purple eyes", "purple hair"]
    assert manifest["target_policy"]["inferred_outfit_features"]["default"] == ["white bikini"]
    assert manifest["summary"]["semantic_factorization_suppressions"] >= 4
