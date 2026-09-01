from __future__ import annotations

import json
from pathlib import Path

from pipeline.models import StepResult, StepStatus
from pipeline.style_caption_policy import apply_style_caption_policy


class _State:
    def __init__(self, root: Path, *, concept_type: str = "style") -> None:
        self.project_dir = root
        self.payload = {
            "project": {
                "type": concept_type,
                "trigger": "zz_style",
            }
        }


def _manifest(root: Path, *, mode: str, text: str) -> tuple[Path, Path]:
    caption = root / "caption.txt"
    caption.write_text(text + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 2,
        "mode": mode,
        "records": [
            {
                "image": "1.png",
                "caption": str(caption),
                "text": text,
                "pruned": [],
                "token_counts": {},
            }
        ],
        "summary": {},
        "input_hash": "old",
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, caption


def test_cleaned_existing_style_caption_suppresses_style_descriptors(tmp_path: Path) -> None:
    state = _State(tmp_path)
    manifest_path, caption_path = _manifest(
        tmp_path,
        mode="existing_taglist_clean",
        text="zz_style, 1girl, watercolor, soft shading, school uniform, outdoors",
    )
    result = StepResult(
        status=StepStatus.DONE,
        input_hash="old",
        output_manifest=str(manifest_path),
        details={},
    )

    updated = apply_style_caption_policy(state, result)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    text = manifest["records"][0]["text"]

    assert updated.status == StepStatus.DONE
    assert text == "zz_style, 1girl, school uniform, outdoors"
    assert manifest["records"][0]["style_descriptors_suppressed"] == [
        "watercolor",
        "soft shading",
    ]
    assert manifest["summary"]["style_descriptors_suppressed"] == 2
    assert caption_path.read_text(encoding="utf-8").strip() == text


def test_style_trigger_is_never_removed_even_if_trigger_looks_like_descriptor(tmp_path: Path) -> None:
    state = _State(tmp_path)
    state.payload["project"]["trigger"] = "watercolor"
    manifest_path, _ = _manifest(
        tmp_path,
        mode="hybrid",
        text="watercolor, painterly, 1girl, city",
    )
    result = StepResult(
        status=StepStatus.DONE,
        input_hash="old",
        output_manifest=str(manifest_path),
        details={},
    )

    apply_style_caption_policy(state, result)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["records"][0]["text"] == "watercolor, 1girl, city"


def test_existing_passthrough_remains_byte_semantic_passthrough(tmp_path: Path) -> None:
    state = _State(tmp_path)
    original = "zz_style, watercolor, painterly, 1girl"
    manifest_path, caption_path = _manifest(
        tmp_path,
        mode="existing_passthrough",
        text=original,
    )
    result = StepResult(
        status=StepStatus.DONE,
        input_hash="old",
        output_manifest=str(manifest_path),
        details={},
    )

    updated = apply_style_caption_policy(state, result)
    assert updated is result
    assert caption_path.read_text(encoding="utf-8").strip() == original


def test_character_caption_is_not_modified(tmp_path: Path) -> None:
    state = _State(tmp_path, concept_type="character")
    manifest_path, caption_path = _manifest(
        tmp_path,
        mode="existing_taglist_clean",
        text="zz_character, watercolor, 1girl",
    )
    result = StepResult(
        status=StepStatus.DONE,
        input_hash="old",
        output_manifest=str(manifest_path),
        details={},
    )

    updated = apply_style_caption_policy(state, result)
    assert updated is result
    assert caption_path.read_text(encoding="utf-8").strip() == "zz_character, watercolor, 1girl"
