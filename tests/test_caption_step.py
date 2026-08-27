from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from pipeline.dataset.tagger import DualTagger, TagResult, TaggerBackend
from pipeline.state import ProjectState
from pipeline.steps.caption import run


class FixedTagger(TaggerBackend):
    def __init__(self, backend: str, tags: dict[str, float], characters: dict[str, float] | None = None):
        self.backend = backend
        self.tags = tags
        self.characters = characters or {}
        self.calls = 0

    def cache_identity(self) -> dict[str, object]:
        return {"backend": self.backend, "tags": self.tags, "characters": self.characters}

    def tag(self, image: Path) -> TagResult:
        del image
        self.calls += 1
        return TagResult(
            ratings={"safe": 1.0},
            tags=self.tags,
            characters=self.characters,
            backend=self.backend,
        )


def _state(tmp_path: Path, *, concept: str = "character") -> ProjectState:
    state = ProjectState.create(
        tmp_path / "project",
        name="caption",
        concept_type=concept,
        base="unused",
        trigger="zz_caption",
        strategy="quality",
    )
    Image.new("RGB", (640, 640), "red").save(
        state.project_dir / "raw" / "sample.png"
    )
    return state


def test_dual_tagger_conflicts_are_preserved_for_review(tmp_path) -> None:
    state = _state(tmp_path)
    tagger = DualTagger(
        FixedTagger("stable", {"shared": 0.9, "stable_only": 0.8}),
        FixedTagger("challenger", {"shared": 0.1, "challenger_only": 0.8}),
        conflict_delta=0.35,
    )
    result = run(state, tagger=tagger)
    manifest = json.loads(Path(result.output_manifest).read_text(encoding="utf-8"))
    record = manifest["records"][0]
    assert record["needs_review"] is True
    assert record["dual_tagger"]["conflicts"] == ["shared"]
    assert record["dual_tagger"]["stable_only"] == ["stable_only"]
    assert record["dual_tagger"]["challenger_only"] == ["challenger_only"]
    assert result.details["dual_tagger_conflict_images"] == 1


def test_existing_passthrough_preserves_caption_bytes(tmp_path) -> None:
    state = _state(tmp_path)
    source = state.project_dir / "raw" / "sample.txt"
    original = b"MyTrigger, Blue_Hair, Mixed CASE\nsecond line\n"
    source.write_bytes(original)
    result = run(state, mode="existing_passthrough")
    manifest = json.loads(Path(result.output_manifest).read_text(encoding="utf-8"))
    destination = Path(manifest["records"][0]["caption"])
    assert destination.read_bytes() == original
    assert manifest["records"][0]["mode"] == "existing_passthrough"


def test_tagger_cache_is_content_addressed_per_image(tmp_path) -> None:
    state = _state(tmp_path)
    tagger = FixedTagger("fixed", {"1girl": 0.99, "portrait": 0.8})
    first = run(state, tagger=tagger)
    second = run(state, tagger=tagger)
    assert tagger.calls == 1
    assert first.details["tagger_cache_hits"] == 0
    assert second.details["tagger_cache_hits"] == 1

    Image.new("RGB", (640, 640), "blue").save(
        state.project_dir / "raw" / "sample.png"
    )
    third = run(state, tagger=tagger)
    assert tagger.calls == 2
    assert third.details["tagger_cache_hits"] == 0


def test_character_tags_flag_mixed_character_candidates(tmp_path) -> None:
    state = _state(tmp_path)
    second_image = state.project_dir / "raw" / "second.png"
    Image.new("RGB", (640, 640), "blue").save(second_image)

    class PerImageTagger(TaggerBackend):
        def cache_identity(self) -> dict[str, object]:
            return {"backend": "per-image-v1"}

        def tag(self, image: Path) -> TagResult:
            character = "character_a" if image.name == "sample.png" else "character_b"
            return TagResult(
                ratings={"safe": 1.0},
                tags={"1girl": 0.99},
                characters={character: 0.95},
                backend="per-image",
            )

    result = run(state, tagger=PerImageTagger())
    manifest = json.loads(Path(result.output_manifest).read_text(encoding="utf-8"))
    assert result.details["character_review_images"] == 1
    assert sum(bool(record.get("needs_review")) for record in manifest["records"]) == 1
