from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from pipeline.dataset.tagger import DualTagger, TagResult, TaggerBackend
from pipeline.state import ProjectState
from pipeline.steps.caption import run


class FixedTagger(TaggerBackend):
    def __init__(self, backend: str, tags: dict[str, float]):
        self.backend = backend
        self.tags = tags

    def tag(self, image: Path) -> TagResult:
        del image
        return TagResult(ratings={"safe": 1.0}, tags=self.tags, characters={}, backend=self.backend)


def test_dual_tagger_conflicts_are_preserved_for_review(tmp_path) -> None:
    state = ProjectState.create(
        tmp_path / "project",
        name="dual",
        concept_type="character",
        base="unused",
        trigger="zz_dual",
        strategy="quality",
    )
    Image.new("RGB", (640, 640), "red").save(state.project_dir / "raw" / "sample.png")
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
