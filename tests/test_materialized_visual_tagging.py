import json
from pathlib import Path

from PIL import Image

from pipeline.dataset.tagger import TagResult, TaggerBackend
from pipeline.materialization.caption import run as caption_run
from pipeline.state import ProjectState


class RecordingTagger(TaggerBackend):
    def __init__(self) -> None:
        self.seen: list[Path] = []

    def cache_identity(self) -> dict[str, object]:
        return {"backend": "recording-v1"}

    def tag(self, image: Path) -> TagResult:
        self.seen.append(image.resolve())
        with Image.open(image) as opened:
            width, height = opened.size
        return TagResult(
            ratings={"safe": 1.0},
            tags={"1girl": 0.99, f"visual_{width}x{height}": 0.9},
            characters={},
            backend="recording",
        )


def test_caption_tagger_reads_materialized_visual_override(tmp_path: Path) -> None:
    state = ProjectState.create(
        tmp_path / "project",
        name="visual-caption",
        concept_type="character",
        base="unused",
        trigger="zz_visual",
        strategy="quality",
    )
    raw = state.project_dir / "raw" / "sample.png"
    Image.new("RGB", (1200, 900), "red").save(raw)

    visual = state.project_dir / "cache" / "visual.png"
    visual.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (600, 800), "blue").save(visual)

    tagger = RecordingTagger()
    result = caption_run(
        state,
        mode="generate",
        tagger=tagger,
        tag_image_overrides={raw: visual},
    )

    assert tagger.seen == [visual.resolve()]
    assert result.details["captions"] == 1
    manifest = json.loads(Path(result.output_manifest).read_text(encoding="utf-8"))
    generated = Path(manifest["records"][0]["caption"])
    text = generated.read_text(encoding="utf-8")
    assert "visual 600x800" in text or "visual_600x800" in text
