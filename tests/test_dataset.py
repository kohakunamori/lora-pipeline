from __future__ import annotations

from PIL import Image

from pipeline.dataset.image_info import inspect_dataset


def test_dataset_manifest_reports_objective_image_properties(tmp_path) -> None:
    Image.new("RGBA", (640, 800), (255, 0, 0, 128)).save(tmp_path / "portrait.png")
    Image.new("RGB", (1280, 720), "blue").save(tmp_path / "wide.jpg")
    (tmp_path / "wide.txt").write_text("landscape\n", encoding="utf-8")
    (tmp_path / "broken.png").write_bytes(b"not an image")
    manifest = inspect_dataset(tmp_path)
    summary = manifest["summary"]
    assert summary["image_count"] == 3
    assert summary["valid_images"] == 2
    assert summary["corrupt_images"] == 1
    assert summary["alpha_images"] == 1
    assert summary["caption_count"] == 1
    assert len(manifest["input_hash"]) == 64
