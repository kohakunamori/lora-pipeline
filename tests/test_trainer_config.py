from __future__ import annotations

import json

from PIL import Image

from pipeline.trainer.sd_scripts import materialize_dataset_snapshot, write_dataset_toml


def test_dataset_snapshot_hashes_content_and_avoids_same_stem_collisions(tmp_path) -> None:
    project = tmp_path / "project"
    images = project / "prepared" / "images"
    captions = project / "prepared" / "captions"
    images.mkdir(parents=True)
    captions.mkdir(parents=True)
    records = []
    for suffix, color, caption_text in (("jpg", "red", "red image"), ("png", "blue", "blue image")):
        image = images / f"same.{suffix}"
        caption = captions / f"same__{suffix}.txt"
        Image.new("RGB", (64, 64), color).save(image)
        caption.write_text(caption_text + "\n", encoding="utf-8")
        records.append(
            {
                "source": f"same.{suffix}",
                "image": f"images/same.{suffix}",
                "caption": f"captions/same__{suffix}.txt",
            }
        )
    (project / "prepared" / "manifest.json").write_text(
        json.dumps({"images": records}), encoding="utf-8"
    )
    count, dataset_hash, captions_hash = materialize_dataset_snapshot(project, tmp_path / "snapshot")
    assert count == 2
    assert len(dataset_hash) == 64
    assert len(captions_hash) == 64
    assert (tmp_path / "snapshot" / "same__jpg.jpg").is_file()
    assert (tmp_path / "snapshot" / "same__jpg.txt").read_text(encoding="utf-8").strip() == "red image"
    assert (tmp_path / "snapshot" / "same__png.png").is_file()
    assert (tmp_path / "snapshot" / "same__png.txt").read_text(encoding="utf-8").strip() == "blue image"


def test_generated_dataset_toml_uses_profile_dimensions(tmp_path) -> None:
    path = tmp_path / "dataset.toml"
    write_dataset_toml(
        path,
        dataset_dir=tmp_path / "images",
        merged={
            "training": {"batch_size": 2},
            "resolution": {
                "default": 1024,
                "enable_bucket": True,
                "bucket_no_upscale": True,
                "bucket_reso_steps": 32,
            },
            "caption": {"shuffle": True, "keep_tokens": 1, "tag_dropout_rate": 0.05},
        },
    )
    text = path.read_text(encoding="utf-8")
    assert "batch_size = 2" in text
    assert "resolution = [1024, 1024]" in text
    assert "bucket_no_upscale = true" in text
    assert "caption_tag_dropout_rate = 0.05" in text
