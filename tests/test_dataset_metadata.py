from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from pipeline.dataset_metadata import (
    classify_composition,
    composition_summary,
    import_composition_records,
    item_metadata,
    set_item_metadata,
)
from pipeline.dataset_metadata_snapshot import attach_dataset_metadata_snapshot
from pipeline.dataset_workspace import DatasetWorkspace
from pipeline.state import ProjectState
from pipeline.video_character import VideoSubject, VideoSubjectReport
from pipeline.video_composition import build_enriched_character_dataset


def _image(path: Path, *, size: tuple[int, int] = (768, 1024), color: str = "white") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def test_composition_classifier_distinguishes_common_character_views() -> None:
    assert classify_composition(
        subject_height_ratio=0.82,
        subject_area_ratio=0.62,
        head_height_ratio=0.34,
        head_to_person_ratio=0.41,
    ) == "portrait"
    assert classify_composition(
        subject_height_ratio=0.72,
        subject_area_ratio=0.44,
        head_height_ratio=0.20,
        head_to_person_ratio=0.29,
    ) == "upper_body"
    assert classify_composition(
        subject_height_ratio=0.70,
        subject_area_ratio=0.36,
        head_height_ratio=0.14,
        head_to_person_ratio=0.20,
    ) == "three_quarter"
    assert classify_composition(
        subject_height_ratio=0.91,
        subject_area_ratio=0.39,
        head_height_ratio=0.12,
        head_to_person_ratio=0.14,
    ) == "full_body"
    assert classify_composition(
        subject_height_ratio=0.43,
        subject_area_ratio=0.12,
        head_height_ratio=0.08,
        head_to_person_ratio=0.19,
    ) == "context"


def test_video_composition_manifest_becomes_source_metadata(tmp_path: Path) -> None:
    incoming = tmp_path / "incoming"
    _image(incoming / "train-00001-upper_body.jpg", size=(913, 1190))
    workspace = DatasetWorkspace.create("demo", root=tmp_path)
    source = workspace.add_source_from_directory(incoming, kind="local_video", label="video")

    changed = import_composition_records(
        workspace,
        str(source["id"]),
        [
            {
                "path": "train-00001-upper_body.jpg",
                "subject_id": "subject-00001",
                "source_frame": "video-000123.jpg",
                "source_group_id": "video-000123:subject-00001",
                "composition_type": "upper_body",
                "variant_kind": "smart_crop",
                "crop_box": [200, 20, 1113, 1210],
                "person_bbox": [270, 80, 1040, 1190],
                "head_bbox": [440, 100, 800, 470],
                "source_resolution": [1152, 2048],
                "frame_subject_count": 1,
                "subject_coverage": 0.72,
                "subject_height_ratio": 0.54,
                "subject_area_ratio": 0.36,
                "head_height_ratio": 0.18,
                "head_to_person_ratio": 0.34,
                "full_keep_score": 0.88,
                "native_resolution": [913, 1190],
                "saved_resolution": [913, 1190],
                "downscaled": False,
                "quality_tier": "high",
            }
        ],
        selected_cluster=2,
    )

    assert changed == 1
    item = workspace.items()[0]
    metadata = item_metadata(workspace, item)
    assert metadata["composition_type"] == "upper_body"
    assert metadata["variant_kind"] == "smart_crop"
    assert metadata["source_group_id"] == "video-000123:subject-00001"
    assert metadata["analysis"]["subject_coverage"] == 0.72
    assert metadata["identity"]["ccip_cluster"] == 2
    summary = composition_summary(workspace)
    assert summary["active_composition_counts"] == {"upper_body": 1}
    assert summary["variant_counts"] == {"smart_crop": 1}


def test_enriched_video_builder_keeps_only_a_small_high_value_full_variant(tmp_path: Path) -> None:
    frame = tmp_path / "frame-000001.jpg"
    image = Image.new("RGB", (1152, 2048), "skyblue")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 220, 2048), fill="navy")
    draw.rectangle((300, 150, 950, 1900), fill="lightpink")
    draw.ellipse((500, 200, 800, 500), fill="royalblue")
    draw.rectangle((350, 150, 900, 1050), outline="white", width=20)
    image.save(frame, quality=95)

    identity_dir = tmp_path / "identity"
    identity_dir.mkdir()
    identity_path = identity_dir / "subject-00001.jpg"
    image.crop((260, 120, 990, 1940)).save(identity_path, quality=95)
    subject = VideoSubject(
        subject_id="subject-00001",
        identity_path=identity_path,
        source_frame=frame,
        source_timestamp_seconds=2.0,
        source_resolution=(1152, 2048),
        person_bbox=(300, 150, 950, 1900),
        head_bbox=(500, 200, 800, 500),
        halfbody_bbox=(350, 150, 900, 1050),
        person_score=0.95,
        head_score=0.94,
        halfbody_score=0.90,
        detection_kind="person",
        quality_tier="high",
        frame_subject_count=1,
        native_identity_resolution=(730, 1820),
        saved_identity_resolution=(730, 1820),
    )
    report = VideoSubjectReport(
        identity_dir=identity_dir,
        subjects=(subject,),
        total_frames=1,
        frames_with_subjects=1,
        detected_persons=1,
        head_fallbacks=0,
        rejected_low_resolution=0,
        detection_proxy_long_edge=1600,
        minimum_person_height=512,
        minimum_head_size=160,
        maximum_saved_long_edge=2048,
        maximum_saved_pixels=4_194_304,
    )

    result = build_enriched_character_dataset(
        report,
        [identity_path],
        tmp_path / "training",
        phash_distance=2,
    )
    variants = [record.variant_kind for record in result.records]
    assert variants.count("smart_crop") == 1
    assert variants.count("original_full") == 1
    assert len(result.records) == 2
    assert len({record.source_group_id for record in result.records}) == 1
    assert all(max(record.saved_resolution) <= 2048 for record in result.records)
    assert all(
        record.saved_resolution[0] <= record.native_resolution[0]
        and record.saved_resolution[1] <= record.native_resolution[1]
        for record in result.records
    )
    assert result.records[0].composition_type in {
        "portrait",
        "upper_body",
        "three_quarter",
        "full_body",
        "context",
    }


def test_metadata_snapshot_is_frozen_with_training_state(tmp_path: Path) -> None:
    incoming = tmp_path / "incoming"
    _image(incoming / "a.png")
    workspace = DatasetWorkspace.create("demo", root=tmp_path)
    workspace.add_source_from_directory(incoming, kind="image_directory")
    item = workspace.items()[0]
    set_item_metadata(
        workspace,
        item,
        {
            "composition_type": "upper_body",
            "variant_kind": "original",
            "analysis": {"status": "analyzed", "subject_area_ratio": 0.55},
        },
    )

    state = ProjectState.create(
        tmp_path / "projects" / "run-demo",
        name="run-demo",
        concept_type="character",
        base="base",
        trigger="zz_demo",
        strategy="quality",
    )
    attach_dataset_metadata_snapshot(state, workspace)
    frozen_hash = state.payload["project"]["dataset_metadata_snapshot"]["snapshot_hash"]
    assert state.payload["project"]["dataset_metadata_snapshot"]["images"][0]["composition_type"] == "upper_body"

    set_item_metadata(workspace, item, {"composition_type": "full_body"})
    reloaded = ProjectState.load(state.project_dir)
    assert reloaded.payload["project"]["dataset_metadata_snapshot"]["snapshot_hash"] == frozen_hash
    assert reloaded.payload["project"]["dataset_metadata_snapshot"]["images"][0]["composition_type"] == "upper_body"
