from __future__ import annotations

from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from pipeline import video_character


class _FakeHash:
    def __init__(self, value: int):
        self.value = value

    def __sub__(self, other: "_FakeHash") -> int:
        return abs(self.value - other.value)


def _frame(path: Path, size: tuple[int, int], marker: int = 0) -> None:
    image = Image.new("RGB", size, "black")
    draw = ImageDraw.Draw(image)
    width, height = size
    draw.rectangle(
        (
            20 + marker * 3,
            20,
            max(21 + marker * 3, width // 2),
            max(21, height // 2),
        ),
        fill="white",
    )
    image.save(path, quality=95)


def test_4k_detection_uses_proxy_but_crops_from_original(tmp_path, monkeypatch) -> None:
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    _frame(frame_dir / "video-000007.jpg", (3840, 2160))
    seen_sizes: list[tuple[int, int]] = []

    def fake_person(image, **kwargs):
        del kwargs
        seen_sizes.append(image.size)
        assert max(image.size) == 1600
        return [((100, 70, 1500, 860), "person", 0.95)]

    def fake_heads(image, **kwargs):
        del kwargs
        assert max(image.size) == 1600
        return [((650, 100, 900, 340), "head", 0.9)]

    def fake_halfbody(image, **kwargs):
        del kwargs
        return [((100, 30, image.width - 100, int(image.height * 0.65)), "halfbody", 0.88)]

    monkeypatch.setattr(
        video_character,
        "_load_detectors",
        lambda: (fake_person, fake_heads, fake_halfbody),
    )

    report = video_character.detect_video_subjects(
        frame_dir,
        tmp_path / "subjects",
        interval_seconds=3,
    )

    assert seen_sizes == [(1600, 900)]
    assert len(report.subjects) == 1
    subject = report.subjects[0]
    assert subject.source_resolution == (3840, 2160)
    assert subject.source_timestamp_seconds == 18.0
    assert subject.quality_tier == "high"
    assert subject.native_identity_resolution[0] > 2048
    assert max(subject.saved_identity_resolution) <= 2048
    assert subject.saved_identity_resolution[0] * subject.saved_identity_resolution[1] <= 4_194_304
    assert subject.identity_path.is_file()


def test_low_resolution_person_is_rejected_but_large_unmatched_head_can_fallback(
    tmp_path, monkeypatch
) -> None:
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    _frame(frame_dir / "video-000001.jpg", (1920, 1080))

    def fake_person(image, **kwargs):
        del image, kwargs
        return [((20, 20, 120, 160), "person", 0.8)]

    def fake_heads(image, **kwargs):
        del image, kwargs
        return [
            ((30, 25, 70, 65), "head", 0.7),
            ((900, 200, 1100, 400), "head", 0.9),
        ]

    def fake_halfbody(image, **kwargs):
        del image, kwargs
        return []

    monkeypatch.setattr(
        video_character,
        "_load_detectors",
        lambda: (fake_person, fake_heads, fake_halfbody),
    )

    report = video_character.detect_video_subjects(
        frame_dir,
        tmp_path / "subjects",
        detection_proxy_long_edge=1920,
    )

    assert report.rejected_low_resolution == 1
    assert report.head_fallbacks == 1
    assert len(report.subjects) == 1
    assert report.subjects[0].detection_kind == "head_fallback"


def _subject(
    tmp_path: Path,
    index: int,
    *,
    frame_subject_count: int = 1,
) -> video_character.VideoSubject:
    source = tmp_path / f"video-{index + 1:06d}.jpg"
    _frame(source, (1400, 1000), marker=index)
    identity_dir = tmp_path / "identity"
    identity_dir.mkdir(exist_ok=True)
    identity = identity_dir / f"subject-{index + 1:05d}.jpg"
    _frame(identity, (900, 900), marker=index)
    return video_character.VideoSubject(
        subject_id=f"subject-{index + 1:05d}",
        identity_path=identity,
        source_frame=source,
        source_timestamp_seconds=float(index * 2),
        source_resolution=(1400, 1000),
        person_bbox=(300, 70, 1100, 950),
        head_bbox=(520, 100, 800, 380),
        halfbody_bbox=(380, 80, 1040, 650),
        person_score=0.9,
        head_score=0.9,
        halfbody_score=0.9,
        detection_kind="person",
        quality_tier="high",
        frame_subject_count=frame_subject_count,
        native_identity_resolution=(900, 900),
        saved_identity_resolution=(900, 900),
    )


def test_balanced_compositions_preserve_real_diversity_without_resolution_copies(
    tmp_path, monkeypatch
) -> None:
    subjects = tuple(_subject(tmp_path, index) for index in range(20))
    report = video_character.VideoSubjectReport(
        identity_dir=tmp_path / "identity",
        subjects=subjects,
        total_frames=20,
        frames_with_subjects=20,
        detected_persons=20,
        head_fallbacks=0,
        rejected_low_resolution=0,
        detection_proxy_long_edge=1600,
        minimum_person_height=512,
        minimum_head_size=160,
        maximum_saved_long_edge=2048,
        maximum_saved_pixels=4_194_304,
    )
    counter = iter(range(0, 1000, 20))
    monkeypatch.setattr(
        video_character.imagehash,
        "phash",
        lambda image: _FakeHash(next(counter)),
    )

    composition = video_character.build_balanced_character_dataset(
        report,
        [subject.identity_path for subject in subjects],
        tmp_path / "training",
        maximum_saved_long_edge=1024,
    )

    assert len(composition.records) == 20
    counts = Counter(record.crop_type for record in composition.records)
    assert counts == {
        "portrait": 5,
        "upper_body": 6,
        "full_body": 6,
        "context": 3,
    }
    assert len({record.subject_id for record in composition.records}) == 20
    assert all(max(record.saved_resolution) <= 1024 for record in composition.records)
    assert all(
        saved <= native
        for record in composition.records
        for saved, native in zip(record.saved_resolution, record.native_resolution, strict=True)
    )


def test_context_compositions_are_not_used_for_multi_character_frames(tmp_path, monkeypatch) -> None:
    subjects = tuple(
        _subject(tmp_path, index, frame_subject_count=2) for index in range(8)
    )
    report = video_character.VideoSubjectReport(
        identity_dir=tmp_path / "identity",
        subjects=subjects,
        total_frames=8,
        frames_with_subjects=8,
        detected_persons=16,
        head_fallbacks=0,
        rejected_low_resolution=0,
        detection_proxy_long_edge=1600,
        minimum_person_height=512,
        minimum_head_size=160,
        maximum_saved_long_edge=2048,
        maximum_saved_pixels=4_194_304,
    )
    counter = iter(range(0, 1000, 20))
    monkeypatch.setattr(
        video_character.imagehash,
        "phash",
        lambda image: _FakeHash(next(counter)),
    )

    composition = video_character.build_balanced_character_dataset(
        report,
        [subject.identity_path for subject in subjects],
        tmp_path / "training",
    )

    assert composition.records
    assert all(record.crop_type != "context" for record in composition.records)
