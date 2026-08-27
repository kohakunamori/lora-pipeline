from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from pipeline.models import PipelineError
from pipeline import video_source


def _pattern(path: Path, *, inverse: bool = False) -> None:
    image = Image.new("RGB", (128, 128), "white" if inverse else "black")
    draw = ImageDraw.Draw(image)
    fill = "black" if inverse else "white"
    draw.rectangle((8, 8, 56, 120), fill=fill)
    draw.ellipse((68, 20, 120, 108), fill=fill)
    image.save(path)


def test_extract_video_frames_filters_near_duplicates(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"placeholder")
    output = tmp_path / "accepted"

    monkeypatch.setattr(video_source, "require_video_tools", lambda **kwargs: None)
    monkeypatch.setattr(
        video_source,
        "_resolve_video",
        lambda source, temporary_dir, remote: Path(source),
    )

    def fake_sample(video_path, output_dir, *, interval_seconds, frame_cap):
        del video_path, interval_seconds, frame_cap
        _pattern(output_dir / "frame-000001.jpg")
        _pattern(output_dir / "frame-000002.jpg")
        _pattern(output_dir / "frame-000003.jpg", inverse=True)

    monkeypatch.setattr(video_source, "_sample_frames", fake_sample)

    report = video_source.extract_video_frames(
        str(source),
        output,
        interval_seconds=2,
        max_frames=10,
        phash_distance=2,
        blur_threshold=0,
    )

    assert report.sampled_frames == 3
    assert report.accepted_frames == 2
    assert report.rejected_near_duplicate == 1
    assert len(list(output.glob("*.jpg"))) == 2


def test_require_video_tools_reports_missing_programs(monkeypatch) -> None:
    monkeypatch.setattr(video_source.shutil, "which", lambda name: None)
    with pytest.raises(PipelineError, match="ffmpeg, yt-dlp"):
        video_source.require_video_tools(remote=True)


def test_url_detection_accepts_http_and_rejects_paths() -> None:
    assert video_source.is_url("https://www.youtube.com/watch?v=abc")
    assert video_source.is_url("http://example.com/video.mp4")
    assert not video_source.is_url("/mnt/media/video.mp4")
