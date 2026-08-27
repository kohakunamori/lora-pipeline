from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from pipeline import video_source


def _pattern(path: Path, marker: int) -> None:
    image = Image.new("RGB", (320, 180), "black")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20 + marker * 25, 20, 150 + marker * 10, 160), fill="white")
    draw.ellipse((180, 30 + marker * 10, 300, 150), fill=(80 + marker * 30,) * 3)
    image.save(path)


def test_remote_video_download_allows_4k_sources(tmp_path, monkeypatch) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], failure: str) -> None:
        del failure
        commands.append(command)
        (tmp_path / "source.webm").write_bytes(b"video")

    monkeypatch.setattr(video_source, "_run", fake_run)

    resolved = video_source._resolve_video(
        "https://example.com/video",
        tmp_path,
        remote=True,
        proxy=video_source.VideoProxy(mode="direct"),
        auth=video_source.VideoAuth(mode="none"),
    )

    assert resolved == tmp_path / "source.webm"
    command = commands[0]
    format_value = command[command.index("-f") + 1]
    assert "height<=2160" in format_value
    assert "height<=1080" not in format_value


def test_filtered_frames_keep_sample_index_for_timestamps(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"placeholder")
    output = tmp_path / "accepted"

    monkeypatch.setattr(video_source, "require_video_tools", lambda **kwargs: None)
    monkeypatch.setattr(
        video_source,
        "_resolve_video",
        lambda source, temporary_dir, remote, proxy, auth: Path(source),
    )

    def fake_sample(video_path, output_dir, *, interval_seconds, frame_cap):
        del video_path, interval_seconds, frame_cap
        _pattern(output_dir / "frame-000001.jpg", 0)
        # This one is intentionally overexposed and should be rejected.
        Image.new("RGB", (320, 180), "white").save(output_dir / "frame-000002.jpg")
        _pattern(output_dir / "frame-000003.jpg", 2)

    monkeypatch.setattr(video_source, "_sample_frames", fake_sample)

    report = video_source.extract_video_frames(
        str(source),
        output,
        max_frames=10,
        blur_threshold=0,
        phash_distance=0,
    )

    assert report.accepted_frames == 2
    assert (output / "video-000001.jpg").is_file()
    assert (output / "video-000003.jpg").is_file()
    assert not (output / "video-000002.jpg").exists()
