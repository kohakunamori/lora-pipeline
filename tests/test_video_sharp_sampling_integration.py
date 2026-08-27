from __future__ import annotations

from pipeline import video_source
from pipeline.video_color import VideoColorInfo


def _sdr_info() -> VideoColorInfo:
    return VideoColorInfo(
        width=3840,
        height=2160,
        color_space="bt709",
        color_transfer="bt709",
        color_primaries="bt709",
    )


def test_sample_frames_prefers_sharp_window_selector(monkeypatch, tmp_path) -> None:
    info = _sdr_info()
    monkeypatch.setattr(video_source, "probe_video_color", lambda path: info)
    calls: list[tuple[int, int]] = []

    def fake_selector(video_path, output_dir, *, color_info, interval_seconds, frame_cap):
        del video_path, output_dir
        assert color_info == info
        calls.append((interval_seconds, frame_cap))
        return True

    monkeypatch.setattr(video_source, "try_sample_sharp_windows", fake_selector)
    monkeypatch.setattr(
        video_source,
        "_run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("fixed sampler should not run")),
    )

    result = video_source._sample_frames(
        tmp_path / "video.mp4",
        tmp_path,
        interval_seconds=2,
        frame_cap=100,
    )

    assert result == info
    assert calls == [(2, 100)]


def test_sample_frames_falls_back_when_blur_selector_is_unavailable(monkeypatch, tmp_path) -> None:
    info = _sdr_info()
    monkeypatch.setattr(video_source, "probe_video_color", lambda path: info)
    monkeypatch.setattr(video_source, "try_sample_sharp_windows", lambda *args, **kwargs: False)
    commands: list[list[str]] = []
    monkeypatch.setattr(video_source, "_run", lambda command, failure: commands.append(command))

    result = video_source._sample_frames(
        tmp_path / "video.mp4",
        tmp_path,
        interval_seconds=3,
        frame_cap=20,
    )

    assert result == info
    assert len(commands) == 1
    vf_index = commands[0].index("-vf")
    assert commands[0][vf_index + 1] == "fps=1/3"
