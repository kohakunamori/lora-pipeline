from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline import video_color, video_source
from pipeline.models import PipelineError


def test_probe_video_color_detects_hdr10_pq(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(video_color.shutil, "which", lambda name: "/usr/bin/ffprobe" if name == "ffprobe" else None)
    payload = {
        "streams": [
            {
                "width": 3840,
                "height": 2160,
                "pix_fmt": "yuv420p10le",
                "color_space": "bt2020nc",
                "color_transfer": "smpte2084",
                "color_primaries": "bt2020",
                "color_range": "tv",
            }
        ]
    }
    monkeypatch.setattr(
        video_color.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr=""),
    )

    info = video_color.probe_video_color(tmp_path / "hdr.mp4")

    assert info.is_hdr is True
    assert info.hdr_kind == "pq"
    assert info.width == 3840
    assert info.height == 2160
    assert info.color_primaries == "bt2020"
    assert info.as_dict()["tone_mapping"]["applied"] is True


def test_probe_video_color_detects_hlg(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(video_color.shutil, "which", lambda name: "/usr/bin/ffprobe")
    payload = {
        "streams": [
            {
                "width": 1920,
                "height": 1080,
                "pix_fmt": "yuv420p10le",
                "color_space": "bt2020nc",
                "color_transfer": "arib-std-b67",
                "color_primaries": "bt2020",
            }
        ]
    }
    monkeypatch.setattr(
        video_color.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr=""),
    )

    info = video_color.probe_video_color(tmp_path / "hlg.mkv")

    assert info.is_hdr is True
    assert info.hdr_kind == "hlg"


def test_sdr_sampling_filter_keeps_original_simple_path() -> None:
    info = video_color.VideoColorInfo(
        width=1920,
        height=1080,
        color_space="bt709",
        color_transfer="bt709",
        color_primaries="bt709",
    )

    assert video_color.sampling_filter(3, info) == "fps=1/3"
    assert info.as_dict()["tone_mapping"] == {"applied": False}


def test_hdr_sampling_filter_converts_to_bt709_without_highlight_desaturation() -> None:
    info = video_color.VideoColorInfo(
        width=3840,
        height=2160,
        pixel_format="yuv420p10le",
        color_space="bt2020nc",
        color_transfer="smpte2084",
        color_primaries="bt2020",
    )

    filter_chain = video_color.sampling_filter(2, info)

    assert filter_chain.startswith("fps=1/2,zscale=t=linear:npl=100")
    assert "format=gbrpf32le" in filter_chain
    assert "zscale=p=bt709" in filter_chain
    assert "tonemap=mobius:param=0.3:desat=0" in filter_chain
    assert "zscale=t=bt709:m=bt709:r=full" in filter_chain
    assert filter_chain.endswith("format=yuvj444p")


def test_sample_frames_uses_hdr_tonemap_chain(monkeypatch, tmp_path) -> None:
    info = video_color.VideoColorInfo(
        width=3840,
        height=2160,
        pixel_format="yuv420p10le",
        color_space="bt2020nc",
        color_transfer="smpte2084",
        color_primaries="bt2020",
    )
    monkeypatch.setattr(video_source, "probe_video_color", lambda path: info)
    monkeypatch.setattr(video_source, "try_sample_sharp_windows", lambda *args, **kwargs: False)
    commands: list[list[str]] = []
    failures: list[str] = []

    def fake_run(command: list[str], failure: str) -> None:
        commands.append(command)
        failures.append(failure)

    monkeypatch.setattr(video_source, "_run", fake_run)

    result = video_source._sample_frames(
        tmp_path / "hdr.mp4",
        tmp_path,
        interval_seconds=3,
        frame_cap=20,
    )

    assert result == info
    command = commands[0]
    vf_index = command.index("-vf")
    assert "tonemap=mobius:param=0.3:desat=0" in command[vf_index + 1]
    assert "PQ" in failures[0]


def test_missing_ffprobe_fails_instead_of_silently_misdecoding_hdr(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(video_color.shutil, "which", lambda name: None)

    with pytest.raises(PipelineError, match="ffprobe"):
        video_color.probe_video_color(tmp_path / "video.mp4")
