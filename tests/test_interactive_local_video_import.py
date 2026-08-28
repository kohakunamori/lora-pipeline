from __future__ import annotations

from pathlib import Path

import pytest

from pipeline import interactive_local_video_import as local_video


@pytest.mark.parametrize(
    ("raw", "seconds"),
    [
        ("0", 0.0),
        ("90", 90.0),
        ("01:30", 90.0),
        ("1:02:03", 3723.0),
        ("00:00:01.5", 1.5),
    ],
)
def test_parse_video_timestamp_accepts_seconds_mmss_and_hhmmss(raw: str, seconds: float) -> None:
    assert local_video.parse_video_timestamp(raw) == pytest.approx(seconds)


@pytest.mark.parametrize("raw", ["", "-1", "00:60", "00:60:00", "1:2:60", "1:2:3:4", "abc"])
def test_parse_video_timestamp_rejects_invalid_values(raw: str) -> None:
    with pytest.raises(ValueError):
        local_video.parse_video_timestamp(raw)


def test_ffmpeg_window_uses_absolute_start_and_end_as_seek_plus_duration() -> None:
    before, after = local_video._ffmpeg_window_args(90.0, 135.5)
    assert before == ["-ss", "90"]
    assert after == ["-t", "45.5"]


def test_advanced_settings_collect_start_end_interval_and_limit() -> None:
    class Console:
        def print(self, *args, **kwargs) -> None:
            del args, kwargs

    class Wizard:
        console = Console()

        def __init__(self) -> None:
            self.answers = iter(["00:01:30", "00:06:45"])
            self.positive = iter([3, 180])

        @staticmethod
        def _b(zh: str, en: str) -> str:
            del en
            return zh

        @staticmethod
        def _menu(title, items, *, default=None):
            del title, items, default
            return "advanced"

        def _ask_text(self, prompt, *, default=None):
            del prompt, default
            return next(self.answers)

        def _ask_positive_int(self, prompt, *, default):
            del prompt, default
            return next(self.positive)

    settings = local_video._choose_local_video_settings(Wizard())
    assert settings is not None
    assert settings.mode == "advanced"
    assert settings.start_seconds == pytest.approx(90.0)
    assert settings.end_seconds == pytest.approx(405.0)
    assert settings.interval_seconds == 3
    assert settings.max_frames == 180


def test_one_click_settings_use_existing_safe_defaults_without_time_window() -> None:
    class Wizard:
        @staticmethod
        def _b(zh: str, en: str) -> str:
            del en
            return zh

        @staticmethod
        def _menu(title, items, *, default=None):
            del title, items, default
            return "one_click"

    settings = local_video._choose_local_video_settings(Wizard())
    assert settings is not None
    assert settings.mode == "one_click"
    assert settings.interval_seconds == 2
    assert settings.max_frames == 250
    assert settings.window is None


def test_windowed_sampler_injects_seek_and_duration_into_ffmpeg(tmp_path, monkeypatch) -> None:
    commands: list[list[str]] = []

    class Color:
        is_hdr = False

    monkeypatch.setattr(local_video, "_try_sample_sharp_windows_in_range", lambda *args, **kwargs: False)
    monkeypatch.setattr(local_video.video_source, "probe_video_color", lambda path: Color())
    monkeypatch.setattr(local_video.video_source, "sampling_filter", lambda interval, color: "fps=1/2")
    monkeypatch.setattr(
        local_video.video_source,
        "_run",
        lambda command, failure: commands.append(command),
    )

    local_video._sample_frames_in_window(
        Path("/video/source.mkv"),
        tmp_path,
        interval_seconds=2,
        frame_cap=25,
        start_seconds=30.0,
        end_seconds=90.0,
    )

    command = commands[0]
    assert command[command.index("-ss") + 1] == "30"
    assert command[command.index("-t") + 1] == "60"
    assert command.index("-ss") < command.index("-i") < command.index("-t")
    assert command[command.index("-frames:v") + 1] == "25"
