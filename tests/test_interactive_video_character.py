from __future__ import annotations

from pathlib import Path

from rich.console import Console

from pipeline import interactive_video_character
from pipeline.models import OptionalBackendUnavailable


def test_smart_character_detection_failure_falls_back_to_whole_frame_ccip(
    tmp_path, monkeypatch
) -> None:
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    expected = (tmp_path / "fallback", {"status": "fallback"})

    monkeypatch.setattr(
        interactive_video_character,
        "detect_video_subjects",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            OptionalBackendUnavailable("detector cache unavailable")
        ),
    )
    monkeypatch.setattr(
        interactive_video_character.BaseInteractiveWizard,
        "_select_video_identity",
        lambda self, directory: expected,
    )

    wizard = interactive_video_character.InteractiveWizard(
        console=Console(file=None, force_terminal=False)
    )
    result = wizard._select_video_identity(frame_dir)

    assert result == expected


def test_video_interval_is_captured_for_subject_timestamp_provenance(monkeypatch) -> None:
    monkeypatch.setattr(
        interactive_video_character.BaseInteractiveWizard,
        "_ask_positive_int",
        lambda self, prompt, default: 3,
    )
    wizard = interactive_video_character.InteractiveWizard()

    value = wizard._ask_positive_int("Sample one frame every N seconds", default=2)

    assert value == 3
    assert wizard._video_interval_seconds == 3
