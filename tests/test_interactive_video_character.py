from __future__ import annotations

from pathlib import Path

from PIL import Image
from rich.console import Console

from pipeline import interactive_video_character
from pipeline.models import OptionalBackendUnavailable


def test_smart_character_detection_failure_keeps_originals_without_ccip(
    tmp_path, monkeypatch
) -> None:
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    Image.new("RGB", (768, 1024), "red").save(frame_dir / "frame-000001.png")

    monkeypatch.setattr(
        interactive_video_character,
        "detect_video_subjects",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            OptionalBackendUnavailable("detector cache unavailable")
        ),
    )

    def fail_if_ccip_called(self, directory):
        raise AssertionError("trusted-identity smart crop must not fall back to CCIP")

    monkeypatch.setattr(
        interactive_video_character.BaseInteractiveWizard,
        "_select_video_identity",
        fail_if_ccip_called,
    )

    wizard = interactive_video_character.InteractiveWizard(
        console=Console(file=None, force_terminal=False)
    )
    result_dir, payload = wizard._select_video_identity(frame_dir)

    assert result_dir == frame_dir
    assert payload["status"] == "subject_detection_unavailable_keep_originals"
    assert payload["identity_assumed_valid"] is True
    assert payload["selected_cluster"] is None
    assert payload["selected_frames"] == 1


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
