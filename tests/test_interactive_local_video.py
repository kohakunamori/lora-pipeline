from __future__ import annotations

from io import StringIO

from rich.console import Console

from pipeline.interactive_sources import InteractiveWizard
from pipeline.interactive_video_auth import InteractiveWizard as BaseInteractiveWizard
from pipeline.wizard import Wizard


def _wizard() -> InteractiveWizard:
    return InteractiveWizard(console=Console(file=StringIO(), force_terminal=False))


def test_source_menu_exposes_local_video_as_first_class_choice(monkeypatch) -> None:
    wizard = _wizard()
    seen: dict[str, object] = {}

    def fake_menu(title, items, *, default=None):
        seen["title"] = title
        seen["values"] = [item.value for item in items]
        seen["default"] = default
        return "local_video"

    def fake_video_flow(self):
        seen["kind_during_video_flow"] = self._selected_video_source_kind
        return "video-result"

    monkeypatch.setattr(wizard, "_menu", fake_menu)
    monkeypatch.setattr(BaseInteractiveWizard, "_new_project_from_video", fake_video_flow)

    assert wizard.new_project() == "video-result"
    assert seen["values"] == ["images", "local_video", "remote_video"]
    assert seen["kind_during_video_flow"] == "local_video"
    assert wizard._selected_video_source_kind is None


def test_image_source_bypasses_old_combined_video_menu(monkeypatch) -> None:
    wizard = _wizard()
    monkeypatch.setattr(wizard, "_menu", lambda *args, **kwargs: "images")
    monkeypatch.setattr(Wizard, "new_project", lambda self: "image-result")

    assert wizard.new_project() == "image-result"


def test_local_video_prompt_resolves_and_confirms_existing_file(tmp_path, monkeypatch) -> None:
    video = tmp_path / "clip with spaces.mkv"
    video.write_bytes(b"not-empty")
    wizard = _wizard()
    wizard._selected_video_source_kind = "local_video"

    monkeypatch.setattr(
        BaseInteractiveWizard,
        "_ask_text",
        lambda self, prompt, default=None, choices=None: str(video),
    )
    monkeypatch.setattr(wizard, "_confirm", lambda *args, **kwargs: True)

    source = wizard._ask_text("YouTube URL or local video path")

    assert source == str(video.resolve())


def test_remote_video_prompt_rejects_local_path_then_accepts_url(tmp_path, monkeypatch) -> None:
    local = tmp_path / "video.mp4"
    local.write_bytes(b"video")
    answers = iter([str(local), "https://www.youtube.com/watch?v=abc"])
    wizard = _wizard()
    wizard._selected_video_source_kind = "remote_video"

    monkeypatch.setattr(
        BaseInteractiveWizard,
        "_ask_text",
        lambda self, prompt, default=None, choices=None: next(answers),
    )

    source = wizard._ask_text("YouTube URL or local video path")

    assert source == "https://www.youtube.com/watch?v=abc"
