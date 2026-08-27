from __future__ import annotations

import sys
from pathlib import Path

from pipeline import video_source


def test_resolve_remote_video_passes_custom_proxy_only_to_ytdlp(tmp_path, monkeypatch) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], failure: str) -> None:
        del failure
        commands.append(command)
        (tmp_path / "source.mp4").write_bytes(b"video")

    monkeypatch.setattr(video_source, "_run", fake_run)
    proxy = video_source.VideoProxy(mode="custom", url="http://127.0.0.1:7890")

    resolved = video_source._resolve_video(
        "https://www.youtube.com/watch?v=abc",
        tmp_path,
        remote=True,
        proxy=proxy,
    )

    assert resolved == tmp_path / "source.mp4"
    assert commands
    command = commands[0]
    assert command[:3] == [sys.executable, "-m", "yt_dlp"]
    proxy_index = command.index("--proxy")
    assert command[proxy_index + 1] == "http://127.0.0.1:7890"
