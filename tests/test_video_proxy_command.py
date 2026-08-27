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
        auth=video_source.VideoAuth(mode="none"),
    )

    assert resolved == tmp_path / "source.mp4"
    assert commands
    command = commands[0]
    assert command[:3] == [sys.executable, "-m", "yt_dlp"]
    proxy_index = command.index("--proxy")
    assert command[proxy_index + 1] == "http://127.0.0.1:7890"
    assert "--cookies" not in command


def test_resolve_remote_video_passes_cookie_file_to_ytdlp(tmp_path, monkeypatch) -> None:
    commands: list[list[str]] = []
    cookies = tmp_path / "youtube-cookies.txt"
    cookies.write_text(
        "# Netscape HTTP Cookie File\n"
        ".youtube.com\tTRUE\t/\tTRUE\t2147483647\tSID\tsecret\n",
        encoding="utf-8",
    )

    def fake_run(command: list[str], failure: str) -> None:
        del failure
        commands.append(command)
        (tmp_path / "source.mp4").write_bytes(b"video")

    monkeypatch.setattr(video_source, "_run", fake_run)
    resolved = video_source._resolve_video(
        "https://www.youtube.com/watch?v=abc",
        tmp_path,
        remote=True,
        proxy=video_source.VideoProxy(mode="direct"),
        auth=video_source.VideoAuth(mode="cookies", cookies_path=str(cookies)),
    )

    assert resolved == tmp_path / "source.mp4"
    command = commands[0]
    cookie_index = command.index("--cookies")
    assert command[cookie_index + 1] == str(cookies.resolve())
