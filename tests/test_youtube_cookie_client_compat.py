from __future__ import annotations

from pathlib import Path

from pipeline import video_source


def _cookies(path: Path) -> Path:
    path.write_text(
        "# Netscape HTTP Cookie File\n"
        ".youtube.com\tTRUE\t/\tTRUE\t2147483647\tSID\ttest\n",
        encoding="utf-8",
    )
    return path


def test_youtube_cookie_download_uses_current_compatible_player_clients(tmp_path, monkeypatch) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], failure: str) -> None:
        del failure
        commands.append(command)
        (tmp_path / "source.mp4").write_bytes(b"video")

    monkeypatch.setattr(video_source, "_run", fake_run)
    auth = video_source.VideoAuth(mode="cookies", cookies_path=str(_cookies(tmp_path / "cookies.txt")))

    resolved = video_source._resolve_video(
        "https://www.youtube.com/watch?v=FHea4xhkNho",
        tmp_path,
        remote=True,
        proxy=video_source.VideoProxy(mode="direct"),
        auth=auth,
    )

    assert resolved == tmp_path / "source.mp4"
    command = commands[0]
    index = command.index("--extractor-args")
    assert command[index + 1] == "youtube:player_client=default,web_embedded"
    cookie_index = command.index("--cookies")
    assert Path(command[cookie_index + 1]).name == "cookies.txt"


def test_anonymous_youtube_download_does_not_force_cookie_client_policy(tmp_path, monkeypatch) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], failure: str) -> None:
        del failure
        commands.append(command)
        (tmp_path / "source.mp4").write_bytes(b"video")

    monkeypatch.setattr(video_source, "_run", fake_run)

    video_source._resolve_video(
        "https://youtu.be/FHea4xhkNho",
        tmp_path,
        remote=True,
        proxy=video_source.VideoProxy(mode="direct"),
        auth=video_source.VideoAuth(mode="none"),
    )

    assert "--extractor-args" not in commands[0]


def test_cookie_policy_is_scoped_to_youtube_hosts(tmp_path, monkeypatch) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], failure: str) -> None:
        del failure
        commands.append(command)
        (tmp_path / "source.mp4").write_bytes(b"video")

    monkeypatch.setattr(video_source, "_run", fake_run)
    auth = video_source.VideoAuth(mode="cookies", cookies_path=str(_cookies(tmp_path / "cookies.txt")))

    video_source._resolve_video(
        "https://example.com/video.mp4",
        tmp_path,
        remote=True,
        proxy=video_source.VideoProxy(mode="direct"),
        auth=auth,
    )

    assert "--extractor-args" not in commands[0]


def test_youtube_host_detection_rejects_lookalikes() -> None:
    assert video_source.is_youtube_url("https://youtube.com/watch?v=abc")
    assert video_source.is_youtube_url("https://www.youtube.com/watch?v=abc")
    assert video_source.is_youtube_url("https://youtu.be/abc")
    assert not video_source.is_youtube_url("https://youtube.com.evil.example/watch?v=abc")
    assert not video_source.is_youtube_url("https://notyoutube.com/watch?v=abc")
