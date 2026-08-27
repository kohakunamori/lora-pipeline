from __future__ import annotations

from pipeline.interactive_video_auth import InteractiveWizard


def test_youtube_bot_challenge_is_detected() -> None:
    message = (
        "yt-dlp could not download the video: ERROR: Sign in to confirm you’re not a bot. "
        "Use --cookies-from-browser or --cookies for the authentication."
    )
    assert InteractiveWizard._looks_like_youtube_auth_challenge(message)


def test_unrelated_download_failure_is_not_auth_challenge() -> None:
    assert not InteractiveWizard._looks_like_youtube_auth_challenge(
        "yt-dlp could not download the video: connection timed out"
    )
