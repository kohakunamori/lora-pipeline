from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import imagehash
from PIL import Image, ImageFilter, ImageStat

from .models import PipelineError


_PROXY_ENV_NAMES = (
    "LORA_VIDEO_PROXY",
    "HTTPS_PROXY",
    "https_proxy",
    "ALL_PROXY",
    "all_proxy",
    "HTTP_PROXY",
    "http_proxy",
)
_SUPPORTED_PROXY_SCHEMES = {"http", "https", "socks4", "socks4a", "socks5", "socks5h"}
_COOKIES_ENV_NAME = "LORA_VIDEO_COOKIES"
_DEFAULT_COOKIES_PATH = Path("~/.config/lora-pipeline/youtube-cookies.txt").expanduser()
_NETSCAPE_COOKIE_HEADERS = {"# HTTP Cookie File", "# Netscape HTTP Cookie File"}


@dataclass(frozen=True)
class VideoProxy:
    """Proxy policy scoped only to yt-dlp video downloads."""

    mode: str = "environment"
    url: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"environment", "direct", "custom"}:
            raise PipelineError(f"Unsupported video proxy mode: {self.mode}")
        if self.mode == "custom":
            if not self.url:
                raise PipelineError("Custom video proxy mode requires a proxy URL")
            validate_proxy_url(self.url)

    def yt_dlp_args(self) -> list[str]:
        if self.mode == "direct":
            # yt-dlp documents an empty proxy URL as an explicit direct connection.
            return ["--proxy", ""]
        if self.mode == "custom":
            return ["--proxy", str(self.url)]
        return []

    def provenance(self) -> dict[str, object]:
        if self.mode == "custom":
            return {
                "mode": self.mode,
                "configured": True,
                "endpoint": redact_proxy_url(str(self.url)),
            }
        if self.mode == "environment":
            env_name, env_value = detect_environment_proxy()
            return {
                "mode": self.mode,
                "configured": bool(env_value),
                "environment_variable": env_name,
                "endpoint": redact_proxy_url(env_value) if env_value else None,
            }
        return {"mode": self.mode, "configured": False, "endpoint": None}


@dataclass(frozen=True)
class VideoAuth:
    """Cookie authentication scoped only to the yt-dlp process.

    Cookie contents and absolute cookie-file paths are deliberately excluded from
    provenance so project metadata cannot become an authentication secret store.
    """

    mode: str = "none"
    cookies_path: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"none", "cookies"}:
            raise PipelineError(f"Unsupported video authentication mode: {self.mode}")
        if self.mode == "cookies":
            if not self.cookies_path:
                raise PipelineError("Cookie authentication requires a cookies.txt path")
            validate_cookies_file(Path(self.cookies_path).expanduser())

    def yt_dlp_args(self) -> list[str]:
        if self.mode == "cookies":
            return ["--cookies", str(Path(str(self.cookies_path)).expanduser().resolve())]
        return []

    def provenance(self) -> dict[str, object]:
        if self.mode == "cookies":
            return {
                "mode": "cookies_file",
                "configured": True,
                # The basename is useful provenance while avoiding disclosure of
                # private NAS directory layout. Never persist the file contents.
                "filename": Path(str(self.cookies_path)).name,
            }
        return {"mode": "none", "configured": False}


@dataclass(frozen=True)
class VideoFrameReport:
    source: str
    downloaded_video: str
    sampled_frames: int
    accepted_frames: int
    rejected_blurry: int
    rejected_near_duplicate: int
    rejected_exposure: int
    interval_seconds: int
    max_frames: int
    proxy: dict[str, object] | None = None
    authentication: dict[str, object] | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "downloaded_video": self.downloaded_video,
            "sampled_frames": self.sampled_frames,
            "accepted_frames": self.accepted_frames,
            "rejected_blurry": self.rejected_blurry,
            "rejected_near_duplicate": self.rejected_near_duplicate,
            "rejected_exposure": self.rejected_exposure,
            "interval_seconds": self.interval_seconds,
            "max_frames": self.max_frames,
            "proxy": self.proxy,
            "authentication": self.authentication,
        }


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def detect_environment_proxy() -> tuple[str | None, str | None]:
    for name in _PROXY_ENV_NAMES:
        value = os.environ.get(name)
        if value:
            return name, value
    return None, None


def detect_cookies_file() -> tuple[str | None, Path | None]:
    """Return a configured cookie file without reading or persisting its contents."""

    env_value = os.environ.get(_COOKIES_ENV_NAME)
    if env_value:
        path = Path(env_value).expanduser()
        if path.is_file():
            return _COOKIES_ENV_NAME, path.resolve()
    if _DEFAULT_COOKIES_PATH.is_file():
        return "default", _DEFAULT_COOKIES_PATH.resolve()
    return None, None


def validate_cookies_file(path: Path) -> None:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise PipelineError(f"Cookies file does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            first = handle.readline().strip().lstrip("\ufeff")
            if first not in _NETSCAPE_COOKIE_HEADERS:
                raise PipelineError(
                    "Cookies file must use Netscape/Mozilla cookies.txt format"
                )
            has_youtube = any("youtube.com" in line.casefold() for line in handle)
    except OSError as exc:
        raise PipelineError(f"Cannot read cookies file: {path}") from exc
    if not has_youtube:
        raise PipelineError("Cookies file contains no youtube.com cookies")


def validate_proxy_url(value: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme.lower() not in _SUPPORTED_PROXY_SCHEMES or not parsed.hostname:
        schemes = ", ".join(sorted(_SUPPORTED_PROXY_SCHEMES))
        raise PipelineError(f"Proxy URL must use one of {schemes} and include a host")
    try:
        port = parsed.port
    except ValueError as exc:
        raise PipelineError("Proxy port must be a valid number between 1 and 65535") from exc
    if port is not None and not (1 <= port <= 65535):
        raise PipelineError("Proxy port must be between 1 and 65535")


def redact_proxy_url(value: str | None) -> str | None:
    if not value:
        return None
    parsed = urlparse(value)
    if not parsed.scheme or not parsed.netloc:
        return "configured"
    hostname = parsed.hostname or ""
    host = f"[{hostname}]" if ":" in hostname and not hostname.startswith("[") else hostname
    try:
        port = parsed.port
    except ValueError:
        return f"{parsed.scheme}://{host}:configured-port"
    if port is not None:
        host = f"{host}:{port}"
    return urlunparse((parsed.scheme, host, parsed.path, "", "", ""))


def require_video_tools(*, remote: bool) -> None:
    missing: list[str] = []
    if shutil.which("ffmpeg") is None:
        missing.append("ffmpeg")
    if remote and importlib.util.find_spec("yt_dlp") is None:
        missing.append("yt-dlp")
    if missing:
        raise PipelineError(
            "Video import needs these executables on PATH: " + ", ".join(missing)
        )


def extract_video_frames(
    source: str,
    output_dir: Path,
    *,
    interval_seconds: int = 2,
    max_frames: int = 250,
    phash_distance: int = 7,
    blur_threshold: float = 55.0,
    proxy: VideoProxy | None = None,
    auth: VideoAuth | None = None,
) -> VideoFrameReport:
    if interval_seconds < 1:
        raise PipelineError("Frame interval must be at least 1 second")
    if max_frames < 1:
        raise PipelineError("Maximum frame count must be at least 1")
    if phash_distance < 0:
        raise PipelineError("Perceptual-hash distance cannot be negative")

    remote = is_url(source)
    require_video_tools(remote=remote)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    proxy = proxy or VideoProxy(mode="environment")
    auth = auth or VideoAuth(mode="none")

    with tempfile.TemporaryDirectory(prefix="lora-video-") as temporary:
        temporary_dir = Path(temporary)
        video_path = _resolve_video(
            source,
            temporary_dir,
            remote=remote,
            proxy=proxy,
            auth=auth,
        )
        candidate_dir = temporary_dir / "frames"
        candidate_dir.mkdir()
        # Sample more candidates than the final cap because filtering can reject many.
        candidate_cap = max(max_frames * 4, max_frames + 20)
        _sample_frames(
            video_path,
            candidate_dir,
            interval_seconds=interval_seconds,
            frame_cap=candidate_cap,
        )
        candidates = sorted(candidate_dir.glob("frame-*.jpg"))
        if not candidates:
            raise PipelineError("ffmpeg produced no frames from the selected video")

        accepted_hashes: list[imagehash.ImageHash] = []
        accepted = 0
        blurry = 0
        duplicates = 0
        exposure = 0
        for candidate in candidates:
            if accepted >= max_frames:
                break
            with Image.open(candidate) as opened:
                image = opened.convert("RGB")
                brightness = float(ImageStat.Stat(image.convert("L")).mean[0])
                if brightness < 18.0 or brightness > 238.0:
                    exposure += 1
                    continue
                if _sharpness_score(image) < blur_threshold:
                    blurry += 1
                    continue
                fingerprint = imagehash.phash(image)
                if any(fingerprint - previous <= phash_distance for previous in accepted_hashes):
                    duplicates += 1
                    continue
                target = output_dir / f"video-{accepted + 1:05d}.jpg"
                image.save(target, quality=95, subsampling=0)
                accepted_hashes.append(fingerprint)
                accepted += 1

        if accepted == 0:
            raise PipelineError(
                "All sampled video frames were rejected as blurry, duplicate, or badly exposed"
            )
        return VideoFrameReport(
            source=source,
            downloaded_video=str(video_path),
            sampled_frames=len(candidates),
            accepted_frames=accepted,
            rejected_blurry=blurry,
            rejected_near_duplicate=duplicates,
            rejected_exposure=exposure,
            interval_seconds=interval_seconds,
            max_frames=max_frames,
            proxy=proxy.provenance() if remote else None,
            authentication=auth.provenance() if remote else None,
        )


def _resolve_video(
    source: str,
    temporary_dir: Path,
    *,
    remote: bool,
    proxy: VideoProxy,
    auth: VideoAuth,
) -> Path:
    if not remote:
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise PipelineError(f"Video file does not exist: {path}")
        return path

    output_template = temporary_dir / "source.%(ext)s"
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-playlist",
        "--no-progress",
        *proxy.yt_dlp_args(),
        *auth.yt_dlp_args(),
        "-f",
        "bestvideo[height<=1080]/best[height<=1080]",
        "-o",
        str(output_template),
        source,
    ]
    _run(command, "yt-dlp could not download the video")
    matches = sorted(temporary_dir.glob("source.*"))
    if not matches:
        raise PipelineError("yt-dlp finished without producing a video file")
    return matches[0]


def _sample_frames(
    video_path: Path,
    output_dir: Path,
    *,
    interval_seconds: int,
    frame_cap: int,
) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"fps=1/{interval_seconds}",
        "-frames:v",
        str(frame_cap),
        "-q:v",
        "2",
        str(output_dir / "frame-%06d.jpg"),
    ]
    _run(command, "ffmpeg could not extract frames from the video")


def _sharpness_score(image: Image.Image) -> float:
    edges = image.convert("L").filter(ImageFilter.FIND_EDGES)
    return float(ImageStat.Stat(edges).var[0])


def _run(command: list[str], failure: str) -> None:
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        tail = detail[-1] if detail else f"exit code {result.returncode}"
        raise PipelineError(f"{failure}: {tail}")
