from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import imagehash
from PIL import Image, ImageFilter, ImageStat

from .models import PipelineError


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
        }


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def require_video_tools(*, remote: bool) -> None:
    missing: list[str] = []
    if shutil.which("ffmpeg") is None:
        missing.append("ffmpeg")
    if remote and shutil.which("yt-dlp") is None:
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

    with tempfile.TemporaryDirectory(prefix="lora-video-") as temporary:
        temporary_dir = Path(temporary)
        video_path = _resolve_video(source, temporary_dir, remote=remote)
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
        )


def _resolve_video(source: str, temporary_dir: Path, *, remote: bool) -> Path:
    if not remote:
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise PipelineError(f"Video file does not exist: {path}")
        return path

    output_template = temporary_dir / "source.%(ext)s"
    command = [
        "yt-dlp",
        "--no-playlist",
        "--no-progress",
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
