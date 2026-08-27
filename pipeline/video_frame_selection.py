from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from .video_color import VideoColorInfo, color_normalization_filters


# Eight candidate positions per second means +/- 2 candidates covers a compact
# 250 ms window on either side of the nominal sampling timestamp. This is wide
# enough to dodge a transient motion-blur frame without drifting to a materially
# different moment in the video.
_CANDIDATE_FPS = 8
_NEIGHBOR_OFFSETS = (-2, -1, 0, 1, 2)
_PROXY_LONG_EDGE = 960
_BLUR_LINE_RE = re.compile(r"\bblur:\s*([0-9]+(?:\.[0-9]+)?)")


def try_sample_sharp_windows(
    video_path: Path,
    output_dir: Path,
    *,
    color_info: VideoColorInfo,
    interval_seconds: int,
    frame_cap: int,
    ffmpeg: str = "ffmpeg",
) -> bool:
    """Select the sharpest nearby decoded frame for each nominal screenshot time.

    This is a two-pass extraction strategy:

    1. score a small low-resolution burst around every nominal timestamp with
       FFmpeg's metadata-only blurdetect filter;
    2. re-run the deterministic candidate grid and write only the winning frames
       at source resolution, applying the normal HDR -> SDR conversion if needed.

    Returning ``False`` means the optimized selector is unavailable or failed and
    the caller should fall back to the legacy fixed-timestamp sampler. No partial
    frame set is left behind on failure.
    """

    if interval_seconds < 1 or frame_cap < 1:
        return False
    if not _ffmpeg_has_filter(ffmpeg, "blurdetect"):
        return False

    period = interval_seconds * _CANDIDATE_FPS
    candidate_indices = _candidate_indices(period, frame_cap)
    if not candidate_indices:
        return False

    probe_filter = _probe_filter(period, color_info)
    probe_command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "verbose",
        "-i",
        str(video_path),
        "-vf",
        probe_filter,
        "-frames:v",
        str(len(candidate_indices)),
        "-an",
        "-f",
        "null",
        "-",
    ]
    probe = subprocess.run(
        probe_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        return False

    scores = _parse_blur_scores(probe.stderr)
    if not scores:
        return False
    selected = _select_best_indices(
        candidate_indices[: len(scores)],
        scores,
        period=period,
        target_count=frame_cap,
    )
    if not selected:
        return False

    staging = output_dir / ".sharp-window"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    try:
        extract_filter = _extract_filter(selected, color_info)
        extract_command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vf",
            extract_filter,
            "-vsync",
            "0",
            "-frames:v",
            str(len(selected)),
            "-q:v",
            "2",
            str(staging / "frame-%06d.jpg"),
        ]
        extracted = subprocess.run(
            extract_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if extracted.returncode != 0:
            return False

        frames = sorted(staging.glob("frame-*.jpg"))
        if not frames:
            return False
        for index, frame in enumerate(frames, start=1):
            frame.replace(output_dir / f"frame-{index:06d}.jpg")
        return True
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _ffmpeg_has_filter(ffmpeg: str, name: str) -> bool:
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-filters"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except OSError:
        return False
    if result.returncode != 0:
        return False
    listing = f"{result.stdout}\n{result.stderr}"
    return re.search(rf"\b{re.escape(name)}\b", listing) is not None


def _candidate_indices(period: int, target_count: int) -> list[int]:
    if period < 1 or target_count < 1:
        return []
    indexes: set[int] = set()
    for target in range(target_count):
        center = target * period
        for offset in _NEIGHBOR_OFFSETS:
            candidate = center + offset
            if candidate >= 0:
                indexes.add(candidate)
    return sorted(indexes)


def _window_select_expression(period: int) -> str:
    residues = sorted({offset % period for offset in _NEIGHBOR_OFFSETS})
    terms = [f"eq(mod(n\\,{period})\\,{residue})" for residue in residues]
    return "+".join(terms)


def _probe_filter(period: int, color_info: VideoColorInfo) -> str:
    filters = [
        f"fps={_CANDIDATE_FPS}:start_time=0:round=near",
        f"select='{_window_select_expression(period)}'",
        # The proxy is only used for relative blur ranking. Downscaling before HDR
        # tone mapping saves substantial work on 4K sources; final pixels are never
        # produced from this proxy.
        (
            f"scale={_PROXY_LONG_EDGE}:{_PROXY_LONG_EDGE}:"
            "force_original_aspect_ratio=decrease:flags=area"
        ),
        *color_normalization_filters(color_info),
        "format=yuv420p",
        "blurdetect=block_width=32:block_height=32:block_pct=80",
    ]
    return ",".join(filters)


def _extract_filter(selected_indices: list[int], color_info: VideoColorInfo) -> str:
    select_expression = "+".join(f"eq(n\\,{index})" for index in selected_indices)
    filters = [
        f"fps={_CANDIDATE_FPS}:start_time=0:round=near",
        f"select='{select_expression}'",
        *color_normalization_filters(color_info),
    ]
    return ",".join(filters)


def _parse_blur_scores(stderr: str) -> list[float]:
    scores: list[float] = []
    for line in stderr.splitlines():
        if "blur mean:" in line:
            continue
        match = _BLUR_LINE_RE.search(line)
        if match is not None:
            scores.append(float(match.group(1)))
    return scores


def _select_best_indices(
    candidate_indices: list[int],
    blur_scores: list[float],
    *,
    period: int,
    target_count: int,
) -> list[int]:
    """Pick minimum-blur candidates, preferring the nominal timestamp on ties."""

    winners: dict[int, tuple[float, int, int]] = {}
    for candidate, score in zip(candidate_indices, blur_scores):
        target = (candidate + period // 2) // period
        if not (0 <= target < target_count):
            continue
        center = target * period
        ranking = (float(score), abs(candidate - center), candidate)
        previous = winners.get(target)
        if previous is None or ranking < previous:
            winners[target] = ranking
    return [winners[target][2] for target in sorted(winners)]
