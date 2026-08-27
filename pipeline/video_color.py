from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import PipelineError


_HDR_TRANSFERS = {
    "smpte2084": "pq",
    "arib-std-b67": "hlg",
}
_TONEMAP_ALGORITHM = "mobius"
_TONEMAP_PARAMETER = 0.3
_TARGET_COLORSPACE = "bt709"
_TARGET_NITS = 100


@dataclass(frozen=True)
class VideoColorInfo:
    """Color metadata for the first video stream and its SDR conversion policy."""

    width: int | None = None
    height: int | None = None
    pixel_format: str | None = None
    color_space: str | None = None
    color_transfer: str | None = None
    color_primaries: str | None = None
    color_range: str | None = None

    @property
    def hdr_kind(self) -> str | None:
        return _HDR_TRANSFERS.get((self.color_transfer or "").casefold())

    @property
    def is_hdr(self) -> bool:
        return self.hdr_kind is not None

    def as_dict(self) -> dict[str, Any]:
        tone_mapping: dict[str, Any] = {"applied": self.is_hdr}
        if self.is_hdr:
            tone_mapping.update(
                {
                    "algorithm": _TONEMAP_ALGORITHM,
                    "parameter": _TONEMAP_PARAMETER,
                    "target_color_space": _TARGET_COLORSPACE,
                    "target_nits": _TARGET_NITS,
                    "highlight_desaturation": False,
                }
            )
        return {
            "width": self.width,
            "height": self.height,
            "pixel_format": self.pixel_format,
            "color_space": self.color_space,
            "color_transfer": self.color_transfer,
            "color_primaries": self.color_primaries,
            "color_range": self.color_range,
            "hdr": self.is_hdr,
            "hdr_kind": self.hdr_kind,
            "tone_mapping": tone_mapping,
        }


def probe_video_color(video_path: Path) -> VideoColorInfo:
    """Read color metadata with ffprobe instead of guessing from decoded RGB values."""

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise PipelineError(
            "Video color detection needs ffprobe on PATH. ffprobe normally ships with FFmpeg; "
            "HDR video must not be sampled without color metadata because the frames may look washed out."
        )

    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,pix_fmt,color_space,color_transfer,color_primaries,color_range",
        "-of",
        "json",
        str(video_path),
    ]
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
        raise PipelineError(f"ffprobe could not inspect video color metadata: {tail}")

    try:
        payload = json.loads(result.stdout or "{}")
        streams = payload.get("streams") or []
        stream = streams[0]
    except (json.JSONDecodeError, IndexError, TypeError, AttributeError) as exc:
        raise PipelineError("ffprobe returned no readable primary video stream metadata") from exc

    def _text(name: str) -> str | None:
        value = stream.get(name)
        if value is None:
            return None
        text = str(value).strip().casefold()
        return text or None

    def _integer(name: str) -> int | None:
        value = stream.get(name)
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    return VideoColorInfo(
        width=_integer("width"),
        height=_integer("height"),
        pixel_format=_text("pix_fmt"),
        color_space=_text("color_space"),
        color_transfer=_text("color_transfer"),
        color_primaries=_text("color_primaries"),
        color_range=_text("color_range"),
    )


def sampling_filter(interval_seconds: int, color: VideoColorInfo) -> str:
    """Build the frame-sampling filter; HDR sources are normalized to SDR BT.709 first."""

    filters = [f"fps=1/{interval_seconds}"]
    if not color.is_hdr:
        return ",".join(filters)

    # zscale converts the HDR transfer function to scene-linear light.  The gamut is
    # brought into BT.709 before Mobius compresses highlights.  desat=0 is deliberate:
    # anime footage often contains saturated emissive colors and automatic highlight
    # desaturation can make character colors look pale.  The final JPEG path is full
    # range BT.709.  Small/normal SDR videos never enter this chain.
    filters.extend(
        [
            f"zscale=t=linear:npl={_TARGET_NITS}",
            "format=gbrpf32le",
            f"zscale=p={_TARGET_COLORSPACE}",
            f"tonemap={_TONEMAP_ALGORITHM}:param={_TONEMAP_PARAMETER}:desat=0",
            f"zscale=t={_TARGET_COLORSPACE}:m={_TARGET_COLORSPACE}:r=full",
            "format=yuvj444p",
        ]
    )
    return ",".join(filters)


def hdr_sampling_failure_message(color: VideoColorInfo) -> str:
    kind = (color.hdr_kind or "HDR").upper()
    return (
        f"ffmpeg could not tone-map/extract {kind} video frames. "
        "Use an FFmpeg build that includes the zscale and tonemap filters"
    )
