from __future__ import annotations

import contextvars
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import video_frame_selection as frame_selection
from . import video_source
from .dataset_workspace import DatasetWorkspace
from .models import PipelineError
from .wizard import MenuItem


@dataclass(frozen=True)
class LocalVideoImportSettings:
    mode: str
    interval_seconds: int
    max_frames: int
    start_seconds: float | None = None
    end_seconds: float | None = None

    @property
    def window(self) -> tuple[float, float] | None:
        if self.start_seconds is None or self.end_seconds is None:
            return None
        return self.start_seconds, self.end_seconds


_ACTIVE_VIDEO_WINDOW: contextvars.ContextVar[tuple[float, float] | None] = contextvars.ContextVar(
    "lora_pipeline_active_video_window",
    default=None,
)
_ORIGINAL_SAMPLE_FRAMES = video_source._sample_frames
_INSTALLED = False


def install_local_video_import_modes(wizard_class: type[Any]) -> None:
    """Add one-click/advanced local-video imports without changing remote-video behavior."""

    global _INSTALLED
    if _INSTALLED:
        return

    original_import = wizard_class._import_video_source

    def import_video_source(
        self: Any,
        workspace: DatasetWorkspace,
        *,
        remote: bool,
    ) -> None:
        if remote:
            return original_import(self, workspace, remote=True)
        return _import_local_video_source(self, workspace)

    setattr(import_video_source, "_lora_local_video_modes", True)
    setattr(import_video_source, "_lora_original", original_import)
    wizard_class._import_video_source = import_video_source

    # extract_video_frames() resolves _sample_frames from video_source globals at
    # call time, so this keeps every existing blur/exposure/pHash filter intact.
    # The ContextVar scopes a requested range to only the current interactive
    # import and avoids leaking range state to remote or later imports.
    video_source._sample_frames = _sample_frames_window_aware
    _INSTALLED = True


def parse_video_timestamp(value: str) -> float:
    """Parse seconds, MM:SS, or HH:MM:SS into seconds."""

    raw = str(value).strip()
    if not raw:
        raise ValueError("time value cannot be empty")
    parts = raw.split(":")
    if len(parts) > 3:
        raise ValueError("use seconds, MM:SS, or HH:MM:SS")

    try:
        if len(parts) == 1:
            seconds = float(parts[0])
            if seconds < 0:
                raise ValueError("time cannot be negative")
            return seconds

        seconds_part = float(parts[-1])
        if not 0 <= seconds_part < 60:
            raise ValueError("seconds must be between 0 and 59.999...")
        minutes = int(parts[-2])
        if minutes < 0:
            raise ValueError("minutes cannot be negative")

        if len(parts) == 2:
            return minutes * 60 + seconds_part

        if minutes >= 60:
            raise ValueError("minutes must be between 0 and 59 in HH:MM:SS")
        hours = int(parts[0])
        if hours < 0:
            raise ValueError("hours cannot be negative")
        return hours * 3600 + minutes * 60 + seconds_part
    except ValueError as exc:
        if str(exc) in {
            "time cannot be negative",
            "seconds must be between 0 and 59.999...",
            "minutes cannot be negative",
            "minutes must be between 0 and 59 in HH:MM:SS",
            "hours cannot be negative",
        }:
            raise
        raise ValueError("use seconds, MM:SS, or HH:MM:SS") from exc


def format_video_timestamp(seconds: float) -> str:
    value = max(0.0, float(seconds))
    hours = int(value // 3600)
    minutes = int((value % 3600) // 60)
    remainder = value % 60
    if abs(remainder - round(remainder)) < 1e-9:
        return f"{hours:02d}:{minutes:02d}:{int(round(remainder)):02d}"
    return f"{hours:02d}:{minutes:02d}:{remainder:06.3f}"


def _import_local_video_source(self: Any, workspace: DatasetWorkspace) -> None:
    source = self._ask_local_video_file()
    settings = _choose_local_video_settings(self)
    if settings is None:
        return

    self._video_interval_seconds = settings.interval_seconds
    proxy = self._select_video_proxy(source)
    work_root = workspace.dataset_dir / "cache" / "work"
    work_root.mkdir(parents=True, exist_ok=True)

    if settings.window is None:
        range_text = self._b(
            "时间范围：默认策略（未限制开始/结束时间）",
            "Time range: default strategy (no explicit start/end bound)",
        )
    else:
        start_seconds, end_seconds = settings.window
        range_text = self._b(
            f"时间范围：{format_video_timestamp(start_seconds)} → {format_video_timestamp(end_seconds)}",
            f"Time range: {format_video_timestamp(start_seconds)} → {format_video_timestamp(end_seconds)}",
        )
    self.console.print(
        self._b(
            f"[cyan]本地视频导入[/cyan] · {'一键导入' if settings.mode == 'one_click' else '高级导入'}\n"
            f"{range_text}\n采样间隔：{settings.interval_seconds} 秒 · 最多保留：{settings.max_frames} 帧",
            f"[cyan]Local video import[/cyan] · {'One-click' if settings.mode == 'one_click' else 'Advanced'}\n"
            f"{range_text}\nSampling interval: {settings.interval_seconds}s · maximum accepted: {settings.max_frames} frames",
        )
    )

    with tempfile.TemporaryDirectory(prefix="video-source-", dir=work_root) as temporary:
        frame_dir = Path(temporary) / "frames"
        token = _ACTIVE_VIDEO_WINDOW.set(settings.window)
        try:
            report, _proxy = self._extract_video_with_retry(
                source,
                frame_dir,
                interval_seconds=settings.interval_seconds,
                max_frames=settings.max_frames,
                proxy=proxy,
            )
        finally:
            _ACTIVE_VIDEO_WINDOW.reset(token)

        self._render_video_report(report.as_dict())
        training_dir, identity = self._select_video_identity(frame_dir)
        processing = report.as_dict()
        processing.pop("downloaded_video", None)
        # The backend report describes extraction mechanics; preserve the original
        # local source and explicitly record the user's import policy alongside it.
        processing["source"] = source
        processing["identity_preselection"] = identity
        processing["source_kind"] = "local_video"
        processing["import_mode"] = settings.mode
        if settings.window is None:
            processing["time_range"] = {"mode": "unbounded"}
        else:
            start_seconds, end_seconds = settings.window
            processing["time_range"] = {
                "mode": "explicit",
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "start": format_video_timestamp(start_seconds),
                "end": format_video_timestamp(end_seconds),
            }

        default_label = Path(source).stem
        label = self._ask_text(
            self._b("来源名称", "Source label"),
            default=default_label or "video",
        ).strip()
        record = workspace.add_source_from_directory(
            training_dir,
            kind="local_video",
            label=label,
            origin=source,
            processing=processing,
        )
    self._render_source_imported(record)


def _choose_local_video_settings(self: Any) -> LocalVideoImportSettings | None:
    mode = self._menu(
        self._b("本地视频导入方式", "Local video import mode"),
        [
            MenuItem(
                "one_click",
                self._b("一键导入", "One-click import"),
                self._b(
                    "不限制开始/结束时间，直接使用默认策略：每 2 秒采样一次，最多保留 250 个通过质量筛选的帧。适合先快速建立数据集。",
                    "Use the default strategy with no explicit start/end bound: sample every 2 seconds and keep at most 250 frames that pass quality filtering. Best for a quick first import.",
                ),
            ),
            MenuItem(
                "advanced",
                self._b("高级导入", "Advanced import"),
                self._b(
                    "输入开始时间点和结束时间点，只处理该片段；同时可调整采样间隔和最大保留帧数。原视频不会被修改。",
                    "Enter explicit start and end timestamps and process only that segment; sampling interval and maximum accepted frames are also configurable. The source video is never modified.",
                ),
            ),
            MenuItem(
                "back",
                self._b("返回", "Back"),
                self._b("取消这次本地视频导入并返回来源类型菜单。", "Cancel this local-video import and return to the source-type menu."),
            ),
        ],
        default="one_click",
    )
    if mode == "back":
        return None
    if mode == "one_click":
        return LocalVideoImportSettings(
            mode="one_click",
            interval_seconds=2,
            max_frames=250,
        )

    start_seconds = _ask_timestamp(
        self,
        self._b(
            "开始时间点（例如 90 / 01:30 / 00:01:30）",
            "Start timestamp (for example 90 / 01:30 / 00:01:30)",
        ),
        default="00:00:00",
    )
    while True:
        end_seconds = _ask_timestamp(
            self,
            self._b(
                "结束时间点（例如 06:45 / 00:06:45）",
                "End timestamp (for example 06:45 / 00:06:45)",
            ),
        )
        if end_seconds > start_seconds:
            break
        self.console.print(
            self._b(
                "[red]结束时间点必须晚于开始时间点。[/red]",
                "[red]End timestamp must be later than start timestamp.[/red]",
            )
        )

    interval_seconds = self._ask_positive_int(
        self._b("每隔多少秒采样一帧", "Sample one frame every N seconds"),
        default=2,
    )
    max_frames = self._ask_positive_int(
        self._b("人物识别前最多保留多少帧", "Maximum accepted frames before identity selection"),
        default=250,
    )
    return LocalVideoImportSettings(
        mode="advanced",
        interval_seconds=interval_seconds,
        max_frames=max_frames,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )


def _ask_timestamp(self: Any, prompt: str, *, default: str | None = None) -> float:
    while True:
        raw = self._ask_text(prompt, default=default).strip()
        try:
            return parse_video_timestamp(raw)
        except ValueError as exc:
            self.console.print(
                self._b(
                    f"[red]时间格式无效：{exc}。可使用秒数、MM:SS 或 HH:MM:SS。[/red]",
                    f"[red]Invalid timestamp: {exc}. Use seconds, MM:SS, or HH:MM:SS.[/red]",
                )
            )


def _sample_frames_window_aware(
    video_path: Path,
    output_dir: Path,
    *,
    interval_seconds: int,
    frame_cap: int,
):
    window = _ACTIVE_VIDEO_WINDOW.get()
    if window is None:
        return _ORIGINAL_SAMPLE_FRAMES(
            video_path,
            output_dir,
            interval_seconds=interval_seconds,
            frame_cap=frame_cap,
        )
    return _sample_frames_in_window(
        video_path,
        output_dir,
        interval_seconds=interval_seconds,
        frame_cap=frame_cap,
        start_seconds=window[0],
        end_seconds=window[1],
    )


def _sample_frames_in_window(
    video_path: Path,
    output_dir: Path,
    *,
    interval_seconds: int,
    frame_cap: int,
    start_seconds: float,
    end_seconds: float,
):
    if start_seconds < 0 or end_seconds <= start_seconds:
        raise PipelineError("Video time range requires 0 <= start < end")

    color_info = video_source.probe_video_color(video_path)
    if _try_sample_sharp_windows_in_range(
        video_path,
        output_dir,
        color_info=color_info,
        interval_seconds=interval_seconds,
        frame_cap=frame_cap,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    ):
        return color_info

    filter_chain = video_source.sampling_filter(interval_seconds, color_info)
    before_input, after_input = _ffmpeg_window_args(start_seconds, end_seconds)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        *before_input,
        "-i",
        str(video_path),
        *after_input,
        "-vf",
        filter_chain,
        "-frames:v",
        str(frame_cap),
        "-q:v",
        "2",
        str(output_dir / "frame-%06d.jpg"),
    ]
    failure = (
        video_source.hdr_sampling_failure_message(color_info)
        if color_info.is_hdr
        else "ffmpeg could not extract frames from the selected video time range"
    )
    video_source._run(command, failure)
    return color_info


def _try_sample_sharp_windows_in_range(
    video_path: Path,
    output_dir: Path,
    *,
    color_info: Any,
    interval_seconds: int,
    frame_cap: int,
    start_seconds: float,
    end_seconds: float,
    ffmpeg: str = "ffmpeg",
) -> bool:
    if interval_seconds < 1 or frame_cap < 1:
        return False
    if not frame_selection._ffmpeg_has_filter(ffmpeg, "blurdetect"):
        return False

    period = interval_seconds * frame_selection._CANDIDATE_FPS
    candidate_indices = frame_selection._candidate_indices(period, frame_cap)
    if not candidate_indices:
        return False

    before_input, after_input = _ffmpeg_window_args(start_seconds, end_seconds)
    probe_command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "verbose",
        *before_input,
        "-i",
        str(video_path),
        *after_input,
        "-vf",
        frame_selection._probe_filter(period, color_info),
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

    scores = frame_selection._parse_blur_scores(probe.stderr)
    if not scores:
        return False
    selected = frame_selection._select_best_indices(
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
        extract_command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            *before_input,
            "-i",
            str(video_path),
            *after_input,
            "-vf",
            frame_selection._extract_filter(selected, color_info),
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


def _ffmpeg_window_args(start_seconds: float, end_seconds: float) -> tuple[list[str], list[str]]:
    if start_seconds < 0 or end_seconds <= start_seconds:
        raise ValueError("video window requires 0 <= start < end")
    before_input: list[str] = []
    if start_seconds > 0:
        before_input = ["-ss", _ffmpeg_seconds(start_seconds)]
    duration = end_seconds - start_seconds
    after_input = ["-t", _ffmpeg_seconds(duration)]
    return before_input, after_input


def _ffmpeg_seconds(value: float) -> str:
    return f"{float(value):.6f}".rstrip("0").rstrip(".") or "0"
