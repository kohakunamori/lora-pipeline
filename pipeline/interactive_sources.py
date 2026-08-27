from __future__ import annotations

from pathlib import Path
from typing import Sequence

from rich.panel import Panel

from .interactive_video_auth import InteractiveWizard as BaseInteractiveWizard
from .video_source import is_url
from .wizard import MenuItem, Wizard


class InteractiveWizard(BaseInteractiveWizard):
    """Interactive entry point with explicit image/local-video/remote-video sources.

    The video backend already supports local files. This layer makes that support
    visible instead of hiding a local path behind the same prompt as a YouTube URL.
    Local files never enter the yt-dlp/proxy/cookie/PO-token path.
    """

    _selected_video_source_kind: str | None = None

    def new_project(self):
        source_kind = self._menu(
            self._b("训练数据来源", "Training data source"),
            [
                MenuItem(
                    "images",
                    self._b("图片目录", "Image directory"),
                    self._b(
                        "使用 NAS 上已有的图片目录和可选的同名 .txt 描述。",
                        "Use an existing image directory on the NAS with optional same-stem .txt captions.",
                    ),
                ),
                MenuItem(
                    "local_video",
                    self._b("本地视频文件", "Local video file"),
                    self._b(
                        "直接处理 NAS 上的 MP4/MKV/MOV/WebM 等视频；不使用网络、代理或 Cookie。",
                        "Process a video already on the NAS; no network, proxy, or cookies are used.",
                    ),
                ),
                MenuItem(
                    "remote_video",
                    self._b("在线视频 / YouTube 链接", "Online video / YouTube URL"),
                    self._b(
                        "通过 yt-dlp 下载后抽帧，可使用代理、cookies.txt 和 YouTube 兼容策略。",
                        "Download with yt-dlp, then extract frames; proxy and cookies.txt support remain available.",
                    ),
                ),
            ],
            default="images",
        )
        if source_kind == "images":
            # Skip the older two-way source chooser and enter the original image
            # project wizard directly.
            return Wizard.new_project(self)

        self._selected_video_source_kind = source_kind
        try:
            return super()._new_project_from_video()
        finally:
            self._selected_video_source_kind = None

    def _ask_text(
        self,
        prompt: str,
        *,
        default: str | None = None,
        choices: Sequence[str] | None = None,
    ) -> str:
        # interactive_app owns the full video project workflow and asks this one
        # legacy combined question. Intercept only that prompt so all downstream
        # frame filtering, CCIP selection, project creation, and provenance remain
        # on the same implementation path.
        if prompt == "YouTube URL or local video path":
            if self._selected_video_source_kind == "local_video":
                return self._ask_local_video_file()
            if self._selected_video_source_kind == "remote_video":
                return self._ask_remote_video_url()
        return super()._ask_text(prompt, default=default, choices=choices)

    def _ask_local_video_file(self) -> str:
        while True:
            raw = super()._ask_text(
                self._b("本地视频文件路径", "Local video file path")
            ).strip()
            if not raw:
                self.console.print(
                    self._b("[red]视频路径不能为空。[/red]", "[red]Video path cannot be empty.[/red]")
                )
                continue

            path = Path(raw).expanduser().resolve()
            if not path.is_file():
                self.console.print(
                    self._b(
                        f"[red]找不到视频文件：{path}[/red]",
                        f"[red]Video file does not exist: {path}[/red]",
                    )
                )
                continue
            try:
                size = path.stat().st_size
                with path.open("rb") as handle:
                    handle.read(1)
            except OSError as exc:
                self.console.print(
                    self._b(
                        f"[red]无法读取视频文件：{path}\n{exc}[/red]",
                        f"[red]Cannot read video file: {path}\n{exc}[/red]",
                    )
                )
                continue
            if size <= 0:
                self.console.print(
                    self._b("[red]视频文件为空。[/red]", "[red]The video file is empty.[/red]")
                )
                continue

            self.console.print(
                Panel.fit(
                    self._b(
                        f"[cyan]本地视频[/cyan]\n路径：{path}\n大小：{self._human_bytes(size)}\n"
                        "处理时只读取原视频，不会修改或移动它。",
                        f"[cyan]Local video[/cyan]\nPath: {path}\nSize: {self._human_bytes(size)}\n"
                        "The source video is read only; it is never modified or moved.",
                    )
                )
            )
            if self._confirm(self._b("使用这个视频吗？", "Use this video?"), default=True):
                return str(path)

    def _ask_remote_video_url(self) -> str:
        while True:
            source = super()._ask_text(
                self._b("视频 / YouTube 链接", "Video / YouTube URL")
            ).strip()
            if not source:
                self.console.print(
                    self._b("[red]视频链接不能为空。[/red]", "[red]Video URL cannot be empty.[/red]")
                )
                continue
            if not is_url(source):
                self.console.print(
                    self._b(
                        "[red]这里需要 http:// 或 https:// 视频链接。若视频已经在 NAS 上，请返回并选择“本地视频文件”。[/red]",
                        "[red]Enter an http:// or https:// URL here. If the video is already on the NAS, go back and choose Local video file.[/red]",
                    )
                )
                continue
            return source
