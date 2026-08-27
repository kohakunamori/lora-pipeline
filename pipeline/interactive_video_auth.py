from __future__ import annotations

import shutil
from pathlib import Path

from rich.panel import Panel

from .i18n import get_language
from .interactive_app import InteractiveWizard as BaseInteractiveWizard
from .models import PipelineError
from .video_source import (
    VideoAuth,
    VideoProxy,
    detect_cookies_file,
    extract_video_frames,
    is_url,
)
from .wizard import MenuItem


class InteractiveWizard(BaseInteractiveWizard):
    """Interactive app with headless-NAS YouTube cookie authentication.

    The NAS is intentionally assumed to have no graphical browser. Authentication
    therefore accepts only a Netscape-format cookies.txt file exported elsewhere.
    """

    def _select_video_auth(self, source: str, *, challenge: bool = False) -> VideoAuth:
        if not is_url(source):
            return VideoAuth(mode="none")

        detected_by, detected_path = detect_cookies_file()
        items: list[MenuItem] = []
        if detected_path is not None:
            items.append(
                MenuItem(
                    "detected",
                    self._b("使用已检测到的 cookies.txt", "Use detected cookies.txt"),
                    self._b(
                        f"{detected_by}: {detected_path}",
                        f"{detected_by}: {detected_path}",
                    ),
                )
            )
        items.extend(
            [
                MenuItem(
                    "file",
                    self._b("选择 cookies.txt 文件", "Choose a cookies.txt file"),
                    self._b(
                        "从桌面浏览器导出的 YouTube Netscape/Mozilla 格式 Cookie。",
                        "YouTube cookies exported from a desktop browser in Netscape/Mozilla format.",
                    ),
                ),
                MenuItem(
                    "help",
                    self._b("查看如何导出 Cookie", "Show cookie export instructions"),
                    self._b(
                        "显示适用于无图形 NAS 的推荐步骤。",
                        "Show the recommended workflow for a headless NAS.",
                    ),
                ),
                MenuItem(
                    "none",
                    self._b("不使用 Cookie", "Continue without cookies"),
                    self._b(
                        "公开视频有时可以直接下载；遇到机器人校验时通常需要 Cookie。",
                        "Some public videos work anonymously; bot checks usually require cookies.",
                    ),
                ),
            ]
        )
        if detected_path is not None:
            default = "detected"
        elif challenge:
            default = "file"
        else:
            default = "none"

        while True:
            choice = self._menu(
                self._b("YouTube Cookie 认证", "YouTube cookie authentication"),
                items,
                default=default,
            )
            if choice == "detected" and detected_path is not None:
                return VideoAuth(mode="cookies", cookies_path=str(detected_path))
            if choice == "file":
                path = Path(
                    self._ask_text(
                        self._b(
                            "cookies.txt 在 NAS 上的路径",
                            "Path to cookies.txt on the NAS",
                        )
                    )
                ).expanduser().resolve()
                return VideoAuth(mode="cookies", cookies_path=str(path))
            if choice == "help":
                self._render_cookie_help()
                default = "file"
                continue
            return VideoAuth(mode="none")

    def _extract_video_with_retry(
        self,
        source: str,
        frame_dir: Path,
        *,
        interval_seconds: int,
        max_frames: int,
        proxy: VideoProxy,
    ):
        auth = self._select_video_auth(source)
        while True:
            if frame_dir.exists():
                shutil.rmtree(frame_dir)
            frame_dir.mkdir(parents=True, exist_ok=True)
            proxy_provenance = proxy.provenance() if is_url(source) else None
            auth_provenance = auth.provenance() if is_url(source) else None
            details = [
                self._b("[cyan]正在准备视频帧[/cyan]", "[cyan]Preparing video frames[/cyan]"),
                self._b(f"来源：{source}", f"Source: {source}"),
                self._b(
                    f"采样间隔：{interval_seconds} 秒",
                    f"Sampling interval: {interval_seconds}s",
                ),
                self._b(
                    f"最多保留帧数：{max_frames}",
                    f"Maximum accepted frames: {max_frames}",
                ),
            ]
            if proxy_provenance:
                endpoint = proxy_provenance.get("endpoint")
                network = str(proxy_provenance.get("mode"))
                if endpoint:
                    network += f" via {endpoint}"
                details.append(self._b(f"网络：{network}", f"Network: {network}"))
            if auth_provenance:
                if auth_provenance.get("configured"):
                    details.append(
                        self._b(
                            f"Cookie 认证：已启用（{auth_provenance.get('filename', 'cookies.txt')}）",
                            f"Cookie authentication: enabled ({auth_provenance.get('filename', 'cookies.txt')})",
                        )
                    )
                else:
                    details.append(self._b("Cookie 认证：未使用", "Cookie authentication: none"))
            self.console.print(Panel.fit("\n".join(details)))

            try:
                report = extract_video_frames(
                    source,
                    frame_dir,
                    interval_seconds=interval_seconds,
                    max_frames=max_frames,
                    proxy=proxy,
                    auth=auth,
                )
                return report, proxy
            except PipelineError as exc:
                if not is_url(source):
                    raise
                message = str(exc)
                challenge = self._looks_like_youtube_auth_challenge(message)
                if challenge:
                    self.console.print(
                        Panel.fit(
                            self._b(
                                "[yellow bold]YouTube 要求登录 Cookie[/yellow bold]\n"
                                "这通常是出口 IP 的机器人校验，不是抽帧或代理代码故障。\n"
                                "请选择从桌面浏览器导出的 cookies.txt 后重试。",
                                "[yellow bold]YouTube requires login cookies[/yellow bold]\n"
                                "This is usually an anti-bot check on the egress IP, not a frame-extraction failure.\n"
                                "Choose a cookies.txt exported from your desktop browser and retry.",
                            )
                        )
                    )
                self.console.print(
                    Panel.fit(
                        self._b(
                            f"[red]视频下载/导入失败[/red]\n{message}",
                            f"[red]Video download/import failed[/red]\n{message}",
                        )
                    )
                )
                action = self._menu(
                    self._b("下载失败处理", "Download recovery"),
                    [
                        MenuItem(
                            "auth",
                            self._b("选择或更换 cookies.txt", "Choose or change cookies.txt"),
                            self._b(
                                "用于 YouTube 登录/机器人校验。",
                                "Use for YouTube login or anti-bot verification.",
                            ),
                        ),
                        MenuItem(
                            "retry",
                            self._b("使用当前设置重试", "Retry with current settings"),
                        ),
                        MenuItem(
                            "proxy",
                            self._b("修改代理设置", "Change proxy settings"),
                        ),
                        MenuItem(
                            "help",
                            self._b("查看 Cookie 导出步骤", "Show cookie export instructions"),
                        ),
                        MenuItem(
                            "cancel",
                            self._b("取消视频导入", "Cancel video import"),
                        ),
                    ],
                    default="auth" if challenge else "retry",
                )
                if action == "cancel":
                    raise PipelineError(
                        self._b("已取消视频导入", "Video import cancelled")
                    ) from exc
                if action == "proxy":
                    proxy = self._select_video_proxy(source)
                elif action == "auth":
                    auth = self._select_video_auth(source, challenge=True)
                elif action == "help":
                    self._render_cookie_help()
                    auth = self._select_video_auth(source, challenge=True)

    def _render_cookie_help(self) -> None:
        if get_language() == "zh-CN":
            body = (
                "[bold]无图形 NAS 的推荐做法[/bold]\n\n"
                "1. 在你的桌面电脑上打开新的无痕/隐私窗口，并登录 YouTube。\n"
                "2. 在同一个窗口、同一个标签页访问 https://www.youtube.com/robots.txt。\n"
                "3. 用可信的 cookies.txt 导出工具，只导出 YouTube Cookie，格式必须是 Netscape/Mozilla。\n"
                "4. 导出后立即关闭整个无痕/隐私窗口，不要再用那个会话浏览 YouTube。\n"
                "5. 把文件安全复制到 NAS，例如：\n"
                "   ~/.config/lora-pipeline/youtube-cookies.txt\n"
                "6. 建议把文件权限设为仅当前 NAS 用户可读写（0600）。\n\n"
                "[yellow]Cookie 等同登录凭据：不要提交 Git、不要发给别人，也不要放进项目目录。[/yellow]\n"
                "YouTube 可能轮换 Cookie；失效时重新导出即可。"
            )
        else:
            body = (
                "[bold]Recommended workflow for a headless NAS[/bold]\n\n"
                "1. On your desktop, open a fresh private/incognito window and sign in to YouTube.\n"
                "2. In the same window and tab, navigate to https://www.youtube.com/robots.txt.\n"
                "3. Use a trusted cookies.txt exporter to export only YouTube cookies in Netscape/Mozilla format.\n"
                "4. Close the entire private/incognito window immediately; do not browse YouTube with that session again.\n"
                "5. Copy the file securely to the NAS, for example:\n"
                "   ~/.config/lora-pipeline/youtube-cookies.txt\n"
                "6. Restrict it to the NAS user (mode 0600 is recommended).\n\n"
                "[yellow]Cookies are login credentials: never commit them to Git, share them, or store them inside a project.[/yellow]\n"
                "YouTube may rotate cookies; export a fresh file when it stops working."
            )
        self.console.print(Panel.fit(body, title=self._b("Cookie 导出帮助", "Cookie export help")))

    @staticmethod
    def _looks_like_youtube_auth_challenge(message: str) -> bool:
        lowered = message.casefold()
        markers = (
            "sign in to confirm you're not a bot",
            "sign in to confirm you’re not a bot",
            "use --cookies",
            "authentication",
        )
        return any(marker in lowered for marker in markers)

    @staticmethod
    def _b(chinese: str, english: str) -> str:
        return chinese if get_language() == "zh-CN" else english
