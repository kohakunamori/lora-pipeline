from __future__ import annotations

from rich.panel import Panel

from .interactive_video_character import InteractiveWizard as BaseInteractiveWizard


class InteractiveWizard(BaseInteractiveWizard):
    """Surface automatic HDR-to-SDR normalization without adding another user setting."""

    def _render_video_report(self, report: dict[str, object]) -> None:
        super()._render_video_report(report)
        color = report.get("video_color")
        if not isinstance(color, dict) or not color.get("hdr"):
            return

        tone_mapping = color.get("tone_mapping")
        tone_mapping = tone_mapping if isinstance(tone_mapping, dict) else {}
        transfer = str(color.get("color_transfer") or color.get("hdr_kind") or "HDR")
        primaries = str(color.get("color_primaries") or "unknown")
        algorithm = str(tone_mapping.get("algorithm") or "mobius")
        self.console.print(
            Panel.fit(
                self._b(
                    "[green bold]已检测到 HDR 视频，并自动转换为训练用 SDR[/green bold]\n"
                    f"源：{primaries} / {transfer}\n"
                    f"转换：{algorithm} tone mapping → BT.709 SDR（100 nits）\n"
                    "高光自动去饱和已关闭，以尽量保持动画人物原本的颜色。\n"
                    "原视频不会被修改；只有临时抽帧经过转换。",
                    "[green bold]HDR video detected and normalized to training SDR[/green bold]\n"
                    f"Source: {primaries} / {transfer}\n"
                    f"Conversion: {algorithm} tone mapping → BT.709 SDR (100 nits)\n"
                    "Automatic highlight desaturation is disabled to preserve character colors.\n"
                    "The source video is unchanged; only temporary extracted frames are converted.",
                ),
                border_style="green",
            )
        )
