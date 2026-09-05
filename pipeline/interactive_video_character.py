from __future__ import annotations

from pathlib import Path

from rich.panel import Panel
from rich.table import Table

from .dataset.image_info import discover_images
from .interactive_sources import InteractiveWizard as BaseInteractiveWizard
from .models import OptionalBackendUnavailable, PipelineError
from .video_character import (
    VideoCompositionReport,
    VideoSubjectReport,
    build_balanced_character_dataset,
    detect_video_subjects,
)


class InteractiveWizard(BaseInteractiveWizard):
    """DeepGHS subject detection and 4K-aware smart cropping.

    Dataset inputs are assumed to have the intended identity. The normal path no
    longer performs CCIP clustering or identity selection: every usable detected
    subject is kept, then cropped from the original-resolution source and reduced
    only after cropping.
    """

    _video_interval_seconds: int = 2
    _training_max_pixels: int = 1_048_576
    _training_max_long_edge: int = 2048

    def _ask_positive_int(self, prompt: str, *, default: int) -> int:
        value = super()._ask_positive_int(prompt, default=default)
        if prompt == "Sample one frame every N seconds":
            self._video_interval_seconds = value
        return value

    def _select_video_identity(self, frame_dir: Path) -> tuple[Path, dict[str, object]]:
        """Compatibility name for subject-aware materialization without identity clustering."""

        self.console.print(
            Panel.fit(
                self._b(
                    "[cyan]自动检测并裁切主体[/cyan]\n"
                    "DeepGHS 在缩小代理图上检测人物/头部，再把 bbox 映射回原始图片裁切。\n"
                    "输入数据默认已保证人物身份正确，因此不会再运行 CCIP 聚类。",
                    "[cyan]Detecting and cropping subjects[/cyan]\n"
                    "DeepGHS detects people/heads on reduced proxies, then maps the boxes back to the original image.\n"
                    "Input identity is treated as trusted, so CCIP clustering is no longer run.",
                )
            )
        )
        subject_root = frame_dir.parent / "character-subjects"
        try:
            subject_report = detect_video_subjects(
                frame_dir,
                subject_root,
                interval_seconds=self._video_interval_seconds,
                maximum_saved_long_edge=self._training_max_long_edge,
                maximum_saved_pixels=self._training_max_pixels,
            )
        except (OptionalBackendUnavailable, PipelineError) as exc:
            self.console.print(
                Panel.fit(
                    self._b(
                        "[yellow]主体检测不可用，保留原始图片继续。[/yellow]\n"
                        f"{exc}\n"
                        "不会尝试 CCIP，也不会因为检测器不可用而丢弃输入。",
                        "[yellow]Subject detection is unavailable; keeping the original images.[/yellow]\n"
                        f"{exc}\n"
                        "CCIP is not attempted and detector failure does not discard the input.",
                    )
                )
            )
            return frame_dir, {
                "status": "subject_detection_unavailable_keep_originals",
                "reason": str(exc),
                "identity_assumed_valid": True,
                "selected_cluster": None,
                "selected_frames": len(discover_images(frame_dir)),
            }

        self._render_subject_detection(subject_report)
        selected_paths = [subject.identity_path for subject in subject_report.subjects]
        return self._build_training_from_subjects(
            subject_report,
            selected_paths,
            cluster_payload={
                "status": "identity_assumed_valid",
                "identity_assumed_valid": True,
                "selected_cluster": None,
                "selected_subjects": len(selected_paths),
            },
        )

    def _build_training_from_subjects(
        self,
        subject_report: VideoSubjectReport,
        selected_paths: list[Path] | tuple[Path, ...],
        *,
        cluster_payload: dict[str, object],
    ) -> tuple[Path, dict[str, object]]:
        output_dir = subject_report.identity_dir.parent.parent / "selected-character"
        composition = build_balanced_character_dataset(
            subject_report,
            selected_paths,
            output_dir,
            maximum_saved_long_edge=self._training_max_long_edge,
            maximum_saved_pixels=self._training_max_pixels,
        )
        self._render_composition_report(composition)
        payload = {
            **cluster_payload,
            "identity_unit": "trusted_input_deepghs_person_crop",
            "subject_detection": subject_report.as_dict(),
            "training_composition": composition.as_dict(),
            "selected_frames": len(discover_images(output_dir)),
        }
        return output_dir, payload

    def _render_subject_detection(self, report: VideoSubjectReport) -> None:
        table = Table(title=self._b("DeepGHS 主体检测", "DeepGHS subject detection"))
        table.add_column(self._b("指标", "Metric"), style="bold")
        table.add_column(self._b("数量", "Count"), justify="right")
        table.add_row(self._b("输入图片/帧", "Input images/frames"), str(report.total_frames))
        table.add_row(self._b("检测到主体", "Images with subjects"), str(report.frames_with_subjects))
        table.add_row(self._b("Person 检测框", "Person detections"), str(report.detected_persons))
        table.add_row(self._b("Head-only 回退", "Head-only fallbacks"), str(report.head_fallbacks))
        table.add_row(
            self._b("原生分辨率过低", "Rejected: low native resolution"),
            str(report.rejected_low_resolution),
        )
        table.add_row(self._b("可训练主体 crop", "Usable subject crops"), str(len(report.subjects)))
        self.console.print(table)
        self.console.print(
            Panel.fit(
                self._b(
                    f"检测代理图最长边：{report.detection_proxy_long_edge}px\n"
                    f"最低人物高度：{report.minimum_person_height}px · 最低头部尺寸：{report.minimum_head_size}px\n"
                    f"中间 crop 保存上限：约 {report.maximum_saved_pixels / 1_048_576:.1f}MP\n"
                    "检测在 proxy 上完成，真正 crop 来自原图；只缩小，不放大。",
                    f"Detection proxy long edge: {report.detection_proxy_long_edge}px\n"
                    f"Minimum person height: {report.minimum_person_height}px · minimum head size: {report.minimum_head_size}px\n"
                    f"Intermediate crop cap: about {report.maximum_saved_pixels / 1_048_576:.1f}MP\n"
                    "Detection uses a proxy; the real crop comes from source pixels and is downscale-only.",
                )
            )
        )

    def _render_composition_report(self, report: VideoCompositionReport) -> None:
        payload = report.as_dict()
        counts = payload["composition_counts"]
        table = Table(title=self._b("最终训练构图", "Final training composition"))
        table.add_column(self._b("构图", "Composition"), style="bold")
        table.add_column(self._b("数量", "Count"), justify="right")
        labels = {
            "portrait": self._b("头肩 / Portrait", "Portrait"),
            "upper_body": self._b("上半身", "Upper body"),
            "full_body": self._b("全身", "Full body"),
            "context": self._b("环境构图", "Context"),
        }
        for key in ("portrait", "upper_body", "full_body", "context"):
            table.add_row(labels[key], str(counts.get(key, 0)))
        table.add_row(
            self._b("最终训练图片", "Final training images"),
            str(payload["training_images"]),
        )
        table.add_row(
            self._b("近重复跳过", "Near-duplicates skipped"),
            str(payload["rejected_near_duplicate"]),
        )
        table.add_row(
            self._b("裁切后尺寸过小", "Too small after crop"),
            str(payload["rejected_too_small"]),
        )
        table.add_row(
            self._b("因像素上限缩小", "Downscaled by pixel cap"),
            str(payload["downscaled_images"]),
        )
        self.console.print(table)
        self.console.print(
            self._b(
                "[dim]每个检测主体只生成一个主要训练构图；最终 prepared generation 还会统一执行约 1MP 的 downscale-only 归一化。[/dim]",
                "[dim]Each detected subject produces one primary training view; the prepared generation then applies the common ~1MP downscale-only normalization.[/dim]",
            )
        )
