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
from .video_identity import VideoIdentityReport, cluster_video_identity
from .wizard import MenuItem


class InteractiveWizard(BaseInteractiveWizard):
    """Video wizard with DeepGHS subject detection and 4K-aware smart cropping."""

    _video_interval_seconds: int = 2

    def _ask_positive_int(self, prompt: str, *, default: int) -> int:
        value = super()._ask_positive_int(prompt, default=default)
        if prompt == "Sample one frame every N seconds":
            self._video_interval_seconds = value
        return value

    def _select_video_identity(self, frame_dir: Path) -> tuple[Path, dict[str, object]]:
        self.console.print(
            Panel.fit(
                self._b(
                    "[cyan]检测视频中的人物[/cyan]\n"
                    "DeepGHS 会在缩小的代理图上检测人物/头部，再把 bbox 映射回原始帧裁切。\n"
                    "4K 源的细节会保留到裁切以后；不会为了凑训练分辨率进行放大。",
                    "[cyan]Detecting characters in video frames[/cyan]\n"
                    "DeepGHS detects people/heads on reduced proxies, then maps the boxes back to the original frames.\n"
                    "4K detail is preserved until after cropping; images are never upscaled just to reach training resolution.",
                )
            )
        )
        subject_root = frame_dir.parent / "character-subjects"
        try:
            subject_report = detect_video_subjects(
                frame_dir,
                subject_root,
                interval_seconds=self._video_interval_seconds,
            )
        except (OptionalBackendUnavailable, PipelineError) as exc:
            self.console.print(
                Panel.fit(
                    self._b(
                        "[yellow]智能人物裁切暂时不可用，将退回整帧 CCIP。[/yellow]\n"
                        f"{exc}\n"
                        "原始视频帧不会丢失，仍可继续创建项目。",
                        "[yellow]Smart character cropping is unavailable; falling back to whole-frame CCIP.[/yellow]\n"
                        f"{exc}\n"
                        "The filtered source frames are still intact and project creation can continue.",
                    )
                )
            )
            return super()._select_video_identity(frame_dir)

        self._render_subject_detection(subject_report)
        identity_dir = subject_report.identity_dir
        try:
            identity_report = cluster_video_identity(identity_dir)
        except OptionalBackendUnavailable as exc:
            self.console.print(
                Panel.fit(
                    self._b(
                        "[yellow]CCIP 不可用，无法自动按人物身份聚类。[/yellow]\n"
                        f"{exc}",
                        "[yellow]CCIP is unavailable, so automatic identity clustering cannot run.[/yellow]\n"
                        f"{exc}",
                    )
                )
            )
            if not self._confirm(
                self._b(
                    "保留全部已检测人物候选，并交给后续 Identity/Review 继续检查吗？",
                    "Keep all detected character candidates and rely on the later Identity/Review stages?",
                ),
                default=True,
            ):
                raise PipelineError(self._b("已取消视频导入", "Video import cancelled"))
            return self._build_training_from_subjects(
                subject_report,
                [subject.identity_path for subject in subject_report.subjects],
                cluster_payload={
                    "status": "ccip_unavailable_keep_all",
                    "reason": str(exc),
                    "selected_cluster": None,
                },
            )
        except PipelineError as exc:
            self.console.print(
                Panel.fit(
                    self._b(
                        "[yellow]CCIP 无法形成稳定人物簇。[/yellow]\n" f"{exc}",
                        "[yellow]CCIP could not form stable character clusters.[/yellow]\n" f"{exc}",
                    )
                )
            )
            if not self._confirm(
                self._b(
                    "保留全部已检测人物候选，并交给后续 Identity/Review 继续检查吗？",
                    "Keep all detected character candidates and rely on the later Identity/Review stages?",
                ),
                default=True,
            ):
                raise
            return self._build_training_from_subjects(
                subject_report,
                [subject.identity_path for subject in subject_report.subjects],
                cluster_payload={
                    "status": "ccip_unstable_keep_all",
                    "reason": str(exc),
                    "selected_cluster": None,
                },
            )

        self._render_subject_clusters(identity_report, identity_dir)
        items = [
            MenuItem(
                f"cluster:{cluster.cluster_id}",
                self._b(
                    f"使用人物簇 {cluster.cluster_id}（{cluster.size} 个候选）",
                    f"Use character cluster {cluster.cluster_id} ({cluster.size} candidates)",
                ),
                self._b(
                    "代表候选：" + ", ".join(path.name for path in cluster.representatives),
                    "Representatives: " + ", ".join(path.name for path in cluster.representatives),
                ),
            )
            for cluster in identity_report.clusters
        ]
        items.extend(
            [
                MenuItem(
                    "all",
                    self._b("保留全部已检测人物候选", "Keep all detected character candidates"),
                    self._b(
                        "可能混入其他人物；后续 Identity/Review 仍会继续检查。",
                        "May include other characters; later Identity/Review stages will still check the dataset.",
                    ),
                ),
                MenuItem("cancel", self._b("取消视频导入", "Cancel video import")),
            ]
        )
        default = f"cluster:{identity_report.clusters[0].cluster_id}"
        base_cluster_payload = identity_report.as_dict(root=identity_dir)
        while True:
            choice = self._menu(self._b("目标人物", "Target character"), items, default=default)
            if choice == "cancel":
                raise PipelineError(self._b("已取消视频导入", "Video import cancelled"))

            cluster_payload = dict(base_cluster_payload)
            if choice == "all":
                if not self._confirm(
                    self._b(
                        "确定保留所有人物簇和 CCIP 离群候选吗？这可能把其他角色带进训练集。",
                        "Keep every cluster and CCIP outlier? This may introduce other characters into training.",
                    ),
                    default=False,
                ):
                    continue
                selected_paths = [subject.identity_path for subject in subject_report.subjects]
                cluster_payload.update(
                    {
                        "status": "kept_all_detected_subjects",
                        "selected_cluster": None,
                        "selected_subjects": len(selected_paths),
                    }
                )
                return self._build_training_from_subjects(
                    subject_report,
                    selected_paths,
                    cluster_payload=cluster_payload,
                )

            cluster_id = int(choice.split(":", 1)[1])
            selected = next(
                cluster for cluster in identity_report.clusters if cluster.cluster_id == cluster_id
            )
            cluster_payload.update(
                {
                    "status": "selected_crop_cluster",
                    "selected_cluster": cluster_id,
                    "selected_subjects": selected.size,
                    "discarded_other_clusters": sum(
                        cluster.size
                        for cluster in identity_report.clusters
                        if cluster.cluster_id != cluster_id
                    ),
                    "discarded_outliers": len(identity_report.outliers),
                }
            )
            return self._build_training_from_subjects(
                subject_report,
                selected.frames,
                cluster_payload=cluster_payload,
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
        )
        self._render_composition_report(composition)
        payload = {
            **cluster_payload,
            "identity_unit": "deepghs_person_crop",
            "subject_detection": subject_report.as_dict(),
            "training_composition": composition.as_dict(),
            "selected_frames": len(discover_images(output_dir)),
        }
        return output_dir, payload

    def _render_subject_detection(self, report: VideoSubjectReport) -> None:
        table = Table(title=self._b("DeepGHS 视频人物检测", "DeepGHS video character detection"))
        table.add_column(self._b("指标", "Metric"), style="bold")
        table.add_column(self._b("数量", "Count"), justify="right")
        table.add_row(self._b("筛选后源帧", "Filtered source frames"), str(report.total_frames))
        table.add_row(self._b("检测到人物的帧", "Frames with characters"), str(report.frames_with_subjects))
        table.add_row(self._b("Person 检测框", "Person detections"), str(report.detected_persons))
        table.add_row(self._b("Head-only 回退", "Head-only fallbacks"), str(report.head_fallbacks))
        table.add_row(
            self._b("低原生分辨率剔除", "Rejected: low native resolution"),
            str(report.rejected_low_resolution),
        )
        table.add_row(self._b("可用于 CCIP 的人物 crop", "Usable CCIP subject crops"), str(len(report.subjects)))
        self.console.print(table)
        self.console.print(
            Panel.fit(
                self._b(
                    f"检测代理图最长边：{report.detection_proxy_long_edge}px\n"
                    f"最低人物高度：{report.minimum_person_height}px · 最低头部尺寸：{report.minimum_head_size}px\n"
                    f"保存上限：最长边 {report.maximum_saved_long_edge}px · 最大约 {report.maximum_saved_pixels / 1_048_576:.1f}MP\n"
                    "只会缩小过大的 crop，不会放大小图。",
                    f"Detection proxy long edge: {report.detection_proxy_long_edge}px\n"
                    f"Minimum person height: {report.minimum_person_height}px · minimum head size: {report.minimum_head_size}px\n"
                    f"Save cap: {report.maximum_saved_long_edge}px long edge · about {report.maximum_saved_pixels / 1_048_576:.1f}MP maximum\n"
                    "Oversized crops may be downscaled; small crops are never upscaled.",
                )
            )
        )

    def _render_subject_clusters(self, report: VideoIdentityReport, root: Path) -> None:
        table = Table(title=self._b("人物 crop 身份聚类", "Character-crop identity clusters"))
        table.add_column(self._b("簇", "Cluster"), style="bold")
        table.add_column(self._b("候选数", "Candidates"), justify="right")
        table.add_column(self._b("代表人物 crop", "Representative character crops"))
        for cluster in report.clusters:
            representatives = ", ".join(
                path.relative_to(root).as_posix() for path in cluster.representatives
            )
            table.add_row(str(cluster.cluster_id), str(cluster.size), representatives)
        table.add_row(
            self._b("CCIP 离群候选", "CCIP outliers"),
            str(len(report.outliers)),
            self._b("选择目标簇时排除", "excluded when a target cluster is selected"),
        )
        self.console.print(table)

    def _render_composition_report(self, report: VideoCompositionReport) -> None:
        payload = report.as_dict()
        counts = payload["composition_counts"]
        table = Table(title=self._b("最终训练构图平衡", "Final training composition balance"))
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
            self._b("crop 级近重复剔除", "Rejected crop-level near-duplicates"),
            str(payload["rejected_near_duplicate"]),
        )
        table.add_row(
            self._b("构图后尺寸过小剔除", "Rejected: too small after composition"),
            str(payload["rejected_too_small"]),
        )
        table.add_row(
            self._b("因尺寸上限缩小", "Downscaled by save cap"),
            str(payload["downscaled_images"]),
        )
        self.console.print(table)
        self.console.print(
            self._b(
                "[dim]不会把同一张人物 crop 人工复制成多个分辨率版本；不同长宽比和人物占比交给真实构图 + SDXL bucket 提供。[/dim]",
                "[dim]The same crop is not duplicated at artificial resolutions; real composition/aspect-ratio diversity plus SDXL buckets provide resolution generalization.[/dim]",
            )
        )
