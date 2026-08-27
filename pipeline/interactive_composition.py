from __future__ import annotations

from pathlib import Path

from rich.table import Table

from .dataset.image_info import discover_images
from .dataset_metadata import composition_summary
from .dataset_workspace import DatasetWorkspace
from .interactive_deletion import InteractiveWizard as BaseInteractiveWizard
from .video_character import VideoSubjectReport
from .video_composition import EnrichedVideoCompositionReport, build_enriched_character_dataset


class InteractiveWizard(BaseInteractiveWizard):
    """CLI layer for enriched composition balancing and Dataset composition summaries."""

    def _build_training_from_subjects(
        self,
        subject_report: VideoSubjectReport,
        selected_paths: list[Path] | tuple[Path, ...],
        *,
        cluster_payload: dict[str, object],
    ) -> tuple[Path, dict[str, object]]:
        output_dir = subject_report.identity_dir.parent.parent / "selected-character"
        composition = build_enriched_character_dataset(
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

    def _render_composition_report(self, report: EnrichedVideoCompositionReport) -> None:
        payload = report.as_dict()
        counts = payload["composition_counts"]
        table = Table(title=self._b("最终训练构图平衡", "Final training composition balance"))
        table.add_column(self._b("构图", "Composition"), style="bold")
        table.add_column(self._b("数量", "Count"), justify="right")
        labels = {
            "portrait": self._b("头肩 / Portrait", "Portrait"),
            "upper_body": self._b("上半身", "Upper body"),
            "three_quarter": self._b("3/4 身", "Three-quarter body"),
            "full_body": self._b("全身", "Full body"),
            "context": self._b("环境构图", "Context"),
        }
        for key in ("portrait", "upper_body", "three_quarter", "full_body", "context"):
            table.add_row(labels[key], str(counts.get(key, 0)))
        table.add_row(self._b("最终训练图片", "Final training images"), str(payload["training_images"]))
        table.add_row(
            self._b("高价值原始全图", "High-value original full frames"),
            str(payload.get("full_variants_kept", 0)),
        )
        table.add_row(
            self._b("crop 级近重复剔除", "Rejected crop-level near-duplicates"),
            str(payload["rejected_near_duplicate"]),
        )
        table.add_row(
            self._b("构图后尺寸过小剔除", "Rejected: too small after composition"),
            str(payload["rejected_too_small"]),
        )
        self.console.print(table)
        self.console.print(
            self._b(
                "[dim]每个人物默认只保留一个主要构图；只有高价值单人物全图才允许第二个 original_full 变体。不会制造多分辨率副本。[/dim]",
                "[dim]Each subject gets one primary composition; only high-value single-character frames may add a second original_full variant. No artificial multi-resolution copies are created.[/dim]",
            )
        )

    def _render_dataset_dashboard(self, workspace: DatasetWorkspace) -> None:
        super()._render_dataset_dashboard(workspace)
        if workspace.concept_type != "character" or not workspace.sources:
            return
        summary = composition_summary(workspace)
        counts = summary["active_composition_counts"]
        table = Table(title=self._b("可训练图片构图分布", "Active composition distribution"))
        table.add_column(self._b("构图", "Composition"))
        table.add_column(self._b("数量", "Count"), justify="right")
        for key, label in (
            ("portrait", self._b("Portrait / 头肩", "Portrait")),
            ("upper_body", self._b("Upper body / 上半身", "Upper body")),
            ("three_quarter", self._b("3/4 body", "Three-quarter body")),
            ("full_body", self._b("Full body / 全身", "Full body")),
            ("context", self._b("Context / 环境", "Context")),
            ("unknown", self._b("未分析", "Unknown")),
        ):
            table.add_row(label, str(counts.get(key, 0)))
        table.add_row(
            self._b("已分析", "Analyzed"),
            f"{summary['analyzed']}/{summary['total']}",
        )
        self.console.print(table)
