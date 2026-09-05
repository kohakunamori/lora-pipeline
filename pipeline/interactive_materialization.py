from __future__ import annotations

import tempfile
from pathlib import Path

from .dataset_workspace import DatasetWorkspace
from .interactive_semantic_concepts import InteractiveWizard as BaseInteractiveWizard
from .models import PipelineError
from .wizard import MenuItem


class InteractiveWizard(BaseInteractiveWizard):
    """Narrow final Dataset UX to the materialization contract.

    Identity is trusted at ingestion. The normal path is therefore:
    import -> subject crop -> sanity check/tag/edit -> TrainingConfig -> prepared
    ~1MP generation. Legacy pHash/CCIP/advanced curation functions remain importable
    for old workspaces but are intentionally absent from this menu.
    """

    def dataset_dashboard(self, name: str) -> None:
        while True:
            workspace = DatasetWorkspace.load(name)
            self._render_dataset_dashboard(workspace)
            action = self._menu(
                self._b("数据集操作", "Dataset actions"),
                [
                    MenuItem(
                        "import",
                        self._b("导入素材", "Import material"),
                        self._b(
                            "人物图片会默认执行主体检测/智能裁剪；视频抽帧后走同一主体流程。",
                            "Character images automatically run subject detection/smart crop; video frames use the same subject path.",
                        ),
                    ),
                    MenuItem(
                        "sources",
                        self._b("来源与裁剪", "Sources and crops"),
                        self._b(
                            "查看来源、启停来源，必要时重新执行单来源智能裁剪。",
                            "Inspect/toggle sources and rerun smart crop for a source when needed.",
                        ),
                    ),
                    MenuItem(
                        "audit",
                        self._b("输入安全检查", "Input sanity check"),
                        self._b(
                            "检查损坏图片和完全重复项；只有确定安全的项目才建议自动排除。",
                            "Check corrupt images and exact duplicates; only deterministic-safe items are suggested for exclusion.",
                        ),
                    ),
                    MenuItem(
                        "tag",
                        self._b("自动打 Tag", "Auto-tag"),
                        self._b(
                            "使用缓存的 WD EVA02 Tagger；已有人工 .txt 默认不覆盖。",
                            "Use the cached WD EVA02 tagger; existing manual .txt captions are preserved by default.",
                        ),
                    ),
                    MenuItem(
                        "edit_tags",
                        self._b("修正 Tag", "Edit tags"),
                        self._b(
                            "只在需要时人工修正自动 Tag。",
                            "Manually correct auto-tags only when necessary.",
                        ),
                    ),
                    MenuItem(
                        "review",
                        self._b("人工排除 / 恢复", "Manual exclude / restore"),
                        self._b(
                            "可选人工兜底；不会物理删除来源文件。",
                            "Optional manual fallback; source files are not physically deleted.",
                        ),
                    ),
                    MenuItem(
                        "training",
                        self._b("开始训练", "Start training"),
                        self._b(
                            "选择 TrainingConfig；TriggerPolicy、caption 与约 1MP 图像归一化在 materialize 阶段冻结。",
                            "Choose a TrainingConfig; TriggerPolicy, captions, and ~1MP image normalization are frozen during materialization.",
                        ),
                    ),
                    MenuItem("back", self._b("返回", "Back")),
                ],
                default="import" if not workspace.sources else "tag",
            )
            if action == "back":
                return
            if action == "import":
                self._import_dataset_source(workspace)
            elif action == "sources":
                self._manage_dataset_sources(workspace.name)
            elif action == "audit":
                self._audit_dataset(workspace)
            elif action == "tag":
                self._auto_tag_dataset(workspace)
            elif action == "edit_tags":
                self._choose_and_edit_tag(workspace)
            elif action == "review":
                self._review_dataset_items(workspace)
            elif action == "training":
                self._start_training_from_dataset_config(prefilled_workspace=workspace)

    def _import_image_directory(self, workspace: DatasetWorkspace) -> None:
        while True:
            directory = Path(
                self._ask_text(self._b("图片目录路径", "Image directory path"))
            ).expanduser().resolve()
            if directory.is_dir():
                break
            self.console.print(
                self._b(
                    f"[red]目录不存在：{directory}[/red]",
                    f"[red]Directory does not exist: {directory}[/red]",
                )
            )
        label = self._ask_text(
            self._b("来源名称", "Source label"),
            default=directory.name or "images",
        ).strip()
        record = workspace.add_source_from_directory(
            directory,
            kind="image_directory",
            label=label,
            origin=str(directory),
        )
        self._render_source_imported(record)

        if workspace.concept_type != "character":
            return
        source_id = str(record["id"])
        try:
            self._smart_crop_source(workspace, source_id)
        except PipelineError as exc:
            self.console.print(
                self._b(
                    f"[yellow]自动主体裁剪未完成，已保留原始来源：{exc}[/yellow]",
                    f"[yellow]Automatic subject crop did not complete; original source retained: {exc}[/yellow]",
                )
            )

    def _smart_crop_source(self, workspace: DatasetWorkspace, source_id: str) -> None:
        """Create one derived subject-crop source with no extra confirmation prompts."""

        source = workspace.sources[source_id]
        work_root = workspace.dataset_dir / "cache" / "work"
        work_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f"crop-{source_id}-", dir=work_root) as temporary:
            frame_dir = Path(temporary) / "frames"
            count = workspace.export_source_active(source_id, frame_dir)
            if count == 0:
                raise PipelineError(
                    self._b(
                        "这个来源没有可裁切的未排除图片。",
                        "This source has no non-excluded images to crop.",
                    )
                )
            training_dir, materialization = self._select_video_identity(frame_dir)

            # Detector/model failure deliberately returns the exported originals.
            # Do not manufacture a duplicate smart_crop source in that case.
            if training_dir.resolve() == frame_dir.resolve():
                self.console.print(
                    self._b(
                        "[yellow]未生成主体裁切；原来源继续启用。[/yellow]",
                        "[yellow]No subject crop was generated; the original source remains enabled.[/yellow]",
                    )
                )
                return

            label = f"{source.get('label', source_id)}-crop"
            record = workspace.add_source_from_directory(
                training_dir,
                kind="smart_crop",
                label=label,
                origin=f"derived:{source_id}",
                parent_source=source_id,
                processing={
                    "subject_materialization": materialization,
                    "input_images": count,
                },
            )

        # A successful derived crop becomes the active representation. The original
        # remains on disk and can be re-enabled from the source manager at any time.
        workspace.set_source_enabled(source_id, False)
        self._render_source_imported(record)
        self.console.print(
            self._b(
                f"[green]已自动启用裁切来源 {record['id']}，并停用原来源 {source_id}。[/green]",
                f"[green]Enabled crop source {record['id']} and disabled original source {source_id} automatically.[/green]",
            )
        )
