from __future__ import annotations

from .dataset_workspace import DatasetWorkspace
from .interactive_semantic_concepts import InteractiveWizard as BaseInteractiveWizard
from .wizard import MenuItem


class InteractiveWizard(BaseInteractiveWizard):
    """Final Dataset UX aligned with the materialization compiler contract.

    Imported Dataset sources stay unchanged. Target-aware crop, downscale, caption
    policy and TriggerPolicy are applied exactly once when a TrainingConfig is
    materialized into an immutable prepared generation.
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
                            "保留原始图片/视频抽帧；训练目标相关裁剪只在 materialize 阶段执行一次。",
                            "Keep imported images/video frames unchanged; target-aware crop runs once during materialization.",
                        ),
                    ),
                    MenuItem(
                        "sources",
                        self._b("来源管理", "Manage sources"),
                        self._b(
                            "查看来源并启用/停用；RAW 来源不会被训练流程改写。",
                            "Inspect and enable/disable sources; training never rewrites RAW sources.",
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
                            "可在 Dataset 阶段准备/修正 caption；训练时 generate/hybrid 会对最终训练像素重新打标。",
                            "Prepare/correct captions in Dataset; generate/hybrid retags the exact final training pixels during materialization.",
                        ),
                    ),
                    MenuItem(
                        "edit_tags",
                        self._b("修正 Tag", "Edit tags"),
                        self._b(
                            "只在需要时人工修正 Tag。",
                            "Manually correct tags only when necessary.",
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
                            "选择 TrainingConfig；materialize 会冻结 crop、约 1MP 训练图、caption/trigger，并生成 preview.html 供检查。",
                            "Choose a TrainingConfig; materialize freezes crop, ~1MP training pixels and caption/trigger, then writes preview.html for review.",
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
