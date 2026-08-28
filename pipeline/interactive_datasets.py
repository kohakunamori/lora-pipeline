from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Sequence

from rich.panel import Panel
from rich.table import Table

from .config import repository_root
from .dataset_workspace import (
    DatasetWorkspace,
    create_project_from_dataset,
    list_datasets,
    parse_number_selection,
)
from .interactive_video_hdr import InteractiveWizard as BaseInteractiveWizard
from .models import PipelineError, StateError
from .service import load_project
from .video_source import is_url
from .wizard import MenuItem, STRATEGIES


class InteractiveWizard(BaseInteractiveWizard):
    """Dataset-first interactive UI.

    Dataset workspaces are mutable curation assets. Projects remain immutable
    training snapshots created from the current enabled/non-excluded dataset view.
    """

    # ------------------------------------------------------------------
    # Home / project creation
    # ------------------------------------------------------------------
    def home(self) -> None:
        while True:
            projects = self.list_projects()
            datasets = list_datasets()
            self._render_home(projects)
            self._render_dataset_home_summary(datasets)
            items = [
                MenuItem(
                    "datasets",
                    self._b("管理数据集", "Manage datasets"),
                    self._b(
                        "导入多个来源、逐来源裁切、自动/人工清洗、Tag 管理。",
                        "Import multiple sources, crop per source, curate, and manage tags.",
                    ),
                ),
                MenuItem(
                    "new",
                    self._b("从数据集创建训练项目", "Create training project from dataset"),
                    self._b(
                        "把当前数据集状态冻结为不可变 project/raw 快照。",
                        "Freeze the current dataset state into an immutable project/raw snapshot.",
                    ),
                ),
            ]
            if projects:
                items.append(
                    MenuItem(
                        "open",
                        self._b("打开训练项目", "Open a training project"),
                        self._b("继续训练、评测或恢复已有项目。", "Resume training, evaluation, or recovery."),
                    )
                )
            items.extend(
                [
                    MenuItem("bases", self._b("管理底模", "Manage base models")),
                    MenuItem("doctor", self._b("检查当前机器", "Check this machine")),
                    MenuItem("quit", self._b("退出", "Exit")),
                ]
            )
            default = "datasets" if not datasets else ("open" if projects else "new")
            action = self._menu(self._b("主页", "Home"), items, default=default)
            if action == "quit":
                self.console.print(self._b("[dim]已退出。[/dim]", "[dim]Goodbye.[/dim]"))
                return
            actions = {
                "datasets": self.dataset_manager,
                "new": self.new_project,
                "bases": self.base_manager,
                "doctor": self.doctor,
            }
            if projects:
                actions["open"] = lambda: self._choose_and_open_project(projects)
            self._guarded(actions[action])

    def new_project(self):
        return self._create_project_from_dataset_interactive()

    # ------------------------------------------------------------------
    # Dataset manager
    # ------------------------------------------------------------------
    def dataset_manager(self) -> None:
        while True:
            workspaces = list_datasets()
            self._render_dataset_list(workspaces)
            items: list[MenuItem] = []
            if workspaces:
                items.append(
                    MenuItem(
                        "open",
                        self._b("打开数据集", "Open a dataset"),
                        self._b("管理来源、图片、Tag 和排除项。", "Manage sources, images, tags, and exclusions."),
                    )
                )
            items.extend(
                [
                    MenuItem("create", self._b("创建数据集", "Create a dataset")),
                    MenuItem("back", self._b("返回", "Back")),
                ]
            )
            action = self._menu(
                self._b("数据集工作区", "Dataset workspace"),
                items,
                default="open" if workspaces else "create",
            )
            if action == "back":
                return
            if action == "create":
                workspace = self._create_dataset()
                if workspace is not None:
                    self.dataset_dashboard(workspace.name)
                continue
            selected = self._select_dataset(workspaces)
            if selected is not None:
                self.dataset_dashboard(selected.name)

    def dataset_dashboard(self, name: str) -> None:
        while True:
            workspace = DatasetWorkspace.load(name)
            self._render_dataset_dashboard(workspace)
            action = self._menu(
                self._b("数据集操作", "Dataset actions"),
                [
                    MenuItem(
                        "import",
                        self._b("导入新的数据来源", "Import a new data source"),
                        self._b("图片目录、本地视频或在线视频会作为独立来源保存。", "Image folders and videos are stored as separate sources."),
                    ),
                    MenuItem(
                        "sources",
                        self._b("按来源管理", "Manage by source"),
                        self._b("启用/停用、单来源裁切、清洗、Tag、人工排除。", "Enable/disable, crop, curate, tag, and review one source."),
                    ),
                    MenuItem(
                        "audit",
                        self._b("全数据集自动检查", "Audit the whole dataset"),
                        self._b("安全排除损坏文件和完全重复副本；其余只标记。", "Safely exclude corrupt/exact duplicates; flag the rest for review."),
                    ),
                    MenuItem(
                        "curate",
                        self._b("高级清洗分析", "Advanced curation analysis"),
                        self._b(
                            "运行 pHash 近重复分析；人物数据集同时运行 CCIP 身份检查，并记录 freshness。",
                            "Run pHash near-duplicate analysis; character datasets also run CCIP identity analysis with freshness tracking.",
                        ),
                    ),
                    MenuItem(
                        "tag",
                        self._b("自动打 Tag", "Auto-tag images"),
                        self._b("使用现有 DeepGHS/WD Tagger，已有人工 Tag 默认不覆盖。", "Use the existing WD tagger; manual captions are preserved by default."),
                    ),
                    MenuItem(
                        "review",
                        self._b("人工图片审核 / 排除", "Manual image review / exclusions"),
                        self._b("按编号或范围快速排除，可随时恢复。", "Exclude by number/range and restore at any time."),
                    ),
                    MenuItem(
                        "edit_tags",
                        self._b("人工修改 Tag", "Edit tags manually"),
                        self._b("逐图片替换、追加或删除 Tag。", "Replace, add, or remove tags for individual images."),
                    ),
                    MenuItem(
                        "project",
                        self._b("用这个数据集创建训练项目", "Create training project from this dataset"),
                        self._b("冻结当前启用且未排除的图片。", "Freeze enabled, non-excluded images into a project snapshot."),
                    ),
                    MenuItem("back", self._b("返回数据集列表", "Back to dataset list")),
                ],
                default="import" if not workspace.sources else "sources",
            )
            if action == "back":
                return
            if action == "import":
                self._import_dataset_source(workspace)
            elif action == "sources":
                self._manage_dataset_sources(workspace.name)
            elif action == "audit":
                self._audit_dataset(workspace)
            elif action == "curate":
                self._curate_dataset(workspace)
            elif action == "tag":
                self._auto_tag_dataset(workspace)
            elif action == "review":
                self._review_dataset_items(workspace)
            elif action == "edit_tags":
                self._choose_and_edit_tag(workspace)
            elif action == "project":
                self._create_project_from_dataset_interactive(workspace=workspace)

    # ------------------------------------------------------------------
    # Dataset creation / import
    # ------------------------------------------------------------------
    def _create_dataset(self) -> DatasetWorkspace | None:
        self.console.print(
            Panel.fit(
                self._b(
                    "[bold blue]创建数据集[/bold blue]\n数据集是可持续维护的数据资产；训练项目只是它的不可变快照。",
                    "[bold blue]Create dataset[/bold blue]\nA dataset is a mutable curated asset; a training project is an immutable snapshot.",
                )
            )
        )
        while True:
            name = self._ask_text(self._b("数据集名称", "Dataset name")).strip()
            try:
                concept = self._menu(
                    self._b("数据集类型", "Dataset type"),
                    [
                        MenuItem("character", self._b("人物", "Character")),
                        MenuItem("style", self._b("风格", "Style")),
                    ],
                    default="character",
                )
                workspace = DatasetWorkspace.create(name, concept_type=concept)
                break
            except (StateError, PipelineError) as exc:
                self.console.print(f"[red]{exc}[/red]")
                if not self._confirm(self._b("重新输入吗？", "Try again?"), default=True):
                    return None
        self.console.print(
            Panel.fit(
                self._b(
                    f"[green bold]数据集已创建[/green bold]\n{workspace.dataset_dir}",
                    f"[green bold]Dataset created[/green bold]\n{workspace.dataset_dir}",
                )
            )
        )
        return workspace

    def _import_dataset_source(self, workspace: DatasetWorkspace) -> None:
        items = [
            MenuItem(
                "images",
                self._b("图片目录", "Image directory"),
                self._b("复制图片和同名 .txt Tag 到一个独立来源。", "Copy images and same-stem .txt tags into one source."),
            )
        ]
        if workspace.concept_type == "character":
            items.extend(
                [
                    MenuItem(
                        "local_video",
                        self._b("本地视频文件", "Local video file"),
                        self._b("抽帧、HDR 转换、人物裁切和 CCIP 后作为一个来源导入。", "Extract, tone-map HDR, crop characters, and CCIP-select before import."),
                    ),
                    MenuItem(
                        "remote_video",
                        self._b("在线视频 / YouTube", "Online video / YouTube"),
                        self._b("下载后走同一套视频人物处理。", "Download, then use the same character-video processing."),
                    ),
                ]
            )
        items.append(MenuItem("back", self._b("返回", "Back")))
        kind = self._menu(self._b("数据来源类型", "Data source type"), items, default="images")
        if kind == "back":
            return
        if kind == "images":
            self._import_image_directory(workspace)
        else:
            self._import_video_source(workspace, remote=kind == "remote_video")

    def _import_image_directory(self, workspace: DatasetWorkspace) -> None:
        while True:
            directory = Path(
                self._ask_text(self._b("图片目录路径", "Image directory path"))
            ).expanduser().resolve()
            if directory.is_dir():
                break
            self.console.print(self._b(f"[red]目录不存在：{directory}[/red]", f"[red]Directory does not exist: {directory}[/red]"))
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
        if workspace.concept_type == "character" and self._confirm(
            self._b(
                "现在对这个来源做 DeepGHS 智能人物裁切吗？原来源会保留。",
                "Run DeepGHS smart character cropping on this source now? The original source is retained.",
            ),
            default=False,
        ):
            self._smart_crop_source(workspace, str(record["id"]))

    def _import_video_source(self, workspace: DatasetWorkspace, *, remote: bool) -> None:
        source = self._ask_remote_video_url() if remote else self._ask_local_video_file()
        interval_seconds = self._ask_positive_int("Sample one frame every N seconds", default=2)
        self._video_interval_seconds = interval_seconds
        max_frames = self._ask_positive_int("Maximum accepted frames before identity selection", default=250)
        proxy = self._select_video_proxy(source)
        work_root = workspace.dataset_dir / "cache" / "work"
        work_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="video-source-", dir=work_root) as temporary:
            frame_dir = Path(temporary) / "frames"
            report, _proxy = self._extract_video_with_retry(
                source,
                frame_dir,
                interval_seconds=interval_seconds,
                max_frames=max_frames,
                proxy=proxy,
            )
            self._render_video_report(report.as_dict())
            training_dir, identity = self._select_video_identity(frame_dir)
            processing = report.as_dict()
            processing.pop("downloaded_video", None)
            processing["identity_preselection"] = identity
            processing["source_kind"] = "remote_url" if remote else "local_video"
            default_label = "online-video" if remote else Path(source).stem
            label = self._ask_text(
                self._b("来源名称", "Source label"),
                default=default_label or "video",
            ).strip()
            record = workspace.add_source_from_directory(
                training_dir,
                kind="remote_video" if remote else "local_video",
                label=label,
                origin=source,
                processing=processing,
            )
        self._render_source_imported(record)

    # ------------------------------------------------------------------
    # Source-specific management
    # ------------------------------------------------------------------
    def _manage_dataset_sources(self, dataset_name: str) -> None:
        while True:
            workspace = DatasetWorkspace.load(dataset_name)
            if not workspace.sources:
                self.console.print(self._b("[yellow]这个数据集还没有来源。[/yellow]", "[yellow]This dataset has no sources yet.[/yellow]"))
                return
            self._render_sources(workspace)
            choices = [
                MenuItem(
                    source_id,
                    str(source.get("label") or source_id),
                    f"{source.get('kind')} · {'enabled' if source.get('enabled', True) else 'disabled'}",
                )
                for source_id, source in sorted(workspace.sources.items())
            ] + [MenuItem("back", self._b("返回", "Back"))]
            source_id = self._menu(self._b("选择来源", "Select source"), choices, default=choices[0].value)
            if source_id == "back":
                return
            self._source_dashboard(workspace, source_id)

    def _source_dashboard(self, workspace: DatasetWorkspace, source_id: str) -> None:
        while True:
            workspace = DatasetWorkspace.load(workspace.name)
            source = workspace.sources[source_id]
            items = workspace.items(source_id=source_id, include_disabled=True, include_excluded=True)
            active = sum(not item.excluded for item in items)
            self.console.print(
                Panel.fit(
                    self._b(
                        f"[bold]{source.get('label', source_id)}[/bold]\n"
                        f"ID：{source_id}\n类型：{source.get('kind')}\n"
                        f"状态：{'启用' if source.get('enabled', True) else '停用'}\n"
                        f"图片：{len(items)} · 未排除：{active}",
                        f"[bold]{source.get('label', source_id)}[/bold]\n"
                        f"ID: {source_id}\nKind: {source.get('kind')}\n"
                        f"State: {'enabled' if source.get('enabled', True) else 'disabled'}\n"
                        f"Images: {len(items)} · not excluded: {active}",
                    )
                )
            )
            actions = [
                MenuItem("review", self._b("审核 / 排除这个来源的图片", "Review / exclude images in this source")),
                MenuItem("tag", self._b("自动打 Tag（仅这个来源）", "Auto-tag this source")),
                MenuItem("edit", self._b("人工修改 Tag（仅这个来源）", "Edit tags in this source")),
                MenuItem("audit", self._b("自动检查这个来源", "Audit this source")),
            ]
            if workspace.concept_type == "character" and source.get("kind") != "smart_crop":
                actions.append(
                    MenuItem(
                        "crop",
                        self._b("从这个来源生成智能人物裁切来源", "Create smart character-crop source from this source"),
                        self._b("不覆盖原图，生成新的派生来源。", "Create a derived source without overwriting originals."),
                    )
                )
            actions.extend(
                [
                    MenuItem(
                        "toggle",
                        self._b(
                            "停用这个来源" if source.get("enabled", True) else "启用这个来源",
                            "Disable this source" if source.get("enabled", True) else "Enable this source",
                        ),
                        self._b("停用后不会进入新项目快照。", "Disabled sources are omitted from new project snapshots."),
                    ),
                    MenuItem("back", self._b("返回来源列表", "Back to source list")),
                ]
            )
            action = self._menu(self._b("来源操作", "Source actions"), actions, default="review")
            if action == "back":
                return
            if action == "toggle":
                workspace.set_source_enabled(source_id, not bool(source.get("enabled", True)))
            elif action == "review":
                self._review_dataset_items(workspace, source_id=source_id)
            elif action == "tag":
                self._auto_tag_dataset(workspace, source_id=source_id)
            elif action == "edit":
                self._choose_and_edit_tag(workspace, source_id=source_id)
            elif action == "audit":
                self._audit_dataset(workspace, source_id=source_id)
            elif action == "crop":
                self._smart_crop_source(workspace, source_id)

    def _smart_crop_source(self, workspace: DatasetWorkspace, source_id: str) -> None:
        source = workspace.sources[source_id]
        work_root = workspace.dataset_dir / "cache" / "work"
        work_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f"crop-{source_id}-", dir=work_root) as temporary:
            frame_dir = Path(temporary) / "frames"
            count = workspace.export_source_active(source_id, frame_dir)
            if count == 0:
                raise PipelineError(self._b("这个来源没有可裁切的未排除图片。", "This source has no non-excluded images to crop."))
            training_dir, identity = self._select_video_identity(frame_dir)
            label = self._ask_text(
                self._b("新裁切来源名称", "New crop-source label"),
                default=f"{source.get('label', source_id)}-crop",
            ).strip()
            record = workspace.add_source_from_directory(
                training_dir,
                kind="smart_crop",
                label=label,
                origin=f"derived:{source_id}",
                parent_source=source_id,
                processing={"identity_preselection": identity, "input_images": count},
            )
        self._render_source_imported(record)
        if self._confirm(
            self._b(
                "停用原来源，只让新的裁切来源进入训练快照吗？",
                "Disable the original source so only the new crop source enters training snapshots?",
            ),
            default=True,
        ):
            workspace.set_source_enabled(source_id, False)

    # ------------------------------------------------------------------
    # Audit / curation / tags
    # ------------------------------------------------------------------
    def _audit_dataset(self, workspace: DatasetWorkspace, *, source_id: str | None = None) -> None:
        audit = workspace.audit(source_id=source_id)
        summary = audit["summary"]
        table = Table(title=self._b("数据检查结果", "Dataset audit"), show_header=False)
        table.add_column(self._b("项目", "Item"), style="bold")
        table.add_column(self._b("数量", "Count"), justify="right")
        table.add_row(self._b("检查图片", "Images checked"), str(summary["images"]))
        table.add_row(self._b("已排除", "Already excluded"), str(summary["already_excluded"]))
        table.add_row(self._b("有标记", "Flagged"), str(summary["flagged"]))
        table.add_row(self._b("可安全自动排除", "Safe auto-exclude"), str(summary["safe_exclude_suggestions"]))
        table.add_row(self._b("完全重复副本", "Exact duplicate copies"), str(summary["exact_duplicate_images"]))
        table.add_row(self._b("损坏图片", "Corrupt images"), str(summary["corrupt_images"]))
        table.add_row(self._b("需要人工判断", "Needs human review"), str(summary["review_only"]))
        self.console.print(table)
        if summary["safe_exclude_suggestions"] and self._confirm(
            self._b(
                "自动排除损坏图片和多余的完全重复副本吗？可在审核界面恢复。",
                "Auto-exclude corrupt images and redundant exact duplicates? They can be restored later.",
            ),
            default=True,
        ):
            result = workspace.apply_safe_audit_exclusions(source_id=source_id)
            self.console.print(
                self._b(
                    f"[green]已自动排除 {result['excluded']} 张。[/green]",
                    f"[green]Automatically excluded {result['excluded']} image(s).[/green]",
                )
            )
        if summary["review_only"]:
            self.console.print(
                self._b(
                    "[yellow]低分辨率、极端长宽比、动画图片不会自动删除；请进入人工审核决定。[/yellow]",
                    "[yellow]Small/extreme-aspect/animated images are only flagged; use manual review to decide.[/yellow]",
                )
            )

    def _curate_dataset(self, workspace: DatasetWorkspace) -> None:
        duplicates = workspace.analyze_duplicates()
        identity = None
        if workspace.concept_type == "character":
            identity = workspace.analyze_identity()
        status = workspace.curation_status()

        table = Table(
            title=self._b("高级清洗分析", "Advanced curation analysis"),
            show_header=False,
        )
        table.add_column(self._b("项目", "Item"), style="bold")
        table.add_column(self._b("结果", "Result"))
        duplicate_summary = duplicates["summary"]
        table.add_row(
            self._b("完全重复组", "Exact duplicate groups"),
            str(duplicate_summary["exact_groups"]),
        )
        table.add_row(
            self._b("近重复组", "Perceptual duplicate groups"),
            str(duplicate_summary["near_groups"]),
        )
        if identity is not None:
            identity_summary = identity["summary"]
            table.add_row(
                self._b("CCIP 离群图片", "CCIP outliers"),
                str(identity_summary["possible_outliers"]),
            )
            table.add_row(
                self._b("可能混入其他人物", "Possible mixed characters"),
                str(identity_summary["possible_mixed_characters"]),
            )
        table.add_row(
            "Freshness",
            "[green]READY[/green]" if status["ready"] else "[yellow]INCOMPLETE[/yellow]",
        )
        self.console.print(table)
        self.console.print(
            self._b(
                "[dim]分析结果绑定当前 active image set。之后新增、删除、排除、恢复、启停来源或修改图片内容都会自动使对应结果 stale；修改 Tag/caption 不会。[/dim]",
                "[dim]Results are bound to the current active image set. Adding/removing/excluding/restoring images, toggling sources, or changing image bytes makes them stale; tag/caption edits do not.[/dim]",
            )
        )
        if duplicate_summary["exact_groups"]:
            self.console.print(
                self._b(
                    "[yellow]仍存在完全重复图片。建议先运行 Dataset audit 的安全自动排除，再重新执行高级清洗分析。[/yellow]",
                    "[yellow]Exact duplicates remain. Run Dataset audit safe exclusions, then rerun advanced curation analysis.[/yellow]",
                )
            )

    def _auto_tag_dataset(self, workspace: DatasetWorkspace, *, source_id: str | None = None) -> None:
        threshold = self._ask_positive_float(self._b("Tag 阈值", "Tag threshold"), default=0.35)
        if threshold > 1.0:
            raise PipelineError(self._b("Tag 阈值必须不大于 1。", "Tag threshold must be <= 1."))
        overwrite = self._confirm(
            self._b(
                "覆盖已经存在的 .txt Tag 吗？默认否，以保护人工修改。",
                "Overwrite existing .txt tags? Default is no to protect manual edits.",
            ),
            default=False,
        )
        result = workspace.auto_tag(
            source_id=source_id,
            threshold=threshold,
            overwrite=overwrite,
        )
        self.console.print(
            Panel.fit(
                self._b(
                    f"[green bold]自动打 Tag 完成[/green bold]\n"
                    f"新写入：{result['tagged']}\n保留已有：{result['skipped_existing']}",
                    f"[green bold]Auto-tagging complete[/green bold]\n"
                    f"Written: {result['tagged']}\nExisting preserved: {result['skipped_existing']}",
                )
            )
        )

    def _review_dataset_items(self, workspace: DatasetWorkspace, *, source_id: str | None = None) -> None:
        page = 0
        page_size = 25
        while True:
            workspace = DatasetWorkspace.load(workspace.name)
            items = workspace.items(
                source_id=source_id,
                include_disabled=True,
                include_excluded=True,
            )
            if not items:
                self.console.print(self._b("[yellow]没有可审核图片。[/yellow]", "[yellow]No images to review.[/yellow]"))
                return
            audit = workspace.audit(source_id=source_id)
            audit_map = {str(record["key"]): record for record in audit["records"]}
            page_count = max(1, (len(items) + page_size - 1) // page_size)
            page = min(page, page_count - 1)
            start = page * page_size
            end = min(len(items), start + page_size)
            table = Table(
                title=self._b(
                    f"图片审核 · 第 {page + 1}/{page_count} 页",
                    f"Image review · page {page + 1}/{page_count}",
                )
            )
            table.add_column("#", justify="right")
            table.add_column(self._b("状态", "State"))
            table.add_column(self._b("来源", "Source"))
            table.add_column(self._b("文件", "File"))
            table.add_column(self._b("分辨率", "Resolution"))
            table.add_column("Tags")
            for index in range(start, end):
                item = items[index]
                record = audit_map.get(item.key, {})
                flags = record.get("flags", [])
                if item.excluded:
                    state = "[red]排除[/red]" if self._b("中", "en") == "中" else "[red]excluded[/red]"
                elif any(flag.get("severity") == "reject" for flag in flags):
                    state = "[red]建议排除[/red]" if self._b("中", "en") == "中" else "[red]reject[/red]"
                elif flags:
                    state = "[yellow]需审核[/yellow]" if self._b("中", "en") == "中" else "[yellow]review[/yellow]"
                else:
                    state = "[green]保留[/green]" if self._b("中", "en") == "中" else "[green]keep[/green]"
                resolution = ""
                if record.get("width") and record.get("height"):
                    resolution = f"{record['width']}×{record['height']}"
                tags = workspace.caption_text(item.key)
                table.add_row(
                    str(index + 1),
                    state,
                    item.source_id,
                    item.relative.as_posix(),
                    resolution,
                    self._truncate(tags, 45),
                )
            self.console.print(table)
            actions = [
                MenuItem("exclude", self._b("按编号/范围排除", "Exclude by number/range")),
                MenuItem("restore", self._b("按编号/范围恢复", "Restore by number/range")),
                MenuItem("edit", self._b("修改某张图片的 Tag", "Edit one image's tags")),
            ]
            if page > 0:
                actions.append(MenuItem("prev", self._b("上一页", "Previous page")))
            if page + 1 < page_count:
                actions.append(MenuItem("next", self._b("下一页", "Next page")))
            actions.append(MenuItem("back", self._b("返回", "Back")))
            action = self._menu(self._b("审核操作", "Review actions"), actions, default="exclude")
            if action == "back":
                return
            if action == "prev":
                page -= 1
                continue
            if action == "next":
                page += 1
                continue
            if action in {"exclude", "restore"}:
                raw = self._ask_text(
                    self._b(
                        "输入编号或范围（例如 1,3-5）",
                        "Enter numbers/ranges (for example 1,3-5)",
                    )
                )
                numbers = parse_number_selection(raw, maximum=len(items))
                keys = [items[number - 1].key for number in numbers]
                if action == "exclude":
                    reason = self._ask_text(
                        self._b("排除原因", "Exclusion reason"),
                        default=self._b("人工审核", "manual review"),
                    ).strip()
                    changed = workspace.exclude(keys, reason=reason or "manual review")
                    self.console.print(self._b(f"[green]已排除 {changed} 张。[/green]", f"[green]Excluded {changed} image(s).[/green]"))
                else:
                    changed = workspace.restore(keys)
                    self.console.print(self._b(f"[green]已恢复 {changed} 张。[/green]", f"[green]Restored {changed} image(s).[/green]"))
            elif action == "edit":
                raw = self._ask_text(self._b("图片编号", "Image number"))
                numbers = parse_number_selection(raw, maximum=len(items))
                if len(numbers) != 1:
                    self.console.print(self._b("[red]一次请选择一张图片。[/red]", "[red]Choose exactly one image.[/red]"))
                    continue
                self._edit_item_tags(workspace, items[numbers[0] - 1].key)

    def _choose_and_edit_tag(self, workspace: DatasetWorkspace, *, source_id: str | None = None) -> None:
        items = workspace.items(
            source_id=source_id,
            include_disabled=True,
            include_excluded=True,
        )
        if not items:
            self.console.print(self._b("[yellow]没有图片。[/yellow]", "[yellow]No images.[/yellow]"))
            return
        table = Table(title=self._b("选择图片修改 Tag", "Choose image to edit tags"))
        table.add_column("#", justify="right")
        table.add_column(self._b("来源", "Source"))
        table.add_column(self._b("文件", "File"))
        table.add_column("Tags")
        limit = min(len(items), 100)
        for index, item in enumerate(items[:limit], start=1):
            table.add_row(str(index), item.source_id, item.relative.as_posix(), self._truncate(workspace.caption_text(item.key), 70))
        self.console.print(table)
        if len(items) > limit:
            self.console.print(self._b(f"[dim]这里只显示前 {limit} 张；大量图片建议从“人工图片审核”分页进入。[/dim]", f"[dim]Showing the first {limit}; use paged Image review for larger sets.[/dim]"))
        raw = self._ask_text(self._b("图片编号", "Image number"))
        numbers = parse_number_selection(raw, maximum=limit)
        if len(numbers) != 1:
            raise PipelineError(self._b("一次只能修改一张图片。", "Edit exactly one image at a time."))
        self._edit_item_tags(workspace, items[numbers[0] - 1].key)

    def _edit_item_tags(self, workspace: DatasetWorkspace, key: str) -> None:
        while True:
            current = workspace.caption_text(key)
            self.console.print(
                Panel.fit(
                    self._b(
                        f"[bold]{key}[/bold]\n当前 Tag：\n{current or '[dim]无[/dim]'}",
                        f"[bold]{key}[/bold]\nCurrent tags:\n{current or '[dim]none[/dim]'}",
                    )
                )
            )
            action = self._menu(
                self._b("Tag 编辑", "Tag editor"),
                [
                    MenuItem("replace", self._b("替换全部 Tag", "Replace all tags")),
                    MenuItem("add", self._b("追加 Tag", "Add tags")),
                    MenuItem("remove", self._b("删除指定 Tag", "Remove selected tags")),
                    MenuItem("clear", self._b("清空 Tag", "Clear tags")),
                    MenuItem("back", self._b("完成 / 返回", "Done / back")),
                ],
                default="add" if current else "replace",
            )
            if action == "back":
                return
            if action == "clear":
                if self._confirm(self._b("清空这张图片的所有 Tag？", "Clear all tags for this image?"), default=False):
                    workspace.replace_caption(key, "")
                continue
            text = self._ask_text(
                self._b(
                    "Tag（逗号分隔）",
                    "Tags (comma-separated)",
                ),
                default=current if action == "replace" else "",
            )
            tags = parse_csv_tags(text)
            if action == "replace":
                workspace.replace_caption(key, ", ".join(tags))
            elif action == "add":
                workspace.add_tags(key, tags)
            elif action == "remove":
                workspace.remove_tags(key, tags)

    # ------------------------------------------------------------------
    # Dataset -> Project snapshot
    # ------------------------------------------------------------------
    def _create_project_from_dataset_interactive(self, *, workspace: DatasetWorkspace | None = None):
        workspaces = list_datasets()
        if workspace is None:
            if not workspaces:
                self.console.print(
                    self._b(
                        "[yellow]还没有数据集。请先创建数据集并导入来源。[/yellow]",
                        "[yellow]No datasets exist yet. Create one and import data first.[/yellow]",
                    )
                )
                if self._confirm(self._b("现在打开数据集管理吗？", "Open dataset manager now?"), default=True):
                    self.dataset_manager()
                return None
            workspace = self._select_dataset(workspaces)
            if workspace is None:
                return None
        workspace = DatasetWorkspace.load(workspace.name)
        summary = workspace.summary()
        if summary["active_images"] < 1:
            raise PipelineError(self._b("数据集没有可用于训练的启用图片。", "Dataset has no enabled images available for training."))

        audit = workspace.audit()
        safe = int(audit["summary"]["safe_exclude_suggestions"])
        if safe and self._confirm(
            self._b(
                f"发现 {safe} 个可安全自动排除的损坏/完全重复项。创建项目前先处理吗？",
                f"Found {safe} safe corrupt/exact-duplicate exclusions. Apply them before creating the project?",
            ),
            default=True,
        ):
            workspace.apply_safe_audit_exclusions()
            workspace = DatasetWorkspace.load(workspace.name)
            summary = workspace.summary()

        registry = self._enabled_bases()
        if not registry:
            self.console.print(self._b("[yellow]还没有已启用底模。[/yellow]", "[yellow]No enabled base model is registered.[/yellow]"))
            if self._confirm(self._b("现在管理底模吗？", "Manage base models now?"), default=True):
                self.base_manager()
                registry = self._enabled_bases()
            if not registry:
                return None

        name = self._ask_project_name()
        base = self._select_base(registry, title="Base checkpoint")
        trigger = self._ask_trigger(name)
        strategy = self._menu("Training strategy", list(STRATEGIES), default="quality")
        active_count = int(summary["active_images"])
        images_seen = self._ask_positive_int("Image exposure budget", default=max(1000, active_count * 8))
        equivalent_epochs = round(images_seen / active_count, 2)

        table = Table(title=self._b("训练项目快照", "Training project snapshot"), show_header=False)
        table.add_column(self._b("项目", "Field"), style="bold")
        table.add_column(self._b("值", "Value"))
        table.add_row(self._b("项目名称", "Project"), name)
        table.add_row(self._b("数据集", "Dataset"), workspace.name)
        table.add_row(self._b("类型", "Type"), workspace.concept_type)
        table.add_row(self._b("启用来源", "Enabled sources"), str(summary["enabled_sources"]))
        table.add_row(self._b("训练图片", "Training images"), str(active_count))
        table.add_row(self._b("已有 Tag", "Captioned"), f"{summary['captioned_active_images']}/{active_count}")
        table.add_row(self._b("底模", "Base"), base)
        table.add_row(self._b("触发词", "Trigger"), trigger)
        table.add_row(self._b("策略", "Strategy"), strategy)
        table.add_row(self._b("图片曝光次数", "Image exposures"), str(images_seen))
        table.add_row(self._b("约等效 Epoch", "Approx. equivalent epochs"), str(equivalent_epochs))
        self.console.print(table)
        if not self._confirm(
            self._b(
                "创建这个不可变训练快照吗？之后数据集继续修改不会改变这个项目。",
                "Create this immutable training snapshot? Later dataset edits will not change this project.",
            ),
            default=True,
        ):
            return None

        state = create_project_from_dataset(
            workspace,
            name=name,
            base=base,
            trigger=trigger,
            strategy=strategy,
            images_seen=images_seen,
        )
        self.console.print(
            Panel.fit(
                self._b(
                    f"[green bold]训练项目已创建[/green bold]\n{state.project_dir}\n"
                    f"数据集快照：{state.payload['project']['dataset_snapshot']['snapshot_hash'][:16]}",
                    f"[green bold]Training project created[/green bold]\n{state.project_dir}\n"
                    f"Dataset snapshot: {state.payload['project']['dataset_snapshot']['snapshot_hash'][:16]}",
                )
            )
        )
        if summary["captioned_active_images"] == active_count:
            self.console.print(
                self._b(
                    "[dim]所有图片已有 Tag；工作流默认使用“清洗已有 Tag 列表”，会在项目层加入 Trigger。[/dim]",
                    "[dim]All images have tags; the workflow defaults to cleaning existing tag lists and adds the project trigger.[/dim]",
                )
            )
        if self._confirm("Configure the guided workflow now?", default=True):
            self.configure_workflow(state.name)
        if self._confirm("Open the project dashboard?", default=True):
            self.project_dashboard(state.name)
        return load_project(state.name)

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    def _render_dataset_home_summary(self, datasets: Sequence[DatasetWorkspace]) -> None:
        if not datasets:
            self.console.print(
                Panel.fit(
                    self._b(
                        "[dim]数据集：0 · 建议先创建数据集，再向其中导入一个或多个来源。[/dim]",
                        "[dim]Datasets: 0 · Create a dataset first, then import one or more sources.[/dim]",
                    )
                )
            )
            return
        total_active = sum(int(workspace.summary()["active_images"]) for workspace in datasets)
        self.console.print(
            self._b(
                f"[dim]数据集：{len(datasets)} · 当前可用图片：{total_active}[/dim]",
                f"[dim]Datasets: {len(datasets)} · active images: {total_active}[/dim]",
            )
        )

    def _render_dataset_list(self, datasets: Sequence[DatasetWorkspace]) -> None:
        if not datasets:
            self.console.print(self._b("[dim]还没有数据集。[/dim]", "[dim]No datasets yet.[/dim]"))
            return
        table = Table(title=self._b("数据集", "Datasets"))
        table.add_column(self._b("名称", "Name"), style="bold")
        table.add_column(self._b("类型", "Type"))
        table.add_column(self._b("来源", "Sources"), justify="right")
        table.add_column(self._b("可用图片", "Active"), justify="right")
        table.add_column(self._b("排除", "Excluded"), justify="right")
        table.add_column("Tags", justify="right")
        for workspace in datasets:
            summary = workspace.summary()
            table.add_row(
                workspace.name,
                workspace.concept_type,
                str(summary["sources"]),
                str(summary["active_images"]),
                str(summary["excluded_images"]),
                f"{summary['captioned_active_images']}/{summary['active_images']}",
            )
        self.console.print(table)

    def _render_dataset_dashboard(self, workspace: DatasetWorkspace) -> None:
        summary = workspace.summary()
        lines = [
            f"[bold]{workspace.name}[/bold] · {workspace.concept_type}",
            self._b(
                f"来源：{summary['enabled_sources']}/{summary['sources']} 启用",
                f"Sources: {summary['enabled_sources']}/{summary['sources']} enabled",
            ),
            self._b(
                f"图片：{summary['active_images']} 可用 · {summary['excluded_images']} 排除",
                f"Images: {summary['active_images']} active · {summary['excluded_images']} excluded",
            ),
            f"Tags: {summary['captioned_active_images']}/{summary['active_images']}",
        ]
        self.console.print(Panel.fit("\n".join(lines), title=self._b("数据集仪表盘", "Dataset dashboard")))
        self._render_sources(workspace)

    def _render_sources(self, workspace: DatasetWorkspace) -> None:
        if not workspace.sources:
            self.console.print(self._b("[dim]尚未导入来源。[/dim]", "[dim]No sources imported.[/dim]"))
            return
        table = Table(title=self._b("数据来源", "Data sources"))
        table.add_column("ID", style="bold")
        table.add_column(self._b("名称", "Label"))
        table.add_column(self._b("类型", "Kind"))
        table.add_column(self._b("状态", "State"))
        table.add_column(self._b("图片", "Images"), justify="right")
        table.add_column(self._b("可用", "Active"), justify="right")
        table.add_column("Tags", justify="right")
        for source_id, source in sorted(workspace.sources.items()):
            items = workspace.items(source_id=source_id, include_disabled=True, include_excluded=True)
            active = [item for item in items if not item.excluded]
            table.add_row(
                source_id,
                str(source.get("label") or source_id),
                str(source.get("kind") or ""),
                self._b("启用", "enabled") if source.get("enabled", True) else self._b("停用", "disabled"),
                str(len(items)),
                str(len(active)),
                f"{sum(item.caption.is_file() for item in active)}/{len(active)}",
            )
        self.console.print(table)

    def _render_source_imported(self, record: dict[str, Any]) -> None:
        self.console.print(
            Panel.fit(
                self._b(
                    f"[green bold]来源已导入[/green bold]\n"
                    f"ID：{record['id']}\n名称：{record['label']}\n图片：{record['imported_images']}\n已有 Tag：{record['imported_captions']}",
                    f"[green bold]Source imported[/green bold]\n"
                    f"ID: {record['id']}\nLabel: {record['label']}\nImages: {record['imported_images']}\nExisting tags: {record['imported_captions']}",
                )
            )
        )

    def _select_dataset(self, datasets: Sequence[DatasetWorkspace]) -> DatasetWorkspace | None:
        if not datasets:
            return None
        choices = [
            MenuItem(
                workspace.name,
                workspace.name,
                self._b(
                    f"{workspace.concept_type} · {workspace.summary()['active_images']} 张可用图片",
                    f"{workspace.concept_type} · {workspace.summary()['active_images']} active images",
                ),
            )
            for workspace in datasets
        ] + [MenuItem("back", self._b("返回", "Back"))]
        selected = self._menu(self._b("选择数据集", "Select dataset"), choices, default=datasets[0].name)
        if selected == "back":
            return None
        return next(workspace for workspace in datasets if workspace.name == selected)


def parse_csv_tags(text: str) -> list[str]:
    return [part.strip() for part in text.replace("\n", ",").replace("，", ",").split(",") if part.strip()]
