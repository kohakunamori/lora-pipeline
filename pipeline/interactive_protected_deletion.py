from __future__ import annotations

from typing import Any

from rich.panel import Panel
from rich.table import Table

from . import interactive_lifecycle as _interactive_lifecycle
from .interactive_training_parameters import InteractiveWizard as BaseInteractiveWizard
from .lifecycle_guard import deletion_blockers
from .resource_deletion import (
    delete_training_config,
    delete_training_project,
    delete_training_run,
    guarded_create_project_from_training_config,
)
from .service import load_project
from .training_config import TrainingConfig
from .wizard import MenuItem


# Make Dataset + TrainingConfig snapshot creation participate in the same lifecycle
# lock as deletion for the final interactive entry point.
_interactive_lifecycle.create_project_from_training_config = guarded_create_project_from_training_config


class InteractiveWizard(BaseInteractiveWizard):
    """Final CLI layer adding protected Config and training-workspace deletion."""

    def _training_config_dashboard(self, name: str) -> None:
        while True:
            config = TrainingConfig.load(name)
            self._render_training_config_detail(config)
            actions = [
                MenuItem("core", self._b("修改基础训练设置", "Edit core training settings")),
                MenuItem("lora", self._b("修改 LoRA 参数", "Edit LoRA parameters")),
                MenuItem(
                    "workflow",
                    self._b("修改数据处理工作流偏好", "Edit data-processing workflow preferences"),
                ),
                MenuItem("evaluation", self._b("修改评测设置", "Edit evaluation settings")),
                MenuItem("start", self._b("用这份配置开始训练", "Start training with this config")),
                MenuItem(
                    "delete",
                    self._b("[red]删除训练配置[/red]", "[red]Delete training config[/red]"),
                    self._b(
                        "历史 Run 保留冻结的 Config snapshot；活动/待启动训练会阻止删除。",
                        "Historical Runs keep frozen Config snapshots; active/pending training blocks deletion.",
                    ),
                ),
                MenuItem("back", self._b("返回", "Back")),
            ]
            action = self._menu(
                self._b("训练配置操作", "Training config actions"),
                actions,
                default="core",
            )
            if action == "back":
                return
            if action == "core":
                self._edit_training_config_core(config)
            elif action == "lora":
                self._edit_training_config_lora(config)
            elif action == "workflow":
                self._edit_training_config_workflow(config)
            elif action == "evaluation":
                self._edit_training_config_evaluation(config)
            elif action == "start":
                self._start_training_from_dataset_config(prefilled_config=config)
            elif action == "delete" and self._delete_training_config_interactive(config):
                return

    def _training_status_detail(self, entry: dict[str, Any]) -> None:
        while True:
            state = load_project(str(entry["project"]))
            run = self._find_run_record(state, entry.get("run_id"))
            project = state.payload["project"]
            identity = project.get("training_identity", {})
            status = str(run.get("status")) if run else self._workspace_status(state)
            table = Table(
                title=self._b("训练状态详情", "Training status detail"),
                show_header=False,
            )
            table.add_column(self._b("项目", "Field"), style="bold")
            table.add_column(self._b("值", "Value"))
            table.add_row(
                self._b("数据集", "Dataset"),
                str(
                    identity.get("dataset")
                    or project.get("dataset_snapshot", {}).get("dataset")
                    or "legacy"
                ),
            )
            table.add_row(
                self._b("训练配置", "Training config"),
                str(identity.get("config") or "legacy"),
            )
            table.add_row(self._b("状态", "Status"), status)
            table.add_row(
                self._b("当前内部步骤", "Current internal step"),
                str(state.next_actionable_step() or "complete"),
            )
            if run:
                table.add_row("Run ID", str(run.get("id")))
                budget = run.get("resolved_budget", {})
                if isinstance(budget, dict):
                    table.add_row(
                        "images_seen",
                        str(budget.get("target_images_seen") or budget.get("images_seen") or ""),
                    )
                metrics = run.get("metrics", {})
                if isinstance(metrics, dict):
                    if metrics.get("seconds_per_step") is not None:
                        table.add_row("sec/step", str(metrics.get("seconds_per_step")))
                    if metrics.get("peak_vram_gb") is not None:
                        table.add_row("peak VRAM", str(metrics.get("peak_vram_gb")))
                if run.get("last_error"):
                    table.add_row(
                        self._b("最后错误", "Last error"),
                        self._truncate(str(run["last_error"]), 180),
                    )
            table.add_row(self._b("技术工作区", "Technical workspace"), state.name)
            self.console.print(table)

            actions: list[MenuItem] = []
            if status not in {"trained", "evaluated", "promoted"} or state.next_actionable_step() is not None:
                actions.append(
                    MenuItem("continue", self._b("继续 / 恢复训练", "Continue / resume training"))
                )
            if run and run.get("status") in {"trained", "evaluated", "promoted"}:
                actions.append(
                    MenuItem("results", self._b("查看这次训练结果", "Open this training result"))
                )
            actions.extend(
                [
                    MenuItem(
                        "technical",
                        self._b("打开技术 Project 仪表盘", "Open technical Project dashboard"),
                    ),
                    MenuItem(
                        "delete",
                        self._b("[red]删除训练工作区[/red]", "[red]Delete training workspace[/red]"),
                        self._b(
                            "删除全部 Run、权重、日志、缓存与评测产物；活动训练会阻止删除。",
                            "Delete all Runs, weights, logs, caches, and evaluation artifacts; active training blocks deletion.",
                        ),
                    ),
                    MenuItem("back", self._b("返回", "Back")),
                ]
            )
            action = self._menu(
                self._b("训练状态操作", "Training status actions"),
                actions,
                default=actions[0].value,
            )
            if action == "back":
                return
            if action == "continue":
                self.continue_project(state.name)
            elif action == "technical":
                self.project_dashboard(state.name)
            elif action == "results" and run:
                self._training_result_detail(
                    {"project": state.name, "run_id": str(run["id"])}
                )
            elif action == "delete" and self._delete_project_interactive(state.name):
                return

    def _training_result_detail(self, entry: dict[str, Any]) -> None:
        while True:
            state = load_project(str(entry["project"]))
            run = self._find_run_record(state, str(entry["run_id"]))
            if run is None:
                return
            self._render_result_detail(state, run)
            evidence = run.get("evaluation", {}) if isinstance(run.get("evaluation"), dict) else {}
            actions = [
                MenuItem(
                    "screening",
                    self._b("运行 / 重跑 Screening 评测", "Run / rerun screening evaluation"),
                    self._b(
                        "生成示例图和 checkpoint × strength 对比图。",
                        "Generate samples and checkpoint × strength sheets.",
                    ),
                ),
                MenuItem(
                    "full",
                    self._b("运行 Full 评测", "Run full evaluation"),
                    self._b(
                        "明确选择 1–2 个 finalist 后生成完整评测。",
                        "Explicitly select 1–2 finalists for full evaluation.",
                    ),
                ),
                MenuItem(
                    "promote",
                    self._b("选择最佳权重", "Promote best weight"),
                    self._b(
                        "基于人工查看过的评测证据创建 best.safetensors。",
                        "Create best.safetensors after human review of evaluation evidence.",
                    ),
                ),
                MenuItem("paths", self._b("查看权重 / 示例图片路径", "Show weight / sample paths")),
                MenuItem(
                    "technical",
                    self._b("打开技术 Project 仪表盘", "Open technical Project dashboard"),
                ),
                MenuItem(
                    "delete",
                    self._b("[red]删除这个训练结果[/red]", "[red]Delete this training result[/red]"),
                    self._b(
                        "只删除当前 Run 的权重、日志、示例图、评测与 Run metadata。",
                        "Delete only this Run's weights, logs, samples, evaluation, and Run metadata.",
                    ),
                ),
                MenuItem("back", self._b("返回", "Back")),
            ]
            default = (
                "screening"
                if "screening" not in evidence
                else ("full" if "full" not in evidence else "promote")
            )
            action = self._menu(self._b("结果操作", "Result actions"), actions, default=default)
            if action == "back":
                return
            if action == "screening":
                self._evaluate_selected_run(state, run, stage="screening")
            elif action == "full":
                self._evaluate_selected_run(state, run, stage="full")
            elif action == "promote":
                self._promote_selected_run(state, run)
            elif action == "paths":
                self._render_result_paths(run)
            elif action == "technical":
                self.project_dashboard(state.name)
            elif action == "delete" and self._delete_run_interactive(state.name, str(run["id"])):
                return

    def _delete_run_interactive(self, project_name: str, run_id: str) -> bool:
        blockers = deletion_blockers("run", f"{project_name}/{run_id}")
        if blockers:
            self._print_blockers(
                self._b("无法删除训练结果", "Cannot delete training result"), blockers
            )
            return False
        self.console.print(
            Panel.fit(
                self._b(
                    f"[bold red]删除训练结果：{run_id}[/bold red]\n"
                    "只删除这个 Run 的权重、日志、示例图、评测和 metadata；同一 Project 的其他 Run、Dataset 与 Training Config 保留。",
                    f"[bold red]Delete training result: {run_id}[/bold red]\n"
                    "Only this Run's weights, logs, samples, evaluation, and metadata are deleted. Other Runs, the Dataset, and Training Config are retained.",
                ),
                border_style="red",
            )
        )
        if not self._confirm(self._b("永久删除这个 Run？", "Delete this Run permanently?"), default=False):
            return False
        typed = self._ask_text(
            self._b(f"输入 Run ID {run_id} 以确认", f"Type Run ID {run_id} to confirm")
        ).strip()
        if typed != run_id:
            self.console.print(self._b("[yellow]确认不匹配，已取消。[/yellow]", "[yellow]Confirmation mismatch; cancelled.[/yellow]"))
            return False
        result = delete_training_run(project_name, run_id)
        size_mb = float(result.get("deleted_bytes", 0)) / (1024 * 1024)
        self.console.print(
            self._b(
                f"[green]训练结果已删除，释放约 {size_mb:.1f} MiB。[/green]",
                f"[green]Training result deleted; freed about {size_mb:.1f} MiB.[/green]",
            )
        )
        return True

    def _delete_training_config_interactive(self, config: TrainingConfig) -> bool:
        blockers = deletion_blockers("training_config", config.name)
        if blockers:
            self._print_blockers(
                self._b("无法删除训练配置", "Cannot delete training config"), blockers
            )
            return False
        self.console.print(
            Panel.fit(
                self._b(
                    f"[bold red]删除训练配置：{config.name}[/bold red]\n"
                    "历史训练已经保存的 Config snapshot 不会被删除。",
                    f"[bold red]Delete training config: {config.name}[/bold red]\n"
                    "Config snapshots frozen into historical training workspaces are retained.",
                ),
                border_style="red",
            )
        )
        if not self._confirm(self._b("永久删除？", "Delete permanently?"), default=False):
            return False
        typed = self._ask_text(
            self._b(
                f"输入训练配置名称 {config.name} 以确认",
                f"Type training config name {config.name} to confirm",
            )
        ).strip()
        if typed != config.name:
            self.console.print(self._b("[yellow]确认不匹配，已取消。[/yellow]", "[yellow]Confirmation mismatch; cancelled.[/yellow]"))
            return False
        delete_training_config(config.name)
        self.console.print(self._b("[green]训练配置已删除。[/green]", "[green]Training config deleted.[/green]"))
        return True

    def _delete_project_interactive(self, project_name: str) -> bool:
        blockers = deletion_blockers("project", project_name)
        if blockers:
            self._print_blockers(
                self._b("无法删除训练工作区", "Cannot delete training workspace"), blockers
            )
            return False
        self.console.print(
            Panel.fit(
                self._b(
                    f"[bold red]删除训练工作区：{project_name}[/bold red]\n"
                    "这会删除该 Project 下所有 Run、权重、日志、缓存和评测产物。\n"
                    "Dataset 与 Training Config 不会被删除。",
                    f"[bold red]Delete training workspace: {project_name}[/bold red]\n"
                    "This deletes all Runs, weights, logs, caches, and evaluation artifacts in the Project.\n"
                    "The Dataset and Training Config are retained.",
                ),
                border_style="red",
            )
        )
        if not self._confirm(self._b("永久删除整个训练工作区？", "Delete the entire training workspace permanently?"), default=False):
            return False
        typed = self._ask_text(
            self._b(
                f"输入工作区名称 {project_name} 以确认",
                f"Type workspace name {project_name} to confirm",
            )
        ).strip()
        if typed != project_name:
            self.console.print(self._b("[yellow]确认不匹配，已取消。[/yellow]", "[yellow]Confirmation mismatch; cancelled.[/yellow]"))
            return False
        result = delete_training_project(project_name)
        size_mb = float(result.get("deleted_bytes", 0)) / (1024 * 1024)
        self.console.print(
            self._b(
                f"[green]训练工作区已删除，释放约 {size_mb:.1f} MiB。[/green]",
                f"[green]Training workspace deleted; freed about {size_mb:.1f} MiB.[/green]",
            )
        )
        return True

    def _print_blockers(self, title: str, blockers: list[dict[str, str]]) -> None:
        lines = [f"• {item['id']} · {item['status']} · {item['reason']}" for item in blockers]
        self.console.print(
            Panel.fit(
                "[bold yellow]" + title + "[/bold yellow]\n" + "\n".join(lines),
                border_style="yellow",
            )
        )
