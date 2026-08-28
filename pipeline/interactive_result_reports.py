from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .models import PipelineError
from .service import load_project, run_single_step
from .steps import promote
from .wizard import MenuItem


def install_result_report_menu(wizard_class: type[Any]) -> None:
    """Install plain-language result/report UX on the final interactive wizard."""

    if getattr(wizard_class, "_lora_result_report_menu", False):
        return
    wizard_class._training_result_detail = _training_result_detail
    wizard_class._evaluation_report_menu = _evaluation_report_menu
    wizard_class._evaluate_selected_run = _evaluate_selected_run
    wizard_class._promote_selected_run = _promote_selected_run
    setattr(wizard_class, "_lora_result_report_menu", True)


def _training_result_detail(self, entry: dict[str, Any]) -> None:
    while True:
        state = load_project(str(entry["project"]))
        run = self._find_run_record(state, str(entry["run_id"]))
        if run is None:
            return
        self._render_result_detail(state, run)
        evidence = run.get("evaluation", {}) if isinstance(run.get("evaluation"), dict) else {}
        actions = result_action_items(self, evidence=evidence, can_delete=hasattr(self, "_delete_run_interactive"))
        action = self._menu(
            self._b("结果操作", "Result actions"),
            actions,
            default="report" if not evidence else "promote",
        )
        if action == "back":
            return
        if action == "report":
            self._evaluation_report_menu(state, run)
        elif action == "promote":
            self._promote_selected_run(state, run)
        elif action == "paths":
            self._render_result_paths(run)
        elif action == "technical":
            self.project_dashboard(state.name)
        elif action == "delete" and self._delete_run_interactive(state.name, str(run["id"])):
            return


def result_action_items(self, *, evidence: dict[str, Any], can_delete: bool) -> list[MenuItem]:
    del evidence
    items = [
        MenuItem(
            "report",
            self._b("生成 / 更新评测报告", "Generate / update evaluation report"),
            self._b(
                "用这次训练保存下来的权重生成对比图片和 HTML 报告；进入后可选择快速比较或深入评测。生成新图片会使用 GPU。",
                "Generate comparison images and an HTML report from this training run; choose a quick comparison or deeper evaluation inside. New images use the GPU.",
            ),
        ),
        MenuItem(
            "promote",
            self._b("选择最终使用的权重", "Choose the final weight"),
            self._b(
                "从已经看过评测结果的权重版本中选出最终使用的一个，复制为 best.safetensors，并记录推荐 LoRA 强度；不会删除其他权重。",
                "Choose the weight version you want to keep after reviewing evaluation results, copy it to best.safetensors, and record the recommended LoRA strength; other weights are kept.",
            ),
        ),
        MenuItem(
            "paths",
            self._b("查看文件保存位置", "Show saved file locations"),
            self._b(
                "只显示这次训练的权重、示例图、HTML 报告和最终 best 文件分别保存在哪里；不会生成或修改任何内容。",
                "Only show where this run's weights, samples, HTML reports, and final best file are stored; nothing is generated or modified.",
            ),
        ),
        MenuItem(
            "technical",
            self._b("打开技术详情", "Open technical details"),
            self._b(
                "查看底层 Project 的步骤状态、日志和内部文件，主要用于排错；日常训练通常不需要进入。",
                "Inspect the underlying Project step state, logs, and internal files for troubleshooting; normally unnecessary for routine training.",
            ),
        ),
    ]
    if can_delete:
        items.append(
            MenuItem(
                "delete",
                self._b("[red]删除这个训练结果[/red]", "[red]Delete this training result[/red]"),
                self._b(
                    "永久删除这一次训练产生的权重、日志、示例图和评测报告；不会删除数据集、训练配置，也不会影响同一工作区中的其他训练记录。",
                    "Permanently delete this run's weights, logs, samples, and evaluation reports; the Dataset, Training Config, and other runs in the same workspace are kept.",
                ),
            )
        )
    items.append(
        MenuItem(
            "back",
            self._b("返回", "Back"),
            self._b("返回上一层，不修改这次训练结果。", "Return to the previous menu without changing this training result."),
        )
    )
    return items


def _evaluation_report_menu(self, state, run: dict[str, Any]) -> None:
    while True:
        fresh_state = load_project(state.name)
        fresh_run = self._find_run_record(fresh_state, str(run["id"]))
        if fresh_run is None:
            return
        evidence = fresh_run.get("evaluation", {}) if isinstance(fresh_run.get("evaluation"), dict) else {}
        items = report_type_items(self, evidence=evidence)
        default = "quick" if "screening" not in evidence else ("deep" if "full" not in evidence else "back")
        action = self._menu(
            self._b("生成评测报告", "Generate evaluation report"),
            items,
            default=default,
        )
        if action == "back":
            return
        stage = "screening" if action == "quick" else "full"
        self._evaluate_selected_run(fresh_state, fresh_run, stage=stage)


def report_type_items(self, *, evidence: dict[str, Any]) -> list[MenuItem]:
    quick_status = self._b("已生成", "generated") if "screening" in evidence else self._b("尚未生成", "not generated")
    deep_status = self._b("已生成", "generated") if "full" in evidence else self._b("尚未生成", "not generated")
    return [
        MenuItem(
            "quick",
            self._b(f"快速比较训练过程中的权重版本（{quick_status}）", f"Quickly compare training weight versions ({quick_status})"),
            self._b(
                "先用较少的测试场景和多档 LoRA 强度，对训练过程中保存下来的候选权重版本生成同条件图片。适合横向比较早期、中期、后期效果，找出最值得继续看的 1–2 个版本。耗时较低，推荐先运行；会使用 GPU。",
                "Generate matched images for the saved candidate weight versions using a smaller set of test scenes and several LoRA strengths. Use this to compare early/middle/late training and find the 1–2 versions worth deeper review. Lower cost and recommended first; uses the GPU.",
            ),
        ),
        MenuItem(
            "deep",
            self._b(f"深入评测 1–2 个候选权重（{deep_status}）", f"Deeply evaluate 1–2 candidate weights ({deep_status})"),
            self._b(
                "从快速比较后看起来最好的权重版本中选择 1–2 个，再用更多测试场景和强度做细致检查，确认角色还原、可控性和触发词表现。最终 HTML 会同时汇总之前已有的快速比较图片。耗时更高；会使用 GPU。",
                "Choose 1–2 promising weight versions after the quick comparison, then test them with more scenes and strengths to inspect fidelity, controllability, and trigger behavior. The final HTML also includes existing quick-comparison images. Higher cost; uses the GPU.",
            ),
        ),
        MenuItem(
            "back",
            self._b("返回", "Back"),
            self._b("返回训练结果页面，不生成新的评测图片。", "Return to the training result page without generating new evaluation images."),
        ),
    ]


def _evaluate_selected_run(self, state, run: dict[str, Any], *, stage: str) -> None:
    checkpoints = [Path(value) for value in run.get("checkpoints", []) if Path(value).is_file()]
    if not checkpoints:
        raise PipelineError(self._b("没有可评测的权重文件。", "No weight files are available for evaluation."))

    checkpoint_names: list[str] | None = None
    if stage == "full":
        selected = self._select_checkpoints(
            checkpoints,
            title=self._b(
                "选择 1–2 个要深入评测的权重版本",
                "Choose 1–2 weight versions for deeper evaluation",
            ),
            minimum=1,
            maximum=2,
        )
        checkpoint_names = [path.name for path in selected]

    evidence = run.get("evaluation", {}) if isinstance(run.get("evaluation"), dict) else {}
    report_name = self._b("快速比较报告", "quick comparison report") if stage == "screening" else self._b("深入评测报告", "deep evaluation report")
    force = stage in evidence
    if force and not self._confirm(
        self._b(
            f"{report_name}已经存在。重新生成会替换这一类型的旧评测结果，继续吗？",
            f"The {report_name} already exists. Regenerating replaces the previous evaluation of this type. Continue?",
        ),
        default=False,
    ):
        return

    count_text = self._b(
        f"{len(checkpoints)} 个训练权重版本" if stage == "screening" else f"{len(checkpoint_names or [])} 个选中的权重版本",
        f"{len(checkpoints)} training weight versions" if stage == "screening" else f"{len(checkpoint_names or [])} selected weight versions",
    )
    if not self._confirm(
        self._b(
            f"现在使用 GPU 为 {count_text} 生成评测图片和 HTML 报告吗？",
            f"Use the GPU now to generate evaluation images and an HTML report for {count_text}?",
        ),
        default=True,
    ):
        return

    result = self._run_with_lock_retry(
        lambda break_lock: run_single_step(
            load_project(state.name),
            "evaluate",
            force=force,
            break_lock=break_lock,
            verbose=self.verbose,
            evaluation_stage=stage,
            evaluation_run=str(run["id"]),
            evaluation_checkpoints=checkpoint_names,
        )
    )
    self._print_step_result("evaluate", result)


def _promote_selected_run(self, state, run: dict[str, Any]) -> None:
    evidence = run.get("evaluation", {}) if isinstance(run.get("evaluation"), dict) else {}
    evaluated = {
        value
        for record in evidence.values()
        if isinstance(record, dict)
        for value in record.get("checkpoints", [])
    }
    checkpoints = [
        Path(value)
        for value in run.get("checkpoints", [])
        if Path(value).is_file() and (Path(value).name in evaluated or Path(value).stem in evaluated)
    ]
    if not checkpoints:
        self.console.print(
            self._b(
                "[yellow]还没有可以选择的已评测权重。请先生成评测报告。[/yellow]",
                "[yellow]No evaluated weight is available yet. Generate an evaluation report first.[/yellow]",
            )
        )
        return

    choice = self._menu(
        self._b("选择最终权重版本", "Choose the final weight version"),
        [
            MenuItem(
                path.name,
                path.name,
                self._b(
                    f"训练过程中保存的权重文件：{path}",
                    f"Weight file saved during training: {path}",
                ),
            )
            for path in checkpoints
        ] + [
            MenuItem(
                "back",
                self._b("返回", "Back"),
                self._b("不选择最终权重，返回上一层。", "Return without choosing a final weight."),
            )
        ],
        default=checkpoints[-1].name,
    )
    if choice == "back":
        return

    strength = self._ask_positive_float(
        self._b(
            "推荐生成时使用的 LoRA 强度（例如 0.8）",
            "Recommended LoRA strength for generation (for example 0.8)",
        ),
        default=0.8,
    )
    if not self._confirm(
        self._b(
            f"确认把 {choice} 设为这次训练的最终权重吗？它会被复制为 best.safetensors，其他权重不会删除。",
            f"Set {choice} as the final weight for this training? It will be copied to best.safetensors; other weights are not deleted.",
        ),
        default=False,
    ):
        return

    payload = promote.run(
        load_project(state.name),
        run_id=str(run["id"]),
        checkpoint_name=choice,
        strength=strength,
    )
    self.console.print(
        self._b(
            f"[green bold]最终权重已生成[/green bold]\n{payload['artifacts']['promoted_lora']}",
            f"[green bold]Final weight created[/green bold]\n{payload['artifacts']['promoted_lora']}",
        )
    )
