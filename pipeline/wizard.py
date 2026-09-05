from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence, TypeVar

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
from rich.table import Table

from .bases import add_base, inspect_base, scan_bases
from .config import load_base_registry, repository_root
from .dataset.image_info import discover_images
from .doctor import run_doctor
from .evaluation import promotion as promote
from .models import STEP_NAMES, PipelineError, StateError, StepResult, StepStatus
from .prepared import load_current_generation
from .service import (
    create_project,
    load_project,
    project_path,
    run_remaining,
    run_single_step,
    skip_preflight_step,
)
from .state import ProjectState, project_lock


T = TypeVar("T")


@dataclass(frozen=True)
class MenuItem:
    value: str
    label: str
    description: str = ""


CAPTION_MODES: tuple[MenuItem, ...] = (
    MenuItem("generate", "Generate captions", "Run the configured tagger and clean the result during materialization."),
    MenuItem(
        "existing_passthrough",
        "Use existing captions unchanged",
        "Preserve every source .txt file byte-for-byte.",
    ),
    MenuItem(
        "existing_taglist_clean",
        "Clean existing tag lists",
        "Treat source captions as Booru-style tags and normalize them during materialization.",
    ),
    MenuItem(
        "hybrid",
        "Hybrid existing + generated",
        "Keep source information and add useful tagger suggestions during materialization.",
    ),
    MenuItem("skip", "Skip caption transformation", "Materialization will use existing sidecars or explicit trigger-only fallback."),
)

STRATEGIES: tuple[MenuItem, ...] = (
    MenuItem("quality", "Quality", "Validated conservative physical batch and full caption variation."),
    MenuItem("fast", "Fast", "Higher physical throughput at the same image-exposure budget."),
    MenuItem("cached", "Cached", "U-Net-only path with text-encoder output caching."),
)

STATUS_STYLE = {
    StepStatus.DONE: ("green", "done"),
    StepStatus.SKIPPED: ("yellow", "skipped"),
    StepStatus.FAILED: ("red", "failed"),
    StepStatus.RUNNING: ("cyan", "running"),
    StepStatus.INTERRUPTED: ("yellow", "interrupted"),
    StepStatus.PENDING: ("white", "pending"),
}

DEFAULT_PREFERENCES: dict[str, Any] = {
    "caption_mode": "generate",
    "allow_trigger_only": False,
}


class Wizard:
    """Interactive Rich UI over the same service layer used by command mode."""

    def __init__(self, *, console: Console | None = None, verbose: int = 0):
        self.console = console or Console()
        self.verbose = verbose

    def home(self) -> None:
        while True:
            projects = self.list_projects()
            self._render_home(projects)
            items = []
            if projects:
                items.append(MenuItem("open", "Open a project", "Resume work from a visual project dashboard."))
            items.extend(
                [
                    MenuItem("new", "Create a project", "Guided project creation with immediate validation."),
                    MenuItem("bases", "Manage base models", "Register, scan, inspect, or verify checkpoints."),
                    MenuItem("doctor", "Check this machine", "Run environment and V100 compatibility checks."),
                    MenuItem("quit", "Exit", "Leave without changing project state."),
                ]
            )
            action = self._menu("Home", items, default="open" if projects else "new")
            if action == "quit":
                self.console.print("[dim]Goodbye.[/dim]")
                return
            self._guarded(
                {
                    "open": lambda: self._choose_and_open_project(projects),
                    "new": self.new_project,
                    "bases": self.base_manager,
                    "doctor": self.doctor,
                }[action]
            )

    def new_project(self) -> ProjectState | None:
        self.console.print(
            Panel.fit(
                "[bold blue]Create a LoRA project[/bold blue]\n"
                "Each answer is validated before the next question. Nothing is created until the summary is confirmed."
            )
        )
        registry = self._enabled_bases()
        if not registry:
            self.console.print("[yellow]No enabled base checkpoint is registered yet.[/yellow]")
            if self._confirm("Open base model manager now?", default=True):
                self.base_manager()
                registry = self._enabled_bases()
            if not registry:
                self.console.print("[red]Project creation needs at least one enabled base model.[/red]")
                return None

        name = self._ask_project_name()
        concept = self._menu(
            "Concept type",
            [
                MenuItem("character", "Character", "Identity consistency, controllability, and leakage review."),
                MenuItem("style", "Style", "Cross-content coverage and dataset-bias diagnostics."),
            ],
            default="character",
        )
        base = self._select_base(registry, title="Base checkpoint")
        dataset, image_count, caption_count = self._ask_dataset()
        trigger = self._ask_trigger(name)
        strategy = self._menu("Training strategy", list(STRATEGIES), default="quality")
        images_seen = self._ask_positive_int("Image exposure budget", default=1000)
        equivalent_epochs = round(images_seen / image_count, 2)

        summary = Table(title="Project summary", show_header=False)
        summary.add_column("Field", style="bold")
        summary.add_column("Value")
        summary.add_row("Name", name)
        summary.add_row("Concept", concept)
        summary.add_row("Base", base)
        summary.add_row("Dataset", str(dataset))
        summary.add_row("Images", str(image_count))
        summary.add_row("Existing captions", f"{caption_count}/{image_count}")
        summary.add_row("Trigger", trigger)
        summary.add_row("Strategy", strategy)
        summary.add_row("Image exposures", str(images_seen))
        summary.add_row("Approx. equivalent epochs", str(equivalent_epochs))
        self.console.print(summary)
        if not self._confirm("Create this project?", default=True):
            self.console.print("[dim]Project creation cancelled.[/dim]")
            return None

        state = create_project(
            name=name,
            concept_type=concept,
            base=base,
            trigger=trigger,
            strategy=strategy,
            dataset=dataset,
            images_seen=images_seen,
        )
        self.console.print(
            Panel.fit(
                f"[green bold]Project created[/green bold]\n{state.project_dir}\n"
                f"Imported {image_count} images and {caption_count} source captions."
            )
        )
        if self._confirm("Configure materialization preferences now?", default=True):
            self.configure_workflow(state.name)
        if self._confirm("Open the project dashboard?", default=True):
            self.project_dashboard(state.name)
        return state

    def open_project(self, name: str, *, resume: bool = False) -> None:
        self.project_dashboard(name, auto_continue=resume)

    def show_state(self, state: ProjectState) -> None:
        self._render_project_steps(state)

    def project_dashboard(self, name: str, *, auto_continue: bool = False) -> None:
        first_iteration = True
        while True:
            state = load_project(name)
            self._render_project_dashboard(state)
            if auto_continue and first_iteration:
                first_iteration = False
                self._guarded(lambda: self.continue_project(name))
                continue
            first_iteration = False
            action = self._menu(
                f"Project: {name}",
                [
                    MenuItem("continue", "Continue recommended work", self._recommended_action(state)),
                    MenuItem("step", "Run one Project stage", "Choose materialize, preflight, or train."),
                    MenuItem("workflow", "Materialization preferences", "Save caption transformation and fallback defaults."),
                    MenuItem("evaluate", "Evaluate checkpoints", "Run run-scoped screening or full evaluation."),
                    MenuItem("promote", "Promote a checkpoint", "Create best.safetensors after human review."),
                    MenuItem("artifacts", "Status and artifacts", "Inspect Project state, runs, reports, and outputs."),
                    MenuItem("advanced", "Advanced recovery", "Preview work or record an expert preflight bypass."),
                    MenuItem("back", "Back to home", "Return to the project list."),
                ],
                default="continue",
            )
            if action == "back":
                return
            handlers: dict[str, Callable[[], Any]] = {
                "continue": lambda: self.continue_project(name),
                "step": lambda: self.run_one_step(name),
                "workflow": lambda: self.configure_workflow(name),
                "evaluate": lambda: self.evaluate_project(name),
                "promote": lambda: self.promote_checkpoint(name),
                "artifacts": lambda: self.show_artifacts(name),
                "advanced": lambda: self.advanced_menu(name),
            }
            self._guarded(handlers[action])

    def continue_project(self, name: str) -> None:
        state = load_project(name)
        next_step = state.next_actionable_step()
        if next_step is None:
            self.console.print("[green]The Project lifecycle has no pending stages.[/green]")
            self._post_pipeline_menu(name)
            return

        preferences = self._preferences(state)
        resume_run = None
        if next_step == "train":
            interrupted = self._interrupted_runs(state)
            if interrupted and self._confirm(
                f"Resume interrupted run {interrupted[-1]['id']} from its latest saved state?",
                default=True,
            ):
                resume_run = str(interrupted[-1]["id"])

        self._render_run_plan(state, preferences, set(), resume_run=resume_run)
        if not self._confirm("Start the guided run now?", default=True):
            return

        def on_step(step: str) -> None:
            current = ProjectState.load(state.project_dir)
            status = current.status(step)
            self.console.print(
                f"\n[bold cyan]▶ {step}[/bold cyan] [dim](current: {status.value})[/dim]"
            )

        results = run_remaining(
            state,
            caption_mode=str(preferences["caption_mode"]),
            allow_trigger_only=bool(preferences["allow_trigger_only"]),
            resume_run=resume_run,
            verbose=self.verbose,
            on_step=on_step,
        )
        if not results:
            self.console.print("[green]Everything in the selected plan was already reusable.[/green]")
        for step, result in results:
            self._print_step_result(step, result)
        refreshed = load_project(name)
        self.console.print(
            Panel.fit(
                f"[green bold]Guided run finished[/green bold]\n"
                f"Recommended next action: {self._recommended_action(refreshed)}"
            )
        )

    def configure_workflow(self, name: str) -> dict[str, Any]:
        state = load_project(name)
        current = self._preferences(state)
        self.console.print(
            Panel.fit(
                "[bold]Materialization preferences[/bold]\n"
                "These choices control caption transformation while freezing the immutable training generation."
            )
        )
        current["caption_mode"] = self._menu(
            "Caption mode",
            list(CAPTION_MODES),
            default=str(current["caption_mode"]),
        )
        current["allow_trigger_only"] = self._confirm(
            "Allow trigger-only fallback when an image has no caption?",
            default=bool(current["allow_trigger_only"]),
        )
        state.payload["project"]["interactive_preferences"] = current
        state.save()
        self.console.print("[green]Materialization preferences saved.[/green]")
        return current

    def run_one_step(self, name: str) -> None:
        state = load_project(name)
        items = [
            MenuItem(step, step, f"Current status: {state.status(step).value}")
            for step in STEP_NAMES
        ] + [MenuItem("back", "Back")]
        step = self._menu(
            "Choose a Project stage",
            items,
            default=state.next_actionable_step() or "train",
        )
        if step == "back":
            return
        if step == "train":
            self.train_project(name)
            return

        state = load_project(name)
        status = state.status(step)
        force = status in {StepStatus.DONE, StepStatus.SKIPPED} and self._confirm(
            f"{step} is already {status.value}. Force a rerun?", default=False
        )
        kwargs: dict[str, Any] = {}
        if step == "materialize":
            preferences = self._preferences(state)
            kwargs["caption_mode"] = str(preferences["caption_mode"])
            kwargs["allow_trigger_only"] = self._confirm(
                "Allow trigger-only captions for otherwise uncaptioned images?",
                default=bool(preferences["allow_trigger_only"]),
            )

        result = self._run_with_lock_retry(
            lambda break_lock: run_single_step(
                load_project(name),
                step,
                force=force,
                break_lock=break_lock,
                verbose=self.verbose,
                **kwargs,
            )
        )
        self._print_step_result(step, result)

    def train_project(self, name: str) -> None:
        state = load_project(name)
        if state.status("preflight") not in {StepStatus.DONE, StepStatus.SKIPPED}:
            if not self._confirm("Preflight is not complete. Run it before training?", default=True):
                self.console.print("[yellow]Training was not started.[/yellow]")
                return
            preflight = self._run_with_lock_retry(
                lambda break_lock: run_single_step(
                    load_project(name),
                    "preflight",
                    break_lock=break_lock,
                    verbose=self.verbose,
                )
            )
            self._print_step_result("preflight", preflight)
            state = load_project(name)

        interrupted = self._interrupted_runs(state)
        resume_run = None
        if interrupted:
            choice = self._menu(
                "Training mode",
                [
                    MenuItem("resume", "Resume interrupted training", str(interrupted[-1]["id"])),
                    MenuItem("new", "Start a new training run", "Keep the interrupted run as historical evidence."),
                    MenuItem("back", "Back"),
                ],
                default="resume",
            )
            if choice == "back":
                return
            if choice == "resume":
                resume_run = str(interrupted[-1]["id"])

        budget = int(state.payload["project"].get("budget", {}).get("value", 1000))
        override: int | None = None
        if not self._confirm(f"Use the saved exposure budget of {budget} images?", default=True):
            override = self._ask_positive_int("Exposure budget for this run", default=budget)

        force = state.status("train") in {StepStatus.DONE, StepStatus.SKIPPED} and self._confirm(
            "Training is already complete. Start another run with these inputs?", default=False
        )
        if state.status("train") in {StepStatus.DONE, StepStatus.SKIPPED} and not force:
            return
        if not self._confirm("Start GPU training now?", default=True):
            return
        result = self._run_with_lock_retry(
            lambda break_lock: run_single_step(
                load_project(name),
                "train",
                force=force,
                break_lock=break_lock,
                images_seen=override,
                resume_run=resume_run,
                verbose=self.verbose,
            )
        )
        self._print_step_result("train", result)
        if self._confirm("Run screening evaluation now?", default=True):
            self.evaluate_project(name, preferred_stage="screening")

    def evaluate_project(self, name: str, *, preferred_stage: str | None = None) -> None:
        state = load_project(name)
        runs = self._successful_runs(state)
        if not runs:
            self.console.print("[yellow]No successful training run is available yet.[/yellow]")
            return
        run = self._select_run(runs, title="Training run to evaluate")
        stage = preferred_stage or self._menu(
            "Evaluation stage",
            [
                MenuItem("screening", "Screening", "Quick matrix across candidate checkpoints."),
                MenuItem("full", "Full", "Detailed matrix for one or two explicit finalists."),
                MenuItem("back", "Back"),
            ],
            default="screening",
        )
        if stage == "back":
            return

        checkpoints = [Path(value) for value in run.get("checkpoints", []) if Path(value).is_file()]
        if not checkpoints:
            raise PipelineError(f"Run {run['id']} has no available checkpoint files")
        selected: list[Path] = checkpoints
        if stage == "full":
            selected = self._select_checkpoints(
                checkpoints,
                title="Select one or two finalists",
                minimum=1,
                maximum=2,
            )
        else:
            self.console.print(
                f"Screening will consider {len(checkpoints)} recorded checkpoint(s); the profile candidate limit still applies."
            )

        evidence = run.get("evaluation", {})
        if evidence:
            self._render_evaluation_evidence(evidence)
        if not self._confirm(
            f"Run {stage} evaluation for {run['id']} on {len(selected)} checkpoint(s)?",
            default=True,
        ):
            return
        result = self._run_with_lock_retry(
            lambda break_lock: run_single_step(
                load_project(name),
                "evaluate",
                break_lock=break_lock,
                evaluation_stage=stage,
                evaluation_run=str(run["id"]),
                evaluation_checkpoints=[path.name for path in selected] if stage == "full" else [],
                verbose=self.verbose,
            )
        )
        self._print_step_result("evaluate", result)
        for label, path in dict(result.details.get("contact_sheets", {})).items():
            self.console.print(f"[bold]{label.replace('_', ' ').title()}:[/bold] {path}")
        if result.details.get("report"):
            self.console.print(f"[bold]Report:[/bold] {result.details['report']}")
        if stage == "full" and self._confirm("Promote a reviewed checkpoint now?", default=False):
            self.promote_checkpoint(name, preferred_run_id=str(run["id"]))

    def promote_checkpoint(self, name: str, *, preferred_run_id: str | None = None) -> None:
        state = load_project(name)
        runs = [record for record in self._successful_runs(state) if record.get("evaluation")]
        if not runs:
            self.console.print(
                "[yellow]No evaluated run is available. Run screening or full evaluation first.[/yellow]"
            )
            return
        run = self._select_run(runs, title="Evaluated run to promote", preferred_id=preferred_run_id)
        evidence = dict(run.get("evaluation", {}))
        self._render_evaluation_evidence(evidence)
        checkpoints = [Path(value) for value in run.get("checkpoints", []) if Path(value).is_file()]
        checkpoint = self._select_checkpoints(
            checkpoints,
            title="Checkpoint to promote",
            minimum=1,
            maximum=1,
        )[0]
        strength = self._ask_positive_float("Recommended LoRA strength", default=0.8)
        summary = Table(title="Promotion summary", show_header=False)
        summary.add_column("Field", style="bold")
        summary.add_column("Value")
        summary.add_row("Run", str(run["id"]))
        summary.add_row("Checkpoint", checkpoint.name)
        summary.add_row("Strength", str(strength))
        summary.add_row("Evidence stages", ", ".join(sorted(evidence)))
        self.console.print(summary)
        if not self._confirm("Create best.safetensors and best.yaml?", default=True):
            return

        def operation(break_lock: bool) -> dict[str, Any]:
            fresh = load_project(name)
            with project_lock(fresh.project_dir, break_lock=break_lock):
                return promote.run(
                    ProjectState.load(fresh.project_dir),
                    run_id=str(run["id"]),
                    checkpoint_name=checkpoint.name,
                    strength=strength,
                )

        payload = self._run_with_lock_retry(operation)
        artifacts = dict(payload.get("artifacts", {}))
        promoted_lora = Path(str(artifacts.get("promoted_lora", "best.safetensors")))
        self.console.print(
            Panel.fit(
                "[green bold]Checkpoint promoted[/green bold]\n"
                f"LoRA: {promoted_lora}\n"
                f"Metadata: {promoted_lora.with_name('best.yaml')}"
            )
        )

    def show_artifacts(self, name: str) -> None:
        state = load_project(name)
        raw_images = discover_images(state.project_dir / "raw")
        validation_images = discover_images(state.project_dir / "validation")
        paths = Table(title="Project paths")
        paths.add_column("Item", style="bold")
        paths.add_column("Path")
        paths.add_column("Details")
        paths.add_row("Project state", str(state.path), "project.yaml")
        paths.add_row("Raw dataset", str(state.project_dir / "raw"), f"{len(raw_images)} image(s)")
        paths.add_row(
            "Validation holdout",
            str(state.project_dir / "validation"),
            f"{len(validation_images)} image(s)",
        )
        try:
            generation = load_current_generation(state.project_dir)
            paths.add_row(
                "Materialized generation",
                str(generation.root),
                f"{len(generation.manifest.get('images', []))} image(s)",
            )
        except PipelineError:
            paths.add_row("Materialized generation", str(state.project_dir / "prepared"), "not created")
        paths.add_row("Runs", str(state.project_dir / "runs"), f"{len(state.payload.get('runs', []))} run(s)")
        self.console.print(paths)

        runs = Table(title="Training and evaluation runs")
        runs.add_column("Run")
        runs.add_column("Status")
        runs.add_column("Checkpoints", justify="right")
        runs.add_column("Evaluation")
        runs.add_column("Promoted")
        runs.add_column("Path")
        for record in reversed(state.payload.get("runs", [])):
            evaluation = ", ".join(sorted(record.get("evaluation", {}))) or "-"
            promotion = record.get("promotion", {}).get("checkpoint", "-")
            runs.add_row(
                str(record.get("id", "?")),
                str(record.get("status", "?")),
                str(len(record.get("checkpoints", []))),
                evaluation,
                str(promotion),
                str(record.get("path", "")),
            )
        if state.payload.get("runs"):
            self.console.print(runs)
        else:
            self.console.print("[dim]No training runs have been recorded yet.[/dim]")
        self._render_project_steps(state)

    def advanced_menu(self, name: str) -> None:
        while True:
            action = self._menu(
                "Advanced recovery",
                [
                    MenuItem("preview", "Preview next action", "Compute fingerprints without changing Project state."),
                    MenuItem(
                        "bypass",
                        "Record preflight bypass",
                        "Expert-only: training safety checks will be marked skipped with a warning.",
                    ),
                    MenuItem("back", "Back"),
                ],
                default="preview",
            )
            if action == "back":
                return
            if action == "preview":
                state = load_project(name)
                preferences = self._preferences(state)
                results = run_remaining(
                    state,
                    dry_run=True,
                    caption_mode=str(preferences["caption_mode"]),
                    allow_trigger_only=bool(preferences["allow_trigger_only"]),
                    verbose=self.verbose,
                )
                if not results:
                    self.console.print("[green]No actionable work remains.[/green]")
                for step, result in results:
                    self._print_step_result(step, result)
            elif action == "bypass":
                typed = self._ask_text(
                    "Type BYPASS to acknowledge that training may be invalid or unsafe",
                    default="",
                )
                if typed != "BYPASS":
                    self.console.print("[yellow]Preflight bypass cancelled.[/yellow]")
                    continue
                result = self._run_with_lock_retry(
                    lambda break_lock: skip_preflight_step(
                        load_project(name), break_lock=break_lock
                    )
                )
                self._print_step_result("preflight", result)

    def base_manager(self) -> None:
        while True:
            registry = load_base_registry()
            self._render_bases(registry)
            action = self._menu(
                "Base model manager",
                [
                    MenuItem("add", "Register a checkpoint", "Add one local .safetensors file."),
                    MenuItem("scan", "Scan a directory", "Find local .safetensors files and register one."),
                    MenuItem("inspect", "Inspect a checkpoint", "Read metadata and use the cached identity hash."),
                    MenuItem("verify", "Fully verify a checkpoint", "Re-read the complete file and compare SHA256."),
                    MenuItem("back", "Back"),
                ],
                default="add" if not registry else "inspect",
            )
            if action == "back":
                return
            self._guarded(
                {
                    "add": self._add_base_interactive,
                    "scan": self._scan_bases_interactive,
                    "inspect": lambda: self._inspect_base_interactive(full_hash=False),
                    "verify": lambda: self._inspect_base_interactive(full_hash=True),
                }[action]
            )

    def doctor(self) -> dict[str, Any]:
        result = run_doctor()
        table = Table(title=f"Machine checks: {result['status']}")
        table.add_column("Status")
        table.add_column("Check")
        table.add_column("Detail")
        for check in result["checks"]:
            color = {"PASS": "green", "WARN": "yellow", "FAIL": "red"}.get(
                check["status"], "white"
            )
            detail = json.dumps(check["detail"], ensure_ascii=False, default=str)
            table.add_row(
                f"[{color}]{check['status']}[/{color}]",
                str(check["name"]),
                self._truncate(detail, 100),
            )
        self.console.print(table)
        return result

    def list_projects(self) -> list[ProjectState]:
        projects_dir = repository_root() / "projects"
        if not projects_dir.is_dir():
            return []
        states: list[ProjectState] = []
        for directory in projects_dir.iterdir():
            if not directory.is_dir() or not (directory / "project.yaml").is_file():
                continue
            try:
                states.append(ProjectState.load(directory))
            except (PipelineError, OSError, ValueError):
                continue
        return sorted(
            states,
            key=lambda state: str(state.payload["project"].get("updated_at", "")),
            reverse=True,
        )

    def _render_home(self, projects: Sequence[ProjectState]) -> None:
        bases = self._enabled_bases()
        self.console.print(
            Panel.fit(
                "[bold blue]LoRA Pipeline[/bold blue]\n"
                "Interactive mode - choose actions by number; command-line flags are optional.\n"
                f"[dim]{len(projects)} project(s) · {len(bases)} enabled base model(s)[/dim]"
            )
        )
        if not projects:
            self.console.print("[dim]No projects yet. Create one to begin.[/dim]")
            return
        table = Table(title="Recent projects")
        table.add_column("Project")
        table.add_column("Type")
        table.add_column("Progress")
        table.add_column("Recommended next action")
        table.add_column("Updated")
        for state in projects[:10]:
            completed = sum(
                state.status(step) in {StepStatus.DONE, StepStatus.SKIPPED}
                for step in STEP_NAMES
            )
            table.add_row(
                state.name,
                state.concept_type,
                f"{completed}/{len(STEP_NAMES)}",
                self._recommended_action(state),
                self._short_timestamp(state.payload["project"].get("updated_at")),
            )
        self.console.print(table)

    def _render_project_dashboard(self, state: ProjectState) -> None:
        project = state.payload["project"]
        latest_run = state.payload.get("runs", [])[-1] if state.payload.get("runs") else None
        budget = project.get("budget", {})
        lines = [
            f"[bold]{state.name}[/bold] · {state.concept_type} · {project.get('strategy')}",
            f"Base: {project.get('base')} · Trigger: {project.get('trigger')}",
            f"Budget: {budget.get('value', '?')} {budget.get('unit', 'images_seen')}",
            f"Recommended: [cyan]{self._recommended_action(state)}[/cyan]",
        ]
        if latest_run:
            lines.append(
                f"Latest run: {latest_run.get('id')} · {latest_run.get('status')} · "
                f"{len(latest_run.get('checkpoints', []))} checkpoint(s)"
            )
        self.console.print(Panel.fit("\n".join(lines), title="Project dashboard"))
        self._render_project_steps(state, compact=True)

    def _render_project_steps(self, state: ProjectState, *, compact: bool = False) -> None:
        table = Table(title="Project lifecycle")
        table.add_column("Stage")
        table.add_column("Status")
        if not compact:
            table.add_column("Attempts", justify="right")
            table.add_column("Details")
        for step in STEP_NAMES:
            record = state.step(step)
            status = state.status(step)
            color, label = STATUS_STYLE[status]
            row = [step, f"[{color}]{label}[/{color}]"]
            if not compact:
                detail = (
                    record.get("last_error")
                    or record.get("invalidation_reason")
                    or record.get("details", {}).get("reason")
                    or ""
                )
                row.extend([str(record.get("attempts", 0)), self._truncate(str(detail), 80)])
            table.add_row(*row)
        self.console.print(table)

    def _render_run_plan(
        self,
        state: ProjectState,
        preferences: dict[str, Any],
        skip: set[str],
        *,
        resume_run: str | None,
    ) -> None:
        del skip
        table = Table(title="Project run plan")
        table.add_column("Stage")
        table.add_column("Current")
        table.add_column("Plan")
        for step in STEP_NAMES:
            if step == "train" and resume_run:
                plan = f"resume {resume_run}"
            elif step == "materialize":
                plan = f"caption mode: {preferences['caption_mode']}"
            else:
                plan = "run or reuse"
            table.add_row(step, state.status(step).value, plan)
        self.console.print(table)

    def _render_bases(self, registry: dict[str, Any]) -> None:
        if not registry:
            self.console.print("[dim]No base models are registered.[/dim]")
            return
        table = Table(title="Registered base models")
        table.add_column("ID")
        table.add_column("Name")
        table.add_column("Available")
        table.add_column("SHA256")
        table.add_column("Path")
        for base_id, base in registry.items():
            table.add_row(
                base_id,
                base.name,
                "[green]yes[/green]" if base.path.is_file() else "[red]no[/red]",
                (base.sha256 or "not inspected")[:16],
                str(base.path),
            )
        self.console.print(table)

    def _render_evaluation_evidence(self, evidence: dict[str, Any]) -> None:
        if not evidence:
            return
        table = Table(title="Existing evaluation evidence")
        table.add_column("Stage")
        table.add_column("Checkpoints")
        table.add_column("Report")
        table.add_column("Completed")
        for stage, record in sorted(evidence.items()):
            table.add_row(
                stage,
                ", ".join(record.get("checkpoints", [])),
                str(record.get("report", "")),
                self._short_timestamp(record.get("completed_at")),
            )
        self.console.print(table)

    def _choose_and_open_project(self, projects: Sequence[ProjectState]) -> None:
        state = self._select_project(projects)
        if state is not None:
            self.project_dashboard(state.name)

    def _select_project(self, projects: Sequence[ProjectState]) -> ProjectState | None:
        if not projects:
            return None
        items = [
            MenuItem(
                state.name,
                state.name,
                f"{state.concept_type}; {self._recommended_action(state)}",
            )
            for state in projects
        ] + [MenuItem("back", "Back")]
        selected = self._menu("Select a project", items, default=projects[0].name)
        if selected == "back":
            return None
        return next(state for state in projects if state.name == selected)

    def _select_base(self, registry: dict[str, Any], *, title: str) -> str:
        items = [
            MenuItem(base_id, base.name, f"{base.path} · {base.family}")
            for base_id, base in registry.items()
        ]
        return self._menu(title, items, default=next(iter(registry)))

    def _select_run(
        self,
        runs: Sequence[dict[str, Any]],
        *,
        title: str,
        preferred_id: str | None = None,
    ) -> dict[str, Any]:
        items = []
        for record in reversed(runs):
            stages = ", ".join(sorted(record.get("evaluation", {}))) or "not evaluated"
            items.append(
                MenuItem(
                    str(record["id"]),
                    str(record["id"]),
                    f"{record.get('status')} · {len(record.get('checkpoints', []))} checkpoint(s) · {stages}",
                )
            )
        selected = self._menu(
            title,
            items,
            default=preferred_id if preferred_id in {item.value for item in items} else items[0].value,
        )
        return next(record for record in runs if str(record["id"]) == selected)

    def _select_checkpoints(
        self,
        checkpoints: Sequence[Path],
        *,
        title: str,
        minimum: int,
        maximum: int,
    ) -> list[Path]:
        table = Table(title=title)
        table.add_column("#", justify="right")
        table.add_column("Checkpoint")
        table.add_column("Size")
        for index, path in enumerate(checkpoints, start=1):
            table.add_row(str(index), path.name, self._human_bytes(path.stat().st_size))
        self.console.print(table)
        while True:
            raw = self._ask_text(
                f"Choose {minimum}" + (f"-{maximum}" if maximum != minimum else "") + " number(s), separated by commas",
                default=str(len(checkpoints)),
            )
            try:
                indexes = [int(value) for value in re.split(r"[\s,]+", raw.strip()) if value]
            except ValueError:
                indexes = []
            indexes = list(dict.fromkeys(indexes))
            if minimum <= len(indexes) <= maximum and all(1 <= value <= len(checkpoints) for value in indexes):
                return [checkpoints[value - 1] for value in indexes]
            self.console.print(
                f"[red]Select between {minimum} and {maximum} valid, distinct checkpoint numbers.[/red]"
            )

    def _preferences(self, state: ProjectState) -> dict[str, Any]:
        preferences = dict(DEFAULT_PREFERENCES)
        stored = state.payload["project"].get("interactive_preferences", {})
        if isinstance(stored, dict):
            if "caption_mode" in stored:
                preferences["caption_mode"] = stored["caption_mode"]
            if "allow_trigger_only" in stored:
                preferences["allow_trigger_only"] = bool(stored["allow_trigger_only"])
        if preferences["caption_mode"] not in {item.value for item in CAPTION_MODES}:
            preferences["caption_mode"] = "generate"
        return preferences

    def _skip_set(self, state: ProjectState, preferences: dict[str, Any]) -> set[str]:
        del state, preferences
        return set()

    def _successful_runs(self, state: ProjectState) -> list[dict[str, Any]]:
        return [
            record
            for record in state.payload.get("runs", [])
            if record.get("status") in {"trained", "evaluated", "promoted"}
            and any(Path(value).is_file() for value in record.get("checkpoints", []))
        ]

    def _interrupted_runs(self, state: ProjectState) -> list[dict[str, Any]]:
        return [record for record in state.payload.get("runs", []) if record.get("status") == "interrupted"]

    def _recommended_action(self, state: ProjectState) -> str:
        next_step = state.next_actionable_step()
        if next_step:
            if state.status(next_step) == StepStatus.FAILED:
                return f"retry {next_step}"
            if state.status(next_step) == StepStatus.INTERRUPTED:
                return f"resume {next_step}"
            return f"continue with {next_step}"
        runs = self._successful_runs(state)
        if not runs:
            return "review project state"
        latest = runs[-1]
        evaluation = latest.get("evaluation", {})
        if "screening" not in evaluation:
            return "run screening evaluation"
        if "full" not in evaluation:
            return "select finalists for full evaluation"
        if not latest.get("promotion"):
            return "review sheets and promote a checkpoint"
        return "complete; inspect artifacts or start another run"

    def _post_pipeline_menu(self, name: str) -> None:
        state = load_project(name)
        action = self._menu(
            "What should happen next?",
            [
                MenuItem("evaluate", "Evaluate checkpoints", self._recommended_action(state)),
                MenuItem("promote", "Promote a checkpoint", "Available after evaluation evidence exists."),
                MenuItem("artifacts", "Inspect artifacts"),
                MenuItem("back", "Back"),
            ],
            default="evaluate",
        )
        if action == "evaluate":
            self.evaluate_project(name)
        elif action == "promote":
            self.promote_checkpoint(name)
        elif action == "artifacts":
            self.show_artifacts(name)

    def _ask_project_name(self) -> str:
        while True:
            value = self._ask_text("Project name").strip()
            try:
                destination = project_path(value)
            except StateError as exc:
                self.console.print(f"[red]{exc}[/red]")
                continue
            if destination.exists():
                self.console.print(f"[red]A project already exists at {destination}.[/red]")
                continue
            return value

    def _ask_dataset(self) -> tuple[Path, int, int]:
        while True:
            path = Path(self._ask_text("Dataset directory")).expanduser().resolve()
            if not path.is_dir():
                self.console.print(f"[red]Directory does not exist: {path}[/red]")
                continue
            images = discover_images(path)
            if not images:
                self.console.print("[red]No supported images were found in that directory.[/red]")
                continue
            captions = sum(image.with_suffix(".txt").is_file() for image in images)
            self.console.print(
                f"Found [bold]{len(images)}[/bold] image(s) and [bold]{captions}[/bold] same-stem caption file(s)."
            )
            if self._confirm("Use this dataset?", default=True):
                return path, len(images), captions

    def _ask_trigger(self, name: str) -> str:
        default = "zz_" + re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")
        while True:
            value = self._ask_text("Trigger token", default=default).strip()
            if value and "," not in value:
                return value
            self.console.print("[red]The trigger must be non-empty and cannot contain a comma.[/red]")

    def _add_base_interactive(self, *, suggested_path: Path | None = None) -> None:
        while True:
            path = suggested_path or Path(self._ask_text("Checkpoint .safetensors path")).expanduser().resolve()
            if path.is_file():
                break
            self.console.print(f"[red]Checkpoint does not exist: {path}[/red]")
            suggested_path = None
        suggested_id = self._slug(path.stem)
        base_id = self._ask_text("Base id", default=suggested_id).strip()
        name = self._ask_text("Display name", default=path.stem).strip()
        base = add_base(base_id, path, name=name, root=repository_root())
        self.console.print(f"[green]Registered {base.id}.[/green]")
        if self._confirm("Inspect metadata and establish the SHA256 identity now?", default=True):
            self._show_base_inspection(
                inspect_base(base.id, root=repository_root(), persist_sha=True, full_hash=True)
            )

    def _scan_bases_interactive(self) -> None:
        directory = Path(self._ask_text("Directory to scan")).expanduser().resolve()
        records = scan_bases(directory, root=repository_root())
        if not records:
            self.console.print("[yellow]No .safetensors files were found.[/yellow]")
            return
        table = Table(title="Discovered checkpoints")
        table.add_column("#", justify="right")
        table.add_column("Status")
        table.add_column("Suggested id")
        table.add_column("Path")
        for index, record in enumerate(records, start=1):
            table.add_row(
                str(index),
                str(record.get("registered_as") or "unregistered"),
                str(record["suggested_id"]),
                str(record["path"]),
            )
        self.console.print(table)
        unregistered = [record for record in records if not record.get("registered_as")]
        if not unregistered:
            self.console.print("[green]Every discovered checkpoint is already registered.[/green]")
            return
        choices = [
            MenuItem(str(index), Path(record["path"]).name, str(record["path"]))
            for index, record in enumerate(unregistered, start=1)
        ] + [MenuItem("back", "Back")]
        selected = self._menu("Register which checkpoint?", choices, default="1")
        if selected != "back":
            self._add_base_interactive(suggested_path=Path(unregistered[int(selected) - 1]["path"]))

    def _inspect_base_interactive(self, *, full_hash: bool) -> None:
        registry = load_base_registry()
        if not registry:
            self.console.print("[yellow]No base models are registered.[/yellow]")
            return
        base_id = self._select_base(registry, title="Checkpoint to inspect")
        if full_hash and not self._confirm(
            "Read and hash the entire checkpoint file? This may take a while on NAS storage.",
            default=True,
        ):
            return
        self._show_base_inspection(
            inspect_base(
                base_id,
                root=repository_root(),
                persist_sha=True,
                full_hash=full_hash,
            )
        )

    def _show_base_inspection(self, result: dict[str, Any]) -> None:
        table = Table(title=f"Base checkpoint: {result['id']}", show_header=False)
        table.add_column("Field", style="bold")
        table.add_column("Value")
        table.add_row("Name", str(result["name"]))
        table.add_row("Path", str(result["path"]))
        table.add_row("Size", self._human_bytes(int(result["bytes"])))
        table.add_row("Family", str(result["family"]))
        table.add_row("SHA256", str(result["sha256"]))
        table.add_row("Identity matches registry", str(result["sha256_matches"]))
        table.add_row("Hash cache reused", str(result["sha256_cache_reused"]))
        table.add_row("Tensor count", str(result["tensor_count"]))
        self.console.print(table)

    def _menu(self, title: str, items: Sequence[MenuItem], *, default: str | None = None) -> str:
        if not items:
            raise ValueError("Menu requires at least one item")
        table = Table(title=title)
        table.add_column("#", justify="right", style="bold cyan")
        table.add_column("Action", style="bold")
        table.add_column("Description")
        for index, item in enumerate(items, start=1):
            table.add_row(str(index), item.label, item.description)
        self.console.print(table)
        by_number = {str(index): item.value for index, item in enumerate(items, start=1)}
        default_number = next(
            (number for number, value in by_number.items() if value == default),
            "1",
        )
        selected = self._ask_text(
            "Choose a number",
            default=default_number,
            choices=list(by_number),
        )
        return by_number[selected]

    def _ask_text(
        self,
        prompt: str,
        *,
        default: str | None = None,
        choices: Sequence[str] | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {}
        if default is not None:
            kwargs["default"] = default
        if choices is not None:
            kwargs["choices"] = list(choices)
        return Prompt.ask(prompt, **kwargs)

    def _confirm(self, prompt: str, *, default: bool = True) -> bool:
        return Confirm.ask(prompt, default=default)

    def _ask_positive_int(self, prompt: str, *, default: int) -> int:
        while True:
            value = IntPrompt.ask(prompt, default=default)
            if value > 0:
                return value
            self.console.print("[red]Enter a positive integer.[/red]")

    def _ask_positive_float(self, prompt: str, *, default: float) -> float:
        while True:
            value = FloatPrompt.ask(prompt, default=default)
            if value > 0:
                return value
            self.console.print("[red]Enter a value greater than zero.[/red]")

    def _guarded(self, action: Callable[[], T]) -> T | None:
        try:
            return action()
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Cancelled. Saved project state was preserved.[/yellow]")
        except (PipelineError, OSError, ValueError) as exc:
            self.console.print(
                Panel.fit(f"[bold red]Action failed[/bold red]\n{exc}", border_style="red")
            )
        return None

    def _run_with_lock_retry(self, action: Callable[[bool], T]) -> T:
        try:
            return action(False)
        except StateError as exc:
            if "lock" not in str(exc).casefold():
                raise
            self.console.print(f"[yellow]{exc}[/yellow]")
            if not self._confirm("Retry after breaking a stale or unverifiable lock?", default=False):
                raise
            return action(True)

    def _print_step_result(self, name: str, result: StepResult) -> None:
        reused = bool(result.details.get("reused"))
        color = "yellow" if result.status == StepStatus.SKIPPED else "green"
        label = "reused" if reused else result.status.value
        details = json.dumps(dict(result.details), ensure_ascii=False, default=str, sort_keys=True)
        self.console.print(
            Panel.fit(
                f"[{color} bold]{name}: {label}[/{color} bold]\n{self._truncate(details, 280)}"
            )
        )

    @staticmethod
    def _enabled_bases() -> dict[str, Any]:
        return {key: value for key, value in load_base_registry().items() if value.enabled}

    @staticmethod
    def _slug(value: str) -> str:
        value = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
        return (value or "checkpoint")[:64]

    @staticmethod
    def _short_timestamp(value: Any) -> str:
        text = str(value or "")
        return text.replace("T", " ")[:19] or "-"

    @staticmethod
    def _truncate(value: str, limit: int) -> str:
        return value if len(value) <= limit else value[: limit - 1] + "…"

    @staticmethod
    def _human_bytes(value: int) -> str:
        size = float(value)
        for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
            if size < 1024 or unit == "TiB":
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TiB"
