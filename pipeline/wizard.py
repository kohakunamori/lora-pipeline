from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from .config import load_base_registry
from .service import create_project, load_project, run_remaining


class Wizard:
    """Rich prompts over the exact same service functions used by command mode."""

    def __init__(self, *, console: Console | None = None, verbose: int = 0):
        self.console = console or Console()
        self.verbose = verbose

    def new_project(self) -> None:
        self.console.print(Panel.fit("[bold blue]LoRA Pipeline[/bold blue]\nNew resumable project"))
        name = Prompt.ask("Project name")
        concept = Prompt.ask("Concept", choices=["character", "style"], default="character")
        registry = {key: value for key, value in load_base_registry().items() if value.enabled}
        if not registry:
            self.console.print("[red]No enabled base models are registered.[/red]")
            return
        table = Table(title="Base models")
        table.add_column("ID")
        table.add_column("Name")
        table.add_column("Path")
        for base_id, base in registry.items():
            table.add_row(base_id, base.name, str(base.path))
        self.console.print(table)
        base = Prompt.ask("Base model id", choices=list(registry), default=next(iter(registry)))
        dataset = Path(Prompt.ask("Dataset directory")).expanduser()
        trigger = Prompt.ask("Trigger token", default=f"zz_{name}")
        strategy = Prompt.ask(
            "Training strategy", choices=["quality", "fast", "cached"], default="quality"
        )
        images_seen = IntPrompt.ask("Image exposure budget", default=1000)
        state = create_project(
            name=name,
            concept_type=concept,
            base=base,
            trigger=trigger,
            strategy=strategy,
            dataset=dataset,
            images_seen=images_seen,
        )
        self.console.print(f"[green]Created[/green] {state.project_dir}")
        self._workflow_choices(state.name)

    def open_project(self, name: str, *, resume: bool = False) -> None:
        state = load_project(name)
        self.show_state(state)
        next_step = state.next_actionable_step()
        if next_step is None:
            self.console.print(
                "[green]All pipeline steps are complete.[/green] "
                "Use evaluate --stage full and promote after reviewing finalists."
            )
            return
        should_resume = resume or Confirm.ask(
            f"Resume from [bold]{next_step}[/bold]?", default=True
        )
        if should_resume:
            self._workflow_choices(name)

    def show_state(self, state: object) -> None:
        table = Table(title=f"Project: {state.name}")
        table.add_column("Step")
        table.add_column("Status")
        table.add_column("Attempts", justify="right")
        table.add_column("Details")
        for name, record in state.payload["steps"].items():
            status = record["status"]
            color = {
                "done": "green",
                "skipped": "yellow",
                "failed": "red",
                "running": "cyan",
                "interrupted": "yellow",
                "pending": "white",
            }.get(status, "white")
            detail = (
                record.get("last_error")
                or record.get("invalidation_reason")
                or record.get("details", {}).get("reason", "")
            )
            table.add_row(
                name,
                f"[{color}]{status}[/{color}]",
                str(record.get("attempts", 0)),
                str(detail),
            )
        self.console.print(table)

    def _workflow_choices(self, name: str) -> None:
        skip: set[str] = set()
        if not Confirm.ask("Run duplicate detection?", default=True):
            skip.add("dedup")
        state = load_project(name)
        if state.concept_type == "style":
            self.console.print("Character consistency: [dim]N/A for style concepts[/dim]")
        elif not Confirm.ask("Run character consistency (CCIP)?", default=True):
            skip.add("identity")
        caption_mode = Prompt.ask(
            "Caption mode",
            choices=[
                "generate",
                "existing_passthrough",
                "existing_taglist_clean",
                "hybrid",
                "skip",
            ],
            default="generate",
        )
        if caption_mode == "skip":
            skip.add("caption")
        allow_trigger_only = Confirm.ask(
            "Allow explicit trigger-only fallback for images without captions?",
            default=False,
        )
        if not Confirm.ask("Create review summary?", default=True):
            skip.add("review")
        if not Confirm.ask("Run screening evaluation after training?", default=True):
            skip.add("evaluate")
        if not Confirm.ask("Start or resume pipeline now?", default=True):
            self.console.print(f"Saved. Resume later with [bold]./lora open {name}[/bold].")
            return
        results = run_remaining(
            state,
            skip=skip,
            caption_mode=caption_mode,
            allow_trigger_only=allow_trigger_only,
            verbose=self.verbose,
        )
        for step, result in results:
            self.console.print(f"[green]OK[/green] {step}: {result.status.value}")
        self.console.print(
            "[bold]Next:[/bold] review screening sheets, run full evaluation for one or two "
            "finalists, then use [bold]./lora promote[/bold]."
        )
