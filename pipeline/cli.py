from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .bases import add_base, inspect_base, scan_bases
from .config import load_base_registry, repository_root
from .doctor import run_doctor
from .models import PipelineError, StepResult
from .service import create_project, load_project, run_remaining, run_single_step
from .wizard import Wizard


app = typer.Typer(
    name="lora",
    help="Resumable Illustrious/SDXL LoRA pipeline around pinned sd-scripts.",
    invoke_without_command=True,
    no_args_is_help=False,
    pretty_exceptions_show_locals=False,
)
base_app = typer.Typer(help="Inspect and register local Illustrious/SDXL checkpoints.")
app.add_typer(base_app, name="base")
console = Console()


@app.callback()
def root_callback(
    ctx: typer.Context,
    verbose: int = typer.Option(0, "--verbose", "-v", count=True, help="Show progressively more backend output."),
) -> None:
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    if ctx.invoked_subcommand is None:
        _invoke(lambda: Wizard(console=console, verbose=verbose).new_project())


@app.command("doctor")
def doctor_command(
    json_output: bool = typer.Option(False, "--json", help="Emit the complete machine-readable report."),
) -> None:
    def action() -> None:
        result = run_doctor()
        if json_output:
            console.print_json(data=result)
        else:
            table = Table(title=f"LoRA doctor — {result['status']}")
            table.add_column("Status")
            table.add_column("Check")
            table.add_column("Detail")
            for check in result["checks"]:
                color = {"PASS": "green", "WARN": "yellow", "FAIL": "red"}[check["status"]]
                table.add_row(f"[{color}]{check['status']}[/{color}]", check["name"], _compact(check["detail"]))
            console.print(table)
        if result["status"] != "PASS":
            raise typer.Exit(1)

    _invoke(action)


@app.command("new")
def new_command(
    ctx: typer.Context,
    name: str | None = typer.Option(None, "--name", help="Project name."),
    concept: str | None = typer.Option(None, "--concept", help="character or style."),
    base: str | None = typer.Option(None, "--base", help="Registered base id."),
    dataset: Path | None = typer.Option(None, "--dataset", help="Source dataset directory."),
    trigger: str | None = typer.Option(None, "--trigger", help="Unique trigger token."),
    strategy: str = typer.Option("quality", "--strategy", help="quality, fast, or cached."),
    steps: int = typer.Option(1000, "--steps", min=1, help="Optimizer-step budget."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Non-interactive creation; all required options must be supplied."),
) -> None:
    def action() -> None:
        verbose = int((ctx.obj or {}).get("verbose", 0))
        if not any((name, concept, base, dataset, trigger)) and not yes:
            Wizard(console=console, verbose=verbose).new_project()
            return
        missing = [key for key, value in {"name": name, "concept": concept, "base": base, "dataset": dataset, "trigger": trigger}.items() if value is None]
        if missing:
            raise PipelineError("Non-interactive project creation is missing: " + ", ".join(missing))
        state = create_project(
            name=str(name),
            concept_type=str(concept),
            base=str(base),
            trigger=str(trigger),
            strategy=strategy,
            dataset=Path(dataset),
            optimizer_steps=steps,
        )
        console.print(f"[green]Created[/green] {state.project_dir}")

    _invoke(action)


@app.command("open")
def open_command(
    ctx: typer.Context,
    project: str = typer.Argument(...),
    yes: bool = typer.Option(False, "--yes", "-y", help="Resume immediately without confirmation."),
) -> None:
    _invoke(lambda: Wizard(console=console, verbose=int((ctx.obj or {}).get("verbose", 0))).open_project(project, resume=yes))


def _simple_step_command(name: str) -> Callable[..., None]:
    def command(
        ctx: typer.Context,
        project: str = typer.Argument(...),
        force: bool = typer.Option(False, "--force", help="Repeat a completed step and override a stale project lock."),
        dry_run: bool = typer.Option(False, "--dry-run", help="Show what would run without changing step state."),
        yes: bool = typer.Option(False, "--yes", "-y", help="Confirm non-interactively."),
    ) -> None:
        del yes
        _invoke(
            lambda: _print_step_result(
                name,
                run_single_step(
                    load_project(project),
                    name,
                    force=force,
                    dry_run=dry_run,
                    verbose=int((ctx.obj or {}).get("verbose", 0)),
                ),
            )
        )

    command.__name__ = f"{name}_command"
    return command


for _step_name in ("inspect", "identity", "prepare", "preflight"):
    app.command(_step_name)(_simple_step_command(_step_name))


@app.command("dedup")
def dedup_command(
    ctx: typer.Context,
    project: str = typer.Argument(...),
    exclude_exact: bool = typer.Option(False, "--exclude-exact", help="Add all but one exact copy to the exclusion manifest."),
    force: bool = typer.Option(False, "--force"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    del yes
    _invoke(lambda: _print_step_result("dedup", run_single_step(load_project(project), "dedup", force=force, dry_run=dry_run, exclude_exact=exclude_exact, verbose=int((ctx.obj or {}).get("verbose", 0)))))


@app.command("caption")
def caption_command(
    ctx: typer.Context,
    project: str = typer.Argument(...),
    mode: str = typer.Option("generate", "--mode", help="generate, existing, or skip."),
    force: bool = typer.Option(False, "--force"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    del yes
    _invoke(lambda: _print_step_result("caption", run_single_step(load_project(project), "caption", force=force, dry_run=dry_run, caption_mode=mode, verbose=int((ctx.obj or {}).get("verbose", 0)))))


@app.command("review")
def review_command(
    ctx: typer.Context,
    project: str = typer.Argument(...),
    exclude: list[str] | None = typer.Option(None, "--exclude", help="Raw-relative image path to exclude; repeatable."),
    force: bool = typer.Option(False, "--force"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    del yes
    _invoke(lambda: _print_step_result("review", run_single_step(load_project(project), "review", force=force, dry_run=dry_run, exclusions=exclude or [], verbose=int((ctx.obj or {}).get("verbose", 0)))))


@app.command("train")
def train_command(
    ctx: typer.Context,
    project: str = typer.Argument(...),
    steps: int | None = typer.Option(None, "--steps", min=1, help="Override this run's optimizer-step budget."),
    skip_preflight: bool = typer.Option(False, "--skip-preflight", help="Expert bypass; recorded with a warning."),
    force: bool = typer.Option(False, "--force"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Generate run configs without launching sd-scripts."),
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    del yes

    def action() -> None:
        state = load_project(project)
        if state.status("preflight").value != "done":
            if skip_preflight:
                console.print("[bold yellow]WARNING:[/bold yellow] preflight explicitly bypassed; this is recorded in project.yaml")
                state.skip_preflight("expert --skip-preflight override")
            elif not dry_run:
                _print_step_result("preflight", run_single_step(state, "preflight", force=force, verbose=int((ctx.obj or {}).get("verbose", 0))))
                state = load_project(project)
        result = run_single_step(state, "train", force=force, dry_run=dry_run, optimizer_steps=steps, verbose=int((ctx.obj or {}).get("verbose", 0)))
        _print_step_result("train", result)

    _invoke(action)


@app.command("evaluate")
def evaluate_command(
    ctx: typer.Context,
    project: str = typer.Argument(...),
    force: bool = typer.Option(False, "--force"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    del yes
    _invoke(lambda: _print_step_result("evaluate", run_single_step(load_project(project), "evaluate", force=force, dry_run=dry_run, verbose=int((ctx.obj or {}).get("verbose", 0)))))


@app.command("run")
def run_command(
    ctx: typer.Context,
    project: str = typer.Argument(...),
    yes: bool = typer.Option(False, "--yes", "-y", help="Run without interactive confirmation."),
    force: bool = typer.Option(False, "--force", help="Repeat completed steps."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview only the next actionable step."),
    caption_mode: str = typer.Option("generate", "--caption-mode", help="generate, existing, or skip."),
    exclude_exact: bool = typer.Option(False, "--exclude-exact"),
    skip_dedup: bool = typer.Option(False, "--skip-dedup"),
    skip_identity: bool = typer.Option(False, "--skip-identity"),
    skip_caption: bool = typer.Option(False, "--skip-caption"),
    skip_review: bool = typer.Option(False, "--skip-review"),
    skip_preflight: bool = typer.Option(False, "--skip-preflight"),
    skip_evaluate: bool = typer.Option(False, "--skip-evaluate"),
) -> None:
    del yes

    def action() -> None:
        skip = {
            name
            for name, enabled in {
                "dedup": skip_dedup,
                "identity": skip_identity,
                "caption": skip_caption,
                "review": skip_review,
                "evaluate": skip_evaluate,
            }.items()
            if enabled
        }
        if skip_preflight:
            console.print("[bold yellow]WARNING:[/bold yellow] preflight will be bypassed and recorded")
        results = run_remaining(
            load_project(project),
            skip=skip,
            skip_preflight=skip_preflight,
            force=force,
            dry_run=dry_run,
            caption_mode="skip" if skip_caption else caption_mode,
            exclude_exact=exclude_exact,
            verbose=int((ctx.obj or {}).get("verbose", 0)),
        )
        for step, result in results:
            _print_step_result(step, result)

    _invoke(action)


@base_app.command("list")
def base_list_command() -> None:
    def action() -> None:
        table = Table(title="Registered base models")
        table.add_column("ID")
        table.add_column("Name")
        table.add_column("Family")
        table.add_column("SHA256")
        table.add_column("Path")
        for base_id, base in load_base_registry().items():
            table.add_row(base_id, base.name, base.family, (base.sha256 or "uninspected")[:16], str(base.path))
        console.print(table)

    _invoke(action)


@base_app.command("add")
def base_add_command(
    base_id: str = typer.Argument(...),
    path: Path = typer.Argument(...),
    name: str | None = typer.Option(None, "--name"),
    family: str = typer.Option("illustrious_sdxl", "--family"),
    prediction_type: str = typer.Option("epsilon", "--prediction-type"),
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    del yes
    _invoke(lambda: console.print(f"[green]Registered[/green] {add_base(base_id, path, name=name, family=family, prediction_type=prediction_type, root=repository_root()).id}"))


@base_app.command("inspect")
def base_inspect_command(
    base_id: str = typer.Argument(...),
    no_persist: bool = typer.Option(False, "--no-persist", help="Do not update a missing/outdated registry SHA256."),
) -> None:
    _invoke(lambda: console.print_json(data=inspect_base(base_id, root=repository_root(), persist_sha=not no_persist)))


@base_app.command("scan")
def base_scan_command(directory: Path = typer.Argument(...)) -> None:
    _invoke(lambda: console.print_json(data=scan_bases(directory, root=repository_root())))


def _print_step_result(name: str, result: StepResult) -> None:
    color = "yellow" if result.status.value == "skipped" else "green"
    console.print(Panel.fit(f"[{color}]{name}: {result.status.value}[/{color}]\n{_compact(dict(result.details))}"))


def _compact(value: Any, limit: int = 240) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str, sort_keys=True) if not isinstance(value, str) else value
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _invoke(action: Callable[[], Any]) -> Any:
    try:
        return action()
    except typer.Exit:
        raise
    except (PipelineError, OSError, ValueError) as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(1) from exc


def main() -> None:
    app()


if __name__ == "__main__":
    main()
