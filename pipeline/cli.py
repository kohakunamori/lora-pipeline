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
from .evaluation.promotion import run as promote_run
from .models import PipelineError, StepResult
from .service import (
    create_project,
    load_project,
    run_remaining,
    run_single_step,
    skip_preflight_step,
)
from .state import ProjectState, project_lock
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
    verbose: int = typer.Option(0, "--verbose", "-v", count=True),
) -> None:
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    if ctx.invoked_subcommand is None:
        _invoke(lambda: Wizard(console=console, verbose=verbose).new_project())


@app.command("doctor")
def doctor_command(json_output: bool = typer.Option(False, "--json")) -> None:
    def action() -> None:
        result = run_doctor()
        if json_output:
            console.print_json(data=result)
        else:
            table = Table(title=f"LoRA doctor - {result['status']}")
            table.add_column("Status")
            table.add_column("Check")
            table.add_column("Detail")
            for check in result["checks"]:
                color = {"PASS": "green", "WARN": "yellow", "FAIL": "red"}[check["status"]]
                table.add_row(
                    f"[{color}]{check['status']}[/{color}]",
                    check["name"],
                    _compact(check["detail"]),
                )
            console.print(table)
        if result["status"] != "PASS":
            raise typer.Exit(1)

    _invoke(action)


@app.command("new")
def new_command(
    ctx: typer.Context,
    name: str | None = typer.Option(None, "--name"),
    concept: str | None = typer.Option(None, "--concept", help="character or style"),
    base: str | None = typer.Option(None, "--base"),
    dataset: Path | None = typer.Option(None, "--dataset"),
    trigger: str | None = typer.Option(None, "--trigger"),
    strategy: str = typer.Option("quality", "--strategy"),
    images_seen: int = typer.Option(1000, "--images-seen", min=1),
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    def action() -> None:
        verbose = int((ctx.obj or {}).get("verbose", 0))
        if not any((name, concept, base, dataset, trigger)) and not yes:
            Wizard(console=console, verbose=verbose).new_project()
            return
        supplied = {"name": name, "concept": concept, "base": base, "dataset": dataset, "trigger": trigger}
        missing = [key for key, value in supplied.items() if value is None]
        if missing:
            raise PipelineError("Project creation is missing: " + ", ".join(missing))
        state = create_project(
            name=str(name),
            concept_type=str(concept),
            base=str(base),
            trigger=str(trigger),
            strategy=strategy,
            dataset=Path(dataset),
            images_seen=images_seen,
        )
        console.print(f"[green]Created[/green] {state.project_dir}")

    _invoke(action)


@app.command("open")
def open_command(
    ctx: typer.Context,
    project: str = typer.Argument(...),
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    _invoke(
        lambda: Wizard(
            console=console, verbose=int((ctx.obj or {}).get("verbose", 0))
        ).open_project(project, resume=yes)
    )


def _run_materialize_command(
    ctx: typer.Context,
    project: str,
    caption_mode: str,
    allow_trigger_only: bool,
    force_step: bool,
    break_lock: bool,
    dry_run: bool,
    *,
    label: str = "materialize",
) -> None:
    _invoke(
        lambda: _print_step_result(
            label,
            run_single_step(
                load_project(project),
                "materialize",
                force=force_step,
                break_lock=break_lock,
                dry_run=dry_run,
                caption_mode=caption_mode,
                allow_trigger_only=allow_trigger_only,
                verbose=int((ctx.obj or {}).get("verbose", 0)),
            ),
        )
    )


@app.command("materialize")
def materialize_command(
    ctx: typer.Context,
    project: str = typer.Argument(...),
    caption_mode: str = typer.Option("generate", "--caption-mode"),
    allow_trigger_only: bool = typer.Option(False, "--allow-trigger-only"),
    force_step: bool = typer.Option(False, "--force-step", "--force"),
    break_lock: bool = typer.Option(False, "--break-lock"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    _run_materialize_command(
        ctx,
        project,
        caption_mode,
        allow_trigger_only,
        force_step,
        break_lock,
        dry_run,
    )


@app.command("prepare", hidden=True)
def prepare_compat_command(
    ctx: typer.Context,
    project: str = typer.Argument(...),
    caption_mode: str = typer.Option("generate", "--caption-mode"),
    allow_trigger_only: bool = typer.Option(False, "--allow-trigger-only"),
    force_step: bool = typer.Option(False, "--force-step", "--force"),
    break_lock: bool = typer.Option(False, "--break-lock"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    console.print("[yellow]prepare is deprecated; use materialize[/yellow]")
    _run_materialize_command(
        ctx,
        project,
        caption_mode,
        allow_trigger_only,
        force_step,
        break_lock,
        dry_run,
        label="materialize",
    )


@app.command("preflight")
def preflight_command(
    ctx: typer.Context,
    project: str = typer.Argument(...),
    force_step: bool = typer.Option(False, "--force-step", "--force"),
    break_lock: bool = typer.Option(False, "--break-lock"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    _invoke(
        lambda: _print_step_result(
            "preflight",
            run_single_step(
                load_project(project),
                "preflight",
                force=force_step,
                break_lock=break_lock,
                dry_run=dry_run,
                verbose=int((ctx.obj or {}).get("verbose", 0)),
            ),
        )
    )


@app.command("train")
def train_command(
    ctx: typer.Context,
    project: str = typer.Argument(...),
    images_seen: int | None = typer.Option(None, "--images-seen", min=1),
    legacy_steps: int | None = typer.Option(None, "--steps", min=1, hidden=True),
    resume_run: str | None = typer.Option(None, "--resume"),
    skip_preflight: bool = typer.Option(False, "--skip-preflight"),
    force_step: bool = typer.Option(False, "--force-step", "--force"),
    break_lock: bool = typer.Option(False, "--break-lock"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    def action() -> None:
        state = load_project(project)
        if skip_preflight:
            console.print("[bold yellow]WARNING:[/bold yellow] preflight bypass recorded")
            _print_step_result("preflight", skip_preflight_step(state, break_lock=break_lock))
            state = load_project(project)
        else:
            preflight_result = run_single_step(
                state,
                "preflight",
                break_lock=break_lock,
                verbose=int((ctx.obj or {}).get("verbose", 0)),
            )
            if not preflight_result.details.get("reused"):
                _print_step_result("preflight", preflight_result)
            state = load_project(project)
        result = run_single_step(
            state,
            "train",
            force=force_step,
            dry_run=dry_run,
            images_seen=images_seen,
            optimizer_steps=legacy_steps,
            resume_run=resume_run,
            verbose=int((ctx.obj or {}).get("verbose", 0)),
        )
        _print_step_result("train", result)

    _invoke(action)


@app.command("run")
def run_command(
    ctx: typer.Context,
    project: str = typer.Argument(...),
    force_step: bool = typer.Option(False, "--force-step", "--force"),
    break_lock: bool = typer.Option(False, "--break-lock"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    caption_mode: str = typer.Option("generate", "--caption-mode"),
    images_seen: int | None = typer.Option(None, "--images-seen", min=1),
    allow_trigger_only: bool = typer.Option(False, "--allow-trigger-only"),
    skip_preflight: bool = typer.Option(False, "--skip-preflight"),
) -> None:
    def action() -> None:
        if skip_preflight:
            console.print("[bold yellow]WARNING:[/bold yellow] preflight bypass recorded")
        results = run_remaining(
            load_project(project),
            skip_preflight=skip_preflight,
            force=force_step,
            break_lock=break_lock,
            dry_run=dry_run,
            caption_mode=caption_mode,
            images_seen=images_seen,
            allow_trigger_only=allow_trigger_only,
            verbose=int((ctx.obj or {}).get("verbose", 0)),
        )
        for step, result in results:
            _print_step_result(step, result)

    _invoke(action)


@app.command("evaluate")
def evaluate_command(
    ctx: typer.Context,
    project: str = typer.Argument(...),
    stage: str = typer.Option("screening", "--stage", help="screening or full"),
    run_id: str | None = typer.Option(None, "--run"),
    checkpoint: list[str] | None = typer.Option(None, "--checkpoint"),
    break_lock: bool = typer.Option(False, "--break-lock"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    _invoke(
        lambda: _print_step_result(
            "evaluate",
            run_single_step(
                load_project(project),
                "evaluate",
                break_lock=break_lock,
                dry_run=dry_run,
                evaluation_stage=stage,
                evaluation_run=run_id,
                evaluation_checkpoints=checkpoint or [],
                verbose=int((ctx.obj or {}).get("verbose", 0)),
            ),
        )
    )


@app.command("promote")
def promote_command(
    project: str = typer.Argument(...),
    run_id: str = typer.Option(..., "--run"),
    checkpoint: str = typer.Option(..., "--checkpoint"),
    strength: float = typer.Option(..., "--strength", min=0.01),
    allow_unreviewed: bool = typer.Option(False, "--allow-unreviewed"),
    break_lock: bool = typer.Option(False, "--break-lock"),
) -> None:
    def action() -> None:
        state = load_project(project)
        with project_lock(state.project_dir, break_lock=break_lock):
            payload = promote_run(
                ProjectState.load(state.project_dir),
                run_id=run_id,
                checkpoint_name=checkpoint,
                strength=strength,
                allow_unreviewed=allow_unreviewed,
            )
        console.print_json(data=payload)

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
            table.add_row(
                base_id,
                base.name,
                base.family,
                (base.sha256 or "uninspected")[:16],
                str(base.path),
            )
        console.print(table)

    _invoke(action)


@base_app.command("add")
def base_add_command(
    base_id: str = typer.Argument(...),
    path: Path = typer.Argument(...),
    name: str | None = typer.Option(None, "--name"),
    family: str = typer.Option("illustrious_sdxl", "--family"),
    prediction_type: str = typer.Option("epsilon", "--prediction-type"),
) -> None:
    _invoke(
        lambda: console.print(
            f"[green]Registered[/green] "
            f"{add_base(base_id, path, name=name, family=family, prediction_type=prediction_type, root=repository_root()).id}"
        )
    )


@base_app.command("inspect")
def base_inspect_command(
    base_id: str = typer.Argument(...),
    no_persist: bool = typer.Option(False, "--no-persist"),
    full_hash: bool = typer.Option(False, "--full-hash"),
) -> None:
    _invoke(
        lambda: console.print_json(
            data=inspect_base(
                base_id,
                root=repository_root(),
                persist_sha=not no_persist,
                full_hash=full_hash,
            )
        )
    )


@base_app.command("verify")
def base_verify_command(base_id: str = typer.Argument(...)) -> None:
    _invoke(
        lambda: console.print_json(
            data=inspect_base(base_id, root=repository_root(), full_hash=True)
        )
    )


@base_app.command("scan")
def base_scan_command(directory: Path = typer.Argument(...)) -> None:
    _invoke(lambda: console.print_json(data=scan_bases(directory, root=repository_root())))


def _print_step_result(name: str, result: StepResult) -> None:
    color = "yellow" if result.status.value in {"skipped", "interrupted"} else "green"
    console.print(
        Panel.fit(
            f"[{color}]{name}: {result.status.value}[/{color}]\n{_compact(dict(result.details))}"
        )
    )


def _compact(value: Any, limit: int = 300) -> str:
    text = (
        json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
        if not isinstance(value, str)
        else value
    )
    return text if len(text) <= limit else text[: limit - 1] + "..."


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
