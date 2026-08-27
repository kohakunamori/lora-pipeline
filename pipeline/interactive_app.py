from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable

from rich.panel import Panel

from .config import sha256_file
from .dataset.image_info import discover_images
from .models import PipelineError
from .service import load_project
from .state import ProjectState, utc_now
from .wizard import MenuItem, Wizard


class InteractiveWizard(Wizard):
    """The full-screen interactive application, including project data utilities."""

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
                    MenuItem("step", "Run one step", "Choose and run, retry, or force a specific pipeline step."),
                    MenuItem("workflow", "Workflow preferences", "Save caption, review, and screening defaults."),
                    MenuItem(
                        "validation",
                        "Import validation images",
                        "Add unseen holdout images without copying files by hand.",
                    ),
                    MenuItem("evaluate", "Evaluate checkpoints", "Run screening or full evaluation without flags."),
                    MenuItem("promote", "Promote a checkpoint", "Create best.safetensors after human review."),
                    MenuItem("artifacts", "Status and artifacts", "Inspect project paths, runs, reports, and outputs."),
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
                "validation": lambda: self.import_validation_images(name),
                "evaluate": lambda: self.evaluate_project(name),
                "promote": lambda: self.promote_checkpoint(name),
                "artifacts": lambda: self.show_artifacts(name),
                "advanced": lambda: self.advanced_menu(name),
            }
            self._guarded(handlers[action])

    def import_validation_images(self, name: str) -> None:
        state = load_project(name)
        source = Path(self._ask_text("Directory containing unseen validation images")).expanduser().resolve()
        if not source.is_dir():
            raise PipelineError(f"Validation source directory does not exist: {source}")
        destination_root = state.project_dir / "validation"
        if source == destination_root.resolve():
            self.console.print("[dim]That directory is already this project's validation directory.[/dim]")
            return
        images = discover_images(source)
        if not images:
            raise PipelineError(f"No supported images were found under {source}")

        self.console.print(
            Panel.fit(
                f"Found [bold]{len(images)}[/bold] validation image(s).\n"
                "Validation images must be independent holdouts and must not duplicate training images."
            )
        )
        if not self._confirm("Check for training overlap and import these images?", default=True):
            return

        raw_hashes = {
            sha256_file(path): path.relative_to(state.project_dir / "raw").as_posix()
            for path in discover_images(state.project_dir / "raw")
        }
        source_records = [(path, sha256_file(path)) for path in images]
        overlap = [
            (path.relative_to(source).as_posix(), raw_hashes[digest])
            for path, digest in source_records
            if digest in raw_hashes
        ]
        if overlap:
            preview = ", ".join(f"{source_name} = raw/{raw_name}" for source_name, raw_name in overlap[:5])
            raise PipelineError(
                f"Validation import blocked: {len(overlap)} image(s) exactly overlap the training set ({preview})"
            )

        existing_hashes = {
            sha256_file(path) for path in discover_images(destination_root)
        }
        imported = 0
        skipped_existing = 0
        for image, digest in source_records:
            if digest in existing_hashes:
                skipped_existing += 1
                continue
            relative = image.relative_to(source)
            target = self._available_validation_target(destination_root / relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, target)
            existing_hashes.add(digest)
            imported += 1

        if imported:
            state = ProjectState.load(state.project_dir)
            state.payload["project"].setdefault("validation_imports", []).append(
                {
                    "source": str(source),
                    "imported_images": imported,
                    "skipped_existing": skipped_existing,
                    "imported_at": utc_now(),
                }
            )
            # Validation is evaluation-only. Invalidate evaluation without touching training.
            state.invalidate_downstream("train", reason="validation holdout changed")
            state.save()
        self.console.print(
            Panel.fit(
                "[green bold]Validation import complete[/green bold]\n"
                f"Imported: {imported}\n"
                f"Already present: {skipped_existing}\n"
                f"Destination: {destination_root}"
            )
        )

    @staticmethod
    def _available_validation_target(target: Path) -> Path:
        if not target.exists():
            return target
        index = 2
        while True:
            candidate = target.with_name(f"{target.stem}__import-{index}{target.suffix}")
            if not candidate.exists():
                return candidate
            index += 1
