from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

from rich.panel import Panel
from rich.table import Table

from .config import repository_root, sha256_file
from .dataset.image_info import discover_images
from .models import OptionalBackendUnavailable, PipelineError
from .service import create_project, load_project
from .state import ProjectState, utc_now
from .video_identity import VideoIdentityReport, cluster_video_identity
from .video_source import (
    VideoProxy,
    detect_environment_proxy,
    extract_video_frames,
    is_url,
    redact_proxy_url,
)
from .wizard import MenuItem, STRATEGIES, Wizard


class InteractiveWizard(Wizard):
    """The full-screen interactive application, including project data utilities."""

    def new_project(self) -> ProjectState | None:
        source_kind = self._menu(
            "Training data source",
            [
                MenuItem(
                    "images",
                    "Image directory",
                    "Use an existing folder of training images and optional caption sidecars.",
                ),
                MenuItem(
                    "video",
                    "Video / YouTube URL",
                    "Download or open a video, sample useful frames, choose the target character, and create a dataset.",
                ),
            ],
            default="images",
        )
        if source_kind == "images":
            return super().new_project()
        return self._new_project_from_video()

    def _new_project_from_video(self) -> ProjectState | None:
        self.console.print(
            Panel.fit(
                "[bold blue]Create a Character LoRA project from video[/bold blue]\n"
                "The importer samples frames, removes blurry or badly exposed frames, filters near-duplicates, "
                "then uses CCIP to let you choose the target character before the normal LoRA pipeline starts."
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
        base = self._select_base(registry, title="Base checkpoint")
        source = self._ask_text("YouTube URL or local video path").strip()
        if not source:
            raise PipelineError("Video source cannot be empty")
        interval_seconds = self._ask_positive_int("Sample one frame every N seconds", default=2)
        max_frames = self._ask_positive_int("Maximum accepted frames before identity selection", default=250)
        proxy = self._select_video_proxy(source)

        cache_root = repository_root() / "cache" / "video-imports"
        cache_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f"{name}-", dir=cache_root) as temporary:
            frame_dir = Path(temporary) / "frames"
            report, proxy = self._extract_video_with_retry(
                source,
                frame_dir,
                interval_seconds=interval_seconds,
                max_frames=max_frames,
                proxy=proxy,
            )
            self._render_video_report(report.as_dict())

            training_dir, identity_provenance = self._select_video_identity(frame_dir)
            image_count = len(discover_images(training_dir))
            if image_count < 5:
                self.console.print(
                    Panel.fit(
                        f"[yellow]Only {image_count} selected frame(s) remain.[/yellow]\n"
                        "That is a very small Character dataset; consider a denser sampling interval or keeping more frames."
                    )
                )
                if not self._confirm("Continue with this small dataset?", default=False):
                    self.console.print("[dim]Project creation cancelled; temporary video frames were discarded.[/dim]")
                    return None

            trigger = self._ask_trigger(name)
            strategy = self._menu("Training strategy", list(STRATEGIES), default="quality")
            images_seen = self._ask_positive_int("Image exposure budget", default=max(1000, image_count * 8))
            equivalent_epochs = round(images_seen / image_count, 2)

            summary = Table(title="Video project summary", show_header=False)
            summary.add_column("Field", style="bold")
            summary.add_column("Value")
            summary.add_row("Name", name)
            summary.add_row("Concept", "character")
            summary.add_row("Base", base)
            summary.add_row("Video source", source)
            summary.add_row("Filtered frames", str(report.accepted_frames))
            summary.add_row("Selected training frames", str(image_count))
            summary.add_row("Sampling interval", f"{interval_seconds}s")
            if identity_provenance.get("selected_cluster") is not None:
                summary.add_row("Selected CCIP cluster", str(identity_provenance["selected_cluster"]))
            proxy_info = report.proxy or {}
            if proxy_info.get("configured"):
                summary.add_row("Download proxy", str(proxy_info.get("endpoint") or proxy_info.get("mode")))
            elif is_url(source):
                summary.add_row("Download proxy", str(proxy_info.get("mode", "direct/environment")))
            summary.add_row("Trigger", trigger)
            summary.add_row("Strategy", strategy)
            summary.add_row("Image exposures", str(images_seen))
            summary.add_row("Approx. equivalent epochs", str(equivalent_epochs))
            self.console.print(summary)
            if not self._confirm("Create this project from the selected frames?", default=True):
                self.console.print("[dim]Project creation cancelled; temporary video frames were discarded.[/dim]")
                return None

            state = create_project(
                name=name,
                concept_type="character",
                base=base,
                trigger=trigger,
                strategy=strategy,
                dataset=training_dir,
                images_seen=images_seen,
            )
            provenance = report.as_dict()
            provenance.pop("downloaded_video", None)
            provenance["source_kind"] = "remote_url" if is_url(source) else "local_video"
            provenance["selected_training_frames"] = image_count
            provenance["identity_preselection"] = identity_provenance
            state.payload["project"]["video_source"] = provenance
            state.save()

        self.console.print(
            Panel.fit(
                f"[green bold]Video project created[/green bold]\n{state.project_dir}\n"
                f"Imported {image_count} target-character frames into the immutable raw dataset."
            )
        )
        if self._confirm("Configure the guided workflow now?", default=True):
            self.configure_workflow(state.name)
        if self._confirm("Open the project dashboard?", default=True):
            self.project_dashboard(state.name)
        return state

    def _select_video_proxy(self, source: str) -> VideoProxy:
        if not is_url(source):
            return VideoProxy(mode="direct")

        env_name, env_value = detect_environment_proxy()
        items: list[MenuItem] = []
        if env_value:
            items.append(
                MenuItem(
                    "environment",
                    "Use detected environment proxy",
                    f"{env_name} = {redact_proxy_url(env_value)}",
                )
            )
        items.extend(
            [
                MenuItem(
                    "direct",
                    "Connect directly (ignore proxy environment variables)",
                    "Pass an explicit direct-connection policy only to yt-dlp.",
                ),
                MenuItem(
                    "custom",
                    "Use a custom proxy for this video",
                    "Supports HTTP(S) and SOCKS proxy URLs; credentials are never written to project metadata.",
                ),
            ]
        )
        default = "environment" if env_value else "direct"
        choice = self._menu("YouTube / video download network", items, default=default)
        if choice == "environment":
            return VideoProxy(mode="environment")
        if choice == "direct":
            return VideoProxy(mode="direct")
        proxy_url = self._ask_text(
            "Proxy URL (for example http://127.0.0.1:7890 or socks5://127.0.0.1:1080)"
        ).strip()
        return VideoProxy(mode="custom", url=proxy_url)

    def _extract_video_with_retry(
        self,
        source: str,
        frame_dir: Path,
        *,
        interval_seconds: int,
        max_frames: int,
        proxy: VideoProxy,
    ):
        while True:
            if frame_dir.exists():
                shutil.rmtree(frame_dir)
            frame_dir.mkdir(parents=True, exist_ok=True)
            proxy_provenance = proxy.provenance() if is_url(source) else None
            proxy_label = ""
            if proxy_provenance:
                endpoint = proxy_provenance.get("endpoint")
                proxy_label = f"\nNetwork: {proxy_provenance.get('mode')}"
                if endpoint:
                    proxy_label += f" via {endpoint}"
            self.console.print(
                Panel.fit(
                    "[cyan]Preparing video frames[/cyan]\n"
                    f"Source: {source}\n"
                    f"Sampling interval: {interval_seconds}s\n"
                    f"Maximum accepted frames: {max_frames}"
                    f"{proxy_label}"
                )
            )
            try:
                report = extract_video_frames(
                    source,
                    frame_dir,
                    interval_seconds=interval_seconds,
                    max_frames=max_frames,
                    proxy=proxy,
                )
                return report, proxy
            except PipelineError as exc:
                if not is_url(source):
                    raise
                self.console.print(Panel.fit(f"[red]Video download/import failed[/red]\n{exc}"))
                action = self._menu(
                    "Download recovery",
                    [
                        MenuItem("retry", "Retry with the same network settings"),
                        MenuItem("proxy", "Change proxy settings"),
                        MenuItem("cancel", "Cancel video import"),
                    ],
                    default="proxy",
                )
                if action == "cancel":
                    raise PipelineError("Video import cancelled") from exc
                if action == "proxy":
                    proxy = self._select_video_proxy(source)

    def _select_video_identity(self, frame_dir: Path) -> tuple[Path, dict[str, object]]:
        self.console.print(
            Panel.fit(
                "[cyan]Finding character identity clusters[/cyan]\n"
                "CCIP will group the filtered frames by likely character identity. You choose the target; "
                "the importer will not automatically assume that the largest cluster is correct."
            )
        )
        try:
            report = cluster_video_identity(frame_dir)
        except OptionalBackendUnavailable as exc:
            self.console.print(
                Panel.fit(
                    "[yellow]CCIP is unavailable, so target-character preselection cannot run.[/yellow]\n"
                    f"{exc}\nThe normal Character identity/review stages can still run later."
                )
            )
            return frame_dir, {
                "status": "unavailable",
                "reason": str(exc),
                "selected_cluster": None,
                "selected_frames": len(discover_images(frame_dir)),
            }
        except PipelineError as exc:
            self.console.print(
                Panel.fit(
                    "[yellow]CCIP could not produce stable pre-import clusters.[/yellow]\n"
                    f"{exc}"
                )
            )
            if not self._confirm("Keep all filtered frames and continue to the normal Identity/Review stages?", default=True):
                raise
            return frame_dir, {
                "status": "fallback_all_frames",
                "reason": str(exc),
                "selected_cluster": None,
                "selected_frames": len(discover_images(frame_dir)),
            }

        self._render_identity_clusters(report, frame_dir)
        items = [
            MenuItem(
                f"cluster:{cluster.cluster_id}",
                f"Use cluster {cluster.cluster_id} ({cluster.size} frames)",
                "Representatives: " + ", ".join(path.name for path in cluster.representatives),
            )
            for cluster in report.clusters
        ]
        items.extend(
            [
                MenuItem(
                    "all",
                    "Keep all filtered frames",
                    "Skip pre-import identity filtering and rely on the later Character Identity/Review steps.",
                ),
                MenuItem("cancel", "Cancel video import"),
            ]
        )
        default = f"cluster:{report.clusters[0].cluster_id}"
        choice = self._menu("Target character", items, default=default)
        if choice == "cancel":
            raise PipelineError("Video import cancelled before target-character selection")

        provenance = report.as_dict(root=frame_dir)
        if choice == "all":
            if not self._confirm(
                "Keep every filtered frame, including other clusters and CCIP outliers?",
                default=False,
            ):
                return self._select_video_identity(frame_dir)
            provenance.update(
                {
                    "status": "kept_all_frames",
                    "selected_cluster": None,
                    "selected_frames": report.total_frames,
                }
            )
            return frame_dir, provenance

        cluster_id = int(choice.split(":", 1)[1])
        selected = next(cluster for cluster in report.clusters if cluster.cluster_id == cluster_id)
        selected_dir = frame_dir.parent / "selected-character"
        if selected_dir.exists():
            shutil.rmtree(selected_dir)
        selected_dir.mkdir(parents=True, exist_ok=True)
        for frame in selected.frames:
            shutil.copy2(frame, selected_dir / frame.name)
        provenance.update(
            {
                "status": "selected_cluster",
                "selected_cluster": cluster_id,
                "selected_frames": selected.size,
                "discarded_other_clusters": sum(
                    cluster.size for cluster in report.clusters if cluster.cluster_id != cluster_id
                ),
                "discarded_outliers": len(report.outliers),
            }
        )
        return selected_dir, provenance

    def _render_identity_clusters(self, report: VideoIdentityReport, frame_dir: Path) -> None:
        table = Table(title="Video character identity clusters")
        table.add_column("Cluster", style="bold")
        table.add_column("Frames", justify="right")
        table.add_column("Representative frame files")
        for cluster in report.clusters:
            representatives = ", ".join(
                path.relative_to(frame_dir).as_posix() for path in cluster.representatives
            )
            table.add_row(str(cluster.cluster_id), str(cluster.size), representatives)
        table.add_row("CCIP outliers", str(len(report.outliers)), "excluded when a target cluster is selected")
        self.console.print(table)
        self.console.print(
            "[dim]Tip: if the same character is split across outfits or extreme camera angles, use the representative "
            "filenames to inspect the candidates before choosing. You can also keep all frames and review later.[/dim]"
        )

    def _render_video_report(self, report: dict[str, object]) -> None:
        table = Table(title="Video frame filtering")
        table.add_column("Metric", style="bold")
        table.add_column("Count", justify="right")
        table.add_row("Sampled candidates", str(report["sampled_frames"]))
        table.add_row("Accepted before identity selection", str(report["accepted_frames"]))
        table.add_row("Rejected: blurry", str(report["rejected_blurry"]))
        table.add_row("Rejected: near-duplicate", str(report["rejected_near_duplicate"]))
        table.add_row("Rejected: exposure", str(report["rejected_exposure"]))
        self.console.print(table)

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
                        "settings",
                        "Project settings",
                        "Change the exposure budget, strategy, or evaluation subject prompt.",
                    ),
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
                "settings": lambda: self.project_settings(name),
                "validation": lambda: self.import_validation_images(name),
                "evaluate": lambda: self.evaluate_project(name),
                "promote": lambda: self.promote_checkpoint(name),
                "artifacts": lambda: self.show_artifacts(name),
                "advanced": lambda: self.advanced_menu(name),
            }
            self._guarded(handlers[action])

    def project_settings(self, name: str) -> None:
        while True:
            state = load_project(name)
            project = state.payload["project"]
            self._render_project_settings(state)
            items = [
                MenuItem(
                    "budget",
                    "Change image-exposure budget",
                    "Updates preflight and future training without touching prepared data.",
                ),
                MenuItem(
                    "strategy",
                    "Change training strategy",
                    "Switch between Quality, Fast, and Cached profiles.",
                ),
            ]
            if state.concept_type == "character":
                items.append(
                    MenuItem(
                        "subject",
                        "Change evaluation subject prompt",
                        "Describe the subject used in Character evaluation prompts.",
                    )
                )
            items.append(MenuItem("back", "Back"))
            action = self._menu("Project settings", items, default="budget")
            if action == "back":
                return
            if action == "budget":
                current = int(project.get("budget", {}).get("value", 1000))
                value = self._ask_positive_int("New image-exposure budget", default=current)
                if value == current:
                    self.console.print("[dim]Budget is unchanged.[/dim]")
                    continue
                project["budget"] = {"unit": "images_seen", "value": value}
                state.invalidate_downstream("prepare", reason="training exposure budget changed")
                state.save()
                self.console.print(f"[green]Exposure budget updated to {value} images.[/green]")
            elif action == "strategy":
                current = str(project.get("strategy", "quality"))
                value = self._menu("Training strategy", list(STRATEGIES), default=current)
                if value == current:
                    self.console.print("[dim]Training strategy is unchanged.[/dim]")
                    continue
                project["strategy"] = value
                state.invalidate_downstream("prepare", reason="training strategy changed")
                state.save()
                self.console.print(f"[green]Training strategy updated to {value}.[/green]")
            elif action == "subject":
                evaluation = project.setdefault("evaluation", {})
                current = str(evaluation.get("subject_prompt", "1girl"))
                value = self._ask_text("Evaluation subject prompt", default=current).strip()
                if not value:
                    raise PipelineError("Evaluation subject prompt cannot be empty")
                if value == current:
                    self.console.print("[dim]Evaluation subject prompt is unchanged.[/dim]")
                    continue
                evaluation["subject_prompt"] = value
                state.invalidate_downstream("train", reason="evaluation subject prompt changed")
                state.save()
                self.console.print("[green]Evaluation subject prompt updated.[/green]")

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

    def _render_project_settings(self, state: ProjectState) -> None:
        project = state.payload["project"]
        table = Table(title="Current project settings", show_header=False)
        table.add_column("Setting", style="bold")
        table.add_column("Value")
        table.add_row(
            "Image exposures",
            str(project.get("budget", {}).get("value", "not set")),
        )
        table.add_row("Training strategy", str(project.get("strategy", "quality")))
        table.add_row("Base", str(project.get("base", "")))
        table.add_row("Trigger", str(project.get("trigger", "")))
        if state.concept_type == "character":
            table.add_row(
                "Evaluation subject",
                str(project.get("evaluation", {}).get("subject_prompt", "1girl")),
            )
        video_source = project.get("video_source")
        if isinstance(video_source, dict):
            table.add_row("Video source", str(video_source.get("source", "")))
            table.add_row("Video filtered frames", str(video_source.get("accepted_frames", "")))
            table.add_row("Video selected frames", str(video_source.get("selected_training_frames", "")))
            identity = video_source.get("identity_preselection", {})
            if isinstance(identity, dict) and identity.get("selected_cluster") is not None:
                table.add_row("Video CCIP cluster", str(identity.get("selected_cluster")))
            proxy = video_source.get("proxy", {})
            if isinstance(proxy, dict) and proxy.get("configured"):
                table.add_row("Video proxy", str(proxy.get("endpoint") or proxy.get("mode")))
        self.console.print(table)

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
