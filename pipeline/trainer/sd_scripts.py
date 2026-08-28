from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time
import warnings
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

from .. import gpu_resources
from ..config import repository_root, sha256_file, stable_hash, write_json_atomic
from ..models import ExternalCommandError, PipelineError, TrainingRequest, TrainingResult
from ..model_artifact import (
    ModelArtifactMetadata,
    build_modelspec_metadata,
    build_sd_scripts_metadata,
    resolve_model_metadata,
    rewrite_safetensors_metadata,
)
from ..prepared import load_current_generation
from .base import TrainerBackend


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if value is None:
        raise ValueError("TOML null values are not supported")
    return str(value)


def write_flat_toml(path: Path, values: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key} = {_toml_value(value)}" for key, value in values.items() if value is not None]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dataset_toml(
    path: Path, *, dataset_dirs: Sequence[Path], merged: Mapping[str, Any]
) -> None:
    resolution = merged.get("resolution", {})
    caption = merged.get("caption", {})
    training = merged.get("training", {})
    dataset_dirs = tuple(dataset_dirs)
    if not dataset_dirs:
        raise PipelineError("Training snapshot has no image-containing subset directories")
    lines = [
        "[general]",
        'caption_extension = ".txt"',
        f"shuffle_caption = {_toml_value(bool(caption.get('shuffle', False)))}",
        f"keep_tokens = {int(caption.get('keep_tokens', 1))}",
        f"caption_dropout_rate = {float(caption.get('dropout_rate', 0))}",
        f"caption_tag_dropout_rate = {float(caption.get('tag_dropout_rate', 0))}",
        f"token_warmup_step = {int(caption.get('token_warmup_step', 0))}",
        "",
        "[[datasets]]",
        f"resolution = [{int(resolution.get('default', 1024))}, {int(resolution.get('default', 1024))}]",
        f"batch_size = {int(training.get('batch_size', 1))}",
        f"enable_bucket = {_toml_value(bool(resolution.get('enable_bucket', True)))}",
        f"bucket_no_upscale = {_toml_value(bool(resolution.get('bucket_no_upscale', True)))}",
        f"bucket_reso_steps = {int(resolution.get('bucket_reso_steps', 32))}",
    ]
    for dataset_dir in dataset_dirs:
        lines.extend(
            [
                "",
                "  [[datasets.subsets]]",
                f"  image_dir = {_toml_value(str(dataset_dir))}",
                "  num_repeats = 1",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _supported_metadata_parameters(sd_scripts: Path, entrypoint: Path) -> set[str]:
    """Discover ModelSpec arguments when a checkout is available.

    The repository records a pinned sd-scripts commit, but deployments keep
    that checkout outside this repository.  Newer sd-scripts versions expose
    ``metadata_*`` arguments; older checkouts need the post-processing fallback
    below.  If no source files are available (for example a config-only test
    fixture), retain the modern default so train.toml remains forward
    compatible.
    """

    all_names = {
        "metadata_title",
        "metadata_author",
        "metadata_description",
        "metadata_license",
        "metadata_tags",
        "metadata_usage_hint",
        "metadata_thumbnail",
        "metadata_merged_from",
        "metadata_trigger_phrase",
    }
    candidates = [
        entrypoint,
        sd_scripts / "train_network.py",
        sd_scripts / "sdxl_train_network.py",
        sd_scripts / "library" / "sai_model_spec.py",
    ]
    existing: list[Path] = []
    seen_paths: set[Path] = set()
    for path in candidates:
        if path in seen_paths or not path.is_file():
            continue
        seen_paths.add(path)
        existing.append(path)
    if len(existing) <= 1 and existing and existing[0] == entrypoint:
        try:
            text = entrypoint.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return all_names
        if "metadata_" not in text:
            return all_names
    snippets: list[str] = []
    for path in existing:
        try:
            snippets.append(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    combined = "\n".join(snippets)
    if not combined or "metadata_" not in combined:
        return set()
    return {name for name in all_names if name in combined}


def _postprocess_checkpoint_metadata(
    checkpoints: Sequence[Path], metadata: ModelArtifactMetadata
) -> None:
    updates = build_modelspec_metadata(metadata)
    if not updates:
        return
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - validated training env owns dependency
        raise PipelineError("safetensors is required to attach LoRA model metadata") from exc
    for checkpoint in checkpoints:
        try:
            with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
                current = dict(handle.metadata() or {})
        except Exception as exc:
            raise PipelineError(f"Could not inspect checkpoint metadata {checkpoint}: {exc}") from exc
        if all(current.get(key) == value for key, value in updates.items()):
            continue
        rewrite_safetensors_metadata(checkpoint, updates)


def materialize_dataset_snapshot(
    project_dir: Path, target: Path
) -> tuple[int, str, str, tuple[Path, ...]]:
    generation = load_current_generation(project_dir)
    manifest = generation.manifest
    target.mkdir(parents=True, exist_ok=False)
    snapshot_records: list[dict[str, Any]] = []
    subset_dirs: set[Path] = set()
    for record in manifest.get("images", []):
        image = generation.root / record["image"]
        caption = generation.root / record["caption"]
        relative = Path(record["source"])
        if not image.is_file() or not caption.is_file():
            raise PipelineError(f"Prepared generation is incomplete for {relative.as_posix()}")
        # sd-scripts pairs captions by stem. Include the source extension in the
        # snapshot stem so foo.jpg and foo.png cannot fight over foo.txt.
        unique_name = f"{relative.stem}__{relative.suffix.lower().lstrip('.')}" + relative.suffix.lower()
        image_target = target / relative.parent / unique_name
        caption_target = image_target.with_suffix(".txt")
        image_target.parent.mkdir(parents=True, exist_ok=True)
        subset_dirs.add(image_target.parent)
        _link_or_copy_immutable(image, image_target)
        _link_or_copy_immutable(caption, caption_target)
        snapshot_records.append(
            {
                "path": relative.as_posix(),
                "generation_id": generation.generation_id,
                "image_bytes": image.stat().st_size,
                "image_sha256": record.get("source_image_sha256") or sha256_file(image),
                "caption_sha256": record.get("caption_sha256") or sha256_file(caption),
                "caption": caption.read_text(encoding="utf-8", errors="replace").strip(),
            }
        )
    if not snapshot_records:
        raise PipelineError("Prepared dataset is empty")
    captions_hash = stable_hash(
        [{"path": record["path"], "caption": record["caption"]} for record in snapshot_records]
    )
    dataset_hash = stable_hash(snapshot_records)
    write_json_atomic(
        target / "snapshot-manifest.json",
        {
            "schema_version": 3,
            "prepared_generation": generation.generation_id,
            "dataset_snapshot_hash": dataset_hash,
            "captions_hash": captions_hash,
            "subset_directories": [
                subset.relative_to(target).as_posix()
                for subset in sorted(
                    subset_dirs,
                    key=lambda item: item.relative_to(target).as_posix().casefold(),
                )
            ],
            "images": snapshot_records,
        },
    )
    ordered_subset_dirs = tuple(
        sorted(
            subset_dirs,
            key=lambda item: item.relative_to(target).as_posix().casefold(),
        )
    )
    return len(snapshot_records), dataset_hash, captions_hash, ordered_subset_dirs


def _link_or_copy_immutable(source: Path, destination: Path) -> None:
    """Hard-link only from an immutable generation; never create mutable symlinks."""

    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def run_command_tee(
    command: list[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    log_path: Path,
    verbose: int = 0,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        log.write("COMMAND " + json.dumps(command, ensure_ascii=False) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                if verbose:
                    print(line, end="")
            return process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise


def _lease_command(value: Any, *, name: str) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not value or not all(
        isinstance(part, str) and part for part in value
    ):
        raise PipelineError(f"gpu_lease.{name} must be a non-empty JSON string array")
    return list(value)


class CommandGpuLease(AbstractContextManager["CommandGpuLease"]):
    """Optional host-provided GPU reservation using shell-free commands."""

    def __init__(self, acquire_command: list[str], release_command: list[str]):
        self.acquire_command = acquire_command
        self.release_command = release_command
        self.acquired = False

    @staticmethod
    def _run(command: list[str], *, action: str) -> None:
        try:
            result = subprocess.run(command, check=False, capture_output=True, text=True)
        except OSError as exc:
            raise PipelineError(f"GPU lease {action} command could not start: {exc}") from exc
        if result.returncode:
            detail = (result.stderr or result.stdout or "no output").strip()
            raise PipelineError(
                f"GPU lease {action} command failed with exit {result.returncode}: {detail}"
            )

    def __enter__(self) -> "CommandGpuLease":
        gpu_resources.release_inprocess_gpu_resources()
        self._run(self.acquire_command, action="acquire")
        self.acquired = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if not self.acquired:
            return None
        self.acquired = False
        try:
            self._run(self.release_command, action="release")
        except PipelineError as release_error:
            if exc_type is None:
                raise
            warnings.warn(str(release_error), RuntimeWarning, stacklevel=2)
        return None


def gpu_lease_from_info(
    info: Mapping[str, Any], *, enabled: bool = True
) -> AbstractContextManager[Any]:
    if not enabled or not info.get("gpu_lease"):
        return NullGpuLease()
    config = info["gpu_lease"]
    if not isinstance(config, Mapping):
        raise PipelineError("gpu_lease must be a JSON object")
    acquire = _lease_command(config.get("acquire_command"), name="acquire_command")
    release = _lease_command(config.get("release_command"), name="release_command")
    if acquire is None or release is None:
        raise PipelineError("gpu_lease requires both acquire_command and release_command")
    return CommandGpuLease(acquire, release)


class NullGpuLease(AbstractContextManager["NullGpuLease"]):
    def __enter__(self) -> "NullGpuLease":
        gpu_resources.release_inprocess_gpu_resources()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        return None


class GpuMonitor(AbstractContextManager["GpuMonitor"]):
    def __init__(self, path: Path):
        self.path = path
        self.process: subprocess.Popen[str] | None = None
        self.handle: TextIO | None = None

    def __enter__(self) -> "GpuMonitor":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", encoding="utf-8")
        try:
            self.process = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,memory.used,utilization.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                    "--loop-ms=500",
                ],
                stdout=self.handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except OSError as exc:
            self.handle.write(f"monitor unavailable: {exc}\n")
            self.handle.flush()
            self.process = None
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self.handle is not None:
            self.handle.close()

    def summary(self) -> dict[str, float | int | None]:
        memory: list[float] = []
        utilization: list[float] = []
        power: list[float] = []
        if not self.path.exists():
            return {"samples": 0, "peak_vram_mib": None, "mean_gpu_utilization": None, "mean_power_w": None}
        for line in self.path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 4:
                continue
            try:
                memory.append(float(parts[1]))
                utilization.append(float(parts[2]))
                power.append(float(parts[3]))
            except ValueError:
                continue
        return {
            "samples": len(memory),
            "peak_vram_mib": int(max(memory)) if memory else None,
            "mean_gpu_utilization": round(sum(utilization) / len(utilization), 3)
            if utilization
            else None,
            "mean_power_w": round(sum(power) / len(power), 3) if power else None,
        }


class SdScriptsTrainer(TrainerBackend):
    def __init__(self, *, root: Path | None = None, use_gpu_lease: bool = True):
        self.root = root or repository_root()
        self.use_gpu_lease = use_gpu_lease

    def train(
        self, request: TrainingRequest, *, dry_run: bool = False, verbose: int = 0
    ) -> TrainingResult:
        info = json.loads(
            (self.root / "environment" / "environment-info.json").read_text(encoding="utf-8")
        )
        python = Path(info["python_path"])
        accelerate = python.parent / "accelerate"
        sd_scripts = Path(info["sd_scripts_path"])
        entrypoint = sd_scripts / "sdxl_train_network.py"
        for required in (python, accelerate, entrypoint):
            if not required.exists():
                raise PipelineError(f"Validated training component is missing: {required}")

        merged = request.config.merged
        training = merged.get("training", {})
        precision = merged.get("precision", {})
        attention = merged.get("attention", {})
        memory = merged.get("memory", {})
        data_loader = merged.get("data_loader", {})
        storage = merged.get("storage", {})

        persistent_config = request.run_dir / "config"
        persistent_checkpoints = request.run_dir / "checkpoints"
        persistent_logs = request.run_dir / "logs"
        for path in (
            persistent_config,
            persistent_checkpoints,
            persistent_logs,
            request.run_dir / "samples",
            request.run_dir / "metrics",
        ):
            path.mkdir(parents=True, exist_ok=True)

        scratch_value = storage.get("scratch_root")
        if scratch_value:
            work_root = (
                Path(str(scratch_value)).expanduser()
                / "lora-pipeline"
                / request.project_dir.name
                / request.run_dir.name
            )
            work_root.mkdir(parents=True, exist_ok=True)
        else:
            work_root = request.run_dir
        work_dataset = work_root / "dataset"
        work_checkpoints = work_root / "checkpoints"
        work_logs = work_root / "logs"
        if work_dataset.exists():
            shutil.rmtree(work_dataset)
        work_checkpoints.mkdir(parents=True, exist_ok=True)
        work_logs.mkdir(parents=True, exist_ok=True)

        image_count, dataset_hash, captions_hash, dataset_dirs = (
            materialize_dataset_snapshot(request.project_dir, work_dataset)
        )
        shutil.copy2(work_dataset / "snapshot-manifest.json", persistent_config / "dataset-snapshot.json")
        snapshot_payload = json.loads(
            (work_dataset / "snapshot-manifest.json").read_text(encoding="utf-8")
        )
        model_metadata = resolve_model_metadata(
            merged,
            run_dir=request.run_dir,
            captions=snapshot_payload.get("images", []),
            samples_dir=request.run_dir / "samples",
        )

        target_candidates = max(
            1, int(merged.get("checkpoints", {}).get("target_candidates", 5))
        )
        save_interval = max(1, math.ceil(request.optimizer_steps / target_candidates))
        save_state = bool(training.get("save_state", True))
        train_values: dict[str, Any] = {
            "pretrained_model_name_or_path": str(request.base.path),
            "network_module": training.get("network_module", "networks.lora"),
            "network_train_unet_only": bool(training.get("network_train_unet_only", True)),
            "network_dim": int(training.get("network_dim", 16)),
            "network_alpha": int(training.get("network_alpha", 8)),
            "optimizer_type": training.get("optimizer", "AdamW8bit"),
            "learning_rate": float(training.get("unet_lr", 0.0001)),
            "unet_lr": float(training.get("unet_lr", 0.0001)),
            "lr_scheduler": training.get("lr_scheduler", "constant"),
            "max_grad_norm": float(training.get("max_grad_norm", 1.0)),
            "mixed_precision": precision.get("mixed_precision", "fp16"),
            "save_precision": precision.get("save_precision", "fp16"),
            "no_half_vae": bool(precision.get("no_half_vae", True)),
            "sdpa": attention.get("backend") == "sdpa",
            "gradient_checkpointing": bool(memory.get("gradient_checkpointing", True)),
            "cache_latents": bool(training.get("cache_latents", True)),
            "cache_latents_to_disk": bool(training.get("cache_latents_to_disk", True)),
            "cache_text_encoder_outputs": bool(
                training.get("cache_text_encoder_outputs", False)
            ),
            "cache_text_encoder_outputs_to_disk": bool(
                training.get("cache_text_encoder_outputs_to_disk", False)
            ),
            "max_data_loader_n_workers": int(data_loader.get("workers", 0)),
            "persistent_data_loader_workers": bool(
                data_loader.get("persistent_workers", False)
            ),
            "max_train_steps": request.optimizer_steps,
            "gradient_accumulation_steps": int(
                training.get("gradient_accumulation_steps", 1)
            ),
            "max_token_length": int(
                merged.get("caption", {}).get(
                    "max_token_length",
                    request.config.hardware.get("caption", {}).get(
                        "default_max_token_length", 75
                    ),
                )
            ),
            "seed": int(training.get("seed", 42)),
            "save_model_as": "safetensors",
            "save_every_n_steps": save_interval,
            "save_state": save_state,
            "resume": str(request.resume_state) if request.resume_state else None,
        }
        metadata_parameters = build_sd_scripts_metadata(model_metadata)
        supported_metadata = _supported_metadata_parameters(sd_scripts, entrypoint)
        train_values.update(
            {
                key: value
                for key, value in metadata_parameters.items()
                if key in supported_metadata
            }
        )
        write_flat_toml(persistent_config / "train.toml", train_values)
        write_dataset_toml(
            persistent_config / "dataset.toml", dataset_dirs=dataset_dirs, merged=merged
        )
        output_name = _safe_output_name(
            request.project_dir.name,
            request.base.id,
            int(training.get("network_dim", 16)),
        )
        command = [
            str(accelerate),
            "launch",
            "--num_processes",
            "1",
            "--num_machines",
            "1",
            "--num_cpu_threads_per_process",
            str(max(1, int(data_loader.get("cpu_threads_per_process", 1)))),
            "--mixed_precision",
            str(precision.get("mixed_precision", "fp16")),
            str(entrypoint),
            "--config_file",
            str(persistent_config / "train.toml"),
            "--dataset_config",
            str(persistent_config / "dataset.toml"),
            "--output_dir",
            str(work_checkpoints),
            "--output_name",
            output_name,
            "--logging_dir",
            str(work_logs / "tensorboard"),
        ]
        physical_batch = int(training.get("batch_size", 1))
        accumulation = int(training.get("gradient_accumulation_steps", 1))
        effective_batch = physical_batch * accumulation
        actual_images_seen = request.optimizer_steps * effective_batch
        accounting = {
            "dataset_images": image_count,
            "dataset_snapshot_hash": dataset_hash,
            "captions_hash": captions_hash,
            "physical_batch": physical_batch,
            "gradient_accumulation": accumulation,
            "effective_batch": effective_batch,
            "optimizer_steps": request.optimizer_steps,
            "target_images_seen": request.target_images_seen,
            "images_seen": actual_images_seen,
            "exposure_rounding_overhead": actual_images_seen - request.target_images_seen,
            "epochs": round(actual_images_seen / image_count, 6),
            "save_every_n_steps": save_interval,
        }
        metadata = {
            "schema_version": 2,
            "run_id": request.run_dir.name,
            "created_at": datetime.now(UTC).isoformat(),
            "base": {
                "id": request.base.id,
                "filename": request.base.path.name,
                "path": str(request.base.path),
                "sha256": request.base.sha256,
            },
            "sd_scripts_commit": info["sd_scripts_commit"],
            "environment": info,
            "profiles": {
                "hardware": request.config.hardware.get("id"),
                "concept": request.config.concept.get("id"),
                "training": request.config.training.get("id"),
            },
            "accounting": accounting,
            "command": command,
            "config_hash": stable_hash({"train": train_values, "merged": merged}),
            "pipeline_git_commit": _git_commit(self.root),
            "cli_command": list(request.command_line),
            "storage": {
                "persistent_run_dir": str(request.run_dir),
                "work_root": str(work_root),
                "scratch_enabled": work_root != request.run_dir,
                "dataset_dir": str(work_dataset),
                "checkpoint_dir": str(work_checkpoints),
                "log_dir": str(work_logs),
            },
            "resume_state": str(request.resume_state) if request.resume_state else None,
            "model_metadata": model_metadata.as_run_dict(),
            "metadata_delivery": {
                "native_parameters": sorted(
                    key for key in metadata_parameters if key in supported_metadata
                ),
                "postprocess_parameters": sorted(
                    key for key in metadata_parameters if key not in supported_metadata
                ),
            },
        }
        write_json_atomic(persistent_config / "run-metadata.json", metadata)
        shutil.copy2(
            self.root / "environment" / "environment-info.json",
            persistent_config / "environment-info.json",
        )
        if dry_run:
            return TrainingResult(
                run_id=request.run_dir.name,
                run_dir=request.run_dir,
                checkpoints=(),
                accounting=accounting,
                metrics={"command": command, "config_only": True},
                dry_run=True,
            )

        environment = os.environ.copy()
        environment.update(
            {
                "HF_HOME": str(self.root / ".cache" / "huggingface"),
                "HUGGINGFACE_HUB_CACHE": str(
                    self.root / ".cache" / "huggingface" / "hub"
                ),
                "TRANSFORMERS_CACHE": str(
                    self.root / ".cache" / "huggingface" / "transformers"
                ),
                "PYTHONUNBUFFERED": "1",
            }
        )
        lease = gpu_lease_from_info(info, enabled=self.use_gpu_lease)
        monitor = GpuMonitor(work_logs / "gpu-monitor.csv")
        started = time.monotonic()
        try:
            with lease, monitor:
                exit_code = run_command_tee(
                    command,
                    cwd=sd_scripts,
                    env=environment,
                    log_path=work_logs / "train.log",
                    verbose=verbose,
                )
        finally:
            if work_root != request.run_dir:
                _sync_tree(work_logs, persistent_logs)
                _sync_tree(work_checkpoints, persistent_checkpoints)
        elapsed = round(time.monotonic() - started, 3)
        checkpoint_source = persistent_checkpoints
        checkpoints = sorted(
            checkpoint_source.glob("*.safetensors"),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
        )
        if exit_code == 0:
            # Samples may only be generated after training. Resolve once more
            # so a deterministic sample preview can be embedded without
            # claiming it was present in the training command. Explicit config
            # metadata remains the same on this second pass.
            model_metadata = resolve_model_metadata(
                merged,
                run_dir=request.run_dir,
                captions=snapshot_payload.get("images", []),
                samples_dir=request.run_dir / "samples",
            )
            _postprocess_checkpoint_metadata(checkpoints, model_metadata)
            metadata["metadata_delivery"]["postprocess_applied"] = sorted(
                build_modelspec_metadata(model_metadata)
            )
        gpu_metrics = monitor.summary()
        metrics = {
            "exit_code": exit_code,
            "elapsed_seconds": elapsed,
            **gpu_metrics,
            "checkpoint_count": len(checkpoints),
            "config_hash": metadata["config_hash"],
            "mean_seconds_per_optimizer_step": round(elapsed / request.optimizer_steps, 6),
            "images_per_second": round(actual_images_seen / elapsed, 6) if elapsed else None,
            "storage": metadata["storage"],
            "resume_states": [
                str(path)
                for path in sorted(
                    persistent_checkpoints.glob("*-state"),
                    key=lambda item: item.stat().st_mtime_ns,
                )
                if path.is_dir()
            ],
        }
        _retain_latest_state(
            persistent_checkpoints,
            enabled=str(training.get("state_retention", "latest")) == "latest",
        )
        metadata["model_metadata"] = model_metadata.as_run_dict()
        metadata["result"] = {**metrics, "checkpoints": [str(path) for path in checkpoints]}
        write_json_atomic(persistent_config / "run-metadata.json", metadata)
        if exit_code or not checkpoints:
            raise ExternalCommandError(
                f"sd-scripts training failed (exit={exit_code}, checkpoints={len(checkpoints)})",
                exit_code=exit_code or 1,
                log_path=persistent_logs / "train.log",
            )
        return TrainingResult(
            run_id=request.run_dir.name,
            run_dir=request.run_dir,
            checkpoints=checkpoints,
            accounting=accounting,
            metrics=metrics,
        )


def _sync_tree(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = destination / item.name
        temporary = destination / f".{item.name}.syncing"
        if temporary.exists():
            if temporary.is_dir():
                shutil.rmtree(temporary)
            else:
                temporary.unlink()
        if item.is_dir():
            shutil.copytree(item, temporary)
            if target.exists():
                shutil.rmtree(target)
            os.replace(temporary, target)
        else:
            shutil.copy2(item, temporary)
            os.replace(temporary, target)


def _retain_latest_state(checkpoints_dir: Path, *, enabled: bool) -> None:
    if not enabled:
        return
    states = sorted(
        (path for path in checkpoints_dir.glob("*-state") if path.is_dir()),
        key=lambda path: path.stat().st_mtime_ns,
    )
    for stale in states[:-1]:
        shutil.rmtree(stale, ignore_errors=True)


def _safe_output_name(project: str, base_id: str, rank: int) -> str:
    project_short = "".join(
        character for character in project.lower() if character.isalnum() or character in "-_"
    )[:32]
    base_short = "".join(part[:8] for part in base_id.split("_")[:2])[:16]
    return f"{project_short}__{base_short}__r{rank}"


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "uncommitted"
