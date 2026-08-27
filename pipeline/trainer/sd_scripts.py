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
from typing import Any, Iterable, Mapping, TextIO

from ..config import repository_root, sha256_file, stable_hash, write_json_atomic
from ..models import ExternalCommandError, PipelineError, TrainingRequest, TrainingResult
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


def write_dataset_toml(path: Path, *, dataset_dir: Path, merged: Mapping[str, Any]) -> None:
    resolution = merged.get("resolution", {})
    caption = merged.get("caption", {})
    training = merged.get("training", {})
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
        "",
        "  [[datasets.subsets]]",
        f"  image_dir = {_toml_value(str(dataset_dir))}",
        "  num_repeats = 1",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def materialize_dataset_snapshot(project_dir: Path, target: Path) -> tuple[int, str, str]:
    manifest_path = project_dir / "prepared" / "manifest.json"
    if not manifest_path.exists():
        raise PipelineError("Prepared dataset manifest is missing; run prepare first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target.mkdir(parents=True, exist_ok=False)
    snapshot_records: list[dict[str, Any]] = []
    for record in manifest.get("images", []):
        image = project_dir / "prepared" / record["image"]
        caption = project_dir / "prepared" / record["caption"]
        relative = Path(record["source"])
        # sd-scripts pairs captions by stem. Include the source extension in the
        # snapshot stem so foo.jpg and foo.png cannot fight over foo.txt.
        unique_name = f"{relative.stem}__{relative.suffix.lower().lstrip('.')}" + relative.suffix.lower()
        image_target = target / relative.parent / unique_name
        caption_target = image_target.with_suffix(".txt")
        image_target.parent.mkdir(parents=True, exist_ok=True)
        _link_or_copy(image, image_target)
        _link_or_copy(caption, caption_target)
        snapshot_records.append(
            {
                "path": relative.as_posix(),
                "image_bytes": image.stat().st_size,
                "image_sha256": sha256_file(image),
                "caption": caption.read_text(encoding="utf-8", errors="replace").strip(),
            }
        )
    if not snapshot_records:
        raise PipelineError("Prepared dataset is empty")
    captions_hash = stable_hash(
        [{"path": record["path"], "caption": record["caption"]} for record in snapshot_records]
    )
    return len(snapshot_records), stable_hash(snapshot_records), captions_hash


def _link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        try:
            destination.symlink_to(source)
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
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            if verbose:
                print(line, end="")
        return process.wait()


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

    def peak_memory_mib(self) -> int | None:
        if not self.path.exists():
            return None
        peak: int | None = None
        for line in self.path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                value = int(float(parts[1].strip()))
            except ValueError:
                continue
            peak = value if peak is None else max(peak, value)
        return peak


class SdScriptsTrainer(TrainerBackend):
    def __init__(self, *, root: Path | None = None, use_gpu_lease: bool = True):
        self.root = root or repository_root()
        self.use_gpu_lease = use_gpu_lease

    def train(self, request: TrainingRequest, *, dry_run: bool = False, verbose: int = 0) -> TrainingResult:
        info = json.loads((self.root / "environment" / "environment-info.json").read_text(encoding="utf-8"))
        python = Path(info["python_path"])
        accelerate = python.parent / "accelerate"
        sd_scripts = Path(info["sd_scripts_path"])
        entrypoint = sd_scripts / "sdxl_train_network.py"
        for required in (python, accelerate, entrypoint):
            if not required.exists():
                raise PipelineError(f"Validated training component is missing: {required}")

        config_dir = request.run_dir / "config"
        checkpoints_dir = request.run_dir / "checkpoints"
        logs_dir = request.run_dir / "logs"
        for path in (config_dir, checkpoints_dir, logs_dir, request.run_dir / "samples", request.run_dir / "metrics"):
            path.mkdir(parents=True, exist_ok=True)
        dataset_dir = config_dir / "dataset"
        image_count, dataset_hash, captions_hash = materialize_dataset_snapshot(request.project_dir, dataset_dir)
        merged = request.config.merged
        training = merged.get("training", {})
        precision = merged.get("precision", {})
        attention = merged.get("attention", {})
        memory = merged.get("memory", {})
        caption = merged.get("caption", {})
        target_candidates = max(1, int(merged.get("checkpoints", {}).get("target_candidates", 5)))
        save_interval = max(1, math.ceil(request.optimizer_steps / target_candidates))
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
            "cache_text_encoder_outputs": bool(training.get("cache_text_encoder_outputs", False)),
            "cache_text_encoder_outputs_to_disk": bool(training.get("cache_text_encoder_outputs_to_disk", False)),
            "max_data_loader_n_workers": 0,
            "persistent_data_loader_workers": False,
            "max_train_steps": request.optimizer_steps,
            "gradient_accumulation_steps": int(training.get("gradient_accumulation_steps", 1)),
            "max_token_length": int(merged.get("caption", {}).get("max_token_length", request.config.hardware.get("caption", {}).get("default_max_token_length", 75))),
            "seed": int(training.get("seed", 42)),
            "save_model_as": "safetensors",
            "save_every_n_steps": save_interval,
            "save_state": False,
        }
        write_flat_toml(config_dir / "train.toml", train_values)
        write_dataset_toml(config_dir / "dataset.toml", dataset_dir=dataset_dir, merged=merged)
        output_name = _safe_output_name(request.project_dir.name, request.base.id, int(training.get("network_dim", 16)))
        command = [
            str(accelerate),
            "launch",
            "--num_processes",
            "1",
            "--num_machines",
            "1",
            "--num_cpu_threads_per_process",
            "1",
            "--mixed_precision",
            str(precision.get("mixed_precision", "fp16")),
            str(entrypoint),
            "--config_file",
            str(config_dir / "train.toml"),
            "--dataset_config",
            str(config_dir / "dataset.toml"),
            "--output_dir",
            str(checkpoints_dir),
            "--output_name",
            output_name,
            "--logging_dir",
            str(logs_dir / "tensorboard"),
        ]
        physical_batch = int(training.get("batch_size", 1))
        accumulation = int(training.get("gradient_accumulation_steps", 1))
        effective_batch = physical_batch * accumulation
        accounting = {
            "dataset_images": image_count,
            "dataset_snapshot_hash": dataset_hash,
            "captions_hash": captions_hash,
            "physical_batch": physical_batch,
            "gradient_accumulation": accumulation,
            "effective_batch": effective_batch,
            "optimizer_steps": request.optimizer_steps,
            "images_seen": request.optimizer_steps * effective_batch,
            "epochs": round(request.optimizer_steps * effective_batch / image_count, 6),
            "save_every_n_steps": save_interval,
        }
        metadata = {
            "schema_version": 1,
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
        }
        write_json_atomic(config_dir / "run-metadata.json", metadata)
        shutil.copy2(self.root / "environment" / "environment-info.json", config_dir / "environment-info.json")
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
                "HUGGINGFACE_HUB_CACHE": str(self.root / ".cache" / "huggingface" / "hub"),
                "TRANSFORMERS_CACHE": str(self.root / ".cache" / "huggingface" / "transformers"),
                "PYTHONUNBUFFERED": "1",
            }
        )
        lease = gpu_lease_from_info(info, enabled=self.use_gpu_lease)
        monitor = GpuMonitor(logs_dir / "gpu-monitor.csv")
        started = time.monotonic()
        with lease, monitor:
            exit_code = run_command_tee(
                command,
                cwd=sd_scripts,
                env=environment,
                log_path=logs_dir / "train.log",
                verbose=verbose,
            )
        elapsed = round(time.monotonic() - started, 3)
        checkpoints = sorted(checkpoints_dir.glob("*.safetensors"), key=lambda path: (path.stat().st_mtime_ns, path.name))
        metrics = {
            "exit_code": exit_code,
            "elapsed_seconds": elapsed,
            "peak_vram_mib": monitor.peak_memory_mib(),
            "checkpoint_count": len(checkpoints),
            "config_hash": metadata["config_hash"],
        }
        metadata["result"] = {**metrics, "checkpoints": [str(path) for path in checkpoints]}
        write_json_atomic(config_dir / "run-metadata.json", metadata)
        if exit_code or not checkpoints:
            raise ExternalCommandError(
                f"sd-scripts training failed (exit={exit_code}, checkpoints={len(checkpoints)})",
                exit_code=exit_code or 1,
                log_path=logs_dir / "train.log",
            )
        return TrainingResult(
            run_id=request.run_dir.name,
            run_dir=request.run_dir,
            checkpoints=checkpoints,
            accounting=accounting,
            metrics=metrics,
        )


def _safe_output_name(project: str, base_id: str, rank: int) -> str:
    project_short = "".join(character for character in project.lower() if character.isalnum() or character in "-_")[:32]
    base_short = "".join(part[:8] for part in base_id.split("_")[:2])[:16]
    return f"{project_short}__{base_short}__r{rank}"


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=False, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "uncommitted"
