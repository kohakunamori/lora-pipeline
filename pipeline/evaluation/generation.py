from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..config import repository_root, write_json_atomic
from ..models import ExternalCommandError, GeneratedImage, GenerationCase, PipelineError
from ..trainer.sd_scripts import gpu_lease_from_info, run_command_tee


class GenerationBackend(ABC):
    @abstractmethod
    def generate(
        self,
        cases: Sequence[GenerationCase],
        *,
        base_path: Path,
        output_dir: Path,
        settings: Mapping[str, Any],
        verbose: int = 0,
    ) -> list[GeneratedImage]:
        raise NotImplementedError


class SdScriptsGenerator(GenerationBackend):
    def __init__(self, *, root: Path | None = None, use_gpu_lease: bool = True):
        self.root = root or repository_root()
        self.use_gpu_lease = use_gpu_lease

    def generate(
        self,
        cases: Sequence[GenerationCase],
        *,
        base_path: Path,
        output_dir: Path,
        settings: Mapping[str, Any],
        verbose: int = 0,
    ) -> list[GeneratedImage]:
        if not cases:
            return []
        info = json.loads((self.root / "environment" / "environment-info.json").read_text(encoding="utf-8"))
        python = Path(info["python_path"])
        sd_scripts = Path(info["sd_scripts_path"])
        entrypoint = sd_scripts / "sdxl_gen_img.py"
        if not python.exists() or not entrypoint.exists():
            raise PipelineError("The validated sd-scripts generation environment is unavailable")
        output_dir.mkdir(parents=True, exist_ok=True)
        logs = output_dir.parent / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        groups: dict[Path, list[GenerationCase]] = defaultdict(list)
        for case in cases:
            groups[case.checkpoint].append(case)
        generated: list[GeneratedImage] = []
        summary_log = logs / "generation.log"
        summary_log.write_text("", encoding="utf-8")
        lease = gpu_lease_from_info(info, enabled=self.use_gpu_lease)
        with lease:
            for index, (checkpoint, checkpoint_cases) in enumerate(groups.items(), start=1):
                checkpoint_dir = output_dir / f"{index:02d}-{_slug(checkpoint.stem)}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                prompt_file = checkpoint_dir / "prompts.txt"
                prompt_file.write_text(
                    "\n".join(_prompt_line(case) for case in checkpoint_cases) + "\n",
                    encoding="utf-8",
                )
                command = [
                    str(python),
                    str(entrypoint),
                    "--ckpt",
                    str(base_path),
                    "--network_module",
                    "networks.lora",
                    "--network_weights",
                    str(checkpoint),
                    "--from_file",
                    str(prompt_file),
                    "--outdir",
                    str(checkpoint_dir),
                    "--sequential_file_name",
                    "--images_per_prompt",
                    "1",
                    "--batch_size",
                    "1",
                    "--W",
                    str(int(settings.get("width", 1024))),
                    "--H",
                    str(int(settings.get("height", 1024))),
                    "--steps",
                    str(int(settings.get("steps", 28))),
                    "--sampler",
                    str(settings.get("sampler", "euler_a")),
                    "--scale",
                    str(float(settings.get("cfg", 4.5))),
                    "--fp16",
                    "--sdpa",
                    "--no_half_vae",
                    "--vae_batch_size",
                    "1",
                    "--console_log_simple",
                ]
                environment = os.environ.copy()
                environment.update(
                    {
                        "HF_HOME": str(self.root / ".cache" / "huggingface"),
                        "HUGGINGFACE_HUB_CACHE": str(self.root / ".cache" / "huggingface" / "hub"),
                        "TRANSFORMERS_CACHE": str(self.root / ".cache" / "huggingface" / "transformers"),
                        "PYTHONUNBUFFERED": "1",
                    }
                )
                log_path = logs / f"generation-{index:02d}.log"
                exit_code = run_command_tee(
                    command,
                    cwd=sd_scripts,
                    env=environment,
                    log_path=log_path,
                    verbose=verbose,
                )
                with summary_log.open("a", encoding="utf-8") as summary:
                    summary.write(f"checkpoint={checkpoint} exit_code={exit_code} log={log_path}\n")
                if exit_code:
                    raise ExternalCommandError(
                        f"Image generation failed for {checkpoint.name}",
                        exit_code=exit_code,
                        log_path=log_path,
                    )
                images = sorted(
                    (path for path in checkpoint_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}),
                    key=lambda path: path.name,
                )
                if len(images) != len(checkpoint_cases):
                    raise ExternalCommandError(
                        f"Generation produced {len(images)} image(s), expected {len(checkpoint_cases)}",
                        exit_code=1,
                        log_path=log_path,
                    )
                generated.extend(
                    GeneratedImage(case=case, path=image)
                    for case, image in zip(checkpoint_cases, images, strict=True)
                )
        manifest = [
            {
                "path": str(item.path),
                "checkpoint": str(item.case.checkpoint),
                "checkpoint_label": item.case.checkpoint_label,
                "strength": item.case.strength,
                "prompt_id": item.case.prompt_id,
                "prompt": item.case.prompt,
                "negative_prompt": item.case.negative_prompt,
                "seed": item.case.seed,
                "contains_trigger": item.case.contains_trigger,
            }
            for item in generated
        ]
        write_json_atomic(output_dir / "generation-manifest.json", {"schema_version": 1, "images": manifest})
        return generated


def _prompt_line(case: GenerationCase) -> str:
    # Per-line sd-scripts prompt arguments keep seed/strength aligned across checkpoints.
    prompt = case.prompt.replace("\n", " ").strip()
    negative = case.negative_prompt.replace("\n", " ").strip()
    return f"{prompt} --d {case.seed} --am {case.strength} --n {negative}"


def _slug(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "-_" else "-" for character in value)
    return cleaned[:60] or "checkpoint"
