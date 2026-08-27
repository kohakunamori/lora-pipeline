from __future__ import annotations

import shutil
from pathlib import Path

import yaml
from PIL import Image

from pipeline.config import repository_root, sha256_file
from pipeline.models import TrainingRequest, TrainingResult
from pipeline.state import ProjectState
from pipeline.steps import prepare, train
from pipeline.trainer.base import TrainerBackend


class CapturingTrainer(TrainerBackend):
    def __init__(self) -> None:
        self.request: TrainingRequest | None = None

    def train(
        self, request: TrainingRequest, *, dry_run: bool = False, verbose: int = 0
    ) -> TrainingResult:
        del verbose
        self.request = request
        checkpoint = request.run_dir / "checkpoints" / "resumed.safetensors"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if not dry_run:
            checkpoint.write_bytes(b"resumed")
        effective_batch = int(request.config.merged["training"]["batch_size"])
        return TrainingResult(
            run_id=request.run_dir.name,
            run_dir=request.run_dir,
            checkpoints=() if dry_run else (checkpoint,),
            accounting={
                "dataset_images": 1,
                "dataset_snapshot_hash": "dataset",
                "captions_hash": "captions",
                "physical_batch": effective_batch,
                "gradient_accumulation": 1,
                "effective_batch": effective_batch,
                "optimizer_steps": request.optimizer_steps,
                "target_images_seen": request.target_images_seen,
                "images_seen": request.optimizer_steps * effective_batch,
                "epochs": request.optimizer_steps * effective_batch,
            },
            metrics={"config_hash": "resumed-config"},
            dry_run=dry_run,
        )


def test_interrupted_run_resumes_from_latest_sd_scripts_state(tmp_path, monkeypatch) -> None:
    source_root = repository_root()
    root = tmp_path / "repo"
    shutil.copytree(source_root / "profiles", root / "profiles")
    (root / "bases").mkdir(parents=True)
    (root / "projects").mkdir()
    base = root / "base.safetensors"
    base.write_bytes(b"base")
    (root / "bases" / "registry.yaml").write_text(
        yaml.safe_dump(
            {
                "bases": {
                    "base": {
                        "name": "Base",
                        "path": str(base),
                        "family": "illustrious_sdxl",
                        "prediction_type": "epsilon",
                        "sha256": sha256_file(base),
                        "enabled": True,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LORA_PIPELINE_ROOT", str(root))
    state = ProjectState.create(
        root / "projects" / "resume",
        name="resume",
        concept_type="character",
        base="base",
        trigger="zz_resume",
        strategy="quality",
    )
    image = state.project_dir / "raw" / "sample.png"
    Image.new("RGB", (64, 64), "red").save(image)
    image.with_suffix(".txt").write_text("zz_resume, portrait\n", encoding="utf-8")
    prepare.run(state)

    run_dir = state.project_dir / "runs" / "run-1"
    old_state = run_dir / "checkpoints" / "000100-state"
    latest_state = run_dir / "checkpoints" / "000200-state"
    old_state.mkdir(parents=True)
    latest_state.mkdir()
    old_state.touch()
    latest_state.touch()
    state.payload["runs"].append(
        {
            "id": "run-1",
            "path": str(run_dir),
            "status": "interrupted",
            "checkpoints": [],
        }
    )
    state.payload["project"]["budget"] = {"unit": "images_seen", "value": 12}
    state.save()

    backend = CapturingTrainer()
    _, result = train.run(state, backend=backend, resume_run="run-1")
    assert result.run_id == "run-1"
    assert backend.request is not None
    assert backend.request.resume_state == latest_state
    assert backend.request.target_images_seen == 12
    reloaded = ProjectState.load(state.project_dir)
    run_record = reloaded.payload["runs"][0]
    assert run_record["status"] == "trained"
    assert run_record["resume_state"] == str(latest_state)
    assert run_record["resolved_budget"]["target_images_seen"] == 12
