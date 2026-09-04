from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

from pipeline.config import repository_root, sha256_file, write_json_atomic
from pipeline.evaluation.generation import GenerationBackend
from pipeline.models import GeneratedImage, GenerationCase, TrainingRequest, TrainingResult
from pipeline.service import create_project, load_project, run_remaining
from pipeline.state import ProjectState, project_lock
from pipeline.steps import preflight as preflight_step
from pipeline.steps import promote
from pipeline.trainer.base import TrainerBackend


class FakeTrainer(TrainerBackend):
    def train(
        self, request: TrainingRequest, *, dry_run: bool = False, verbose: int = 0
    ) -> TrainingResult:
        del verbose
        for relative in ("config", "checkpoints", "logs", "samples", "metrics"):
            (request.run_dir / relative).mkdir(parents=True, exist_ok=True)
        (request.run_dir / "config" / "train.toml").write_text(
            f"max_train_steps = {request.optimizer_steps}\n", encoding="utf-8"
        )
        (request.run_dir / "config" / "dataset.toml").write_text("# fake\n", encoding="utf-8")
        write_json_atomic(
            request.run_dir / "config" / "run-metadata.json",
            {"sd_scripts_commit": "fake", "base_sha256": request.base.sha256},
        )
        checkpoint = request.run_dir / "checkpoints" / "candidate-000003.safetensors"
        if not dry_run:
            checkpoint.write_bytes(b"fake-lora")
        effective_batch = int(request.config.merged["training"]["batch_size"])
        actual_images_seen = request.optimizer_steps * effective_batch
        accounting = {
            "dataset_images": 3,
            "dataset_snapshot_hash": "fake-dataset",
            "captions_hash": "fake-captions",
            "physical_batch": effective_batch,
            "gradient_accumulation": 1,
            "effective_batch": effective_batch,
            "optimizer_steps": request.optimizer_steps,
            "target_images_seen": request.target_images_seen,
            "images_seen": actual_images_seen,
            "epochs": actual_images_seen / 3,
        }
        return TrainingResult(
            run_id=request.run_dir.name,
            run_dir=request.run_dir,
            checkpoints=() if dry_run else (checkpoint,),
            accounting=accounting,
            metrics={"fake": True, "config_hash": "fake-config"},
            dry_run=dry_run,
        )


class FakeGenerator(GenerationBackend):
    def generate(
        self,
        cases: list[GenerationCase],
        *,
        base_path: Path,
        output_dir: Path,
        settings: dict,
        verbose: int = 0,
    ) -> list[GeneratedImage]:
        del base_path, settings, verbose
        output_dir.mkdir(parents=True, exist_ok=True)
        generated = []
        # Deliberately reverse filesystem naming relative to case order. Evaluation
        # must bind metrics to case objects, never sorted filenames.
        for index, case in enumerate(cases):
            path = output_dir / f"reverse-{len(cases) - index:04d}.png"
            color = (40 + index % 180, 80 if case.contains_trigger else 30, 140)
            image = Image.new("RGB", (96, 96), color)
            ImageDraw.Draw(image).text((4, 4), case.case_id[-6:], fill="white")
            image.save(path)
            generated.append(GeneratedImage(case=case, path=path))
        return generated


def test_complete_style_pipeline_with_fake_backends(tmp_path, monkeypatch) -> None:
    source_root = repository_root()
    test_root = tmp_path / "repo"
    shutil.copytree(source_root / "profiles", test_root / "profiles")
    (test_root / "bases").mkdir(parents=True)
    (test_root / "projects").mkdir()
    base_path = test_root / "fake-base.safetensors"
    base_path.write_bytes(b"fake-base")
    registry = {
        "bases": {
            "fake_base": {
                "name": "Fake base",
                "path": str(base_path),
                "family": "illustrious_sdxl",
                "prediction_type": "epsilon",
                "sha256": sha256_file(base_path),
                "enabled": True,
                "generation_defaults": {"sampler": "euler_a", "cfg": 4.5, "steps": 2},
            }
        }
    }
    (test_root / "bases" / "registry.yaml").write_text(
        yaml.safe_dump(registry), encoding="utf-8"
    )
    monkeypatch.setenv("LORA_PIPELINE_ROOT", str(test_root))
    real_preflight = preflight_step.run
    monkeypatch.setattr(
        preflight_step,
        "run",
        lambda state: real_preflight(state, minimum_free_gib=0),
    )
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    for index, size in enumerate(((640, 832), (1024, 1024), (1216, 832))):
        path = dataset / f"image-{index}.png"
        Image.new("RGB", size, (index * 40, 80, 120)).save(path)
        path.with_suffix(".txt").write_text(
            ["1girl, portrait", "landscape, outdoors, day", "interior, complex_background"][index]
            + "\n",
            encoding="utf-8",
        )
    state = create_project(
        name="style-test",
        concept_type="style",
        base="fake_base",
        trigger="zz_style_test",
        strategy="quality",
        dataset=dataset,
        images_seen=3,
    )
    raw_before = {
        path.name: sha256_file(path) for path in (state.project_dir / "raw").glob("*.png")
    }
    results = run_remaining(
        state,
        skip={"dedup", "caption", "review"},
        trainer_backend=FakeTrainer(),
        generation_backend=FakeGenerator(),
    )
    assert [name for name, _ in results] == [
        "caption",
        "prepare",
        "preflight",
        "train",
        "evaluate",
    ]
    completed = load_project("style-test")
    assert completed.next_actionable_step() is None
    for step_name in ("inspect", "prepare", "preflight", "train", "evaluate"):
        assert completed.step(step_name).get("input_hash"), step_name
    raw_after = {
        path.name: sha256_file(path) for path in (state.project_dir / "raw").glob("*.png")
    }
    assert raw_after == raw_before
    run_dir = Path(completed.payload["runs"][-1]["path"])
    assert not (run_dir / "best.safetensors").exists()
    assert not (run_dir / "best.yaml").exists()
    assert (run_dir / "contact-sheet.jpg").is_file()
    assert (run_dir / "report.html").is_file()
    assert (run_dir / "contact-sheets" / "screening" / "trigger-leakage.jpg").is_file()
    assert (run_dir / "logs" / "pipeline.log").is_file()

    checkpoint = Path(completed.payload["runs"][-1]["checkpoints"][0])
    with project_lock(completed.project_dir):
        promoted = promote.run(
            ProjectState.load(completed.project_dir),
            run_id=run_dir.name,
            checkpoint_name=checkpoint.name,
            strength=0.7,
        )
    assert (run_dir / "best.safetensors").is_file()
    assert (run_dir / "best.yaml").is_file()
    assert promoted["base"]["sha256"] == sha256_file(base_path)
    assert promoted["selection"]["method"] == "manual"
    assert promoted["selection"]["recommended_strength"] == 0.7
