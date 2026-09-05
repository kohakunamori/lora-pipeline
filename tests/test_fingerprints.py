from __future__ import annotations

import shutil
from pathlib import Path

import yaml
from PIL import Image

from pipeline.config import repository_root, sha256_file
from pipeline.fingerprints import compute_step_signature
from pipeline.materialization import run as materialize
from pipeline.state import ProjectState


def _repo(tmp_path, monkeypatch) -> tuple[Path, ProjectState]:
    source = repository_root()
    root = tmp_path / "repo"
    shutil.copytree(source / "profiles", root / "profiles")
    (root / "bases").mkdir(parents=True)
    (root / "projects").mkdir()
    base_a = root / "base-a.safetensors"
    base_b = root / "base-b.safetensors"
    base_a.write_bytes(b"base-a")
    base_b.write_bytes(b"base-b")
    registry = {
        "bases": {
            "base_a": {
                "name": "Base A",
                "path": str(base_a),
                "family": "illustrious_sdxl",
                "prediction_type": "epsilon",
                "sha256": sha256_file(base_a),
                "enabled": True,
            },
            "base_b": {
                "name": "Base B",
                "path": str(base_b),
                "family": "illustrious_sdxl",
                "prediction_type": "epsilon",
                "sha256": sha256_file(base_b),
                "enabled": True,
            },
        }
    }
    (root / "bases" / "registry.yaml").write_text(
        yaml.safe_dump(registry), encoding="utf-8"
    )
    monkeypatch.setenv("LORA_PIPELINE_ROOT", str(root))
    state = ProjectState.create(
        root / "projects" / "fingerprint",
        name="fingerprint",
        concept_type="character",
        base="base_a",
        trigger="zz_fp",
        strategy="quality",
    )
    image = state.project_dir / "raw" / "sample.png"
    Image.new("RGB", (64, 64), "red").save(image)
    image.with_suffix(".txt").write_text("zz_fp, portrait\n", encoding="utf-8")
    materialize(state)
    return root, state


def _add_run(state: ProjectState, run_id: str, checkpoint_bytes: bytes) -> Path:
    checkpoint = (
        state.project_dir / "runs" / run_id / "checkpoints" / "candidate.safetensors"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(checkpoint_bytes)
    state.payload["runs"].append(
        {
            "id": run_id,
            "path": str(checkpoint.parents[1]),
            "status": "trained",
            "checkpoints": [str(checkpoint)],
            "accounting": {"images_seen": 100},
        }
    )
    state.save()
    return checkpoint


def test_raw_and_caption_changes_affect_materialization_fingerprint(tmp_path, monkeypatch) -> None:
    _, state = _repo(tmp_path, monkeypatch)
    options = {"caption_mode": "existing_passthrough"}
    before = compute_step_signature(state, "materialize", options=options)

    (state.project_dir / "raw" / "sample.txt").write_text(
        "zz_fp, full body\n", encoding="utf-8"
    )
    after_caption = compute_step_signature(state, "materialize", options=options)
    assert after_caption != before

    Image.new("RGB", (64, 64), "blue").save(state.project_dir / "raw" / "sample.png")
    assert compute_step_signature(state, "materialize", options=options) != after_caption


def test_base_and_training_profile_changes_invalidate_only_relevant_inputs(
    tmp_path, monkeypatch
) -> None:
    _, state = _repo(tmp_path, monkeypatch)
    preflight_before = compute_step_signature(state, "preflight")
    train_before = compute_step_signature(state, "train")
    materialize_before = compute_step_signature(state, "materialize")

    state.payload["project"]["base"] = "base_b"
    state.save()
    assert compute_step_signature(state, "preflight") != preflight_before
    assert compute_step_signature(state, "train") != train_before
    assert compute_step_signature(state, "materialize") == materialize_before

    train_after_base = compute_step_signature(state, "train")
    state.payload["project"]["overrides"] = {"training": {"network_dim": 32}}
    state.save()
    assert compute_step_signature(state, "train") != train_after_base
    assert compute_step_signature(state, "materialize") == materialize_before


def test_evaluation_config_does_not_change_project_fingerprints(tmp_path, monkeypatch) -> None:
    _, state = _repo(tmp_path, monkeypatch)
    materialize_before = compute_step_signature(state, "materialize")
    preflight_before = compute_step_signature(state, "preflight")
    train_before = compute_step_signature(state, "train")

    state.payload["project"]["overrides"] = {
        "evaluation": {"screening_prompts": ["portrait", "night"]}
    }
    state.save()

    assert compute_step_signature(state, "materialize") == materialize_before
    assert compute_step_signature(state, "preflight") == preflight_before
    assert compute_step_signature(state, "train") == train_before


def test_validation_changes_do_not_change_project_fingerprints(tmp_path, monkeypatch) -> None:
    _, state = _repo(tmp_path, monkeypatch)
    materialize_before = compute_step_signature(state, "materialize")
    preflight_before = compute_step_signature(state, "preflight")
    train_before = compute_step_signature(state, "train")

    validation = state.project_dir / "validation" / "holdout.png"
    Image.new("RGB", (64, 64), "green").save(validation)

    assert compute_step_signature(state, "materialize") == materialize_before
    assert compute_step_signature(state, "preflight") == preflight_before
    assert compute_step_signature(state, "train") == train_before


def test_results_history_is_not_part_of_project_fingerprint(tmp_path, monkeypatch) -> None:
    _, state = _repo(tmp_path, monkeypatch)
    train_before = compute_step_signature(state, "train")
    _add_run(state, "run-1", b"lora-one")
    _add_run(state, "run-2", b"lora-two")
    assert compute_step_signature(state, "train") == train_before
