from __future__ import annotations

import json
from pathlib import Path

from pipeline.evaluation.report_samples import build_sample_history


def _sample(
    run_dir: Path,
    *,
    stage: str,
    checkpoint: Path,
    name: str,
    prompt_id: str,
    strength: float,
) -> dict[str, object]:
    cases = run_dir / "samples" / stage / "cases"
    cases.mkdir(parents=True, exist_ok=True)
    image = cases / f"{name}.png"
    image.write_bytes(b"image")
    return {
        "case_id": name,
        "path": str(image),
        "checkpoint": str(checkpoint),
        "checkpoint_label": checkpoint.stem,
        "strength": strength,
        "prompt_id": prompt_id,
        "prompt": f"zz_test, {prompt_id}",
        "negative_prompt": "bad",
        "seed": 42,
        "contains_trigger": True,
    }


def _write_manifest(run_dir: Path, stage: str, images: list[dict[str, object]]) -> None:
    path = run_dir / "samples" / stage / "generation-manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema_version": 2, "images": images}),
        encoding="utf-8",
    )


def test_full_report_history_keeps_screening_samples_for_every_exported_checkpoint(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    checkpoints = [run_dir / "checkpoints" / f"candidate-{index}.safetensors" for index in range(3)]
    for checkpoint in checkpoints:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"lora")

    screening = [
        _sample(
            run_dir,
            stage="screening",
            checkpoint=checkpoint,
            name=f"screening-{index}",
            prompt_id="portrait",
            strength=0.8,
        )
        for index, checkpoint in enumerate(checkpoints)
    ]
    _write_manifest(run_dir, "screening", screening)
    _write_manifest(
        run_dir,
        "full",
        [
            _sample(
                run_dir,
                stage="full",
                checkpoint=checkpoints[-1],
                name="full-finalist",
                prompt_id="full_body",
                strength=1.0,
            )
        ],
    )

    rendered, summary = build_sample_history(
        run_dir,
        checkpoints,
        include_stages=("screening", "full"),
    )

    assert summary["exported_checkpoints"] == 3
    assert summary["checkpoints_with_samples"] == 3
    assert summary["sample_images"] == 4
    assert summary["stages"] == ["screening", "full"]
    for checkpoint in checkpoints:
        assert checkpoint.name in rendered
    assert "samples/screening/cases/screening-0.png" in rendered
    assert "samples/screening/cases/screening-1.png" in rendered
    assert "samples/screening/cases/screening-2.png" in rendered
    assert "samples/full/cases/full-finalist.png" in rendered
    assert "Screening" in rendered
    assert "Full" in rendered


def test_exported_checkpoint_without_samples_is_visible_in_coverage(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    sampled = run_dir / "checkpoints" / "sampled.safetensors"
    missing = run_dir / "checkpoints" / "missing.safetensors"
    sampled.parent.mkdir(parents=True, exist_ok=True)
    sampled.write_bytes(b"lora")
    missing.write_bytes(b"lora")
    _write_manifest(
        run_dir,
        "screening",
        [
            _sample(
                run_dir,
                stage="screening",
                checkpoint=sampled,
                name="sampled",
                prompt_id="portrait",
                strength=0.8,
            )
        ],
    )

    rendered, summary = build_sample_history(
        run_dir,
        [sampled, missing],
        include_stages=("screening", "full"),
    )

    assert summary["exported_checkpoints"] == 2
    assert summary["checkpoints_with_samples"] == 1
    missing_coverage = next(
        item for item in summary["coverage"] if item["checkpoint"] == missing.name
    )
    assert missing_coverage["sample_count"] == 0
    assert missing_coverage["has_sample_evidence"] is False
    assert missing.name in rendered
    assert "no sample evidence" in rendered


def test_stage_filter_prevents_future_full_samples_from_leaking_into_screening_report(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "checkpoints" / "candidate.safetensors"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"lora")
    _write_manifest(
        run_dir,
        "screening",
        [
            _sample(
                run_dir,
                stage="screening",
                checkpoint=checkpoint,
                name="screening-only",
                prompt_id="portrait",
                strength=0.8,
            )
        ],
    )
    _write_manifest(
        run_dir,
        "full",
        [
            _sample(
                run_dir,
                stage="full",
                checkpoint=checkpoint,
                name="full-existing",
                prompt_id="full_body",
                strength=1.0,
            )
        ],
    )

    rendered, summary = build_sample_history(
        run_dir,
        [checkpoint],
        include_stages=("screening",),
    )

    assert summary["sample_images"] == 1
    assert summary["stages"] == ["screening"]
    assert "screening-only.png" in rendered
    assert "full-existing.png" not in rendered
