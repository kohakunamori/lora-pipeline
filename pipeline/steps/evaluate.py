from __future__ import annotations

import html
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import load_base_registry, resolve_profiles, stable_hash, write_json_atomic, write_yaml_atomic
from ..dataset.caption_cleaner import parse_caption
from ..dataset.style import distribution_summary
from ..evaluation.character import controllability_proxy, identity_metrics
from ..evaluation.contact_sheet import create_contact_sheet
from ..evaluation.generation import GenerationBackend, SdScriptsGenerator
from ..evaluation.leakage import character_trigger_leakage, style_trigger_leakage
from ..evaluation.style import cross_content_metrics
from ..models import GeneratedImage, GenerationCase, PipelineError, StepResult
from ..state import ProjectState


DEFAULT_NEGATIVE_PROMPT = "low quality, worst quality, lowres, bad anatomy, text, watermark, signature"


def run(
    state: ProjectState,
    *,
    backend: GenerationBackend | None = None,
    verbose: int = 0,
) -> StepResult:
    project = state.payload["project"]
    run_record = _latest_trained_run(state)
    run_dir = Path(run_record["path"])
    _pipeline_log(run_dir, "evaluate.start", {"concept_type": state.concept_type})
    checkpoints = [Path(path) for path in run_record.get("checkpoints", []) if Path(path).is_file()]
    if not checkpoints:
        raise PipelineError("No candidate checkpoints exist for the latest trained run")
    registry = load_base_registry()
    base = registry[str(project["base"])]
    profiles = resolve_profiles(
        str(project.get("hardware", "v100_16gb")),
        str(project["type"]),
        str(project.get("strategy", "quality")),
        project_overrides=project.get("overrides", {}),
    )
    evaluation = profiles.merged.get("evaluation", {})
    strengths = [float(value) for value in evaluation.get("strengths", [0.4, 0.6, 0.8, 1.0])]
    prompts = [str(value) for value in evaluation.get("prompts", [])]
    if not prompts or not strengths:
        raise PipelineError("Evaluation prompt matrix or strength matrix is empty")
    target_candidates = int(profiles.merged.get("checkpoints", {}).get("target_candidates", 5))
    checkpoints = checkpoints[-target_candidates:]
    seed = int(profiles.merged.get("training", {}).get("seed", 42))
    cases = _build_cases(
        checkpoints,
        strengths=strengths,
        prompt_ids=prompts,
        trigger=str(project["trigger"]),
        concept_type=str(project["type"]),
        seed=seed,
    )
    settings = {
        "width": 1024,
        "height": 1024,
        "steps": int(base.generation_defaults.get("steps", 28)),
        "sampler": str(base.generation_defaults.get("sampler", "euler_a")),
        "scheduler": str(base.generation_defaults.get("scheduler", "normal")),
        "cfg": float(base.generation_defaults.get("cfg", 4.5)),
        "seed": seed,
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
    }
    backend = backend or SdScriptsGenerator()
    samples_dir = run_dir / "samples"
    try:
        generated = backend.generate(
            cases,
            base_path=base.path,
            output_dir=samples_dir,
            settings=settings,
            verbose=verbose,
        )
    except BaseException as exc:
        _pipeline_log(run_dir, "evaluate.failed", {"error": f"{type(exc).__name__}: {exc}"})
        raise
    if not generated:
        raise PipelineError("Evaluation backend produced no images")
    contact_sheet = create_contact_sheet(generated, run_dir / "contact-sheet.jpg", prompt_id=prompts[0])
    prepared_manifest_path = state.project_dir / "prepared" / "manifest.json"
    prepared_manifest = json.loads(prepared_manifest_path.read_text(encoding="utf-8"))
    prepared_records = list(prepared_manifest.get("images", []))
    prepared_images = [state.project_dir / "prepared" / record["image"] for record in prepared_records]
    dataset_bias: dict[str, Any] | None = None
    metrics: dict[str, Any] = {
        "automatic_scores_are_ground_truth": False,
        "generation_settings": settings,
        "matrix": {
            "candidate_checkpoints": len(checkpoints),
            "strengths": strengths,
            "prompts": prompts,
            "positive_and_no_trigger": True,
            "generated_images": len(generated),
        },
    }
    if state.concept_type == "character":
        metrics["identity"] = identity_metrics(prepared_images, generated)
        metrics["prompt_controllability"] = controllability_proxy(generated)
        metrics["trigger_leakage"] = character_trigger_leakage(prepared_images, generated)
    else:
        captions = [
            parse_caption(
                (state.project_dir / "prepared" / record["caption"]).read_text(
                    encoding="utf-8", errors="replace"
                )
            )
            for record in prepared_records
        ]
        inspection_path = state.project_dir / "dataset-manifest.json"
        aspect_ratios: list[float] = []
        if inspection_path.exists():
            inspection = json.loads(inspection_path.read_text(encoding="utf-8"))
            aspect_ratios = [
                float(record["aspect_ratio"])
                for record in inspection.get("images", [])
                if not record.get("corrupt") and record.get("aspect_ratio") is not None
            ]
        dataset_bias = distribution_summary(
            captions,
            profiles.concept.get("distribution", {}),
            aspect_ratios=aspect_ratios,
        )
        metrics["cross_content"] = cross_content_metrics(generated, dataset_bias=dataset_bias)
        metrics["trigger_leakage"] = style_trigger_leakage(generated)
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "evaluation.json"
    write_json_atomic(metrics_path, metrics)

    recommended_checkpoint = checkpoints[-1]
    recommended_strength = min(strengths, key=lambda value: abs(value - 0.8))
    best_path = run_dir / "best.safetensors"
    _link_or_copy(recommended_checkpoint, best_path)
    accounting = run_record.get("accounting", {})
    best_payload = {
        "schema_version": 1,
        "project": state.name,
        "type": state.concept_type,
        "trigger": project["trigger"],
        "base": {"id": base.id, "filename": base.path.name, "sha256": base.sha256},
        "training": {
            "rank": profiles.merged.get("training", {}).get("network_dim"),
            "alpha": profiles.merged.get("training", {}).get("network_alpha"),
            "learning_rate": profiles.merged.get("training", {}).get("unet_lr"),
            "physical_batch": accounting.get("physical_batch"),
            "effective_batch": accounting.get("effective_batch"),
            "images_seen": accounting.get("images_seen"),
            "optimizer_steps": accounting.get("optimizer_steps"),
            "epochs": accounting.get("epochs"),
        },
        "recommended": {
            "checkpoint": recommended_checkpoint.name,
            "strength": recommended_strength,
            "status": "provisional_pending_manual_contact_sheet_review",
            "selection_rule": "latest candidate; no automatic quality claim",
        },
        "evaluation": {
            "metrics": str(metrics_path),
            "contact_sheet": str(contact_sheet),
            "automatic_scores_are_ground_truth": False,
        },
    }
    best_yaml = run_dir / "best.yaml"
    write_yaml_atomic(best_yaml, best_payload)
    report = _write_report(
        state=state,
        run_dir=run_dir,
        base=base,
        profiles=profiles.merged,
        run_record=run_record,
        metrics=metrics,
        checkpoints=checkpoints,
        best=best_payload,
        dataset_bias=dataset_bias,
    )
    run_record.update(
        {
            "status": "evaluated",
            "evaluation": {
                "metrics": str(metrics_path),
                "contact_sheet": str(contact_sheet),
                "report": str(report),
                "best": str(best_path),
                "best_metadata": str(best_yaml),
            },
        }
    )
    state.save()
    input_hash = stable_hash(
        {
            "checkpoints": [{"path": str(path), "size": path.stat().st_size} for path in checkpoints],
            "settings": settings,
            "prompts": prompts,
            "strengths": strengths,
        }
    )
    _pipeline_log(
        run_dir,
        "evaluate.finished",
        {
            "generated_images": len(generated),
            "contact_sheet": str(contact_sheet),
            "report": str(report),
            "best": str(best_path),
        },
    )
    return StepResult(
        input_hash=input_hash,
        output_manifest=str(metrics_path),
        details={
            "run_id": run_dir.name,
            "generated_images": len(generated),
            "contact_sheet": str(contact_sheet),
            "report": str(report),
            "best": str(best_path),
            "best_yaml": str(best_yaml),
            "recommendation": best_payload["recommended"],
        },
    )


def _pipeline_log(run_dir: Path, event: str, details: dict[str, Any]) -> None:
    path = run_dir / "logs" / "pipeline.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"at": datetime.now(UTC).isoformat(), "event": event, "details": details}
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def _latest_trained_run(state: ProjectState) -> dict[str, Any]:
    for record in reversed(state.payload.get("runs", [])):
        if record.get("status") in {"trained", "evaluated"}:
            return record
    raise PipelineError("No successful training run is available for evaluation")


def _build_cases(
    checkpoints: list[Path],
    *,
    strengths: list[float],
    prompt_ids: list[str],
    trigger: str,
    concept_type: str,
    seed: int,
) -> list[GenerationCase]:
    cases: list[GenerationCase] = []
    for checkpoint in checkpoints:
        for prompt_index, prompt_id in enumerate(prompt_ids):
            content = _prompt_content(prompt_id, concept_type)
            case_seed = seed + prompt_index
            for strength in strengths:
                cases.append(
                    GenerationCase(
                        checkpoint=checkpoint,
                        checkpoint_label=checkpoint.stem,
                        strength=strength,
                        prompt_id=prompt_id,
                        prompt=f"{trigger}, {content}",
                        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                        seed=case_seed,
                        contains_trigger=True,
                    )
                )
                cases.append(
                    GenerationCase(
                        checkpoint=checkpoint,
                        checkpoint_label=checkpoint.stem,
                        strength=strength,
                        prompt_id=prompt_id,
                        prompt=content,
                        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                        seed=case_seed,
                        contains_trigger=False,
                    )
                )
    return cases


def _prompt_content(prompt_id: str, concept_type: str) -> str:
    normalized = prompt_id.replace("_", " ")
    if concept_type == "character":
        return f"1girl, {normalized}, detailed eyes, high quality"
    return f"{normalized}, detailed composition, high quality"


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.unlink(missing_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _write_report(
    *,
    state: ProjectState,
    run_dir: Path,
    base: Any,
    profiles: dict[str, Any],
    run_record: dict[str, Any],
    metrics: dict[str, Any],
    checkpoints: list[Path],
    best: dict[str, Any],
    dataset_bias: dict[str, Any] | None,
) -> Path:
    inspection_path = state.project_dir / "dataset-manifest.json"
    inspection = json.loads(inspection_path.read_text(encoding="utf-8")) if inspection_path.exists() else {}
    preflight_path = state.project_dir / "preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8")) if preflight_path.exists() else {}
    train_config_path = run_dir / "config" / "train.toml"
    dataset_config_path = run_dir / "config" / "dataset.toml"
    metadata_path = run_dir / "config" / "run-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    payload = {
        "project": state.name,
        "concept_type": state.concept_type,
        "trigger": state.payload["project"].get("trigger"),
        "base": {"id": base.id, "filename": base.path.name, "sha256": base.sha256},
        "dataset_stats": inspection.get("summary", {}),
        "preflight": preflight,
        "training_profiles": {
            "hardware": state.payload["project"].get("hardware"),
            "concept": state.concept_type,
            "strategy": state.payload["project"].get("strategy"),
        },
        "sd_scripts_commit": metadata.get("sd_scripts_commit"),
        "optimizer": profiles.get("training", {}).get("optimizer"),
        "rank": profiles.get("training", {}).get("network_dim"),
        "alpha": profiles.get("training", {}).get("network_alpha"),
        "learning_rate": profiles.get("training", {}).get("unet_lr"),
        "accounting": run_record.get("accounting", {}),
        "candidate_checkpoints": [path.name for path in checkpoints],
        "evaluation_metrics": metrics,
        "style_distribution": dataset_bias,
        "recommended": best["recommended"],
    }
    report = run_dir / "report.html"
    report.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>LoRA run report</title>"
        "<style>body{font:15px system-ui;max-width:1200px;margin:32px auto;padding:0 24px;color:#202124}"
        "h1,h2{color:#174ea6}pre{background:#f1f3f4;padding:16px;overflow:auto;border-radius:8px}"
        ".warning{background:#fef7e0;border-left:4px solid #f9ab00;padding:12px}img{max-width:100%;height:auto}</style>"
        "</head><body>"
        f"<h1>{html.escape(state.name)} — LoRA run {html.escape(run_dir.name)}</h1>"
        "<p class='warning'><strong>Manual review required.</strong> The selected best checkpoint is provisional; "
        "automatic metrics are auxiliary and are not image-quality ground truth.</p>"
        "<h2>Contact sheet</h2><img src='contact-sheet.jpg' alt='checkpoint by LoRA strength contact sheet'>"
        "<h2>Run summary</h2>"
        f"<pre>{html.escape(json.dumps(payload, indent=2, ensure_ascii=False))}</pre>"
        "<h2>Training config</h2>"
        f"<pre>{html.escape(train_config_path.read_text(encoding='utf-8') if train_config_path.exists() else '')}</pre>"
        "<h2>Dataset config</h2>"
        f"<pre>{html.escape(dataset_config_path.read_text(encoding='utf-8') if dataset_config_path.exists() else '')}</pre>"
        "</body></html>\n",
        encoding="utf-8",
    )
    return report
