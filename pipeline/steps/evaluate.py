from __future__ import annotations

import html
import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from ..config import load_base_registry, resolve_profiles, sha256_file, stable_hash, write_json_atomic
from ..dataset.caption_cleaner import normalize_tag, parse_caption
from ..dataset.image_info import discover_images
from ..dataset.style import distribution_summary
from ..evaluation.character import controllability_proxy, identity_metrics
from ..evaluation.contact_sheet import (
    create_contact_sheet,
    create_leakage_sheet,
    create_prompt_checkpoint_sheet,
)
from ..evaluation.generation import GenerationBackend, SdScriptsGenerator
from ..evaluation.leakage import character_trigger_leakage, style_trigger_leakage
from ..evaluation.outfit import outfit_retention_proxy, outfit_trigger_leakage_proxy
from ..evaluation.style import cross_content_metrics
from ..models import GeneratedImage, GenerationCase, PipelineError, StepResult
from ..prepared import load_current_generation
from ..state import ProjectState


DEFAULT_NEGATIVE_PROMPT = (
    "low quality, worst quality, lowres, bad anatomy, text, watermark, signature"
)
EVALUATION_STAGES = {"screening", "full"}


def run(
    state: ProjectState,
    *,
    backend: GenerationBackend | None = None,
    verbose: int = 0,
    stage: str = "screening",
    run_id: str | None = None,
    checkpoint_names: list[str] | None = None,
) -> StepResult:
    if stage not in EVALUATION_STAGES:
        raise PipelineError(f"Evaluation stage must be one of {sorted(EVALUATION_STAGES)}")
    project = state.payload["project"]
    training_target_type = str(project.get("training_target_type", project["type"]))
    run_record = _select_trained_run(state, run_id)
    run_dir = Path(run_record["path"])
    _pipeline_log(
        run_dir,
        "evaluate.start",
        {
            "concept_type": state.concept_type,
            "training_target_type": training_target_type,
            "stage": stage,
            "checkpoint_names": checkpoint_names,
        },
    )
    checkpoints = [
        Path(path) for path in run_record.get("checkpoints", []) if Path(path).is_file()
    ]
    if not checkpoints:
        raise PipelineError("No candidate checkpoints exist for the selected trained run")
    if checkpoint_names:
        requested = set(checkpoint_names)
        checkpoints = [
            path for path in checkpoints if path.name in requested or path.stem in requested
        ]
        missing = requested - {value for path in checkpoints for value in (path.name, path.stem)}
        if missing:
            raise PipelineError("Unknown checkpoint selection: " + ", ".join(sorted(missing)))
    registry = load_base_registry()
    base = registry[str(project["base"])]
    profiles = resolve_profiles(
        str(project.get("hardware", "v100_16gb")),
        str(project["type"]),
        str(project.get("strategy", "quality")),
        project_overrides=project.get("overrides", {}),
    )
    evaluation = profiles.merged.get("evaluation", {})
    prompts, strengths = _stage_matrix(evaluation, stage=stage, concept_type=state.concept_type)
    if not prompts or not strengths:
        raise PipelineError("Evaluation prompt matrix or strength matrix is empty")
    target_candidates = int(
        profiles.merged.get("checkpoints", {}).get("target_candidates", 5)
    )
    checkpoints = checkpoints[-target_candidates:]
    if stage == "full" and len(checkpoints) > 2:
        if checkpoint_names:
            raise PipelineError("Full evaluation accepts only one or two finalists")
        raise PipelineError(
            "Full evaluation requires explicit finalists when more than two checkpoints exist; "
            "choose one or two candidates in the interactive checkpoint picker"
        )
    seed = int(profiles.merged.get("training", {}).get("seed", 42))
    subject_prompt = str(
        project.get("evaluation", {}).get(
            "subject_prompt", evaluation.get("subject_prompt", "1girl")
        )
    )
    cases = _build_cases(
        checkpoints,
        strengths=strengths,
        prompt_ids=prompts,
        trigger=str(project["trigger"]),
        concept_type=str(project["type"]),
        subject_prompt=subject_prompt,
        seed=seed,
    )
    settings = {
        "width": int(evaluation.get("width", 1024)),
        "height": int(evaluation.get("height", 1024)),
        "steps": int(base.generation_defaults.get("steps", 28)),
        "sampler": str(base.generation_defaults.get("sampler", "euler_a")),
        "cfg": float(base.generation_defaults.get("cfg", 4.5)),
        "seed": seed,
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
    }
    backend = backend or SdScriptsGenerator()
    samples_dir = run_dir / "samples" / stage
    try:
        generated = backend.generate(
            cases,
            base_path=base.path,
            output_dir=samples_dir,
            settings=settings,
            verbose=verbose,
        )
    except BaseException as exc:
        _pipeline_log(
            run_dir,
            "evaluate.failed",
            {"stage": stage, "error": f"{type(exc).__name__}: {exc}"},
        )
        raise
    if not generated:
        raise PipelineError("Evaluation backend produced no images")

    sheets_dir = run_dir / "contact-sheets" / stage
    checkpoint_strength_sheet = create_contact_sheet(
        generated,
        sheets_dir / "checkpoint-strength.jpg",
        prompt_id=prompts[0],
    )
    prompt_checkpoint_sheet = create_prompt_checkpoint_sheet(
        generated, sheets_dir / "prompt-checkpoint.jpg"
    )
    leakage_sheet = create_leakage_sheet(generated, sheets_dir / "trigger-leakage.jpg")
    # Compatibility pointer to the primary sheet, without claiming a best checkpoint.
    shutil.copy2(checkpoint_strength_sheet, run_dir / "contact-sheet.jpg")

    generation = load_current_generation(state.project_dir)
    prepared_records = list(generation.manifest.get("images", []))
    prepared_images = [generation.root / record["image"] for record in prepared_records]
    validation_images = discover_images(state.project_dir / "validation")
    reference_images = validation_images or prepared_images
    reference_source = "validation" if validation_images else "training_fallback"
    dataset_bias: dict[str, Any] | None = None
    metrics: dict[str, Any] = {
        "schema_version": 2,
        "stage": stage,
        "concept_type": state.concept_type,
        "training_target_type": training_target_type,
        "automatic_scores_are_ground_truth": False,
        "manual_selection_required": True,
        "generation_settings": settings,
        "reference_source": reference_source,
        "matrix": {
            "candidate_checkpoints": len(checkpoints),
            "checkpoint_names": [path.name for path in checkpoints],
            "strengths": strengths,
            "prompts": prompts,
            "positive_and_no_trigger": True,
            "subject_prompt": subject_prompt,
            "generated_images": len(generated),
        },
        "contact_sheets": {
            "checkpoint_strength": str(checkpoint_strength_sheet),
            "prompt_checkpoint": str(prompt_checkpoint_sheet),
            "trigger_leakage": str(leakage_sheet),
        },
    }
    if state.concept_type == "character":
        metrics["identity"] = identity_metrics(reference_images, generated)
        metrics["prompt_controllability"] = controllability_proxy(generated)
        if training_target_type == "character_outfit":
            anchor_tags = project.get("caption_anchor_tags", [])
            metrics["outfit_retention"] = outfit_retention_proxy(generated)
            metrics["trigger_leakage"] = outfit_trigger_leakage_proxy(
                generated, anchor_tags=anchor_tags
            )
            metrics["outfit_review_note"] = (
                "CCIP identity remains auxiliary. Outfit fidelity and outfit leakage are "
                "reviewed on aligned trigger-on/off contact sheets rather than inferred from "
                "identity similarity."
            )
        else:
            metrics["trigger_leakage"] = character_trigger_leakage(
                reference_images, generated
            )
        if reference_source == "training_fallback":
            metrics["identity_warning"] = (
                "No validation images were supplied; identity scores may reward memorization"
            )
    else:
        captions = [
            parse_caption(
                (generation.root / record["caption"]).read_text(
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
        metrics["cross_content"] = cross_content_metrics(
            generated, dataset_bias=dataset_bias
        )
        metrics["trigger_leakage"] = style_trigger_leakage(generated)

    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"evaluation-{stage}.json"
    write_json_atomic(metrics_path, metrics)
    report = _write_report(
        state=state,
        run_dir=run_dir,
        stage=stage,
        base=base,
        profiles=profiles.merged,
        run_record=run_record,
        metrics=metrics,
        checkpoints=checkpoints,
        dataset_bias=dataset_bias,
    )
    shutil.copy2(report, run_dir / "report.html")
    evaluation_record = run_record.setdefault("evaluation", {})
    evaluation_record[stage] = {
        "metrics": str(metrics_path),
        "report": str(report),
        "contact_sheets": metrics["contact_sheets"],
        "checkpoints": [path.name for path in checkpoints],
        "completed_at": datetime.now(UTC).isoformat(),
    }
    run_record["status"] = "evaluated"
    state.save()
    input_hash = stable_hash(
        {
            "stage": stage,
            "training_target_type": training_target_type,
            "checkpoints": [
                {"path": str(path), "sha256": sha256_file(path)} for path in checkpoints
            ],
            "settings": settings,
            "subject_prompt": subject_prompt,
            "prompts": prompts,
            "strengths": strengths,
        }
    )
    _pipeline_log(
        run_dir,
        "evaluate.finished",
        {
            "stage": stage,
            "training_target_type": training_target_type,
            "generated_images": len(generated),
            "contact_sheets": metrics["contact_sheets"],
            "report": str(report),
            "manual_selection_required": True,
        },
    )
    return StepResult(
        input_hash=input_hash,
        output_manifest=str(metrics_path),
        details={
            "run_id": run_dir.name,
            "stage": stage,
            "training_target_type": training_target_type,
            "generated_images": len(generated),
            "contact_sheets": metrics["contact_sheets"],
            "report": str(report),
            "manual_selection_required": True,
            "next": "Open the project dashboard and choose Promote a checkpoint after review",
        },
    )


def _pipeline_log(run_dir: Path, event: str, details: dict[str, Any]) -> None:
    path = run_dir / "logs" / "pipeline.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"at": datetime.now(UTC).isoformat(), "event": event, "details": details}
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def _select_trained_run(state: ProjectState, run_id: str | None) -> dict[str, Any]:
    for record in reversed(state.payload.get("runs", [])):
        if run_id is not None and record.get("id") != run_id:
            continue
        if record.get("status") in {"trained", "evaluated", "promoted"}:
            return record
    if run_id:
        raise PipelineError(f"No successful trained run exists with id {run_id}")
    raise PipelineError("No successful training run is available for evaluation")


def _stage_matrix(
    evaluation: dict[str, Any], *, stage: str, concept_type: str
) -> tuple[list[str], list[float]]:
    prompts = [str(value) for value in evaluation.get("prompts", [])]
    strengths = [float(value) for value in evaluation.get("strengths", [])]
    if stage == "screening":
        configured_prompts = evaluation.get("screening_prompts")
        if configured_prompts:
            prompts = [str(value) for value in configured_prompts]
        else:
            defaults = (
                ["portrait", "full body", "different outfit"]
                if concept_type == "character"
                else ["female portrait", "male portrait", "landscape"]
            )
            prompts = [value for value in defaults if value in prompts] or prompts[:3]
        strengths = [
            float(value)
            for value in evaluation.get("screening_strengths", [0.6, 0.8, 1.0])
        ]
    return prompts, strengths


def _build_cases(
    checkpoints: list[Path],
    *,
    strengths: list[float],
    prompt_ids: list[str],
    trigger: str,
    concept_type: str,
    subject_prompt: str,
    seed: int,
) -> list[GenerationCase]:
    subject_tags = {normalize_tag(tag) for tag in parse_caption(subject_prompt)}
    if normalize_tag(trigger) in subject_tags:
        raise PipelineError(
            "Evaluation subject prompt must not contain the LoRA trigger; "
            "trigger-on/off cases add it automatically"
        )
    cases: list[GenerationCase] = []
    for checkpoint in checkpoints:
        for prompt_index, prompt_id in enumerate(prompt_ids):
            content = _prompt_content(prompt_id, concept_type, subject_prompt)
            case_seed = seed + prompt_index
            for strength in strengths:
                for contains_trigger in (True, False):
                    prompt = f"{trigger}, {content}" if contains_trigger else content
                    case_id = _case_id(
                        checkpoint=checkpoint,
                        prompt_id=prompt_id,
                        strength=strength,
                        contains_trigger=contains_trigger,
                        seed=case_seed,
                    )
                    cases.append(
                        GenerationCase(
                            case_id=case_id,
                            checkpoint=checkpoint,
                            checkpoint_label=checkpoint.stem,
                            strength=strength,
                            prompt_id=prompt_id,
                            prompt=prompt,
                            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                            seed=case_seed,
                            contains_trigger=contains_trigger,
                        )
                    )
    return cases


def _prompt_content(prompt_id: str, concept_type: str, subject_prompt: str) -> str:
    normalized = prompt_id.replace("_", " ")
    if concept_type == "character":
        return f"{subject_prompt}, {normalized}, detailed eyes, high quality"
    return f"{normalized}, detailed composition, high quality"


def _case_id(
    *,
    checkpoint: Path,
    prompt_id: str,
    strength: float,
    contains_trigger: bool,
    seed: int,
) -> str:
    readable = "__".join(
        [
            checkpoint.stem,
            prompt_id,
            f"s{int(round(strength * 1000)):04d}",
            "trigger-on" if contains_trigger else "trigger-off",
            f"seed-{seed}",
        ]
    )
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", readable).strip("-._")[:112]
    suffix = stable_hash(
        {
            "checkpoint": str(checkpoint),
            "prompt_id": prompt_id,
            "strength": strength,
            "contains_trigger": contains_trigger,
            "seed": seed,
        }
    )[:10]
    return f"{slug}__{suffix}"


def _write_report(
    *,
    state: ProjectState,
    run_dir: Path,
    stage: str,
    base: Any,
    profiles: dict[str, Any],
    run_record: dict[str, Any],
    metrics: dict[str, Any],
    checkpoints: list[Path],
    dataset_bias: dict[str, Any] | None,
) -> Path:
    inspection_path = state.project_dir / "dataset-manifest.json"
    inspection = (
        json.loads(inspection_path.read_text(encoding="utf-8"))
        if inspection_path.exists()
        else {}
    )
    preflight_path = state.project_dir / "preflight.json"
    preflight = (
        json.loads(preflight_path.read_text(encoding="utf-8"))
        if preflight_path.exists()
        else {}
    )
    train_config_path = run_dir / "config" / "train.toml"
    dataset_config_path = run_dir / "config" / "dataset.toml"
    metadata_path = run_dir / "config" / "run-metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    payload = {
        "project": state.name,
        "concept_type": state.concept_type,
        "training_target_type": state.payload["project"].get(
            "training_target_type", state.concept_type
        ),
        "trigger": state.payload["project"].get("trigger"),
        "caption_anchor_tags": state.payload["project"].get("caption_anchor_tags", []),
        "stage": stage,
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
        "selection": "Manual promotion is required; evaluation does not create best.safetensors",
    }
    report = run_dir / f"report-{stage}.html"
    sheet_html = "".join(
        f"<h3>{html.escape(name.replace('_', ' ').title())}</h3>"
        f"<img src='{html.escape(str(Path(path).relative_to(run_dir)))}' alt='{html.escape(name)}'>"
        for name, path in metrics.get("contact_sheets", {}).items()
    )
    report.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>LoRA run report</title>"
        "<style>body{font:15px system-ui;max-width:1400px;margin:32px auto;padding:0 24px;color:#202124}"
        "h1,h2{color:#174ea6}pre{background:#f1f3f4;padding:16px;overflow:auto;border-radius:8px}"
        ".warning{background:#fef7e0;border-left:4px solid #f9ab00;padding:12px}img{max-width:100%;height:auto}</style>"
        "</head><body>"
        f"<h1>{html.escape(state.name)} — {html.escape(stage)} evaluation</h1>"
        "<p class='warning'><strong>Manual selection required.</strong> Automatic metrics are auxiliary. "
        "Use the project dashboard's Promote a checkpoint action only after reviewing aligned sheets.</p>"
        f"{sheet_html}"
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
