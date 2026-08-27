from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import resolve_profiles, stable_hash, write_json_atomic
from ..dataset.caption_cleaner import clean_caption, parse_caption
from ..dataset.image_info import discover_images, unique_caption_relative
from ..dataset.tagger import DualTagger, ImgutilsWdTagger, TaggerBackend
from ..models import PipelineError, StepResult, StepStatus
from ..state import ProjectState


def run(
    state: ProjectState,
    *,
    mode: str = "generate",
    tagger: TaggerBackend | None = None,
    threshold: float = 0.35,
) -> StepResult:
    if mode == "skip":
        return StepResult(status=StepStatus.SKIPPED, details={"reason": "caption explicitly skipped"})
    project = state.payload["project"]
    raw = state.project_dir / "raw"
    images = discover_images(raw)
    generated_root = state.project_dir / "review" / "captions" / "generated"
    generated_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    conflict_count = 0
    profiles = resolve_profiles(
        project.get("hardware", "v100_16gb"),
        project["type"],
        project.get("strategy", "quality"),
        project_overrides=project.get("overrides", {}),
    )
    caption_config = profiles.concept.get("caption", {})
    tagger_config = profiles.concept.get("tagger", {})
    max_tokens = int(profiles.hardware.get("caption", {}).get("default_max_token_length", 75))
    if mode == "existing":
        missing = []
        for image in images:
            relative = image.relative_to(raw)
            source = image.with_suffix(".txt")
            if not source.exists():
                missing.append(relative.as_posix())
                continue
            destination = generated_root / unique_caption_relative(relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            text = source.read_text(encoding="utf-8", errors="replace").strip()
            cleaned = clean_caption(
                parse_caption(text),
                trigger=project["trigger"],
                blacklist=caption_config.get("blacklist", []),
                replacements=caption_config.get("replacements", {}),
                max_token_length=max_tokens,
                concept_type=project["type"],
                preserve_existing_style_descriptors=True,
                ordering=caption_config.get("ordering", []),
            )
            destination.write_text(cleaned.text + "\n", encoding="utf-8")
            records.append(
                {
                    "image": relative.as_posix(),
                    "caption": str(destination),
                    "mode": "existing",
                    "estimated_tokens": cleaned.estimated_tokens,
                    "pruned": list(cleaned.pruned),
                }
            )
        if missing:
            raise PipelineError(f"Existing captions are missing for {len(missing)} image(s)")
    elif mode == "generate":
        if tagger is None and tagger_config.get("dual_enabled"):
            raise PipelineError(
                "Dual tagger is enabled but no optional challenger backend was supplied; "
                "disable it or install/configure a TaggerBackend challenger"
            )
        stable_backend = str(tagger_config.get("stable_backend", "wd_eva02_large_v3"))
        if tagger is None and stable_backend != "wd_eva02_large_v3":
            raise PipelineError(f"Unsupported configured stable tagger backend: {stable_backend}")
        tagger = tagger or ImgutilsWdTagger(model_name="EVA02_Large", threshold=threshold)
        for image in images:
            relative = image.relative_to(raw)
            dual_details: dict[str, Any] | None = None
            if isinstance(tagger, DualTagger):
                comparison = tagger.compare(image)
                result = comparison.stable
                dual_details = {
                    "challenger_backend": comparison.challenger.backend,
                    "agreement": comparison.agreement,
                    "stable_only": comparison.stable_only,
                    "challenger_only": comparison.challenger_only,
                    "conflicts": comparison.conflicts,
                }
                if comparison.conflicts:
                    conflict_count += 1
            else:
                result = tagger.tag(image)
            cleaned = clean_caption(
                result.tags,
                trigger=project["trigger"],
                threshold=threshold,
                blacklist=caption_config.get("blacklist", []),
                replacements=caption_config.get("replacements", {}),
                max_token_length=max_tokens,
                concept_type=project["type"],
                identity_mode=caption_config.get("caption_identity_features", {}).get("mode", "conservative"),
                ordering=caption_config.get("ordering", []),
            )
            destination = generated_root / unique_caption_relative(relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(cleaned.text + "\n", encoding="utf-8")
            records.append(
                {
                    "image": relative.as_posix(),
                    "caption": str(destination),
                    "mode": "generate",
                    "backend": result.backend,
                    "backend_metadata": dict(result.metadata),
                    "ratings": dict(result.ratings),
                    "characters": dict(result.characters),
                    "tags": dict(result.tags),
                    "estimated_tokens": cleaned.estimated_tokens,
                    "pruned": list(cleaned.pruned),
                    "dual_tagger": dual_details,
                    "needs_review": bool(dual_details and dual_details["conflicts"]),
                }
            )
    else:
        raise PipelineError(f"Unknown caption mode: {mode}")
    style_distribution = None
    if project["type"] == "style":
        from ..dataset.style import distribution_summary

        captions = [
            parse_caption(Path(record["caption"]).read_text(encoding="utf-8", errors="replace"))
            for record in records
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
        style_distribution = distribution_summary(
            captions,
            profiles.concept.get("distribution", {}),
            aspect_ratios=aspect_ratios,
        )
    manifest = {
        "schema_version": 1,
        "mode": mode,
        "records": records,
        "style_distribution": style_distribution,
        "summary": {
            "captions": len(records),
            "dual_tagger_conflict_images": conflict_count,
        },
        "input_hash": stable_hash(records),
    }
    manifest_path = state.project_dir / "review" / "captions" / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    state.payload["project"]["caption_mode"] = mode
    return StepResult(
        input_hash=manifest["input_hash"],
        output_manifest=str(manifest_path),
        details={
            "captions": len(records),
            "mode": mode,
            "dual_tagger_conflict_images": conflict_count,
            "style_distribution": style_distribution,
        },
    )
