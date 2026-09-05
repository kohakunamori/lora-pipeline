from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from ..config import read_yaml, sha256_file, stable_hash, write_json_atomic
from ..dataset.caption_cleaner import caption_prefix
from ..dataset.crop import CROP_POLICY_VERSION, plan_target_crop
from ..dataset.image_info import discover_images, unique_caption_relative
from ..dataset.image_normalizer import (
    DEFAULT_BUCKET_STEP,
    DEFAULT_MAX_PIXELS,
    normalize_training_image,
    plan_training_image,
)
from ..dataset.subject import SubjectDetector
from ..models import PipelineError, StepResult
from ..prepared import generation_path, generations_root, set_current_generation
from ..state import ProjectState


def run(
    state: ProjectState,
    *,
    allow_trigger_only: bool | None = None,
    caption_mode: str | None = None,
) -> StepResult:
    project_dir = state.project_dir
    raw = project_dir / "raw"
    images = discover_images(raw)
    project = state.payload["project"]
    resolved_caption_mode = _resolve_caption_mode(project, caption_mode)

    # Captions still use the existing compatibility transform in this phase. The
    # visual side is now target-aware and deterministic; the follow-up phase moves
    # generated tagging onto these transformed images so tag semantics also reflect
    # the final crop.
    caption_details: dict[str, Any] = {}
    if resolved_caption_mode != "skip":
        from . import caption as caption_step

        caption_result = caption_step.run(state, mode=resolved_caption_mode)
        caption_details = dict(caption_result.details)
    else:
        project["caption_mode"] = "skip"

    exclusions_path = project_dir / "review" / "exclusions.yaml"
    exclusions = (
        set(read_yaml(exclusions_path).get("excluded", []))
        if exclusions_path.exists()
        else set()
    )
    generated = project_dir / "review" / "captions" / "generated"
    trigger = str(project["trigger"])
    fixed_prefix = caption_prefix(trigger, project.get("caption_anchor_tags", []))
    fallback_caption = ", ".join(fixed_prefix)
    if allow_trigger_only is not None:
        project["allow_trigger_only"] = bool(allow_trigger_only)
        state.save()
    allow_fallback = bool(project.get("allow_trigger_only", False))

    target_type = str(project.get("training_target_type", project.get("type", "")))
    if target_type not in {"character", "character_outfit", "style"}:
        raise PipelineError(f"Unsupported training target for materialization: {target_type}")
    detector = SubjectDetector() if target_type != "style" else None

    normalization = {
        "max_pixels": DEFAULT_MAX_PIXELS,
        "bucket_step": DEFAULT_BUCKET_STEP,
        "no_upscale": True,
        "order": ["target_crop", "downscale"],
    }
    crop_policy = {
        "version": CROP_POLICY_VERSION,
        "target_type": target_type,
        "character": "subject_aware",
        "character_outfit": "outfit_preserving",
        "style": "composition_preserving",
    }

    planned: list[dict[str, object]] = []
    missing: list[str] = []
    for image in images:
        relative = image.relative_to(raw)
        relative_text = relative.as_posix()
        if relative_text in exclusions:
            continue
        caption_relative = unique_caption_relative(relative)
        generated_caption = generated / caption_relative
        raw_caption = image.with_suffix(".txt")
        if resolved_caption_mode != "skip" and generated_caption.is_file():
            caption_bytes = generated_caption.read_bytes()
            caption_source = "caption-transform"
        elif raw_caption.is_file():
            caption_bytes = raw_caption.read_bytes()
            caption_source = "existing-passthrough"
        elif allow_fallback:
            caption_bytes = (fallback_caption + "\n").encode("utf-8")
            caption_source = "explicit-trigger-only"
        else:
            missing.append(relative_text)
            continue
        caption_text = caption_bytes.decode("utf-8", errors="replace").strip()
        if not caption_text:
            if allow_fallback:
                caption_bytes = (fallback_caption + "\n").encode("utf-8")
                caption_text = fallback_caption
                caption_source = "explicit-trigger-only"
            else:
                missing.append(relative_text)
                continue

        crop = plan_target_crop(
            image,
            target_type=target_type,
            detector=detector,
        )
        image_plan = plan_training_image(
            image,
            crop_box=crop.crop_box,
            max_pixels=DEFAULT_MAX_PIXELS,
            bucket_step=DEFAULT_BUCKET_STEP,
        )
        planned.append(
            {
                "source": relative_text,
                "source_image": image,
                "source_image_sha256": sha256_file(image),
                "source_size": list(image_plan.source_size),
                "crop_size": list(image_plan.input_size),
                "prepared_size": list(image_plan.target_size),
                "cropped": image_plan.cropped,
                "downscaled": image_plan.downscaled,
                "crop": crop.as_dict(),
                "caption_bytes": caption_bytes,
                "caption_source": caption_source,
                "caption_sha256": hashlib.sha256(caption_bytes).hexdigest(),
                "image": (Path("images") / relative).as_posix(),
                "caption": (Path("captions") / caption_relative).as_posix(),
            }
        )
    if missing:
        preview = ", ".join(missing[:5])
        raise PipelineError(
            f"{len(missing)} image(s) have no usable caption ({preview}). "
            "Add captions in the Dataset workspace or explicitly enable --allow-trigger-only."
        )
    if not planned:
        raise PipelineError("No images remain after exclusions")

    manifest_basis = {
        "schema_version": 5,
        "images": [
            {
                key: value
                for key, value in record.items()
                if key not in {"source_image", "caption_bytes"}
            }
            for record in planned
        ],
        "excluded": sorted(exclusions),
        "trigger": trigger,
        "fixed_prefix": list(fixed_prefix),
        "training_target_type": target_type,
        "caption_mode": resolved_caption_mode,
        "allow_trigger_only": allow_fallback,
        "crop_policy": crop_policy,
        "image_normalization": normalization,
    }
    manifest_hash = stable_hash(manifest_basis)
    generation_id = manifest_hash
    target = generation_path(project_dir, generation_id)
    manifest_path = target / "manifest.json"
    reused_generation = target.exists()

    if reused_generation:
        if not manifest_path.is_file():
            raise PipelineError(f"Prepared generation exists without a manifest: {target}")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("manifest_hash") != manifest_hash:
            raise PipelineError(f"Prepared generation hash collision or corruption: {target}")
    else:
        root = generations_root(project_dir)
        root.mkdir(parents=True, exist_ok=True)
        cache = project_dir / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix="prepared-generation-", dir=cache))
        try:
            for record in planned:
                image_destination = stage / str(record["image"])
                caption_destination = stage / str(record["caption"])
                crop_record = record.get("crop", {})
                raw_box = (
                    crop_record.get("crop_box")
                    if isinstance(crop_record, Mapping)
                    else None
                )
                crop_box = (
                    tuple(int(value) for value in raw_box)
                    if isinstance(raw_box, list) and len(raw_box) == 4
                    else None
                )
                normalize_training_image(
                    Path(record["source_image"]),
                    image_destination,
                    crop_box=crop_box,
                    max_pixels=DEFAULT_MAX_PIXELS,
                    bucket_step=DEFAULT_BUCKET_STEP,
                )
                caption_destination.parent.mkdir(parents=True, exist_ok=True)
                caption_destination.write_bytes(bytes(record["caption_bytes"]))
            manifest = {
                **manifest_basis,
                "manifest_hash": manifest_hash,
                "generation_id": generation_id,
            }
            write_json_atomic(stage / "manifest.json", manifest)
            os.replace(stage, target)
            stage = Path()
        finally:
            if stage and stage.exists() and stage != Path("."):
                shutil.rmtree(stage, ignore_errors=True)

    project["caption_mode"] = resolved_caption_mode
    pointer = set_current_generation(
        project_dir,
        generation_id=generation_id,
        manifest_hash=manifest_hash,
        image_count=len(planned),
    )
    crop_reasons = Counter(
        str(record.get("crop", {}).get("reason", "unknown"))
        for record in planned
        if isinstance(record.get("crop"), Mapping)
    )
    return StepResult(
        input_hash=manifest_hash,
        output_manifest=str(manifest_path),
        details={
            "prepared_images": len(planned),
            "excluded": len(exclusions),
            "generation_id": generation_id,
            "generation_path": str(target),
            "current_pointer": str(pointer),
            "reused_generation": reused_generation,
            "fixed_prefix": list(fixed_prefix),
            "caption_mode": resolved_caption_mode,
            "caption_transform": caption_details,
            "crop_policy": crop_policy,
            "cropped_images": sum(bool(record["cropped"]) for record in planned),
            "crop_reasons": dict(sorted(crop_reasons.items())),
            "image_normalization": normalization,
            "downscaled_images": sum(bool(record["downscaled"]) for record in planned),
            "trigger_only_captions": sum(
                record["caption_source"] == "explicit-trigger-only"
                for record in planned
            ),
        },
    )


def _resolve_caption_mode(project: Mapping[str, Any], requested: str | None) -> str:
    # A direct legacy `prepare.run(state)` call historically consumed raw
    # sidecars (or explicit trigger-only fallback) and never inherited a guided
    # workflow preference that could unexpectedly start a tagger. Guided
    # training passes its caption mode explicitly through service.run_remaining.
    mode = str(requested or project.get("caption_mode") or "skip")
    if mode != "auto":
        return mode

    snapshot = project.get("dataset_snapshot", {})
    if isinstance(snapshot, Mapping):
        image_count = int(snapshot.get("image_count", 0) or 0)
        caption_count = int(snapshot.get("caption_count", 0) or 0)
        if image_count > 0 and caption_count == image_count:
            return "existing_taglist_clean"
    return "generate"
