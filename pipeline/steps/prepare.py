from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from ..config import read_yaml, sha256_file, stable_hash, write_json_atomic
from ..dataset.image_info import discover_images, unique_caption_relative
from ..models import PipelineError, StepResult
from ..prepared import generation_path, generations_root, set_current_generation
from ..state import ProjectState


def run(state: ProjectState, *, allow_trigger_only: bool | None = None) -> StepResult:
    project_dir = state.project_dir
    raw = project_dir / "raw"
    images = discover_images(raw)
    exclusions_path = project_dir / "review" / "exclusions.yaml"
    exclusions = (
        set(read_yaml(exclusions_path).get("excluded", []))
        if exclusions_path.exists()
        else set()
    )
    generated = project_dir / "review" / "captions" / "generated"
    trigger = str(state.payload["project"]["trigger"])
    project = state.payload["project"]
    if allow_trigger_only is not None:
        project["allow_trigger_only"] = bool(allow_trigger_only)
        state.save()
    allow_fallback = bool(project.get("allow_trigger_only", False))

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
        if generated_caption.is_file():
            caption_text = generated_caption.read_text(
                encoding="utf-8", errors="replace"
            ).strip()
            caption_source = "caption-step"
        elif raw_caption.is_file():
            caption_text = raw_caption.read_text(
                encoding="utf-8", errors="replace"
            ).strip()
            caption_source = "existing-passthrough"
        elif allow_fallback:
            caption_text = trigger
            caption_source = "explicit-trigger-only"
        else:
            missing.append(relative_text)
            continue
        if not caption_text:
            if allow_fallback:
                caption_text = trigger
                caption_source = "explicit-trigger-only"
            else:
                missing.append(relative_text)
                continue
        planned.append(
            {
                "source": relative_text,
                "source_image": image,
                "source_image_sha256": sha256_file(image),
                "caption_text": caption_text,
                "caption_source": caption_source,
                "caption_sha256": stable_hash(caption_text),
                "image": (Path("images") / relative).as_posix(),
                "caption": (Path("captions") / caption_relative).as_posix(),
            }
        )
    if missing:
        preview = ", ".join(missing[:5])
        raise PipelineError(
            f"{len(missing)} image(s) have no usable caption ({preview}). "
            "Run caption, provide sidecars, or explicitly enable --allow-trigger-only."
        )
    if not planned:
        raise PipelineError("No images remain after exclusions")

    manifest_basis = {
        "schema_version": 2,
        "images": [
            {
                key: value
                for key, value in record.items()
                if key not in {"source_image", "caption_text"}
            }
            for record in planned
        ],
        "excluded": sorted(exclusions),
        "trigger": trigger,
        "caption_mode": project.get("caption_mode"),
        "allow_trigger_only": allow_fallback,
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
                image_destination.parent.mkdir(parents=True, exist_ok=True)
                caption_destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(Path(record["source_image"]), image_destination)
                caption_destination.write_text(
                    str(record["caption_text"]) + "\n", encoding="utf-8"
                )
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

    pointer = set_current_generation(
        project_dir,
        generation_id=generation_id,
        manifest_hash=manifest_hash,
        image_count=len(planned),
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
            "trigger_only_captions": sum(
                record["caption_source"] == "explicit-trigger-only"
                for record in planned
            ),
        },
    )
