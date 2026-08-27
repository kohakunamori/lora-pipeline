from __future__ import annotations

import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from ..config import read_yaml, resolve_profiles, stable_hash, write_json_atomic
from ..dataset.caption_cleaner import clean_caption, parse_caption
from ..dataset.image_info import discover_images, unique_caption_relative
from ..models import PipelineError, StepResult
from ..state import ProjectState


def run(state: ProjectState) -> StepResult:
    project_dir = state.project_dir
    raw = project_dir / "raw"
    images = discover_images(raw)
    exclusions_path = project_dir / "review" / "exclusions.yaml"
    exclusions = set(read_yaml(exclusions_path).get("excluded", [])) if exclusions_path.exists() else set()
    generated = project_dir / "review" / "captions" / "generated"
    trigger = str(state.payload["project"]["trigger"])
    project = state.payload["project"]
    profiles = resolve_profiles(
        project.get("hardware", "v100_16gb"),
        project["type"],
        project.get("strategy", "quality"),
        project_overrides=project.get("overrides", {}),
    )
    caption_config = profiles.concept.get("caption", {})
    max_tokens = int(profiles.hardware.get("caption", {}).get("default_max_token_length", 75))
    selected: list[dict[str, str]] = []
    cache = project_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="prepared-", dir=cache))
    try:
        image_root = stage / "images"
        caption_root = stage / "captions"
        image_root.mkdir(parents=True)
        caption_root.mkdir(parents=True)
        for image in images:
            relative = image.relative_to(raw)
            relative_text = relative.as_posix()
            if relative_text in exclusions:
                continue
            destination = image_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, destination)
            caption_relative = unique_caption_relative(relative)
            generated_caption = generated / caption_relative
            raw_caption = image.with_suffix(".txt")
            if generated_caption.exists():
                caption_text = generated_caption.read_text(encoding="utf-8", errors="replace").strip()
                caption_source = "generated"
            elif raw_caption.exists():
                caption_text = raw_caption.read_text(encoding="utf-8", errors="replace").strip()
                caption_source = "existing"
            else:
                caption_text = trigger
                caption_source = "trigger-only fallback"
            cleaned = clean_caption(
                parse_caption(caption_text),
                trigger=trigger,
                blacklist=caption_config.get("blacklist", []),
                max_token_length=max_tokens,
                concept_type=project["type"],
                preserve_existing_style_descriptors=caption_source != "generated",
                ordering=caption_config.get("ordering", []),
            )
            caption_destination = caption_root / caption_relative
            caption_destination.parent.mkdir(parents=True, exist_ok=True)
            caption_destination.write_text(cleaned.text + "\n", encoding="utf-8")
            selected.append(
                {
                    "source": relative_text,
                    "image": destination.relative_to(stage).as_posix(),
                    "caption": caption_destination.relative_to(stage).as_posix(),
                    "caption_source": caption_source,
                    "estimated_tokens": cleaned.estimated_tokens,
                    "pruned": list(cleaned.pruned),
                }
            )
        if not selected:
            raise PipelineError("No images remain after exclusions")
        manifest = {
            "schema_version": 1,
            "images": selected,
            "excluded": sorted(exclusions),
            "input_hash": stable_hash(
                {"selected": selected, "excluded": sorted(exclusions), "raw": state.step("inspect").get("input_hash")}
            ),
        }
        write_json_atomic(stage / "manifest.json", manifest)
        prepared = project_dir / "prepared"
        backup: Path | None = None
        if prepared.exists():
            timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
            backup = cache / "prepare-backups" / timestamp
            backup.parent.mkdir(parents=True, exist_ok=True)
            os.replace(prepared, backup)
        try:
            os.replace(stage, prepared)
        except BaseException:
            if backup is not None and not prepared.exists():
                os.replace(backup, prepared)
            raise
        stage = Path()
        manifest_path = prepared / "manifest.json"
        return StepResult(
            input_hash=manifest["input_hash"],
            output_manifest=str(manifest_path),
            details={"prepared_images": len(selected), "excluded": len(exclusions), "backup": str(backup) if backup else None},
        )
    finally:
        if stage and stage.exists() and stage != Path("."):
            shutil.rmtree(stage, ignore_errors=True)
