from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from ..config import resolve_profiles, stable_hash, write_json_atomic
from ..dataset.caption_cleaner import clean_caption, parse_caption
from ..dataset.image_info import discover_images, unique_caption_relative
from ..dataset.tagger import (
    CachedTagger,
    DualTagger,
    ImgutilsWdTagger,
    TagResult,
    TaggerBackend,
)
from ..models import PipelineError, StepResult, StepStatus
from ..state import ProjectState
from ..tokenizers import TokenCounts, count_sdxl_tokens


CAPTION_MODES = {
    "generate",
    "existing_passthrough",
    "existing_taglist_clean",
    "hybrid",
    "skip",
}


def run(
    state: ProjectState,
    *,
    mode: str = "generate",
    tagger: TaggerBackend | None = None,
    threshold: float = 0.35,
) -> StepResult:
    if mode == "existing":  # backward-compatible name, explicit behavior going forward
        mode = "existing_taglist_clean"
    if mode == "skip":
        return StepResult(status=StepStatus.SKIPPED, details={"reason": "caption explicitly skipped"})
    if mode not in CAPTION_MODES:
        raise PipelineError(f"Unknown caption mode: {mode}")

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
    max_tokens = int(
        profiles.merged.get("caption", {}).get(
            "max_token_length",
            profiles.hardware.get("caption", {}).get("default_max_token_length", 75),
        )
    )

    requires_tagger = mode in {"generate", "hybrid"}
    if requires_tagger:
        tagger = _build_cached_tagger(
            state,
            tagger=tagger,
            tagger_config=tagger_config,
            threshold=threshold,
        )

    missing: list[str] = []
    for image in images:
        relative = image.relative_to(raw)
        source = image.with_suffix(".txt")
        destination = generated_root / unique_caption_relative(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)

        if mode == "existing_passthrough":
            if not source.is_file():
                missing.append(relative.as_posix())
                continue
            shutil.copy2(source, destination)
            text = source.read_text(encoding="utf-8", errors="replace").strip()
            counts = count_sdxl_tokens(text)
            records.append(
                _base_record(
                    relative=relative,
                    destination=destination,
                    mode=mode,
                    text=text,
                    counts=counts,
                    pruned=[],
                )
            )
            continue

        if mode == "existing_taglist_clean":
            if not source.is_file():
                missing.append(relative.as_posix())
                continue
            source_text = source.read_text(encoding="utf-8", errors="replace").strip()
            text, pruned, counts = _clean_and_prune(
                parse_caption(source_text),
                project=project,
                caption_config=caption_config,
                threshold=threshold,
                max_tokens=max_tokens,
                preserve_existing_style_descriptors=True,
            )
            destination.write_text(text + "\n", encoding="utf-8")
            records.append(
                _base_record(
                    relative=relative,
                    destination=destination,
                    mode=mode,
                    text=text,
                    counts=counts,
                    pruned=pruned,
                )
            )
            continue

        assert tagger is not None
        result, dual_details = _tag_image(tagger, image)
        if dual_details and dual_details["conflicts"]:
            conflict_count += 1
        if mode == "hybrid" and source.is_file():
            existing_tags = parse_caption(
                source.read_text(encoding="utf-8", errors="replace").strip()
            )
            combined: dict[str, float] = {str(tag): 1.0 for tag in existing_tags}
            for key, value in result.tags.items():
                combined.setdefault(str(key), float(value))
            caption_input: Mapping[str, float] = combined
            preserve_style = True
            existing_tag_set = {tag.replace("_", " ").casefold() for tag in existing_tags}
            suggested = sorted(
                tag
                for tag, score in result.tags.items()
                if float(score) >= threshold
                and tag.replace("_", " ").casefold() not in existing_tag_set
            )
        else:
            caption_input = result.tags
            preserve_style = False
            suggested = []
        text, pruned, counts = _clean_and_prune(
            caption_input,
            project=project,
            caption_config=caption_config,
            threshold=threshold,
            max_tokens=max_tokens,
            preserve_existing_style_descriptors=preserve_style,
        )
        destination.write_text(text + "\n", encoding="utf-8")
        record = _base_record(
            relative=relative,
            destination=destination,
            mode=mode,
            text=text,
            counts=counts,
            pruned=pruned,
        )
        record.update(
            {
                "backend": result.backend,
                "backend_metadata": dict(result.metadata),
                "ratings": dict(result.ratings),
                "characters": dict(result.characters),
                "tags": dict(result.tags),
                "hybrid_suggested_tags": suggested,
                "dual_tagger": dual_details,
                "needs_review": bool(dual_details and dual_details["conflicts"]),
            }
        )
        records.append(record)

    if missing:
        preview = ", ".join(missing[:5])
        raise PipelineError(
            f"{mode} requires an existing caption for every image; missing {len(missing)} ({preview})"
        )

    character_review = _annotate_character_review(records)
    if character_review["flagged_images"]:
        for record in records:
            if record["image"] in character_review["flagged_images"]:
                record["needs_review"] = True
                record["character_review"] = character_review["flagged_images"][record["image"]]
    write_json_atomic(
        state.project_dir / "review" / "captions" / "character-review.json",
        character_review,
    )

    style_distribution = None
    if project["type"] == "style":
        from ..dataset.style import distribution_summary

        captions = [parse_caption(str(record["text"])) for record in records]
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

    cache_hits = sum(
        bool(record.get("backend_metadata", {}).get("cache_hit")) for record in records
    )
    manifest = {
        "schema_version": 2,
        "mode": mode,
        "records": records,
        "style_distribution": style_distribution,
        "character_review": character_review,
        "summary": {
            "captions": len(records),
            "dual_tagger_conflict_images": conflict_count,
            "character_review_images": len(character_review["flagged_images"]),
            "tagger_cache_hits": cache_hits,
            "needs_review": sum(bool(record.get("needs_review")) for record in records),
        },
    }
    manifest["input_hash"] = stable_hash(manifest)
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
            "character_review_images": len(character_review["flagged_images"]),
            "tagger_cache_hits": cache_hits,
            "style_distribution": style_distribution,
        },
    )


def _build_cached_tagger(
    state: ProjectState,
    *,
    tagger: TaggerBackend | None,
    tagger_config: Mapping[str, Any],
    threshold: float,
) -> TaggerBackend:
    if tagger is None and tagger_config.get("dual_enabled"):
        raise PipelineError(
            "Dual tagger is enabled but no challenger backend was supplied; "
            "disable it or configure a DualTagger challenger"
        )
    stable_backend = str(tagger_config.get("stable_backend", "wd_eva02_large_v3"))
    if tagger is None and stable_backend != "wd_eva02_large_v3":
        raise PipelineError(f"Unsupported configured stable tagger backend: {stable_backend}")
    tagger = tagger or ImgutilsWdTagger(model_name="EVA02_Large", threshold=threshold)
    cache_root = state.project_dir / "cache" / "tagger"
    if isinstance(tagger, DualTagger):
        return DualTagger(
            CachedTagger(tagger.stable, cache_root / "stable"),
            CachedTagger(tagger.challenger, cache_root / "challenger"),
            conflict_delta=tagger.conflict_delta,
        )
    return CachedTagger(tagger, cache_root / "stable")


def _tag_image(
    tagger: TaggerBackend, image: Path
) -> tuple[TagResult, dict[str, Any] | None]:
    if isinstance(tagger, DualTagger):
        comparison = tagger.compare(image)
        return comparison.stable, {
            "challenger_backend": comparison.challenger.backend,
            "agreement": comparison.agreement,
            "stable_only": comparison.stable_only,
            "challenger_only": comparison.challenger_only,
            "conflicts": comparison.conflicts,
        }
    return tagger.tag(image), None


def _clean_and_prune(
    tags: Mapping[str, float] | list[str],
    *,
    project: Mapping[str, Any],
    caption_config: Mapping[str, Any],
    threshold: float,
    max_tokens: int,
    preserve_existing_style_descriptors: bool,
) -> tuple[str, list[str], TokenCounts]:
    cleaned = clean_caption(
        tags,
        trigger=str(project["trigger"]),
        threshold=threshold,
        blacklist=caption_config.get("blacklist", []),
        replacements=caption_config.get("replacements", {}),
        max_token_length=1_000_000,
        concept_type=str(project["type"]),
        identity_mode=caption_config.get("caption_identity_features", {}).get(
            "mode", "conservative"
        ),
        preserve_existing_style_descriptors=preserve_existing_style_descriptors,
        ordering=caption_config.get("ordering", []),
    )
    retained = list(cleaned.tags)
    pruned = list(cleaned.pruned)
    counts = count_sdxl_tokens(", ".join(retained))
    while len(retained) > 1 and counts.maximum > max_tokens:
        pruned.append(retained.pop())
        counts = count_sdxl_tokens(", ".join(retained))
    return ", ".join(retained), pruned, counts


def _base_record(
    *,
    relative: Path,
    destination: Path,
    mode: str,
    text: str,
    counts: TokenCounts,
    pruned: list[str],
) -> dict[str, Any]:
    return {
        "image": relative.as_posix(),
        "caption": str(destination),
        "mode": mode,
        "text": text,
        "token_counts": {
            "clip_l": counts.clip_l,
            "clip_g": counts.clip_g,
            "exact": counts.exact,
            "backend": counts.backend,
            "error": counts.error,
        },
        "pruned": pruned,
    }


def _annotate_character_review(records: list[dict[str, Any]]) -> dict[str, Any]:
    top_characters: list[str] = []
    for record in records:
        characters = record.get("characters", {})
        if characters:
            top_characters.append(max(characters, key=characters.get))
    dominant = Counter(top_characters).most_common(1)
    dominant_character = dominant[0][0] if dominant else None
    flagged: dict[str, Any] = {}
    for record in records:
        characters = {
            str(key): float(value) for key, value in record.get("characters", {}).items()
        }
        if not characters:
            continue
        top = max(characters, key=characters.get)
        reasons: list[str] = []
        if len(characters) > 1:
            reasons.append("multiple_character_tags")
        if dominant_character and top != dominant_character:
            reasons.append("top_character_differs_from_dataset_mode")
        if reasons:
            flagged[record["image"]] = {
                "reasons": reasons,
                "top_character": top,
                "characters": characters,
            }
    return {
        "schema_version": 1,
        "dominant_character": dominant_character,
        "dominant_character_counts": Counter(top_characters).most_common(10),
        "flagged_images": flagged,
    }
