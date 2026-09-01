from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from .config import stable_hash, write_json_atomic
from .dataset.caption_cleaner import CATEGORY_PATTERNS, normalize_tag, parse_caption
from .models import StepResult, StepStatus
from .tokenizers import count_sdxl_tokens


_IDENTITY_INVARIANT_MIN_COVERAGE = 0.85
_OUTFIT_INVARIANT_MIN_COVERAGE = 0.80
_OUTFIT_SPECIFICITY_MIN_DELTA = 0.50
_MIN_INFERENCE_SAMPLES = 3

_IDENTITY_PATTERNS = (
    re.compile(
        r"^(?:black|blonde|brown|red|blue|green|purple|pink|white|grey|gray|silver|"
        r"aqua|orange|multicolored|two-tone) hair$"
    ),
    re.compile(
        r"^(?:black|brown|red|blue|green|purple|pink|yellow|gold|golden|white|grey|"
        r"gray|silver|aqua|orange) eyes$"
    ),
    re.compile(
        r"^(?:very long hair|long hair|medium hair|short hair|twintails|twin tails|"
        r"ponytail|side ponytail|braid|double braid|single braid|bob cut|hime cut|"
        r"straight hair|wavy hair|curly hair|ahoge|blunt bangs|bangs|hair over one eye|"
        r"heterochromia)$"
    ),
)


def apply_character_semantic_factorization(state, result: StepResult) -> StepResult:
    """Move stable identity/outfit descriptors from captions into semantic tokens.

    This runs after the existing Character semantic composer has injected the frozen
    character and per-image outfit tokens. It only removes high-confidence intrinsic
    identity tags plus manually selected or strongly outfit-specific garment tags.
    """

    project = state.payload.get("project", {})
    snapshot = project.get("dataset_semantics_snapshot")
    if not snapshot or project.get("type") != "character":
        return result
    if result.status not in {StepStatus.DONE, StepStatus.SKIPPED} or not result.output_manifest:
        return result

    manifest_path = Path(result.output_manifest)
    if not manifest_path.is_file():
        return result
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("mode") == "existing_passthrough":
        return result
    records = manifest.get("records", [])
    if not isinstance(records, list):
        return result

    bindings = snapshot.get("images", {})
    outfits = snapshot.get("outfits", {})
    character = snapshot.get("character", {})
    character_token = str(character.get("token") or project.get("trigger") or "").strip()
    manual_character_features = {normalize_tag(tag) for tag in character.get("features", [])}
    inferred_identity = infer_invariant_identity_tags(records) - manual_character_features
    inferred_outfits = infer_outfit_features_by_group(records, bindings)
    anchors = {
        normalize_tag(value)
        for value in project.get("caption_anchor_tags", [])
        if str(value).strip()
    }

    changed = 0
    total_suppressed = 0
    for record in records:
        image_key = str(record.get("image") or "")
        outfit_id = str(bindings.get(image_key, {}).get("outfit") or "default")
        outfit = outfits.get(outfit_id, outfits.get("default", {}))
        outfit_token = str(outfit.get("token") or "").strip()
        manual_outfit = {normalize_tag(tag) for tag in outfit.get("features", [])}
        suppressed = inferred_identity | manual_outfit | inferred_outfits.get(outfit_id, set())
        protected = {
            normalize_tag(value)
            for value in (character_token, outfit_token)
            if value
        } | anchors

        retained: list[str] = []
        suppressed_here: list[str] = []
        seen: set[str] = set()
        for tag in parse_caption(str(record.get("text") or "")):
            normalized = normalize_tag(tag)
            if not normalized or normalized in seen:
                continue
            if normalized in suppressed and normalized not in protected:
                suppressed_here.append(tag)
                continue
            seen.add(normalized)
            retained.append(tag)

        text = ", ".join(retained)
        counts = count_sdxl_tokens(text)
        if text != str(record.get("text") or ""):
            changed += 1
        total_suppressed += len(suppressed_here)
        Path(str(record["caption"])).write_text(text + "\n", encoding="utf-8")
        record["text"] = text
        record["token_counts"] = {
            "clip_l": counts.clip_l,
            "clip_g": counts.clip_g,
            "exact": counts.exact,
            "backend": counts.backend,
            "error": counts.error,
        }
        concepts = dict(record.get("semantic_concepts") or {})
        concepts["factorized_identity_features"] = sorted(
            normalized for normalized in inferred_identity if normalized in {normalize_tag(tag) for tag in suppressed_here}
        )
        concepts["factorized_outfit_features"] = sorted(
            normalized
            for normalized in (manual_outfit | inferred_outfits.get(outfit_id, set()))
            if normalized in {normalize_tag(tag) for tag in suppressed_here}
        )
        record["semantic_concepts"] = concepts

    manifest.pop("input_hash", None)
    manifest["target_policy"] = {
        "target_type": "character",
        "identity_invariant_min_coverage": _IDENTITY_INVARIANT_MIN_COVERAGE,
        "outfit_invariant_min_coverage": _OUTFIT_INVARIANT_MIN_COVERAGE,
        "outfit_specificity_min_delta": _OUTFIT_SPECIFICITY_MIN_DELTA,
        "min_inference_samples": _MIN_INFERENCE_SAMPLES,
        "manual_character_features": sorted(manual_character_features),
        "inferred_identity_features": sorted(inferred_identity),
        "inferred_outfit_features": {
            outfit_id: sorted(tags) for outfit_id, tags in sorted(inferred_outfits.items()) if tags
        },
    }
    manifest.setdefault("summary", {})["semantic_factorization_updates"] = changed
    manifest["summary"]["semantic_factorization_suppressions"] = total_suppressed
    manifest["input_hash"] = stable_hash(manifest)
    write_json_atomic(manifest_path, manifest)

    details = dict(result.details)
    details.update(
        {
            "semantic_factorization_updates": changed,
            "semantic_factorization_suppressions": total_suppressed,
            "inferred_identity_features": sorted(inferred_identity),
            "inferred_outfit_features": {
                outfit_id: sorted(tags)
                for outfit_id, tags in sorted(inferred_outfits.items())
                if tags
            },
        }
    )
    return StepResult(
        status=result.status,
        input_hash=manifest["input_hash"],
        output_manifest=str(manifest_path),
        details=details,
    )


def infer_invariant_identity_tags(records: list[Mapping[str, Any]]) -> set[str]:
    if len(records) < _MIN_INFERENCE_SAMPLES:
        return set()
    counts: Counter[str] = Counter()
    for record in records:
        tags = {
            normalize_tag(tag)
            for tag in parse_caption(str(record.get("text") or ""))
            if normalize_tag(tag)
        }
        counts.update(tags)
    total = len(records)
    return {
        tag
        for tag, count in counts.items()
        if count / total >= _IDENTITY_INVARIANT_MIN_COVERAGE and _is_identity_tag(tag)
    }


def infer_outfit_features_by_group(
    records: list[Mapping[str, Any]], bindings: Mapping[str, Any]
) -> dict[str, set[str]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        key = str(record.get("image") or "")
        outfit_id = str(bindings.get(key, {}).get("outfit") or "default")
        grouped[outfit_id].append(record)

    result: dict[str, set[str]] = {}
    all_records = list(records)
    for outfit_id, selected in grouped.items():
        if len(selected) < _MIN_INFERENCE_SAMPLES:
            result[outfit_id] = set()
            continue
        selected_counts = _record_tag_counts(selected)
        selected_ids = {id(record) for record in selected}
        others = [record for record in all_records if id(record) not in selected_ids]
        other_counts = _record_tag_counts(others)
        selected_total = len(selected)
        other_total = len(others)
        tags: set[str] = set()
        for tag, count in selected_counts.items():
            coverage = count / selected_total
            other_coverage = other_counts.get(tag, 0) / other_total if other_total else 0.0
            if (
                coverage >= _OUTFIT_INVARIANT_MIN_COVERAGE
                and coverage - other_coverage >= _OUTFIT_SPECIFICITY_MIN_DELTA
                and _is_outfit_tag(tag)
            ):
                tags.add(tag)
        result[outfit_id] = tags
    return result


def _record_tag_counts(records: list[Mapping[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        tags = {
            normalize_tag(tag)
            for tag in parse_caption(str(record.get("text") or ""))
            if normalize_tag(tag)
        }
        counts.update(tags)
    return counts


def _is_identity_tag(tag: str) -> bool:
    return any(pattern.fullmatch(tag) for pattern in _IDENTITY_PATTERNS)


def _is_outfit_tag(tag: str) -> bool:
    return any(pattern.search(tag) for pattern in CATEGORY_PATTERNS.get("outfit", ()))
