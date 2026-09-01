from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType
from typing import Any

from .config import stable_hash, write_json_atomic
from .dataset.caption_cleaner import STYLE_DESCRIPTOR_PATTERNS, normalize_tag, parse_caption
from .models import StepResult, StepStatus
from .tokenizers import count_sdxl_tokens


def install_style_caption_policy_hook(caption_module: ModuleType) -> None:
    """Ensure cleaned/hybrid Style captions do not caption away the target style."""

    original = caption_module.run
    if getattr(original, "_style_caption_policy_wrapped", False):
        return

    def run(state, *args, **kwargs):
        result = original(state, *args, **kwargs)
        return apply_style_caption_policy(state, result)

    run._style_caption_policy_wrapped = True
    run._style_caption_policy_original = original
    caption_module.run = run


def apply_style_caption_policy(state, result: StepResult) -> StepResult:
    project = state.payload.get("project", {})
    if project.get("type") != "style":
        return result
    if result.status not in {StepStatus.DONE, StepStatus.SKIPPED} or not result.output_manifest:
        return result

    manifest_path = Path(result.output_manifest)
    if not manifest_path.is_file():
        return result
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mode = str(manifest.get("mode") or "")
    if mode in {"existing_passthrough", "skip"}:
        return result

    trigger = str(project.get("trigger") or "").strip()
    protected = normalize_tag(trigger)
    changed = 0
    suppressed_total = 0
    for record in manifest.get("records", []):
        tags = parse_caption(str(record.get("text") or ""))
        retained: list[str] = []
        suppressed: list[str] = []
        seen: set[str] = set()
        for tag in tags:
            normalized = normalize_tag(tag)
            if not normalized or normalized in seen:
                continue
            if normalized != protected and _is_style_descriptor(normalized):
                suppressed.append(tag)
                continue
            seen.add(normalized)
            retained.append(tag)

        text = ", ".join(retained)
        if text != str(record.get("text") or ""):
            changed += 1
        suppressed_total += len(suppressed)
        destination = Path(str(record["caption"]))
        destination.write_text(text + "\n", encoding="utf-8")
        counts = count_sdxl_tokens(text)
        record["text"] = text
        record["token_counts"] = {
            "clip_l": counts.clip_l,
            "clip_g": counts.clip_g,
            "exact": counts.exact,
            "backend": counts.backend,
            "error": counts.error,
        }
        record["style_descriptors_suppressed"] = suppressed

    manifest.pop("input_hash", None)
    manifest.setdefault("summary", {})["style_caption_updates"] = changed
    manifest["summary"]["style_descriptors_suppressed"] = suppressed_total
    manifest["style_caption_policy"] = {
        "suppress_style_descriptors": True,
        "applies_to": ["generate", "existing_taglist_clean", "hybrid"],
        "existing_passthrough_preserved": True,
    }
    manifest["input_hash"] = stable_hash(manifest)
    write_json_atomic(manifest_path, manifest)

    details = dict(result.details)
    details.update(
        {
            "style_caption_updates": changed,
            "style_descriptors_suppressed": suppressed_total,
        }
    )
    return StepResult(
        status=result.status,
        input_hash=manifest["input_hash"],
        output_manifest=str(manifest_path),
        details=details,
    )


def _is_style_descriptor(tag: str) -> bool:
    return any(pattern.search(tag) for pattern in STYLE_DESCRIPTOR_PATTERNS)
