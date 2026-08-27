from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal

from .dataset.caption_cleaner import normalize_tag, parse_caption
from .dataset_workspace import DatasetWorkspace
from .models import PipelineError


BatchTagAction = Literal["prepend", "append", "remove"]


def parse_tag_input(text: str) -> list[str]:
    """Parse comma/newline separated tags while preserving the user's spelling."""

    parts = text.replace("\n", ",").replace("，", ",").split(",")
    return _unique_tags(part.strip() for part in parts if part.strip())


def batch_edit_tags(
    workspace: DatasetWorkspace,
    keys: Sequence[str],
    tags: Iterable[str],
    *,
    action: BatchTagAction,
) -> dict[str, object]:
    """Apply one tag operation to multiple dataset captions."""

    if action not in {"prepend", "append", "remove"}:
        raise PipelineError(f"Unknown batch tag action: {action}")
    selected = _unique_keys(keys)
    if not selected:
        raise PipelineError("No dataset items were selected for batch tag editing")
    requested = _unique_tags(tags)
    if not requested:
        raise PipelineError("No tags were provided for batch tag editing")

    before = {key: workspace.caption_text(key) for key in selected}
    requested_norm = {normalize_tag(tag) for tag in requested}
    changed = 0
    unchanged = 0
    for key in selected:
        existing = _unique_tags(parse_caption(before[key]))
        if action == "remove":
            updated = [tag for tag in existing if normalize_tag(tag) not in requested_norm]
        else:
            retained = [tag for tag in existing if normalize_tag(tag) not in requested_norm]
            updated = [*requested, *retained] if action == "prepend" else [*retained, *requested]
        new_text = ", ".join(updated)
        if new_text == before[key].strip():
            unchanged += 1
            continue
        workspace.replace_caption(key, new_text)
        changed += 1
    return {
        "action": action,
        "selected": len(selected),
        "changed": changed,
        "unchanged": unchanged,
        "tags": requested,
    }


def _unique_keys(keys: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for key in keys:
        value = str(key).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _unique_tags(tags: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        cleaned = str(tag).strip()
        normalized = normalize_tag(cleaned)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(cleaned)
    return result
