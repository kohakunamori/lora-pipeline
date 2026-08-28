from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
_STAGE_LABELS = {
    "screening": "Screening",
    "full": "Full",
    "evaluation": "Evaluation",
}


def build_sample_history(
    run_dir: Path,
    exported_checkpoints: Sequence[str | Path],
    *,
    include_stages: Iterable[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Render all existing evaluation samples grouped by exported checkpoint.

    Evaluation generation already writes a generation-manifest.json containing
    the checkpoint, prompt, strength, seed, trigger state, and image path for
    every sample. The report should consume those manifests instead of assuming
    that the current evaluation stage is the complete history of a training run.

    Missing sample evidence is kept visible in the coverage table: an exported
    checkpoint must never silently disappear from a full report just because it
    was not part of the current finalist evaluation.
    """

    run_dir = run_dir.resolve()
    requested_stages = {str(value) for value in include_stages} if include_stages else None
    checkpoint_order = _checkpoint_order(exported_checkpoints)
    records = _collect_records(run_dir, requested_stages=requested_stages)

    # Preserve training/export order first. Manifests may also reference an older
    # checkpoint no longer listed in run metadata; retain it after known exports
    # rather than dropping evidence that already exists on disk.
    known_names = {entry["name"] for entry in checkpoint_order}
    for record in records:
        name = str(record["checkpoint"])
        if name not in known_names:
            checkpoint_order.append({"name": name, "stem": Path(name).stem})
            known_names.add(name)

    grouped: dict[str, list[dict[str, Any]]] = {entry["name"]: [] for entry in checkpoint_order}
    by_stem = {entry["stem"]: entry["name"] for entry in checkpoint_order}
    for record in records:
        name = str(record["checkpoint"])
        destination = name if name in grouped else by_stem.get(Path(name).stem)
        if destination is None:
            destination = name
            grouped.setdefault(destination, [])
        grouped[destination].append(record)

    coverage: list[dict[str, Any]] = []
    total_images = 0
    stage_names: set[str] = set()
    for checkpoint in checkpoint_order:
        name = checkpoint["name"]
        samples = grouped.get(name, [])
        stages = _ordered_stages({str(sample["stage"]) for sample in samples})
        total_images += len(samples)
        stage_names.update(stages)
        coverage.append(
            {
                "checkpoint": name,
                "sample_count": len(samples),
                "stages": stages,
                "has_sample_evidence": bool(samples),
            }
        )

    summary = {
        "schema_version": 1,
        "exported_checkpoints": len(checkpoint_order),
        "checkpoints_with_samples": sum(bool(item["has_sample_evidence"]) for item in coverage),
        "sample_images": total_images,
        "stages": _ordered_stages(stage_names),
        "coverage": coverage,
    }
    return _render_history_html(run_dir, checkpoint_order, grouped, summary), summary


def _checkpoint_order(values: Sequence[str | Path]) -> list[dict[str, str]]:
    ordered: list[dict[str, str]] = []
    seen: set[str] = set()
    for value in values:
        path = Path(str(value))
        name = path.name
        if not name or name in seen:
            continue
        seen.add(name)
        ordered.append({"name": name, "stem": path.stem})
    return ordered


def _collect_records(
    run_dir: Path,
    *,
    requested_stages: set[str] | None,
) -> list[dict[str, Any]]:
    samples_root = run_dir / "samples"
    if not samples_root.is_dir():
        return []

    manifests = sorted(samples_root.glob("**/generation-manifest.json"))
    records: list[dict[str, Any]] = []
    for manifest in manifests:
        relative_parent = manifest.parent.relative_to(samples_root)
        stage = relative_parent.parts[0] if relative_parent.parts else "evaluation"
        if requested_stages is not None and stage not in requested_stages:
            continue
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        images = payload.get("images", [])
        if not isinstance(images, list):
            continue
        for raw in images:
            if not isinstance(raw, dict):
                continue
            image_path = _resolve_sample_path(run_dir, manifest.parent, raw.get("path"))
            if image_path is None:
                continue
            checkpoint_raw = str(raw.get("checkpoint") or raw.get("checkpoint_label") or "").strip()
            if not checkpoint_raw:
                continue
            checkpoint_name = Path(checkpoint_raw).name
            if not Path(checkpoint_name).suffix:
                checkpoint_name += ".safetensors"
            records.append(
                {
                    "stage": stage,
                    "checkpoint": checkpoint_name,
                    "checkpoint_label": str(raw.get("checkpoint_label") or Path(checkpoint_name).stem),
                    "path": image_path,
                    "prompt_id": str(raw.get("prompt_id") or ""),
                    "prompt": str(raw.get("prompt") or ""),
                    "strength": raw.get("strength"),
                    "seed": raw.get("seed"),
                    "contains_trigger": raw.get("contains_trigger"),
                    "case_id": str(raw.get("case_id") or image_path.stem),
                }
            )
    return records


def _resolve_sample_path(run_dir: Path, manifest_dir: Path, value: object) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    path = Path(raw)
    candidates = [path] if path.is_absolute() else [manifest_dir / path, run_dir / path]
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            resolved.relative_to(run_dir)
        except (OSError, ValueError):
            continue
        if resolved.is_file() and resolved.suffix.lower() in _IMAGE_SUFFIXES:
            return resolved
    return None


def _ordered_stages(values: Iterable[str]) -> list[str]:
    unique = {str(value) for value in values if str(value)}
    priority = {"screening": 0, "full": 1, "evaluation": 2}
    return sorted(unique, key=lambda value: (priority.get(value, 99), value.casefold()))


def _render_history_html(
    run_dir: Path,
    checkpoint_order: Sequence[dict[str, str]],
    grouped: dict[str, list[dict[str, Any]]],
    summary: dict[str, Any],
) -> str:
    coverage_rows: list[str] = []
    sections: list[str] = []
    for checkpoint in checkpoint_order:
        name = checkpoint["name"]
        samples = grouped.get(name, [])
        stage_names = _ordered_stages({str(sample["stage"]) for sample in samples})
        coverage_rows.append(
            "<tr>"
            f"<td><code>{html.escape(name)}</code></td>"
            f"<td>{len(samples)}</td>"
            f"<td>{html.escape(', '.join(_stage_label(stage) for stage in stage_names) or '—')}</td>"
            f"<td>{'yes' if samples else '<strong class=\"missing\">no sample evidence</strong>'}</td>"
            "</tr>"
        )
        if not samples:
            sections.append(
                f"<section class='checkpoint-block'><h3>{html.escape(name)}</h3>"
                "<p class='missing'>No generated sample evidence exists for this exported checkpoint.</p></section>"
            )
            continue

        stage_blocks: list[str] = []
        for stage in stage_names:
            stage_samples = [sample for sample in samples if sample["stage"] == stage]
            cards = "".join(_sample_card(run_dir, sample) for sample in stage_samples)
            stage_blocks.append(
                f"<h4>{html.escape(_stage_label(stage))} · {len(stage_samples)} images</h4>"
                f"<div class='sample-grid'>{cards}</div>"
            )
        sections.append(
            f"<section class='checkpoint-block'><h3>{html.escape(name)}</h3>{''.join(stage_blocks)}</section>"
        )

    if not checkpoint_order:
        coverage_rows.append("<tr><td colspan='4'>No exported checkpoints were recorded.</td></tr>")

    return (
        "<section id='checkpoint-sample-history'>"
        "<style>"
        ".coverage{border-collapse:collapse;width:100%;margin:12px 0 24px}.coverage th,.coverage td{border:1px solid #dadce0;padding:8px;text-align:left}"
        ".coverage th{background:#f8f9fa}.checkpoint-block{margin:28px 0 40px}.sample-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:14px}"
        ".sample-card{border:1px solid #dadce0;border-radius:10px;overflow:hidden;background:#fff}.sample-card img{display:block;width:100%;aspect-ratio:1/1;object-fit:contain;background:#f8f9fa}"
        ".sample-meta{padding:9px;font-size:12px;line-height:1.45}.sample-meta code{font-size:11px}.missing{color:#b3261e;font-weight:600}"
        "</style>"
        "<h2>Checkpoint sample history</h2>"
        "<p>This section aggregates existing generated samples across evaluation stages. "
        "Full evaluation may contain only 1–2 finalists; Screening evidence remains visible here so checkpoint evolution is not lost.</p>"
        f"<p><strong>{summary['checkpoints_with_samples']}/{summary['exported_checkpoints']}</strong> checkpoints have sample evidence · "
        f"<strong>{summary['sample_images']}</strong> images total.</p>"
        "<table class='coverage'><thead><tr><th>Exported checkpoint</th><th>Samples</th><th>Stages</th><th>Evidence</th></tr></thead>"
        f"<tbody>{''.join(coverage_rows)}</tbody></table>"
        f"{''.join(sections)}"
        "</section>"
    )


def _sample_card(run_dir: Path, sample: dict[str, Any]) -> str:
    path = Path(sample["path"])
    relative = path.relative_to(run_dir).as_posix()
    trigger_state = sample.get("contains_trigger")
    if trigger_state is True:
        trigger_text = "trigger on"
    elif trigger_state is False:
        trigger_text = "trigger off"
    else:
        trigger_text = "trigger unknown"
    strength = sample.get("strength")
    strength_text = "—" if strength is None else str(strength)
    seed = sample.get("seed")
    seed_text = "—" if seed is None else str(seed)
    prompt_id = str(sample.get("prompt_id") or "")
    prompt = str(sample.get("prompt") or "")
    return (
        "<figure class='sample-card'>"
        f"<a href='{html.escape(relative)}'><img loading='lazy' src='{html.escape(relative)}' alt='{html.escape(str(sample.get('case_id') or path.stem))}'></a>"
        "<figcaption class='sample-meta'>"
        f"<strong>{html.escape(prompt_id or 'sample')}</strong><br>"
        f"strength {html.escape(strength_text)} · {html.escape(trigger_text)} · seed {html.escape(seed_text)}"
        + (f"<br><span>{html.escape(prompt)}</span>" if prompt else "")
        + "</figcaption></figure>"
    )


def _stage_label(stage: str) -> str:
    return _STAGE_LABELS.get(stage, stage.replace("_", " ").title())
