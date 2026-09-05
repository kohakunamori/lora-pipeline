from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import read_yaml, repository_root, stable_hash, write_json_atomic
from .dataset.caption_cleaner import parse_caption
from .dataset.image_info import inspect_image
from .dataset.style import distribution_summary
from .dataset_semantics import load_semantics
from .dataset_workspace import DatasetWorkspace
from .target_dataset_diagnostics import target_dataset_diagnostics
from .target_preflight import assess_style_distribution
from .state import utc_now


def analyze_acquisition_gaps(
    workspace: DatasetWorkspace,
    *,
    target_type: str | None = None,
    limits: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Turn curation diagnostics into non-destructive acquisition guidance.

    Dataset pruning answers "what can I remove?". This report answers the equally
    important inverse question: "what information is under-represented and should I
    add before training?". It never mutates the DatasetWorkspace.
    """

    target = str(target_type or workspace.concept_type)
    if target not in {"character", "character_outfit", "style"}:
        raise ValueError("target_type must be character, character_outfit, or style")

    items = workspace.items(include_disabled=False, include_excluded=False)
    caption_records = [
        {
            "image": item.key,
            "text": (
                item.caption.read_text(encoding="utf-8", errors="replace").strip()
                if item.caption.is_file()
                else ""
            ),
        }
        for item in items
    ]
    missing_captions = sum(not bool(record["text"]) for record in caption_records)

    if target == "style":
        payload = _style_acquisition(workspace, caption_records, limits=limits)
    else:
        semantics = load_semantics(workspace, create=False) or {}
        diagnostics = target_dataset_diagnostics(
            target,
            caption_records=caption_records,
            dataset_semantics=semantics,
            limits=limits,
        )
        actions = _character_actions(diagnostics)
        payload = {
            "target_type": target,
            "diagnostics": diagnostics,
            "actions": actions,
        }

    report = {
        "schema_version": 1,
        "dataset": workspace.name,
        "target_type": target,
        "generated_at": utc_now(),
        "image_count": len(items),
        "captioned_images": len(items) - missing_captions,
        "missing_captions": missing_captions,
        **payload,
    }
    if missing_captions:
        report.setdefault("actions", []).insert(
            0,
            {
                "priority": "metadata",
                "dimension": "caption_coverage",
                "reason": f"{missing_captions} active image(s) have no caption, so diversity diagnostics are incomplete.",
                "suggestion": "Tag or caption the missing images before using caption-derived diversity as an acquisition signal.",
            },
        )
    report["status"] = "needs_data" if report.get("actions") else "balanced"
    report["input_hash"] = stable_hash(
        {
            "target_type": target,
            "items": [
                {
                    "image": record["image"],
                    "text": record["text"],
                }
                for record in caption_records
            ],
            "payload": payload,
        }
    )
    path = workspace.dataset_dir / "review" / "optimization" / "acquisition.json"
    write_json_atomic(path, report)
    report["manifest"] = str(path)
    return report


def _character_actions(diagnostics: Mapping[str, Any]) -> list[dict[str, str]]:
    actions: list[dict[str, str]] = []
    checks = diagnostics.get("checks", {})
    limits = diagnostics.get("limits", {})
    concentration = float(limits.get("concentration_warning_fraction", 0.8))
    min_coverage = float(limits.get("minimum_category_caption_coverage", 0.25))

    for category in ("pose", "expression", "composition", "background", "lighting"):
        summary = checks.get(category, {})
        coverage = float(summary.get("caption_coverage_fraction", 0.0) or 0.0)
        dominant = float(summary.get("dominant_fraction", 0.0) or 0.0)
        dominant_tag = str(summary.get("dominant_tag") or "").strip()
        if coverage < min_coverage:
            actions.append(
                {
                    "priority": "metadata",
                    "dimension": category,
                    "reason": f"Only {coverage:.1%} of captions expose a recognized {category} tag.",
                    "suggestion": f"Improve {category} tagging before deciding whether more {category} coverage is needed.",
                }
            )
        elif dominant >= concentration and dominant_tag:
            actions.append(
                {
                    "priority": "acquire",
                    "dimension": category,
                    "reason": f"{dominant_tag!r} dominates {category} coverage at {dominant:.1%}.",
                    "suggestion": f"Add images with {category} conditions different from {dominant_tag!r}; preserve rare existing variants during deduplication.",
                }
            )

    semantic = checks.get("semantic_outfits", {})
    required = int(limits.get("character_min_semantic_outfits", 2) or 2)
    represented = int(semantic.get("represented_outfits", 0) or 0)
    if semantic and represented < required:
        actions.append(
            {
                "priority": "acquire",
                "dimension": "outfit",
                "reason": f"Only {represented} semantic outfit(s) are represented; target policy expects about {required} for clothing-independent identity.",
                "suggestion": "Add or correctly bind another outfit when the intended Character LoRA should generalize across clothing.",
            }
        )

    framing = checks.get("outfit_framing", {})
    if framing:
        full_body = float(framing.get("full_body_fraction", 0.0) or 0.0)
        minimum = float(framing.get("minimum_full_body_fraction", 0.15) or 0.15)
        if full_body < minimum:
            actions.append(
                {
                    "priority": "acquire",
                    "dimension": "full_body",
                    "reason": f"Full-body coverage is {full_body:.1%}, below the target floor of {minimum:.1%}.",
                    "suggestion": "Add full-body views that expose lower-garment and footwear details; avoid filling the gap with near-identical poses.",
                }
            )

    return _dedupe_actions(actions)


def _style_acquisition(
    workspace: DatasetWorkspace,
    caption_records: Sequence[Mapping[str, Any]],
    *,
    limits: Mapping[str, Any] | None,
) -> dict[str, Any]:
    profile_path = repository_root() / "profiles" / "concepts" / "style.yaml"
    profile = read_yaml(profile_path) if profile_path.is_file() else {}
    distribution_config = dict(profile.get("distribution", {}))
    style_limits = dict(profile.get("limits", {}).get("style_bias", {}))
    if limits:
        style_limits.update(dict(limits))

    captions = [parse_caption(str(record.get("text") or "")) for record in caption_records]
    aspect_ratios: list[float] = []
    for item in workspace.items(include_disabled=False, include_excluded=False):
        inspected = inspect_image(item.image, workspace.source_images_dir(item.source_id))
        if not inspected.get("corrupt") and inspected.get("aspect_ratio") is not None:
            aspect_ratios.append(float(inspected["aspect_ratio"]))

    distribution = distribution_summary(
        captions,
        distribution_config,
        aspect_ratios=aspect_ratios,
    )
    assessment = assess_style_distribution(distribution, limits=style_limits)
    actions: list[dict[str, str]] = []
    for warning in distribution.get("warnings", []):
        code = str(warning.get("code") or "")
        if code == "high_subject_concentration":
            suggestion = "Add different subject categories so the style is not bound to one character/subject type."
            dimension = "subject"
        elif code == "high_portrait_concentration":
            suggestion = "Add full-body, medium-shot, landscape, or wider compositions in the same style."
            dimension = "composition"
        elif code == "high_simple_background_concentration":
            suggestion = "Add complex/background-rich scenes in the same style."
            dimension = "background"
        elif code == "low_multi_subject_coverage":
            suggestion = "Add a small number of multi-subject scenes to test whether the style generalizes beyond solo compositions."
            dimension = "subject_count"
        elif code == "low_aspect_ratio_diversity":
            suggestion = "Add at least one additional broad aspect-ratio class (portrait/square/landscape)."
            dimension = "aspect_ratio"
        else:
            suggestion = str(warning.get("message") or "Diversify this axis before training.")
            dimension = code or "style_distribution"
        actions.append(
            {
                "priority": "acquire",
                "dimension": dimension,
                "reason": str(warning.get("message") or code),
                "suggestion": suggestion,
            }
        )

    for reason in assessment.get("blocking", []):
        actions.insert(
            0,
            {
                "priority": "blocking_acquire",
                "dimension": "style_entanglement",
                "reason": str(reason),
                "suggestion": "Diversify both subject identity and composition/background before starting a Style training run.",
            },
        )

    return {
        "style_distribution": distribution,
        "style_assessment": assessment,
        "actions": _dedupe_actions(actions),
    }


def _dedupe_actions(actions: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for action in actions:
        key = (str(action.get("priority")), str(action.get("dimension")))
        if key in seen:
            continue
        seen.add(key)
        result.append(dict(action))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report what a DatasetWorkspace should acquire, not only what it can prune"
    )
    parser.add_argument("dataset")
    parser.add_argument("--root", type=Path)
    parser.add_argument(
        "--target-type",
        choices=("character", "character_outfit", "style"),
        default=None,
    )
    args = parser.parse_args(argv)
    workspace = DatasetWorkspace.load(args.dataset, root=args.root)
    report = analyze_acquisition_gaps(workspace, target_type=args.target_type)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
