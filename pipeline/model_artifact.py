"""Portable metadata and thumbnail helpers for trained LoRA artifacts.

The training backend owns the sd-scripts invocation, while this module keeps
ModelSpec resolution, conservative trigger inference, and safetensors
post-processing independent and easy to test.  Metadata values written to a
safetensors header are always strings; the run metadata keeps the richer,
human-friendly representation (for example, tags as a list).
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image, ImageOps

from .models import PipelineError


SUPPORTED_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})
DEFAULT_THUMBNAIL_SIZE = 256

# This is deliberately a small vocabulary plus deterministic pattern checks,
# rather than a dataset-specific trigger blacklist.  These tags describe
# composition, generic subjects, or common photographic attributes and are
# therefore poor trigger candidates in most Danbooru-style datasets.
GENERIC_TAGS = frozenset(
    {
        "1girl",
        "1boy",
        "2girls",
        "2boys",
        "3girls",
        "3boys",
        "4girls",
        "4boys",
        "multiple_girls",
        "multiple_boys",
        "multiple_people",
        "solo",
        "duo",
        "group",
        "person",
        "people",
        "looking at viewer",
        "looking_at_viewer",
        "smile",
        "grin",
        "laughing",
        "long hair",
        "long_hair",
        "short hair",
        "short_hair",
        "medium hair",
        "medium_hair",
        "standing",
        "sitting",
        "lying",
        "indoors",
        "outdoors",
        "upper body",
        "upper_body",
        "full body",
        "full_body",
        "portrait",
        "close-up",
        "close_up",
        "simple background",
        "simple_background",
        "white background",
        "white_background",
        "black background",
        "black_background",
        "no humans",
        "no_humans",
        "day",
        "night",
        "interior",
        "landscape",
        "scenery",
        "solo focus",
        "solo_focus",
    }
)
_COUNTED_SUBJECT_RE = re.compile(
    r"^(?:\d+|many|multiple)?(?:girl|girls|boy|boys|person|people|humans?)$"
)


@dataclass(frozen=True)
class TriggerInference:
    phrase: str | None
    confidence: float | None = None
    candidates: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class ModelArtifactMetadata:
    """Resolved user-facing metadata for one LoRA run."""

    title: str | None = None
    author: str | None = None
    description: str | None = None
    license: str | None = None
    tags: tuple[str, ...] = ()
    trigger_phrase: str | None = None
    trigger_source: str = "none"
    trigger_confidence: float | None = None
    trigger_candidates: tuple[dict[str, Any], ...] = ()
    usage_hint: str | None = None
    thumbnail_source: str = "none"
    thumbnail_path: Path | None = None
    thumbnail_source_path: Path | None = None
    merged_from: tuple[str, ...] = ()

    def as_run_dict(self) -> dict[str, Any]:
        """Return the stable, JSON-serializable run-metadata representation."""

        return {
            "title": self.title,
            "author": self.author,
            "description": self.description,
            "license": self.license,
            "tags": list(self.tags),
            "trigger_phrase": self.trigger_phrase,
            "trigger_source": self.trigger_source,
            "trigger_confidence": self.trigger_confidence,
            "trigger_candidates": [dict(item) for item in self.trigger_candidates],
            "usage_hint": self.usage_hint,
            "merged_from": list(self.merged_from),
            "thumbnail_source": self.thumbnail_source,
            "thumbnail_path": str(self.thumbnail_path) if self.thumbnail_path else None,
            "thumbnail_source_path": (
                str(self.thumbnail_source_path) if self.thumbnail_source_path else None
            ),
        }


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw: Iterable[Any] = value.split(",")
    elif isinstance(value, (list, tuple, set, frozenset)):
        raw = sorted(value, key=lambda item: str(item).casefold()) if isinstance(value, (set, frozenset)) else value
    else:
        raise PipelineError("metadata.tags must be a string or a list of strings")
    result: list[str] = []
    seen: set[str] = set()
    for item in raw:
        text = _optional_text(item)
        if text is None:
            continue
        key = text.casefold()
        if key not in seen:
            result.append(text)
            seen.add(key)
    return result


def _normalize_merged_from(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values: Iterable[Any] = value.split(",")
    elif isinstance(value, (list, tuple, set, frozenset)):
        values = value
    else:
        raise PipelineError("metadata.merged_from must be a string or a list of strings")
    return [text for item in values if (text := _optional_text(item)) is not None]


def normalize_metadata_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize the optional ``metadata`` section without changing defaults.

    The returned mapping is intentionally plain data so it can be included in
    fingerprints and safely merged with project/CLI configuration.
    """

    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise PipelineError("metadata must be a mapping")
    thumbnail_value = config.get("thumbnail", {})
    if thumbnail_value is None:
        thumbnail_value = {}
    if not isinstance(thumbnail_value, Mapping):
        raise PipelineError("metadata.thumbnail must be a mapping")
    path_value = thumbnail_value.get("path")
    thumbnail_path = str(path_value).strip() if path_value is not None else None
    if thumbnail_path == "":
        thumbnail_path = None
    return {
        "title": _optional_text(config.get("title")),
        "author": _optional_text(config.get("author")),
        "description": _optional_text(config.get("description")),
        "license": _optional_text(config.get("license")),
        "tags": _normalize_tags(config.get("tags")),
        "trigger_phrase": _optional_text(config.get("trigger_phrase")),
        "usage_hint": _optional_text(config.get("usage_hint")),
        "merged_from": _normalize_merged_from(config.get("merged_from")),
        "thumbnail": {
            "enabled": bool(thumbnail_value.get("enabled", False)),
            "path": thumbnail_path,
        },
    }


def _caption_values(captions: Any) -> list[str]:
    """Accept a caption sequence, snapshot records, or a snapshot manifest."""

    if isinstance(captions, (str, Path)):
        payload = json.loads(Path(captions).read_text(encoding="utf-8"))
        return _caption_values(payload)
    if isinstance(captions, Mapping):
        if "images" in captions:
            return _caption_values(captions["images"])
        if "caption" in captions:
            value = captions["caption"]
            return [str(value)] if value is not None else []
        return []
    if captions is None:
        return []
    result: list[str] = []
    for item in captions:
        if isinstance(item, Mapping):
            value = item.get("caption")
        else:
            value = item
        if value is not None:
            text = str(value).strip()
            if text:
                result.append(text)
    return result


def _is_generic_tag(tag: str) -> bool:
    normalized = tag.casefold().replace("-", "_")
    if normalized in {item.replace("-", "_") for item in GENERIC_TAGS}:
        return True
    if _COUNTED_SUBJECT_RE.fullmatch(normalized):
        return True
    if "background" in normalized or normalized.endswith(("_view", "_viewer")):
        return True
    # Generic quality/composition descriptors are poor triggers even when a
    # particular spelling was not in the static vocabulary above.
    generic_words = {
        "anime",
        "style",
        "illustration",
        "artwork",
        "masterpiece",
        "highres",
        "absurdres",
        "general",
        "tagme",
        "character",
        "clothing",
        "dress",
        "expression",
        "hair",
        "head",
        "body",
        "face",
        "eye",
        "eyes",
        "solo",
        "standing",
        "sitting",
        "outdoor",
        "indoor",
    }
    if normalized in generic_words:
        return True
    # Appearance/composition attributes commonly occur as compound tags
    # (``blue_hair``, ``red_eyes``, ``school_uniform``).  Treat those as
    # generic unless they have an identifier-like token of their own.
    compound_generic_words = {
        "hair",
        "eyes",
        "eye",
        "dress",
        "shirt",
        "skirt",
        "outfit",
        "clothing",
        "uniform",
        "female",
        "male",
        "person",
        "people",
        "body",
        "face",
        "portrait",
        "background",
    }
    return any(token in compound_generic_words for token in normalized.split("_"))


def infer_trigger_analysis(captions: Any) -> TriggerInference:
    """Infer a trigger conservatively from prepared captions.

    A candidate must occur in nearly every image and must not look like a
    generic composition/subject tag.  Ranking favours coverage, early caption
    position, and identifier-like tags.  All calculations are deterministic.
    """

    values = _caption_values(captions)
    image_count = len(values)
    # One caption cannot establish dataset-wide coverage, so auto mode stays
    # conservative and asks for an explicit trigger in that case.
    if image_count < 2:
        return TriggerInference(None, None, ())
    image_counts: Counter[str] = Counter()
    display: dict[str, str] = {}
    positions: defaultdict[str, list[int]] = defaultdict(list)
    for caption in values:
        tags = [part.strip() for part in caption.split(",") if part.strip()]
        seen_in_image: set[str] = set()
        for index, tag in enumerate(tags):
            key = tag.casefold()
            if key in seen_in_image:
                continue
            seen_in_image.add(key)
            display.setdefault(key, tag)
            image_counts[key] += 1
            positions[key].append(index)

    candidates: list[dict[str, Any]] = []
    for key, count in image_counts.items():
        tag = display[key]
        if _is_generic_tag(tag):
            continue
        # Prepared Danbooru captions represent multi-word tags with
        # underscores.  Plain prose fragments are too ambiguous for an
        # automatic trigger and are therefore left to explicit configuration.
        if " " in tag and "_" not in tag:
            continue
        coverage = count / image_count
        required = 1.0 if image_count == 1 else max(0.5, 1.0 - 1.0 / image_count)
        if coverage < required:
            continue
        average_position = sum(positions[key]) / len(positions[key])
        max_position = max((len(item.split(",")) for item in values), default=1) - 1
        front_score = 1.0 if max_position <= 0 else max(0.0, 1.0 - average_position / max_position)
        # Identifier-like forms are more distinctive than ordinary prose.
        identifier_score = 0.25 if ("_" in tag or re.fullmatch(r"[A-Za-z0-9]+", tag)) else 0.0
        score = round(0.58 * coverage + 0.27 * front_score + 0.15 * (0.5 + identifier_score), 6)
        candidates.append(
            {
                "tag": tag,
                "image_count": count,
                "dataset_coverage": round(coverage, 6),
                "average_position": round(average_position, 6),
                "score": score,
            }
        )
    candidates.sort(
        key=lambda item: (
            -float(item["score"]),
            -float(item["dataset_coverage"]),
            float(item["average_position"]),
            str(item["tag"]).casefold(),
        )
    )
    if not candidates:
        return TriggerInference(None, None, ())
    winner = candidates[0]
    confidence = round(
        min(
            1.0,
            0.65 * float(winner["dataset_coverage"])
            + 0.25 * (1.0 - min(1.0, float(winner["average_position"]) / max(1, len(values))))
            + 0.1 * (1.0 if "_" in str(winner["tag"]) else 0.5),
        ),
        6,
    )
    # A single surviving candidate is still rejected when it is weakly
    # distinctive (for example, a caption set containing only "anime").
    if confidence < 0.55:
        return TriggerInference(None, confidence, tuple(candidates))
    return TriggerInference(str(winner["tag"]), confidence, tuple(candidates))


def infer_trigger_phrase(captions: Any) -> str | None:
    """Return only the inferred phrase for callers that do not need evidence."""

    return infer_trigger_analysis(captions).phrase


def _validate_image_path(path: Path) -> None:
    if path.suffix.casefold() not in SUPPORTED_IMAGE_SUFFIXES:
        allowed = ", ".join(sorted(SUPPORTED_IMAGE_SUFFIXES))
        raise PipelineError(f"Thumbnail must use a common image format ({allowed}): {path}")
    if not path.is_file():
        raise PipelineError(f"Thumbnail file does not exist: {path}")
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:
        raise PipelineError(f"Thumbnail is not a readable image: {path} ({exc})") from exc


def find_preview_image(samples_dir: Path) -> Path | None:
    """Select a stable preview from ``samples_dir`` without relying on fs order."""

    if not samples_dir.is_dir():
        return None
    files = sorted(
        (
            path
            for path in samples_dir.iterdir()
            if path.is_file() and path.suffix.casefold() in SUPPORTED_IMAGE_SUFFIXES
        ),
        key=lambda path: path.name.casefold(),
    )
    preferred = [path for path in files if path.stem.casefold() == "preview"]
    return (preferred or files)[0] if (preferred or files) else None


def prepare_thumbnail(source: Path, output_dir: Path, *, max_size: int = DEFAULT_THUMBNAIL_SIZE) -> Path:
    """Create a derived JPEG thumbnail, preserving transparency and the source."""

    source = Path(source).expanduser()
    _validate_image_path(source)
    if max_size < 1:
        raise PipelineError("Thumbnail max_size must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "model-thumbnail.jpg"
    try:
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened)
            if image.mode in ("RGBA", "LA") or "transparency" in image.info:
                rgba = image.convert("RGBA")
                background = Image.new("RGB", rgba.size, (255, 255, 255))
                background.paste(rgba, mask=rgba.getchannel("A"))
                image = background
            else:
                image = image.convert("RGB")
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            image.save(destination, format="JPEG", quality=88, optimize=True)
    except Exception as exc:
        raise PipelineError(f"Could not prepare thumbnail {source}: {exc}") from exc
    return destination


def resolve_model_metadata(
    merged: Mapping[str, Any],
    *,
    run_dir: Path,
    captions: Any = None,
    samples_dir: Path | None = None,
    allow_sample: bool = True,
    thumbnail_max_size: int = DEFAULT_THUMBNAIL_SIZE,
) -> ModelArtifactMetadata:
    """Resolve config, trigger evidence, and the best available thumbnail."""

    raw_config = merged.get("metadata", {})
    config = normalize_metadata_config(raw_config)
    usage_hint_explicit = (
        isinstance(raw_config, Mapping)
        and "usage_hint" in raw_config
        and raw_config.get("usage_hint") is not None
    )
    requested_trigger = config["trigger_phrase"]
    inference = TriggerInference(None, None, ())
    if requested_trigger and requested_trigger.casefold() == "auto":
        inference = infer_trigger_analysis(captions)
        trigger = inference.phrase
        trigger_source = "auto" if trigger else "none"
    elif requested_trigger:
        trigger = requested_trigger
        trigger_source = "explicit"
    else:
        trigger = None
        trigger_source = "none"
    usage_hint = config["usage_hint"]
    if usage_hint is None and trigger and not usage_hint_explicit:
        usage_hint = f"Use trigger phrase: {trigger}"

    thumbnail_source = "none"
    thumbnail_source_path: Path | None = None
    thumbnail_path: Path | None = None
    thumbnail_config = config["thumbnail"]
    if thumbnail_config["enabled"]:
        if not thumbnail_config["path"]:
            raise PipelineError("metadata.thumbnail.enabled is true but metadata.thumbnail.path is empty")
        thumbnail_source_path = Path(thumbnail_config["path"]).expanduser()
        thumbnail_path = prepare_thumbnail(
            thumbnail_source_path,
            Path(run_dir) / "config",
            max_size=thumbnail_max_size,
        )
        thumbnail_source = "explicit"
    elif allow_sample:
        thumbnail_source_path = find_preview_image(samples_dir or (Path(run_dir) / "samples"))
        if thumbnail_source_path is not None:
            thumbnail_path = prepare_thumbnail(
                thumbnail_source_path,
                Path(run_dir) / "config",
                max_size=thumbnail_max_size,
            )
            thumbnail_source = "sample"

    return ModelArtifactMetadata(
        title=config["title"],
        author=config["author"],
        description=config["description"],
        license=config["license"],
        tags=tuple(config["tags"]),
        trigger_phrase=trigger,
        trigger_source=trigger_source,
        trigger_confidence=inference.confidence,
        trigger_candidates=inference.candidates,
        usage_hint=usage_hint,
        thumbnail_source=thumbnail_source,
        thumbnail_path=thumbnail_path,
        thumbnail_source_path=thumbnail_source_path,
        merged_from=tuple(config["merged_from"]),
    )


def _thumbnail_data_url(path: Path) -> str:
    _validate_image_path(path)
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def build_modelspec_metadata(metadata: ModelArtifactMetadata) -> dict[str, str]:
    """Build standard ``modelspec.*`` keys for post-processing."""

    values: dict[str, str] = {}
    for field_name, key in (
        ("title", "modelspec.title"),
        ("author", "modelspec.author"),
        ("description", "modelspec.description"),
        ("license", "modelspec.license"),
        ("usage_hint", "modelspec.usage_hint"),
        ("trigger_phrase", "modelspec.trigger_phrase"),
    ):
        value = getattr(metadata, field_name)
        if value is not None:
            values[key] = str(value)
    if metadata.tags:
        values["modelspec.tags"] = ", ".join(metadata.tags)
    if metadata.merged_from:
        values["modelspec.merged_from"] = ", ".join(metadata.merged_from)
    if metadata.thumbnail_path:
        values["modelspec.thumbnail"] = _thumbnail_data_url(metadata.thumbnail_path)
    return values


def build_sd_scripts_metadata(metadata: ModelArtifactMetadata) -> dict[str, str]:
    """Build sd-scripts TOML keys (thumbnail remains a source file path)."""

    values: dict[str, str] = {}
    for field_name, key in (
        ("title", "metadata_title"),
        ("author", "metadata_author"),
        ("description", "metadata_description"),
        ("license", "metadata_license"),
        ("usage_hint", "metadata_usage_hint"),
        ("trigger_phrase", "metadata_trigger_phrase"),
    ):
        value = getattr(metadata, field_name)
        if value is not None:
            values[key] = str(value)
    if metadata.tags:
        values["metadata_tags"] = ", ".join(metadata.tags)
    if metadata.merged_from:
        values["metadata_merged_from"] = ", ".join(metadata.merged_from)
    if metadata.thumbnail_path:
        values["metadata_thumbnail"] = str(metadata.thumbnail_path)
    return values


def rewrite_safetensors_metadata(path: Path, updates: Mapping[str, Any]) -> dict[str, str]:
    """Atomically merge metadata into a safetensors file.

    Tensors are loaded through the official ``safe_open`` API and written with
    ``save_file``.  The original file is only replaced after the temporary file
    has been completely written and fsynced, so failures leave it readable.
    """

    try:
        from safetensors import safe_open
        try:
            from safetensors.torch import save_file
            framework = "pt"
        except ImportError:
            # Config-only/CPU tooling may not install torch.  The format and
            # atomic rewrite semantics are identical for NumPy tensors.
            from safetensors.numpy import save_file

            framework = "np"
    except ImportError as exc:  # pragma: no cover - exercised in env smoke tests
        raise PipelineError("safetensors is required to rewrite model metadata") from exc
    path = Path(path)
    if not path.is_file():
        raise PipelineError(f"Safetensors checkpoint does not exist: {path}")
    clean_updates = {str(key): str(value) for key, value in updates.items() if value is not None}
    try:
        with safe_open(str(path), framework=framework, device="cpu") as handle:
            tensors = {key: handle.get_tensor(key) for key in handle.keys()}
            old_metadata = dict(handle.metadata() or {})
    except Exception as exc:
        raise PipelineError(f"Could not read safetensors checkpoint {path}: {exc}") from exc
    new_metadata = {
        str(key): str(value)
        for key, value in {**old_metadata, **clean_updates}.items()
        if value is not None
    }
    temporary_name: str | None = None
    try:
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        os.close(fd)
        save_file(tensors, temporary_name, metadata=new_metadata)
        with open(temporary_name, "rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    except Exception as exc:
        raise PipelineError(f"Could not atomically rewrite safetensors metadata {path}: {exc}") from exc
    finally:
        if temporary_name:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
    return new_metadata
