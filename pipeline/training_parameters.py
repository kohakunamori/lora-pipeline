from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .config import read_yaml, repository_root
from .models import PipelineError


DEFAULT_TEXT_ENCODER_LR = 1e-5


@dataclass(frozen=True)
class TrainingParameterSpec:
    key: str
    label_zh: str
    label_en: str
    description_zh: str
    description_en: str
    recommendation_zh: str
    recommendation_en: str


TRAINING_PARAMETER_SPECS = (
    TrainingParameterSpec(
        "network_dim",
        "LoRA Rank",
        "LoRA rank",
        "决定 LoRA 的容量。Rank 越高可表达的细节越多，同时参数量、显存和过拟合风险也会上升。",
        "Controls LoRA capacity. Higher rank can represent more detail, while increasing parameter count, VRAM use, and overfitting risk.",
        "常用 8-64；本项目预设为 16。人物/衣装通常先从 16 或 32 开始。",
        "Commonly 8-64; repository presets use 16. Character/outfit LoRAs usually start at 16 or 32.",
    ),
    TrainingParameterSpec(
        "network_alpha",
        "LoRA Alpha",
        "LoRA alpha",
        "LoRA 的缩放系数，会与 Rank 一起影响更新强度。Alpha 不是越大越好；改变 Alpha 会改变同一学习率下的有效更新尺度。",
        "LoRA scaling factor used together with rank. Larger alpha is not automatically better; changing it changes the effective update scale at the same learning rate.",
        "常见为 Rank 的 1/2 到 1 倍；本项目预设为 Rank 16 / Alpha 8。",
        "Often between one-half of rank and rank; repository presets use rank 16 / alpha 8.",
    ),
    TrainingParameterSpec(
        "unet_lr",
        "UNet 学习率",
        "UNet learning rate",
        "控制 LoRA 在去噪网络中的更新速度。过高更容易过拟合、颜色/构图漂移；过低则可能学不住衣装或人物特征。",
        "Controls LoRA update speed in the denoising network. Too high can overfit or shift color/composition; too low can underlearn identity or outfit detail.",
        "Illustrious/SDXL LoRA 可先用 1e-4；需要更保守时可尝试 5e-5 到 8e-5。",
        "For Illustrious/SDXL LoRA, 1e-4 is a practical starting point; 5e-5 to 8e-5 is more conservative.",
    ),
    TrainingParameterSpec(
        "batch_size",
        "物理 Batch Size",
        "Physical batch size",
        "每个 micro-step 同时放进 GPU 的图片数，直接影响显存。这里不再设置人工上限；是否可用以实际显存/OOM 为准。",
        "Number of images placed on the GPU per micro-step. It directly affects VRAM. The pipeline no longer imposes an artificial upper limit; actual VRAM/OOM determines feasibility.",
        "V100 16GB 预设仍保守使用 1 或 2；可根据实际显存继续提高。固定 images_seen 时，Batch 增大也会减少 optimizer step 数。",
        "V100 16GB presets remain conservative at 1 or 2, but you may raise it based on observed VRAM. With fixed images_seen, larger batches also reduce optimizer-step count.",
    ),
    TrainingParameterSpec(
        "gradient_accumulation_steps",
        "梯度累积步数",
        "Gradient accumulation steps",
        "在执行一次 optimizer step 前累计多个 micro-batch。它可在不按比例增加峰值显存的情况下提高有效 Batch，但会增加一次 optimizer step 的计算时间。",
        "Accumulates multiple micro-batches before one optimizer step. It raises effective batch without proportionally increasing peak VRAM, but each optimizer step takes more compute.",
        "有效 Batch = 物理 Batch x 梯度累积。固定 images_seen 时，提高它同样会减少 optimizer step 数。",
        "Effective batch = physical batch x gradient accumulation. With fixed images_seen, increasing it also reduces optimizer-step count.",
    ),
    TrainingParameterSpec(
        "train_text_encoder",
        "训练 Text Encoder",
        "Train text encoders",
        "SDXL 有两个 Text Encoder。训练它们可让触发词/角色语义绑定更强，但更容易把数据集措辞学死，并增加显存与训练成本。",
        "SDXL has two text encoders. Training them can strengthen trigger/identity binding, but can overfit wording and increases VRAM/compute cost.",
        "默认关闭。只有在触发词绑定明显不足时再开启；开启后 Text Encoder 输出缓存会自动关闭。",
        "Disabled by default. Enable only when trigger binding is clearly weak; text-encoder output caching is automatically disabled.",
    ),
    TrainingParameterSpec(
        "text_encoder_lr1",
        "Text Encoder 1 学习率",
        "Text encoder 1 learning rate",
        "SDXL 第一个 Text Encoder 的 LoRA 学习率。应明显低于 UNet LR，避免语义空间过快漂移。",
        "LoRA learning rate for SDXL text encoder 1. Keep it substantially below the UNet LR to avoid rapid semantic drift.",
        "建议从 1e-5 开始。",
        "Start around 1e-5.",
    ),
    TrainingParameterSpec(
        "text_encoder_lr2",
        "Text Encoder 2 学习率",
        "Text encoder 2 learning rate",
        "SDXL 第二个 Text Encoder 的 LoRA 学习率。与 Encoder 1 一样通常应低于 UNet LR。",
        "LoRA learning rate for SDXL text encoder 2. As with encoder 1, it should normally stay below the UNet LR.",
        "建议从 1e-5 开始。",
        "Start around 1e-5.",
    ),
)


MANAGED_TRAINING_KEYS = frozenset(
    {
        "network_dim",
        "network_alpha",
        "unet_lr",
        "batch_size",
        "gradient_accumulation_steps",
        "network_train_unet_only",
        "cache_text_encoder_outputs",
        "cache_text_encoder_outputs_to_disk",
        "text_encoder_lr1",
        "text_encoder_lr2",
    }
)

_INT_KEYS = ("network_dim", "network_alpha", "batch_size", "gradient_accumulation_steps")
_FLOAT_KEYS = ("unet_lr", "text_encoder_lr1", "text_encoder_lr2")
_BOOL_KEYS = (
    "network_train_unet_only",
    "cache_text_encoder_outputs",
    "cache_text_encoder_outputs_to_disk",
)


def strategy_training_defaults(strategy: str, *, root: Path | None = None) -> dict[str, Any]:
    base = root or repository_root()
    payload = read_yaml(base / "profiles" / "training" / f"{strategy}.yaml")
    training = payload.get("training", {})
    if not isinstance(training, Mapping):
        raise PipelineError(f"Training profile {strategy!r} does not contain a training mapping")
    return copy.deepcopy(dict(training))


def effective_training_settings(
    strategy: str,
    overrides: Mapping[str, Any] | None = None,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    result = strategy_training_defaults(strategy, root=root)
    training_override = dict((overrides or {}).get("training", {}))
    result.update(copy.deepcopy(training_override))
    return result


def update_key_training_overrides(
    overrides: Mapping[str, Any] | None,
    *,
    strategy: str,
    values: Mapping[str, Any],
    train_text_encoder: bool,
    text_encoder_lr1: float | None = None,
    text_encoder_lr2: float | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Replace the user-facing key parameter overrides while preserving unknown expert overrides."""

    result = copy.deepcopy(dict(overrides or {}))
    training = dict(result.get("training", {}))
    for key in MANAGED_TRAINING_KEYS:
        training.pop(key, None)

    defaults = strategy_training_defaults(strategy, root=root)
    for key in _INT_KEYS + ("unet_lr",):
        if key not in values:
            continue
        value = values[key]
        if value != defaults.get(key):
            training[key] = value

    if train_text_encoder:
        training["network_train_unet_only"] = False
        # sd-scripts cannot train text-encoder LoRA modules from cached text-encoder outputs.
        training["cache_text_encoder_outputs"] = False
        training["cache_text_encoder_outputs_to_disk"] = False
        training["text_encoder_lr1"] = float(
            DEFAULT_TEXT_ENCODER_LR if text_encoder_lr1 is None else text_encoder_lr1
        )
        training["text_encoder_lr2"] = float(
            DEFAULT_TEXT_ENCODER_LR if text_encoder_lr2 is None else text_encoder_lr2
        )

    if training:
        result["training"] = training
    else:
        result.pop("training", None)
    validate_training_override_values(result)
    return result


def reset_key_training_overrides(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    result = copy.deepcopy(dict(overrides or {}))
    training = dict(result.get("training", {}))
    for key in MANAGED_TRAINING_KEYS:
        training.pop(key, None)
    if training:
        result["training"] = training
    else:
        result.pop("training", None)
    return result


def validate_training_override_values(overrides: Mapping[str, Any] | None) -> None:
    if not overrides:
        return
    training = overrides.get("training", {})
    if not isinstance(training, Mapping):
        raise PipelineError("Training overrides must contain a training mapping")

    for key in _INT_KEYS:
        if key not in training:
            continue
        value = training[key]
        if isinstance(value, bool):
            raise PipelineError(f"Training parameter {key} must be an integer >= 1")
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise PipelineError(f"Training parameter {key} must be an integer >= 1") from exc
        if parsed < 1 or parsed != value:
            raise PipelineError(f"Training parameter {key} must be an integer >= 1")

    for key in _FLOAT_KEYS:
        if key not in training:
            continue
        value = training[key]
        if isinstance(value, bool):
            raise PipelineError(f"Training parameter {key} must be a positive number")
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise PipelineError(f"Training parameter {key} must be a positive number") from exc
        if parsed <= 0:
            raise PipelineError(f"Training parameter {key} must be a positive number")

    for key in _BOOL_KEYS:
        if key in training and not isinstance(training[key], bool):
            raise PipelineError(f"Training parameter {key} must be true or false")

    has_text_encoder_lr = any(key in training for key in ("text_encoder_lr1", "text_encoder_lr2"))
    unet_only = training.get("network_train_unet_only", True)
    if has_text_encoder_lr and unet_only is not False:
        raise PipelineError(
            "Text-encoder learning rates require network_train_unet_only=false"
        )
    if unet_only is False:
        if "text_encoder_lr1" not in training or "text_encoder_lr2" not in training:
            raise PipelineError(
                "Training SDXL text encoders requires both text_encoder_lr1 and text_encoder_lr2"
            )
        if training.get("cache_text_encoder_outputs") is True or training.get(
            "cache_text_encoder_outputs_to_disk"
        ) is True:
            raise PipelineError(
                "Text-encoder training is incompatible with text-encoder output caching"
            )
