from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .config import read_yaml, repository_root
from .models import PipelineError


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
        "images_seen",
        "图片曝光预算 images_seen",
        "Image exposure budget (images_seen)",
        "整个训练计划希望模型累计看到多少张训练图片，是本项目比较不同 Batch 策略时的主预算。",
        "Canonical exposure budget: how many training images the model should see in total. It is the primary budget used to compare different batch strategies.",
        "人物/衣装 LoRA 可先从 1000-3000 起步，再根据评测决定是否追加；数据很少时注意过拟合。",
        "Character/outfit LoRAs often start around 1000-3000 exposures, then extend only if evaluation supports it; watch overfitting on small datasets.",
    ),
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
        "每个 micro-step 同时放进 GPU 的图片数，直接影响显存。这里不设置人工上限；是否可用以实际显存/OOM 为准。",
        "Number of images placed on the GPU per micro-step. It directly affects VRAM. The pipeline does not impose an artificial upper limit; actual VRAM/OOM determines feasibility.",
        "V100 16GB 预设仍保守使用 1 或 2；可按实测显存继续提高。固定 images_seen 时，Batch 增大也会减少 optimizer step 数。",
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
        "seed",
        "随机种子 Seed",
        "Random seed",
        "控制训练中的随机顺序和随机过程，主要用于复现实验。更换 Seed 可能让最终结果略有差异，但不存在固定的“更好 Seed”。",
        "Controls stochastic ordering and random processes for reproducibility. Different seeds can produce slightly different results, but there is no universally better seed.",
        "默认 42。对比参数时尽量固定 Seed；做鲁棒性验证时再换 Seed 重跑。",
        "Default is 42. Keep it fixed when comparing parameters; vary it only when testing robustness.",
    ),
)


MANAGED_TRAINING_KEYS = frozenset(
    {
        "network_dim",
        "network_alpha",
        "unet_lr",
        "batch_size",
        "gradient_accumulation_steps",
        "seed",
    }
)

_POSITIVE_INT_KEYS = (
    "network_dim",
    "network_alpha",
    "batch_size",
    "gradient_accumulation_steps",
)


def strategy_training_defaults(strategy: str, *, root: Path | None = None) -> dict[str, Any]:
    requested_base = root or repository_root()
    profile = requested_base / "profiles" / "training" / f"{strategy}.yaml"
    if not profile.is_file():
        bundled_base = Path(__file__).resolve().parents[1]
        bundled_profile = bundled_base / "profiles" / "training" / f"{strategy}.yaml"
        if bundled_profile.is_file():
            profile = bundled_profile
    payload = read_yaml(profile)
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
    root: Path | None = None,
) -> dict[str, Any]:
    """Replace user-facing key overrides while preserving unknown expert overrides."""

    result = copy.deepcopy(dict(overrides or {}))
    training = dict(result.get("training", {}))
    for key in MANAGED_TRAINING_KEYS:
        training.pop(key, None)

    defaults = strategy_training_defaults(strategy, root=root)
    for key in MANAGED_TRAINING_KEYS:
        if key not in values:
            continue
        value = values[key]
        if value != defaults.get(key):
            training[key] = value

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

    for key in _POSITIVE_INT_KEYS:
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

    if "seed" in training:
        value = training["seed"]
        if isinstance(value, bool):
            raise PipelineError("Training parameter seed must be an integer >= 0")
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise PipelineError("Training parameter seed must be an integer >= 0") from exc
        if parsed < 0 or parsed != value:
            raise PipelineError("Training parameter seed must be an integer >= 0")

    if "unet_lr" in training:
        value = training["unet_lr"]
        if isinstance(value, bool):
            raise PipelineError("Training parameter unet_lr must be a positive number")
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise PipelineError("Training parameter unet_lr must be a positive number") from exc
        if parsed <= 0:
            raise PipelineError("Training parameter unet_lr must be a positive number")
