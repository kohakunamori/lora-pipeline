# 训练参数：预设与关键参数自定义

LoRA Pipeline 继续以 `quality`、`fast`、`cached` 作为推荐起点，但训练配置可以只覆盖少数关键参数。未覆盖的参数始终跟随所选预设，因此用户不需要复制整份 `sd-scripts` 配置。

## 三套预设

| 参数 | quality | fast | cached |
| --- | ---: | ---: | ---: |
| 物理 Batch | 1 | 2 | 1 |
| 梯度累积 | 1 | 1 | 1 |
| LoRA Rank | 16 | 16 | 16 |
| LoRA Alpha | 8 | 8 | 8 |
| UNet LR | 1e-4 | 1e-4 | 1e-4 |
| Seed | 42 | 42 | 42 |
| Text Encoder 输出缓存 | 关闭 | 关闭 | 开启 |

`cached` 是 U-Net-only + Text Encoder 输出缓存路径；当前关键参数面板不会悄悄切换到另一种 Text Encoder 训练模式。

## 可自定义的关键参数

### `images_seen`

训练图片的累计曝光预算，是本项目比较不同 Batch 策略时的主预算。比如 `images_seen=2000` 表示训练计划目标是累计处理约 2000 张训练图片，而不是固定跑相同 optimizer steps。

人物/衣装 LoRA 可以先从 1000-3000 起步，再根据评测决定是否继续训练。数据集很小时要特别注意过拟合。

### `network_dim` / LoRA Rank

决定 LoRA 容量。Rank 越高，可表达的细节通常越多，同时参数量、显存占用和过拟合风险也会上升。

常见范围是 8-64。仓库默认 16；人物或复杂衣装可以从 16 或 32 开始，不建议仅因为显存有余量就盲目增大。

### `network_alpha` / LoRA Alpha

LoRA 的缩放系数，与 Rank 一起影响更新强度。Alpha 不是越大越好，同一学习率下改变 Alpha 也会改变有效更新尺度。

常见做法是 Alpha 取 Rank 的 1/2 到 1 倍；仓库默认 Rank 16 / Alpha 8。

### `unet_lr` / UNet 学习率

控制 LoRA 在去噪网络中的更新速度。过高更容易出现过拟合、颜色漂移、构图黏死等问题；过低则可能学不住人物或衣装细节。

Illustrious / SDXL LoRA 可以先用 `1e-4`。需要更保守时可以尝试 `5e-5` 到 `8e-5`。

### `batch_size` / 物理 Batch Size

每个 micro-step 同时放入 GPU 的图片数，直接影响显存占用。

**当前不再设置 `max_physical_batch_1024=2` 之类的人工上限。** 用户可以填写 3、4 或更高，只要实际训练能够放进显存。预设仍保守使用 1 或 2，但这是默认值，不是安全上限。

显存占用会随分辨率、bucket、Rank、缓存方式和模型变化，因此不要根据 Batch 1 的显存简单线性推算最大 Batch；最终以真实运行是否 OOM 为准。

### `gradient_accumulation_steps` / 梯度累积

在执行一次 optimizer step 前累计多个 micro-batch。它可以在不按比例增加峰值显存的情况下提高有效 Batch，但每次 optimizer step 需要更多 micro-step 计算。

```text
effective_batch = batch_size * gradient_accumulation_steps
```

### `seed`

控制训练中的随机顺序和随机过程，主要用于实验复现。没有固定的“更好 Seed”。比较 Rank/LR/Batch 等参数时应尽量固定 Seed；只有做鲁棒性验证时再换 Seed 重跑。

## Batch 与 `images_seen` 的关系

本项目以 `images_seen` 作为主预算，而不是强行让不同 Batch 跑相同 optimizer steps：

```text
effective_batch = physical_batch * gradient_accumulation
optimizer_steps = ceil(images_seen / effective_batch)
actual_images_seen = optimizer_steps * effective_batch
```

因此，把 Batch 从 1 提到 4 时：

- 通常吞吐会提高；
- optimizer step 数会减少；
- 梯度噪声和优化轨迹也会变化；
- 所以它不是“完全等价、只变快”的开关，也不能简单理解成“一定降低效果”。

正确做法是保持相同 `images_seen`，再用固定评测矩阵比较结果。

## 配置示例

只覆盖 Batch 与 Rank：

```yaml
strategy: quality
images_seen: 2000
overrides:
  training:
    batch_size: 4
    network_dim: 32
```

其余参数仍来自 `quality` 预设。如果之后把策略切换为 `fast`，这里明确写出的 Batch 4 / Rank 32 会保留，而未覆盖参数会采用 `fast` 的默认值。

人物衣装示例：

```yaml
concept_type: character
target_type: character_outfit
trigger: misuzu_nic26
anchor_tags:
  - hataya misuzu
strategy: quality
images_seen: 2000
overrides:
  training:
    batch_size: 4
    network_dim: 32
    network_alpha: 16
    unet_lr: 0.00008
```

CLI 和 Web 都会显示这些参数的用途、建议以及当前预设值。Web 中留空即恢复策略默认值；CLI 的“恢复关键参数预设”只删除这些用户可见的关键覆盖项，不会误删其他专家级 override。
