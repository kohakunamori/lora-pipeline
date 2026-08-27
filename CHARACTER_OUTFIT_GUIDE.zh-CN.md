# 人物衣装 LoRA 训练指南

`character_outfit` 用于学习**某个人物的一套特定服装 / SSR 衣装 / 水着 / 舞台服**。

它不是新的底层 Project 类型。Dataset 和运行时仍使用 `character`，因此继续复用人物身份检查、Tagger、训练器和已有 Character 管线；`character_outfit` 只是更精确的 Training Config 目标。

## 三类信息必须分开

以秦谷美鈴特定水着衣装为例：

```yaml
concept_type: character
target_type: character_outfit
trigger: misuzu_nic26
anchor_tags:
  - hataya misuzu

evaluation:
  subject_prompt: 1girl
```

实际运行时评测主体会自动解析为：

```text
1girl, hataya misuzu
```

训练 Caption 在 `generate`、`hybrid`、`existing_taglist_clean` 模式下会固定以以下前缀开头：

```text
misuzu_nic26, hataya misuzu, ...可变 Tag...
```

含义：

- `misuzu_nic26`：唯一 LoRA Trigger，负责激活所学习的衣装概念。
- `hataya misuzu`：人物身份 Anchor，让底模知道衣装属于谁，但不把人物名本身变成 LoRA Trigger。
- 后续 Tag：服装细节、姿势、表情、构图、背景、光照等可变内容，降低错误绑定。

## 不要这样配置

错误：

```text
Trigger: hataya misuzu, misuzu_nic26,
```

Trigger 必须是单个 token / 短语，不能是逗号分隔的 Prompt。

错误：

```text
Evaluation subject: 1girl, hataya misuzu, misuzu_nic26
```

评测会自动生成 Trigger ON / OFF 对照。如果基础评测 Prompt 已经包含 Trigger，OFF 组也会被激活，泄漏测试失效。Training Config、Preflight 和 Evaluation case builder 都会阻止这种配置。

## Caption 安全规则

`character_outfit` 会把 Trigger + Anchor 当成不可裁剪的固定前缀。Token 预算不足时，优先裁剪后面的可变 Tag，不会先删人物 Anchor。

如果固定前缀本身已经超过 Caption token 预算，Caption 阶段会直接报错，而不是悄悄丢 Anchor。

`allow_trigger_only` 回退在衣装目标中实际写入：

```text
trigger, anchor1, anchor2, ...
```

因此即使没有普通 Caption，也不会丢失人物身份上下文。

### passthrough / skip

`existing_passthrough` 的语义仍然是“原样使用已有文本”，不会偷偷改写用户文件；`skip` 也不会自动清洗已有 Caption。

为了避免这两个显式高级模式绕过衣装上下文，Preflight 会检查每条最终 Prepared Caption：

- 必须包含 LoRA Trigger；
- 必须包含所有 Character Anchor；
- 缺任意一项都会 BLOCKED，训练不会启动。

日常衣装训练推荐使用默认 `auto`。当 Dataset 每张图都有 Tag 时，它会选择 `existing_taglist_clean` 并自动补齐固定前缀；否则选择自动生成。

## 衣装专用评测

普通 Character LoRA 需要测试 `different outfit`，因为目标是让人物身份脱离训练服装。

人物衣装 LoRA 的目标相反：服装本身就是需要学习的概念。因此 `character_outfit` 使用独立评测矩阵，不再把 `different outfit` 当核心成功条件。

Screening 默认覆盖：

- portrait
- full body
- different expression

Full evaluation 默认覆盖：

- portrait
- upper body
- full body
- different expression
- dynamic pose
- complex background
- indoor
- outdoor
- day
- night

每个 case 仍生成严格对齐的 Trigger ON / OFF 两组：

```text
ON : misuzu_nic26, 1girl, hataya misuzu, <case>
OFF:                1girl, hataya misuzu, <case>
```

这样可以观察：

1. **Identity preservation**：Trigger ON 后人物身份是否仍稳定；CCIP 仅作为辅助指标。
2. **Outfit retention**：改变姿势、表情、构图、背景、时间后目标衣装是否保持。
3. **Outfit leakage**：不写 Trigger、只保留人物 Anchor 时，目标衣装是否仍系统性出现。
4. **Strength sensitivity**：不同 LoRA strength 下衣装是否从不足到过拟合出现合理变化。

当前依赖中没有经过验证的服装语义自动评分器，因此项目不会用 CCIP 身份距离冒充“衣装相似度”。衣装保持和衣装泄漏被明确记录为 `manual_review_required`，并提供对齐 contact sheet 与 ON/OFF pair coverage；最终 Promote 仍需人工确认。

## CLI 推荐流程

进入训练配置后选择：

```text
训练目标
1 人物
2 人物衣装
3 风格
```

衣装示例：

```text
训练目标: 人物衣装
唯一 Trigger token: misuzu_nic26
人物锚点 Tag: hataya misuzu
images_seen: 2000
评测主体基础 Prompt: 1girl
```

若 Trigger 输入逗号列表，CLI 会在该字段立即要求重输，不会等填写完整份配置后才失败。

## Web UI

Web 的 Training Config 页面也提供：

- Training target
- Single Trigger
- Character anchors
- Evaluation base subject prompt

服务端使用与 CLI 相同的 `TrainingConfig.validate()`，因此 Web 表单不能绕过 Trigger、Anchor 或评测污染规则。

## 兼容性

旧配置无需迁移：

- 旧 `concept_type: character` 自动视为 `target_type: character`。
- 旧 `concept_type: style` 自动视为 `target_type: style`。
- `character_outfit` 才要求 `anchor_tags`。
- Project/Dataset 的底层 concept schema 仍只有 `character` / `style`，避免破坏已有训练、身份检查和历史快照。
