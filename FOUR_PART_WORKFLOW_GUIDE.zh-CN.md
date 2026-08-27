# 四区交互工作流

新的无参数交互界面把日常 LoRA 工作明确拆成四个一级区域：

1. **数据集（Dataset）**
2. **训练配置（Training Config）**
3. **训练状态（Training Status）**
4. **训练结果（Training Results）**

`ProjectState` 仍保留在内部，主要用于兼容旧项目、锁、步骤指纹、resume 和现有 sd-scripts 编排。新工作流不要求用户把 Project 当成日常一级概念。

## 1. 数据集

Dataset 是长期、可持续编辑的数据资产。

一个 Dataset 可以包含多个独立 Source：

- 图片目录
- 本地视频
- 在线 / YouTube 视频
- 从某个来源派生的 DeepGHS 智能人物裁切来源

每个 Source 分开保存，可以单独：

- 启用 / 停用
- 人物裁切
- 自动检查
- 自动 Tag
- 人工修改 Tag
- 人工排除 / 恢复图片

排除默认是 soft delete，不物理删除唯一原图。

视频来源继续复用已有能力：HDR → SDR tone mapping、4K 原图人物 crop、DeepGHS 人物/头部检测、crop-level CCIP、构图平衡、代理和 cookies.txt。

## 2. 训练配置

Training Config 是可复用的“怎么训练”配方，不绑定某一个 Dataset 版本。

当前可管理：

- Character / Style 类型
- 底模
- Trigger
- quality / fast / cached 策略
- `images_seen`
- LoRA rank (`network_dim`)
- LoRA alpha (`network_alpha`)
- UNet learning rate
- dedup / identity / caption / review 工作流偏好
- Character 评测 subject prompt

默认情况下 rank / alpha / LR 继承对应训练策略 profile；只有显式自定义时才写入 project overrides。

Training Config 的 `caption_mode=auto` 表示：

- Dataset 所有 active 图片都有 Tag：运行时使用 `existing_taglist_clean`
- Dataset 存在缺失 Tag：运行时使用 `generate`

评测不在训练阶段自动执行；四区 UI 中评测属于“训练结果”。

## 3. 训练状态

“训练状态”负责：

- 开始一次训练
- 查看当前准备 / 训练状态
- 恢复中断训练
- 重试失败步骤
- 查看底层技术 Project（高级）

开始训练时选择：

```text
Dataset + Training Config
```

然后同时冻结：

```text
Dataset Snapshot
+
Training Config Snapshot
```

例如：

```text
Dataset: misuzu
snapshot: 7f32...

Training Config: character-quality
snapshot: 91ac...
```

之后即使继续修改 `misuzu` 数据集或 `character-quality` 配置，这次训练的输入也不会变化。

内部仍创建一个兼容 Project workspace，名称类似：

```text
run-misuzu-character-qua-20260827-120000
```

这是技术实现细节；日常 UI 以 Dataset / Config / Run 为主。

每一个真正的 sd-scripts Run 还会额外保存：

```text
runs/<run-id>/config/run-snapshot.yaml
```

其中记录该次 Run 的 Dataset 和 Training Config 快照。Resume 时不会重写这个 manifest。

## 4. 训练结果

只有已经产出有效 checkpoint 的 Run 才进入“训练结果”。

结果页负责：

- 查看所有 `.safetensors` checkpoint
- 查看示例图片路径
- 查看 contact sheet
- 运行 / 重跑 Screening
- 选择 1–2 个 finalist 进行 Full evaluation
- 人工选择最佳 checkpoint
- 创建 `best.safetensors` 和 `best.yaml`

结果目录仍沿用已有 Run artifact 结构，例如：

```text
runs/<run-id>/
├── checkpoints/
├── samples/
│   ├── screening/
│   └── full/
├── contact-sheets/
├── metrics/
├── report.html
├── best.safetensors
└── best.yaml
```

## 推荐日常流程

```text
./lora
  ↓
数据集
  ↓
创建 / 导入多个 Source
  ↓
裁切、Tag、人工审核、排除差图
  ↓
返回
  ↓
训练配置
  ↓
创建或调整训练配方
  ↓
返回
  ↓
训练状态
  ↓
开始一次新训练
  ↓
选择 Dataset + Training Config
  ↓
冻结两个 Snapshot
  ↓
准备 / 训练 / resume
  ↓
返回
  ↓
训练结果
  ↓
查看权重和示例图
  ↓
Screening / Full evaluation
  ↓
人工选择 best
```

## 与旧 Project 的兼容

旧 `projects/*` 不迁移、不删除，仍可以恢复和继续运行。

主页中的“系统与高级 → 技术 Project 视图”保留原有 Project dashboard，用于：

- 旧项目
- 专家级单步执行
- 锁恢复
- 技术排障

新创建的四区训练也会在内部使用 ProjectState，因此已有的步骤指纹、resume、不可变 prepared generations、evaluation 和 promotion 逻辑仍然复用，而不是重写训练器。
