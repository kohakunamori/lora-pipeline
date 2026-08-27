# 视频人物 LoRA 数据准备指南

本页说明 LoRA Pipeline 当前的视频人物数据处理逻辑，重点针对 4K 视频、远景小人物、多人同框和 SDXL/Illustrious 1024 训练。

日常仍然只需要运行：

```text
./lora
```

创建项目时选择：

```text
训练数据来源

1. 图片目录
2. 本地视频文件
3. 在线视频 / YouTube 链接
```

## 总体流程

视频 Character 项目现在按以下顺序处理：

```text
视频
  ↓
按时间间隔抽帧
  ↓
整帧曝光 / 模糊 / 粗粒度近重复过滤
  ↓
DeepGHS 人物 / 头部检测
  ↓
代理图 bbox 映射回原始帧
  ↓
从原始高分辨率帧裁人物
  ↓
crop-level CCIP 身份聚类
  ↓
用户选择目标人物簇
  ↓
按整个数据集平衡 Portrait / 上半身 / 全身 / 环境构图
  ↓
crop-level 近重复过滤
  ↓
最终 Character raw 数据集
  ↓
Caption / Review / Prepare / Train
```

## 4K 视频如何处理

本地 4K 视频不会在抽帧阶段先缩成 1080p。抽出的源帧会保留原始视频分辨率，用于后续人物裁切。

在线视频会优先选择最高不超过 2160p 的视频格式，因此 YouTube 有 4K 源时可以保留到 4K 级别用于人物裁切。

DeepGHS 检测不会直接让检测器处理完整 3840×2160 图像。默认先生成最长边 1600px 的内存代理图：

```text
3840×2160 原始帧
       ↓
1600×900 检测代理图
       ↓
DeepGHS detect_person / detect_heads
       ↓
bbox 按比例映射回 3840×2160
       ↓
从原始 4K 帧真正 crop
```

这样可以降低检测开销，同时保留远景小人物在 4K 源中的真实细节。

## DeepGHS 检测

项目继续使用已经固定依赖的：

```text
dghs-imgutils==0.19.0
```

没有新增另一套人物检测框架，也不会完整引入 waifuc 作为运行时流水线。

处理方式参考 waifuc 的 PersonSplitAction / ThreeStageSplitAction：

- 先 `detect_person` 分离多人画面；
- 全图同时运行 `detect_heads` 作为更稳定的头部证据；
- 对每个单人 person crop 再运行 `detect_halfbody`；
- Person Detector 漏检、但 Head Detector 找到足够大的头部时，可以使用 head-only fallback 推断一个候选人物区域。

如果 DeepGHS 检测 backend 或模型缓存不可用，交互界面会明确警告并退回原来的整帧 CCIP 流程，不会丢弃已经抽出的帧。

## 原生像素质量门槛

系统不会把很小的人物裁出来以后强行 upscale 成 1024。

当前默认质量门槛：

```text
高质量：
head >= 256px
或 person height >= 800px

可用：
head >= 160px
或 person height >= 512px

低质量：
低于以上条件
→ 建项前剔除
```

这些判断依据的是映射回原始帧后的原生像素尺寸，不是放大后的尺寸。

因此：

```text
300×500 小人物
↓
放大成 1200×2000
```

不会被错误地视为高质量素材。

## 保存分辨率上限

人物 crop 完成以后才限制文件尺寸。

默认上限：

```text
最长边 <= 2048px
总像素 <= 4,194,304（约 4MP）
```

规则是：

```text
大于上限
→ LANCZOS downscale

小于上限
→ 保留原始 crop 像素

任何情况
→ 不自动 upscale
```

这样既不会让 4K 整图一直拖到 Tagger / CCIP / latent cache 阶段，也不会提前损失小人物的细节。

## crop-level CCIP

旧流程是：

```text
整张 16:9 视频帧
↓
CCIP
```

现在优先改为：

```text
Frame 001
 ├─ 人物 A crop → CCIP
 ├─ 人物 B crop → CCIP
 └─ 人物 C crop → CCIP

Frame 002
 ├─ 人物 A crop → CCIP
 └─ 人物 C crop → CCIP
```

然后界面显示人物 crop cluster：

```text
人物 crop 身份聚类

簇    候选数    代表人物 crop
0     86        subject-00001.jpg, ...
1     37        subject-00004.jpg, ...
2     15        subject-00019.jpg, ...
```

最大簇仍然只是默认高亮项，不会自动当成真值。用户必须明确选择目标人物。

## 构图多样性而不是“尺寸副本”

选择目标人物以后，不会把同一个 crop 制作成：

```text
768版
1024版
1536版
```

然后重复加入训练。

这样只是在提高同一视觉样本的训练权重，并不是真正的数据多样性。

当前目标是让真实视频帧形成约以下构图分布：

```text
Portrait / 头肩    25%
上半身             30%
全身               30%
环境构图           15%
```

实际比例会根据检测结果调整。例如没有可靠 head bbox 的候选不会强行做 Portrait；多人同框帧不会生成宽松的 Context crop，以避免把其他人物重新带回训练图片。

每个选中的人物候选最多生成一个最终训练构图。因此不会因为三阶段裁切把同一个视频时刻机械扩增成三四张高度相关图片。

## crop-level 去重

整帧预过滤只负责粗粒度去重，默认阈值已经降低，以避免“背景几乎不变、远处人物动作改变”的帧被过早删除。

选定目标人物并生成最终构图以后，会再进行一次人物 crop 级 perceptual hash 去重。

如果某一构图和已有训练图片过于相似，系统会优先尝试该人物候选的其他可用构图；仍然重复才放弃该候选。

## 时间信息

过滤后的帧文件名保留原始采样序号，例如：

```text
video-000001.jpg
video-000003.jpg
video-000007.jpg
```

中间的编号缺口表示候选帧曾被曝光、模糊或粗去重过滤掉。

因此可以由：

```text
(frame_index - 1) × sampling_interval
```

恢复该帧的近似视频时间位置，用于后续排序和 provenance。

## SDXL / Illustrious 训练分辨率

视频预处理不会把同一图片人工制作成多种训练尺寸。

训练端继续使用当前 V100 profile：

```text
resolution = 1024
enable_bucket = true
bucket_no_upscale = true
bucket_reso_steps = 32
```

因此不同生成分辨率和长宽比的泛化主要来自：

- Portrait / 上半身 / 全身 / Context 的人物尺度差异；
- 不同真实 crop 的长宽比；
- SDXL aspect-ratio buckets；
- Illustrious/SDXL 底模自身已经具备的多分辨率能力。

不是来自同一图的多尺寸副本。

## 为什么 Context 只保留一部分

环境构图对于防止人物 LoRA 只会“大头照”很重要，但多人视频里过宽的 crop 也容易重新包含其他人物。

当前默认规则：

- 一帧只检测到一个可用人物时，可以进入 Context 候选；
- 一帧检测到多个可用人物时，只生成较紧的人物构图，不生成 Context。

后续正常 Character Identity / Review 仍然建议开启，作为第二层安全检查。

## 最终项目记录

`project.yaml` 的 `video_source.identity_preselection` 会记录摘要，包括：

- DeepGHS 检测方法；
- 输入帧数量；
- 可用人物 crop 数量；
- head-only fallback 数量；
- 原生分辨率过低的剔除数量；
- 检测代理图尺寸；
- 人物/头部最低像素门槛；
- crop 保存像素上限；
- CCIP 人物簇选择；
- 最终 Portrait / 上半身 / 全身 / Context 数量；
- crop-level 重复剔除数量；
- 因保存尺寸上限发生 downscale 的数量；
- `upscale_generated: false`。

因此后续可以判断一个 LoRA 的数据到底来自什么筛选策略，而不需要猜当时如何处理视频。

## NAS / V100 注意事项

DeepGHS 模型第一次运行时可能需要从模型缓存或网络加载权重。Pipeline 不提供安装向导，也不会自行升级 Torch/CUDA。

部署仍由 Codex 负责准备环境。日常用户只操作 `./lora` 交互菜单。

GitHub CPU CI 使用 fake detector 验证 bbox 映射、尺寸限制、构图平衡和 fallback 行为；真实 DeepGHS 模型和 V100 推理仍应在 NAS 上做一次实际 smoke test。
