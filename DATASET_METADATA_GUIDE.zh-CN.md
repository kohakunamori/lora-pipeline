# Dataset 构图与图片 Metadata

Dataset 现在不仅保存图片和 Tag，还为每个 Source 保存图片级 provenance：

```text
datasets/<dataset>/sources/<source-id>/
├── images/
└── metadata.json
```

`metadata.json` 不进入训练图片目录，不会被 sd-scripts 当成训练样本。

## 构图类型

人物 Dataset 使用以下统一分类：

- `portrait`：头肩 / 胸像，身份细节优先
- `upper_body`：上半身
- `three_quarter`：3/4 身
- `full_body`：全身
- `context`：人物较小或环境构图本身有价值
- `unknown`：尚未分析或无法稳定判断

构图标签不是训练 Tag 的替代品。它们属于 Dataset provenance，用于审核、筛选、统计、构图平衡和训练快照追溯。

## 图片变体

- `original`：普通原始导入图
- `original_full`：智能裁切流程额外保留的高价值完整画面
- `smart_crop`：DeepGHS 检测后生成的人物智能裁切
- `derived_crop`：其他派生裁切

同一原图/视频帧的多个变体共享 `source_group_id`，便于识别“它们其实来自同一个视觉样本”。

## 视频构图策略 v2

视频人物处理仍然先在低分辨率代理图上做 DeepGHS 检测，再从原始高分辨率帧裁切。新的构图目标约为：

```text
portrait       20%
upper_body     25%
three_quarter  20%
full_body      20%
context        15%
```

每个检测人物默认只产生一个主要训练构图。不会机械生成 portrait + upper-body + full-body 三份，也不会制造 768/1024/1536 等多分辨率副本。

为了保留高质量完整构图，单人物、高原生质量、主体占比合理、分辨率足够的帧可以额外保留 `original_full`。默认要求 `full_keep_score >= 0.72`，目标量约为检测人物数的 12%，因此它是少量补充而不是整套数据翻倍。

upper-body / portrait / full-body 的 margin 也比旧版略宽，降低帽带、手臂、裙摆等人物组成刚好贴边或被截断的概率。

## 普通图片来源

普通图片目录导入后会先显示：

```text
composition: unknown
analysis.status: not_analyzed
```

Web Dataset 页面可以针对整个 Dataset 或单个 Source 点击“后台分析构图”。人物 Dataset 会使用现有 `dghs-imgutils` 的动漫人物/头部检测器补充：

- `person_bbox`
- `head_bbox`
- `person_count`
- `subject_height_ratio`
- `subject_area_ratio`
- `head_height_ratio`
- `head_to_person_ratio`
- `full_keep_score`
- resolution / quality tier

分析只写 metadata，不修改图片、不裁图片，也不自动排除图片。

## Web 图片墙

`数据集 -> 来源` 页面可以按：

- composition
- variant
- active / excluded

筛选图片。

每张图片卡片会显示主要 provenance，例如：

```text
[Upper body / 上半身] [智能裁切] [保留] [CCIP 0]
913×1190 · high · 人物占比 72% · 头/人物 34%
组：video-000123:subject-00001 · 分析：derived_from_video_detection
```

Tag 仍然可以在同一张卡片直接人工修改。

## Source / Dataset 统计

Dataset 和每个 Source 都会汇总 active 图片的构图分布以及 metadata 分析进度，从而快速发现：

- upper-body 过多
- full-body / context 不足
- 大量图片尚未分析
- 某个 Source 的构图结构明显失衡

## Training Run 快照

真正启动训练时，除了原来的 Dataset image/caption snapshot，还会冻结一份独立的 `dataset_metadata_snapshot`，包括每张 active 图片当时的：

- `composition_type`
- `variant_kind`
- `source_group_id`
- resolution
- person/head 比例和 bbox
- quality
- CCIP identity metadata

并记录独立 `dataset_metadata_snapshot_hash`。

因此 Run 启动后，即使继续修改 Dataset 的构图 metadata，历史 Run 的 provenance 也不会漂移。

## 删除语义

- 删除整个 Source：Source 目录和它的 `metadata.json` 一起删除。
- 删除 Dataset：整个 Dataset metadata 一起删除，但不影响已经冻结的 Run。
- Web 永久删除单张图片：同时清理对应 stale metadata。
- 日常数据清洗仍推荐优先使用可恢复的“排除”。
