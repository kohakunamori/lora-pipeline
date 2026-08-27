# Dataset Workspace 数据集工作区

新的交互结构把 **数据集维护** 和 **训练项目** 分开。

```text
Dataset（可持续修改）
  ├─ Source A：图片目录
  ├─ Source B：本地视频处理结果
  ├─ Source C：在线视频处理结果
  └─ Source D：从某个 Source 派生的智能人物裁切
       │
       ├─ 自动检查
       ├─ 自动 Tag
       ├─ 人工 Tag 修改
       ├─ 人工排除 / 恢复
       └─ 来源启用 / 停用
              │
              ▼
       immutable snapshot
              │
              ▼
Project（训练项目，不随 Dataset 后续修改）
              │
              ▼
sd-scripts / Evaluation
```

## 为什么分开

以前“创建项目”同时承担：选择数据来源、视频下载、抽帧、人物筛选、Tag、训练配置。随着视频 HDR、4K 裁切、CCIP、Cookie 等功能增加，单个向导会越来越重。

现在：

- **Dataset** 是数据资产，可以持续追加和清理；
- **Source** 是一次导入或一次派生处理，始终独立保存；
- **Project** 只关心某个时刻的数据集快照和训练配置。

因此，同一个人物可以先导入：

- 角色截图目录；
- 第一段 4K HDR 视频；
- 第二段普通 SDR 视频；
- 后续补充的高质量官方图；

全部组成一个 Dataset，而不需要创建四个训练项目再手工合并文件。

## 首页

`./lora` 的主要入口现在是：

1. 管理数据集；
2. 从数据集创建训练项目；
3. 打开已有训练项目；
4. 管理底模；
5. 检查当前机器；
6. 退出。

日常数据处理应优先进入 **管理数据集**。

## Dataset 目录结构

示例：

```text
datasets/my-character/
├── dataset.yaml
├── sources/
│   ├── image-directory-001/
│   │   └── images/
│   ├── local-video-001/
│   │   └── images/
│   ├── remote-video-001/
│   │   └── images/
│   └── smart-crop-001/
│       └── images/
├── review/
│   ├── exclusions.yaml
│   ├── audit.json
│   └── tagging-last.json
└── cache/
    ├── tagger/
    └── work/
```

不同 Source 即使存在完全相同的文件名也不会冲突。

## 导入多个来源

Dataset Dashboard 中选择 **导入新的数据来源**。

Character Dataset 支持：

- 图片目录；
- 本地视频文件；
- 在线视频 / YouTube。

Style Dataset 第一版以图片目录为主；Character 专用的视频人物检测和 CCIP 不会误套到 Style Dataset。

视频来源继续复用已有能力：

- 代理 / cookies.txt；
- HDR PQ / HLG 自动 tone mapping；
- 4K 高分辨率人物裁切；
- DeepGHS person/head/halfbody detection；
- crop-level CCIP；
- 构图平衡；
- 模糊、曝光、近重复过滤。

视频处理结束后，结果作为一个独立 Source 加入 Dataset，而不是立刻创建训练项目。

## 按来源管理

每个 Source 都可以单独：

- 启用 / 停用；
- 自动检查；
- 自动打 Tag；
- 人工修改 Tag；
- 人工审核 / 排除；
- Character Dataset 中从这个 Source 生成新的智能人物裁切 Source。

**停用来源不会删除任何文件。** 它只是不会进入之后新建的训练项目快照。

### 派生人物裁切来源

图片 Source 可以做：

```text
原 Source
  ↓
DeepGHS person/head detection
  ↓
原图 bbox crop
  ↓
crop-level CCIP
  ↓
选择目标人物
  ↓
Portrait / upper-body / full-body / context balance
  ↓
新的 smart-crop Source
```

原 Source 保持不变。创建派生 Source 后，界面默认会询问是否停用原 Source，避免“完整原图 + 裁切图”同时进入训练导致重复权重。

## 自动检查与自动排除

自动检查会读取图片并记录：

- 损坏文件；
- 完全重复内容；
- 过小分辨率；
- 极端长宽比；
- 动画图片。

自动处理遵循保守原则：

### 可以安全自动排除

- 损坏文件；
- 完全重复图片中除一个 canonical copy 以外的副本。

### 只标记，不自动排除

- 边长低于 512 的图片；
- 极端长宽比；
- animated GIF / animated WebP 等。

这些可能仍然有训练价值，所以留给人工审核决定。

## 人工排除：默认是可恢复的 soft delete

审核页面按 25 张分页，并提供稳定编号。

可以输入：

```text
1,3-5,18
```

一次排除多张图片。

排除结果写入：

```text
review/exclusions.yaml
```

图片文件本身不会删除。被排除的图片：

- 不进入 Dataset active view；
- 不进入新 Project snapshot；
- 随时可以用同样的编号操作恢复。

这相当于训练侧“删除”，但避免误删唯一源素材。

## 自动 Tag

Dataset 可以直接使用现有的 DeepGHS / WD Tagger。

默认行为：

- Tag threshold：0.35；
- 已经存在 `.txt` 的图片 **不覆盖**；
- 只处理启用、未排除图片；
- Tagger cache 保存在 Dataset 自己的 `cache/tagger/`；
- WD character-name suggestions 只记录在 tagging report，不直接写入通用 Tag sidecar，降低人物名泄漏风险。

如果需要完全重打，可以在界面明确选择覆盖已有 Tag。

## 人工修改 Tag

每张图片都可以：

- 替换全部 Tag；
- 追加 Tag；
- 删除指定 Tag；
- 清空 Tag。

重复 Tag 会按规范化名称去重。

人工 Tag 是 Dataset 数据资产的一部分，不依赖某个具体训练项目。

## 从 Dataset 创建 Project

创建训练项目时：

1. 只选取 **enabled source**；
2. 跳过所有 **excluded image**；
3. 计算每张图片 SHA256；
4. 计算每个 caption SHA256；
5. 生成 Dataset snapshot hash；
6. 按 Source ID 保存到 Project 的 `raw/`；
7. snapshot 信息写入 `project.yaml`。

例如：

```text
projects/train-v1/raw/
├── image-directory-001/
├── local-video-001/
└── smart-crop-001/
```

Project 创建完成后，即使你继续：

- 给 Dataset 加 Source；
- 修改 Tag；
- 排除图片；
- 停用 Source；

这个 Project 的 `raw/` 都不会变化。

要训练新的 Dataset 状态，就再创建一个新的 Project snapshot。

## Dataset Tag 和项目 Trigger

Dataset 层的 Tag 不绑定训练 Trigger。

例如 Dataset 中保存：

```text
1girl, blue hair, school uniform, upper body, smile
```

从 Dataset 创建 Character Project，Trigger 为：

```text
zz_my_character
```

如果所有 active image 都已有 Tag，Project 的交互工作流默认选择：

```text
existing_taglist_clean
```

Project Caption 步骤会变成：

```text
zz_my_character, 1girl, blue hair, school uniform, upper body, smile
```

因此 Dataset 可以被多个不同 Project / Trigger 重用，人工 Tag 也不会被自动 Tagger 再次覆盖。

## 推荐日常流程

```text
./lora
  ↓
管理数据集
  ↓
创建 Character Dataset
  ↓
导入图片 Source A
  ↓
导入本地 4K HDR 视频 Source B
  ↓
按来源检查 / 裁切
  ↓
全数据集自动检查
  ↓
自动 Tag
  ↓
人工审核：排除不合适图片
  ↓
人工修改重点图片 Tag
  ↓
从数据集创建训练项目
  ↓
运行现有 inspect / dedup / identity / caption / review / prepare / train / evaluate
```

Dataset 层的清洗不会取消 Project 层的 `dedup / identity / review`。Project 层仍作为最后一道训练前一致性检查，以防 Dataset 中存在跨来源重复、身份混入或 caption 问题。
