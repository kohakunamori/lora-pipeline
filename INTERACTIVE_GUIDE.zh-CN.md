# LoRA Pipeline 交互式使用指南

日常使用不需要记忆子命令、参数或 Bash 脚本。

在仓库目录中只需要启动一次：

```text
./lora
```

通过 Python 包安装时也可以使用：

```text
lora-pipeline
```

无参数启动后会进入持续运行的编号菜单。输入数字并按回车即可。

## 首页

首页会显示：

- 最近使用的项目；
- 每个项目的完成进度；
- 系统推荐的下一步；
- 已启用的底模数量。

首页操作包括：

1. 打开已有项目；
2. 创建新项目；
3. 管理底模；
4. 检查当前机器；
5. 退出。

## 创建项目

选择 **Create a project** 后会先询问训练数据来源：

- **Image directory**：使用现有图片目录；
- **Video / YouTube URL**：从本地视频或公开视频 URL 自动抽取人物 LoRA 训练帧。

普通图片目录模式会依次询问并即时检查：

- 项目名称；
- Character 或 Style；
- 底模；
- 数据集目录；
- Trigger；
- Quality、Fast 或 Cached 策略；
- `images_seen` 训练曝光预算。

选择数据集后会先显示图片数量和现有 caption 数量。最终确认页还会显示估算的等效 epoch，确认后才真正创建项目。

没有注册底模时，向导会直接进入底模管理器，而不是要求退出后手写命令。

## 从视频或 YouTube 创建人物 LoRA

选择 **Video / YouTube URL** 后，只需要粘贴 YouTube 链接或输入本地视频路径。

视频导入依赖：

- `ffmpeg`：本地视频和 URL 视频都需要；
- `yt-dlp`：只有 HTTP/HTTPS 视频 URL 需要。

缺少依赖时会在交互界面直接提示，不会影响普通图片项目。

向导会询问：

- 项目名称；
- 底模；
- YouTube URL 或本地视频路径；
- 每隔多少秒采样一个候选帧；
- 在人物识别之前最多保留多少过滤帧；
- URL 下载使用的网络/代理方式；
- CCIP 识别出的目标人物簇；
- Trigger；
- 训练策略；
- `images_seen` 曝光预算。

### YouTube 网络与代理

远程 URL 会单独询问下载网络方式。代理只传给当前视频下载的 `yt-dlp` 进程，不会修改训练、Hugging Face 或系统的全局网络设置。

如果当前环境已经定义代理，向导会自动检测，并提供：

- **Use detected environment proxy**：读取 `LORA_VIDEO_PROXY`、`HTTPS_PROXY`、`ALL_PROXY` 等环境变量；
- **Connect directly**：显式忽略代理环境变量；
- **Use a custom proxy for this video**：仅本次下载使用手工输入的代理。

支持常见形式，例如：

```text
http://127.0.0.1:7890
socks5://127.0.0.1:1080
```

如果代理 URL 包含用户名或密码，凭据只存在于当前进程的 `yt-dlp` 参数中。项目配置不会保存代理密码，只记录代理模式和脱敏后的 endpoint。

下载失败时不会直接退出建项流程，而会让你选择：

- 使用相同网络设置重试；
- 修改代理；
- 取消视频导入。

### 抽帧与画面质量过滤

抽帧不是简单地把所有连续帧都塞进训练集。导入器会在正式创建项目之前执行：

- 时间间隔采样；
- 过暗和过曝帧过滤；
- 模糊帧过滤；
- perceptual hash 近重复过滤；
- 最大帧数限制。

完成后会显示：候选帧数量、过滤后数量、模糊帧淘汰数量、近重复淘汰数量和曝光异常淘汰数量。

### 建项前选择目标人物

过滤完成后，视频模式会在项目真正创建之前调用与 Character Identity 步骤相同的 CCIP backend，对保留下来的画面进行人物身份聚类。

界面会列出：

- 每个 CCIP cluster 的编号；
- cluster 内帧数；
- 若干代表帧文件名；
- CCIP outlier 数量。

系统不会自动假设“最大 cluster 就是目标角色”。最大 cluster 只是默认高亮项，你需要在菜单里明确确认目标人物 cluster。

选择某个 cluster 后：

- 只有这个人物簇的画面会进入新项目的 `raw/`；
- 其他人物 cluster 不进入训练集；
- CCIP outlier 不进入训练集；
- cluster 统计和最终选择会写入 `project.yaml` 供以后追溯。

如果同一角色因为不同服装、极端角度或强遮挡被 CCIP 拆成多个 cluster，可以选择 **Keep all filtered frames**，然后在正常的 Identity/Review 阶段继续人工清理。CCIP backend 不可用或无法形成稳定 cluster 时，也会提供这个安全回退，而不会丢弃已经抽出的画面。

确认创建项目后，只有最终选中的训练帧会被复制到 `projects/<name>/raw/`，临时下载的视频和未选候选帧会被删除。项目配置会记录原始视频 URL、采样间隔、过滤统计、代理的脱敏信息以及 CCIP 选择结果，便于以后追溯数据来源。

视频模式固定创建 **Character** 项目。即使已经做过建项前 CCIP 选择，默认工作流中的 Identity 和 Review 仍然建议保持开启，因为多人同框、背影、严重遮挡和极远景仍可能逃过单次聚类筛选。

对于人物视频，通常不要把采样间隔设得过小。相邻视频帧的信息增量很低，大量近似姿势会增加过拟合风险。默认每 2 秒采样一次、CCIP 前最多保留 250 帧是偏保守的起点，最终应以选定人物簇中的视角、表情、姿势和服装覆盖情况为准。

## 项目仪表盘

打开项目后会显示：

- 项目类型、底模、Trigger 和训练策略；
- 训练曝光预算；
- 每个流水线步骤的状态；
- 最近一次训练 run；
- 系统推荐的下一项操作。

常用操作：

1. **Continue recommended work**：按照保存的工作流偏好继续执行；
2. **Run one step**：单独运行、重试或强制重跑某个步骤；
3. **Workflow preferences**：保存 caption、去重、审核和评测偏好；
4. **Project settings**：修改曝光预算、训练策略或 Character 评测主体描述；
5. **Import validation images**：从目录导入独立验证集并自动检查训练集泄漏；
6. **Evaluate checkpoints**：通过编号选择 Screening 或 Full Evaluation；
7. **Promote a checkpoint**：人工确认后生成 `best.safetensors`；
8. **Status and artifacts**：查看数据、报告、contact sheet、run 和输出路径；
9. **Advanced recovery**：只用于预览、锁恢复或显式跳过 preflight。

## 可保存的工作流偏好

项目会记住以下选择：

- 是否运行重复图片检测；
- 是否自动排除多余的完全重复图片；
- Character 项目是否运行身份一致性检查；
- Caption 模式；
- 是否允许缺失 caption 时使用 trigger-only；
- 是否生成审核摘要；
- 训练后是否自动运行 Screening Evaluation。

以后在项目仪表盘选择 **Continue recommended work** 即可复用这些设置。

## 项目设置

在 **Project settings** 中可以直接修改：

- `images_seen` 训练曝光预算；
- Quality、Fast 或 Cached 训练策略；
- Character Evaluation 使用的 subject prompt。

修改曝光预算或训练策略只会让 Preflight、Train 和 Evaluation 重新变为待执行，不会重新复制原图或重建不相关的前处理结果。

修改 Evaluation subject prompt 只会让 Evaluation 失效，不会要求重新训练。

## 导入独立验证集

在项目仪表盘选择 **Import validation images**，再输入保存 holdout 图片的目录。

向导会：

- 统计可用图片；
- 对训练集和待导入图片计算内容哈希；
- 检测到与训练集完全相同的图片时阻止整次导入；
- 跳过验证集中已经存在的相同内容；
- 在文件名冲突但内容不同时生成安全的新文件名；
- 只让 Evaluation 失效，不会要求重新训练。

因此不需要使用 `cp`、文件管理脚本或手工进入 `projects/<name>/validation/`。

## Caption 模式

菜单提供五种明确模式：

- **Generate captions**：使用 Tagger 生成并清洗；
- **Use existing captions unchanged**：原样保留已有 `.txt`；
- **Clean existing tag lists**：把已有 caption 当作标签列表清洗；
- **Hybrid existing + generated**：合并已有内容与 Tagger 建议；
- **Skip the caption step**：跳过该步骤，prepare 时必须已有可用 caption。

## 训练与断点恢复

进入训练时，向导会：

- 检查是否已完成 preflight；
- 询问是否使用保存的 `images_seen` 预算；
- 检测中断的 run；
- 在发现可恢复状态时优先提供 **Resume interrupted training**；
- 训练成功后询问是否立即进行 Screening Evaluation。

不需要手写 run ID 或 `--resume` 参数。

## 评测与晋升

Screening Evaluation 会快速比较候选 checkpoint。

Full Evaluation 会显示 checkpoint 编号，并且只允许选择一到两个终选模型，从界面层和后端层同时阻止过大的评测矩阵。

评测完成后会直接显示：

- checkpoint × strength 对比图；
- prompt × checkpoint 对比图；
- trigger on/off 泄漏对比图；
- HTML 报告路径。

人工审阅后，在仪表盘选择 **Promote a checkpoint**，再选择 run、checkpoint 和推荐强度。只有这一步会生成：

```text
best.safetensors
best.yaml
```

## 底模管理

底模管理器支持：

- 注册单个 checkpoint；
- 扫描目录并通过编号注册；
- 查看 checkpoint 元数据；
- 完整重新计算 SHA256 并核对底模身份。

## 错误与项目锁

普通错误会显示在红色面板中，并返回当前菜单，不会直接退出整个程序。

检测到 stale 或无法验证的项目锁时，界面会解释原因，并询问是否重试解除；仍在运行的真实进程锁不会被覆盖。

按 `Ctrl+C` 会取消当前交互操作，并保留已经写入的项目状态。

## 高级命令仍然可用

原有的非交互式子命令继续保留，适合自动化或调试。但正常的创建、视频下载与代理选择、视频人物簇筛选、训练、恢复、评测、晋升、验证集导入、项目设置和底模管理均可在编号菜单中完成。
