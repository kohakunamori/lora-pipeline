from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable


DEFAULT_LANGUAGE = "zh-CN"
SUPPORTED_LANGUAGES = {"zh-CN", "en"}
_LANGUAGE = "en"
_HOOKS_INSTALLED = False


# Exact values are translated only when the whole visible cell/prompt matches.
_EXACT_ZH: dict[str, str] = {
    "Home": "主页",
    "Open a project": "打开项目",
    "Create a project": "创建项目",
    "Manage base models": "管理底模",
    "Check this machine": "检查当前机器",
    "Exit": "退出",
    "Back": "返回",
    "Back to home": "返回主页",
    "Character": "人物",
    "Style": "风格",
    "character": "人物",
    "style": "风格",
    "Quality": "质量优先",
    "Fast": "速度优先",
    "Cached": "缓存模式",
    "quality": "质量优先",
    "fast": "速度优先",
    "cached": "缓存模式",
    "done": "已完成",
    "skipped": "已跳过",
    "failed": "失败",
    "running": "运行中",
    "interrupted": "已中断",
    "pending": "待执行",
    "trained": "已训练",
    "evaluated": "已评测",
    "promoted": "已晋升",
    "inspect": "检查数据集",
    "dedup": "去重",
    "identity": "人物一致性",
    "caption": "标签/描述",
    "review": "人工审核",
    "prepare": "准备训练集",
    "preflight": "训练前检查",
    "train": "训练",
    "evaluate": "评测",
    "Screening": "快速筛选",
    "Full": "完整评测",
    "screening": "快速筛选",
    "full": "完整评测",
    "Generate captions": "自动生成标签/描述",
    "Use existing captions unchanged": "原样使用已有描述",
    "Clean existing tag lists": "清洗已有标签列表",
    "Hybrid existing + generated": "已有描述 + 自动生成混合",
    "Skip the caption step": "跳过标签/描述步骤",
    "Project settings": "项目设置",
    "Workflow preferences": "工作流偏好",
    "Evaluate checkpoints": "评测 Checkpoint",
    "Promote a checkpoint": "晋升 Checkpoint",
    "Status and artifacts": "状态与产物",
    "Advanced recovery": "高级恢复",
    "Import validation images": "导入验证集图片",
    "Change image-exposure budget": "修改训练曝光预算",
    "Change training strategy": "修改训练策略",
    "Change evaluation subject prompt": "修改评测主体提示词",
    "Image directory": "图片目录",
    "Video / YouTube URL": "视频 / YouTube 链接",
    "Training data source": "训练数据来源",
    "Base checkpoint": "训练底模",
    "Training strategy": "训练策略",
    "Caption mode": "标签/描述模式",
    "Project summary": "项目摘要",
    "Video project summary": "视频项目摘要",
    "Video frame filtering": "视频帧筛选",
    "Video character identity clusters": "视频人物身份聚类",
    "Target character": "目标人物",
    "Project dashboard": "项目仪表盘",
    "Pipeline steps": "流水线步骤",
    "Guided run plan": "引导式运行计划",
    "Recent projects": "最近项目",
    "Registered base models": "已注册底模",
    "Base model manager": "底模管理",
    "Project paths": "项目路径",
    "Training and evaluation runs": "训练与评测记录",
    "Existing evaluation evidence": "已有评测结果",
    "Current project settings": "当前项目设置",
    "Promotion summary": "晋升摘要",
    "Machine checks": "机器检查",
    "Name": "名称",
    "Concept": "类型",
    "Base": "底模",
    "Dataset": "数据集",
    "Images": "图片",
    "Existing captions": "已有描述",
    "Trigger": "触发词",
    "Strategy": "策略",
    "Image exposures": "图片曝光次数",
    "Approx. equivalent epochs": "约等效 Epoch",
    "Field": "字段",
    "Value": "值",
    "Setting": "设置",
    "Video source": "视频来源",
    "Filtered frames": "筛选后帧数",
    "Selected training frames": "选中训练帧",
    "Sampling interval": "采样间隔",
    "Selected CCIP cluster": "选中的 CCIP 簇",
    "Download proxy": "下载代理",
    "Video filtered frames": "视频筛选后帧数",
    "Video selected frames": "视频选中帧数",
    "Video CCIP cluster": "视频 CCIP 簇",
    "Video proxy": "视频代理",
    "Evaluation subject": "评测主体",
    "Project": "项目",
    "Type": "类型",
    "Progress": "进度",
    "Recommended next action": "建议下一步",
    "Updated": "更新时间",
    "Step": "步骤",
    "Status": "状态",
    "Attempts": "尝试次数",
    "Details": "详情",
    "Current": "当前状态",
    "Plan": "计划",
    "Available": "可用",
    "Path": "路径",
    "Stage": "阶段",
    "Checkpoints": "Checkpoint",
    "Checkpoint": "Checkpoint",
    "Report": "报告",
    "Completed": "完成时间",
    "Size": "大小",
    "Run": "运行记录",
    "Evaluation": "评测",
    "Promoted": "已晋升",
    "Item": "项目",
    "Metric": "指标",
    "Count": "数量",
    "Cluster": "簇",
    "Frames": "帧数",
    "Representative frame files": "代表帧文件",
    "Sampled candidates": "采样候选帧",
    "Accepted before identity selection": "人物筛选前保留帧",
    "Rejected: blurry": "剔除：模糊",
    "Rejected: near-duplicate": "剔除：近重复",
    "Rejected: exposure": "剔除：曝光异常",
    "Run one step": "单独运行一步",
    "Continue recommended work": "继续建议的工作",
    "Preview next action": "预览下一步",
    "Record preflight bypass": "记录跳过训练前检查",
    "Register a checkpoint": "注册 Checkpoint",
    "Scan a directory": "扫描目录",
    "Inspect a checkpoint": "检查 Checkpoint",
    "Fully verify a checkpoint": "完整校验 Checkpoint",
    "Training mode": "训练模式",
    "Resume interrupted training": "恢复中断的训练",
    "Start a new training run": "开始新的训练",
    "Evaluation stage": "评测阶段",
    "Training run to evaluate": "选择要评测的训练记录",
    "Evaluated run to promote": "选择要晋升的已评测记录",
    "Checkpoint to promote": "选择要晋升的 Checkpoint",
    "Select one or two finalists": "选择 1-2 个终选 Checkpoint",
    "Choose a pipeline step": "选择流水线步骤",
    "Select a project": "选择项目",
    "Project name": "项目名称",
    "Dataset directory": "数据集目录",
    "Trigger token": "触发词",
    "Checkpoint .safetensors path": "Checkpoint .safetensors 路径",
    "Base id": "底模 ID",
    "Display name": "显示名称",
    "Directory to scan": "要扫描的目录",
    "Checkpoint to inspect": "要检查的 Checkpoint",
    "Image exposure budget": "图片曝光预算",
    "New image-exposure budget": "新的图片曝光预算",
    "Evaluation subject prompt": "评测主体提示词",
    "YouTube URL or local video path": "YouTube 链接或本地视频路径",
    "Sample one frame every N seconds": "每隔 N 秒采样一帧",
    "Maximum accepted frames before identity selection": "人物筛选前最多保留帧数",
    "YouTube / video download network": "YouTube / 视频下载网络",
    "Download recovery": "下载失败处理",
    "Retry with the same network settings": "使用相同网络设置重试",
    "Change proxy settings": "修改代理设置",
    "Cancel video import": "取消视频导入",
    "Use detected environment proxy": "使用检测到的环境代理",
    "Connect directly (ignore proxy environment variables)": "直接连接（忽略代理环境变量）",
    "Use a custom proxy for this video": "为本次视频使用自定义代理",
    "Keep all filtered frames": "保留全部筛选后帧",
    "CCIP outliers": "CCIP 离群帧",
    "Network": "网络",
    "Source": "来源",
    "Destination": "目标目录",
    "Imported": "已导入",
    "Already present": "已存在",
    "PASS": "通过",
    "WARN": "警告",
    "FAIL": "失败",
    "yes": "是",
    "no": "否",
    "not set": "未设置",
    "not created": "尚未创建",
    "not inspected": "未检查",
    "not evaluated": "未评测",
    "unregistered": "未注册",
    "N/A": "不适用",
    "skip": "跳过",
    "run or reuse": "运行或复用",
}


# Longer visible phrases are translated inside dynamic strings and Rich markup.
_REPLACEMENTS_ZH: tuple[tuple[str, str], ...] = tuple(
    sorted(
        {
            "Interactive mode - choose actions by number; command-line flags are optional.": "交互模式——输入编号选择操作，不需要记忆命令行参数。",
            "Resume work from a visual project dashboard.": "从项目仪表盘继续之前的工作。",
            "Guided project creation with immediate validation.": "通过向导创建项目，并立即检查输入。",
            "Register, scan, inspect, or verify checkpoints.": "注册、扫描、检查或校验本地 Checkpoint。",
            "Run environment and V100 compatibility checks.": "检查环境与 V100 兼容性。",
            "Leave without changing project state.": "退出，不修改项目状态。",
            "Create a LoRA project": "创建 LoRA 项目",
            "Each answer is validated before the next question. Nothing is created until the summary is confirmed.": "每一项输入都会先验证；确认最终摘要之前不会真正创建项目。",
            "No enabled base checkpoint is registered yet.": "当前还没有已启用的训练底模。",
            "Open base model manager now?": "现在打开底模管理器吗？",
            "Project creation needs at least one enabled base model.": "创建项目至少需要一个已启用的底模。",
            "Concept type": "LoRA 类型",
            "Identity consistency, controllability, and leakage review.": "人物身份一致性、可控性与触发词泄漏检查。",
            "Cross-content coverage and dataset-bias diagnostics.": "跨内容覆盖与数据集偏置诊断。",
            "Create this project?": "创建这个项目吗？",
            "Project creation cancelled.": "已取消创建项目。",
            "Project created": "项目已创建",
            "Configure the guided workflow now?": "现在配置引导式工作流吗？",
            "Open the project dashboard?": "打开项目仪表盘吗？",
            "Choose and run, retry, or force a specific pipeline step.": "选择某一步进行运行、重试或强制重跑。",
            "Save caption, review, and screening defaults.": "保存标签、审核和快速评测的默认偏好。",
            "Run screening or full evaluation without flags.": "无需命令参数即可运行快速或完整评测。",
            "Create best.safetensors after human review.": "人工审核后生成 best.safetensors。",
            "Inspect project paths, runs, reports, and outputs.": "查看项目路径、训练记录、报告和输出。",
            "Preview work or record an expert preflight bypass.": "预览待执行工作，或以专家模式跳过训练前检查。",
            "Return to the project list.": "返回项目列表。",
            "The core pipeline has no pending steps.": "核心流水线没有待执行步骤。",
            "Start the guided run now?": "现在开始引导式运行吗？",
            "Everything in the selected plan was already reusable.": "所选计划中的所有结果都可以直接复用。",
            "Guided run finished": "引导式运行完成",
            "Recommended next action:": "建议下一步：",
            "Guided workflow preferences": "引导式工作流偏好",
            "These choices are saved in project.yaml and reused the next time you press Continue.": "这些选择会保存到 project.yaml，下次选择“继续建议的工作”时自动复用。",
            "Run duplicate detection?": "运行重复图片检测吗？",
            "Automatically exclude redundant exact copies?": "自动排除完全重复的多余副本吗？",
            "Run character identity consistency checks?": "运行人物身份一致性检查吗？",
            "Character identity checks are not applicable to Style projects.": "风格 LoRA 不需要人物身份一致性检查。",
            "Run the configured tagger and clean the result.": "运行已配置的 Tagger 并清洗结果。",
            "Preserve every source .txt file byte-for-byte.": "逐字节保留原始 .txt 描述，不做改写。",
            "Treat source captions as Booru-style tags and normalize them.": "把已有描述按 Booru 标签列表清洗规范化。",
            "Keep source information and add useful tagger suggestions.": "保留已有信息，并补充有价值的 Tagger 建议。",
            "Preparation will require existing sidecars.": "准备训练集时将要求已有对应 .txt 文件。",
            "Allow trigger-only fallback when an image has no caption?": "图片没有描述时允许只使用触发词吗？",
            "Create a review summary before preparation?": "准备训练集之前生成审核摘要吗？",
            "Run screening evaluation after training?": "训练完成后自动运行快速筛选评测吗？",
            "Workflow preferences saved.": "工作流偏好已保存。",
            "Current status:": "当前状态：",
            "Force a rerun?": "强制重新运行吗？",
            "Exclude all but one image in each exact-duplicate group?": "每组完全重复图片只保留一张吗？",
            "Optional raw-relative image paths to exclude (comma-separated; blank for none)": "可选：要排除的 raw 相对路径（逗号分隔，留空表示不排除）",
            "Allow trigger-only captions for otherwise uncaptioned images?": "对于没有描述的图片，允许只使用触发词吗？",
            "Preflight is not complete. Run it before training?": "训练前检查尚未完成。现在运行吗？",
            "Training was not started.": "训练未启动。",
            "Keep the interrupted run as historical evidence.": "保留中断的训练记录作为历史证据。",
            "Use the saved exposure budget of": "使用项目保存的曝光预算：",
            "Exposure budget for this run": "本次训练的曝光预算",
            "Training is already complete. Start another run with these inputs?": "训练已经完成。使用相同输入再启动一次训练吗？",
            "Start GPU training now?": "现在开始 GPU 训练吗？",
            "Run screening evaluation now?": "现在运行快速筛选评测吗？",
            "No successful training run is available yet.": "目前没有可用于评测的成功训练记录。",
            "Quick matrix across candidate checkpoints.": "快速比较候选 Checkpoint。",
            "Detailed matrix for one or two explicit finalists.": "对 1-2 个明确的终选 Checkpoint 做详细评测。",
            "has no available checkpoint files": "没有可用的 Checkpoint 文件",
            "Screening will consider": "快速筛选将考虑",
            "recorded checkpoint(s); the profile candidate limit still applies.": "个已记录 Checkpoint；仍受配置中的候选数量上限约束。",
            "Regenerate images even if this exact evaluation is reusable?": "即使相同评测结果可复用，也重新生成图片吗？",
            "Promote a reviewed checkpoint now?": "现在晋升一个已经审核的 Checkpoint 吗？",
            "No evaluated run is available. Run screening or full evaluation first.": "没有可晋升的已评测记录，请先运行快速或完整评测。",
            "Recommended LoRA strength": "推荐 LoRA 强度",
            "Evidence stages": "评测依据阶段",
            "Create best.safetensors and best.yaml?": "生成 best.safetensors 和 best.yaml 吗？",
            "Checkpoint promoted": "Checkpoint 已晋升",
            "No training runs have been recorded yet.": "还没有训练记录。",
            "Compute fingerprints without changing step state.": "计算输入指纹，但不修改步骤状态。",
            "Expert-only: training safety checks will be marked skipped with a warning.": "仅限专家：训练安全检查会被标记为跳过并显示警告。",
            "No actionable work remains.": "没有剩余的可执行工作。",
            "Type BYPASS to acknowledge that training may be invalid or unsafe": "输入 BYPASS，确认你理解跳过检查可能导致训练无效或不安全",
            "Preflight bypass cancelled.": "已取消跳过训练前检查。",
            "Add one local .safetensors file.": "注册一个本地 .safetensors 文件。",
            "Find local .safetensors files and register one.": "扫描本地 .safetensors 文件并选择注册。",
            "Read metadata and use the cached identity hash.": "读取元数据并使用缓存的身份哈希。",
            "Re-read the complete file and compare SHA256.": "重新读取完整文件并校验 SHA256。",
            "No base models are registered.": "还没有注册底模。",
            "No projects yet. Create one to begin.": "还没有项目，先创建一个开始吧。",
            "No projects": "没有项目",
            "continue with": "继续：",
            "retry": "重试",
            "resume": "恢复",
            "review project state": "检查项目状态",
            "run screening evaluation": "运行快速筛选评测",
            "select finalists for full evaluation": "选择终选模型并运行完整评测",
            "review sheets and promote a checkpoint": "查看对比图并晋升 Checkpoint",
            "complete; inspect artifacts or start another run": "已完成；查看产物或开始新的训练",
            "What should happen next?": "接下来做什么？",
            "Available after evaluation evidence exists.": "有评测结果后可用。",
            "Inspect artifacts": "查看产物",
            "A project already exists at": "该位置已经存在项目：",
            "Directory does not exist:": "目录不存在：",
            "No supported images were found in that directory.": "该目录中没有找到支持的图片。",
            "same-stem caption file(s).": "个同名描述文件。",
            "Use this dataset?": "使用这个数据集吗？",
            "The trigger must be non-empty and cannot contain a comma.": "触发词不能为空，也不能包含逗号。",
            "Checkpoint does not exist:": "Checkpoint 不存在：",
            "Registered": "已注册",
            "Inspect metadata and establish the SHA256 identity now?": "现在检查元数据并建立 SHA256 身份吗？",
            "No .safetensors files were found.": "没有找到 .safetensors 文件。",
            "Discovered checkpoints": "发现的 Checkpoint",
            "Suggested id": "建议 ID",
            "Every discovered checkpoint is already registered.": "扫描到的所有 Checkpoint 都已经注册。",
            "Register which checkpoint?": "注册哪个 Checkpoint？",
            "Read and hash the entire checkpoint file? This may take a while on NAS storage.": "读取并哈希整个 Checkpoint 文件吗？在 NAS 上可能需要一些时间。",
            "Identity matches registry": "身份与注册表一致",
            "Hash cache reused": "复用了哈希缓存",
            "Tensor count": "Tensor 数量",
            "Menu requires at least one item": "菜单至少需要一个选项",
            "Action": "操作",
            "Description": "说明",
            "Choose a number": "请输入编号",
            "Enter a positive integer.": "请输入正整数。",
            "Enter a value greater than zero.": "请输入大于 0 的数值。",
            "Cancelled. Saved project state was preserved.": "已取消；已经保存的项目状态不会丢失。",
            "Action failed": "操作失败",
            "Retry after breaking a stale or unverifiable lock?": "确认旧锁已失效后解除锁并重试吗？",
            "reused": "已复用",
            "Training data source": "训练数据来源",
            "Use an existing folder of training images and optional caption sidecars.": "使用现有训练图片目录，可同时包含同名 .txt 描述。",
            "Download or open a video, sample useful frames, choose the target character, and create a dataset.": "下载或打开视频，抽取有效帧、选择目标人物并创建数据集。",
            "Create a Character LoRA project from video": "从视频创建人物 LoRA 项目",
            "The importer samples frames, removes blurry or badly exposed frames, filters near-duplicates, then uses CCIP to let you choose the target character before the normal LoRA pipeline starts.": "导入器会采样视频帧、过滤模糊/曝光异常/近重复图片，再用 CCIP 聚类让你选择目标人物，之后进入正常 LoRA 流程。",
            "Video source cannot be empty": "视频来源不能为空",
            "Only": "仅剩",
            "selected frame(s) remain.": "张选中帧。",
            "That is a very small Character dataset; consider a denser sampling interval or keeping more frames.": "人物数据集过小；建议缩短采样间隔或保留更多帧。",
            "Continue with this small dataset?": "仍使用这个小数据集继续吗？",
            "Project creation cancelled; temporary video frames were discarded.": "已取消创建项目；临时视频帧已清理。",
            "Create this project from the selected frames?": "使用选中的帧创建项目吗？",
            "Video project created": "视频项目已创建",
            "target-character frames into the immutable raw dataset.": "张目标人物帧导入不可变 raw 数据集。",
            "Pass an explicit direct-connection policy only to yt-dlp.": "仅对 yt-dlp 显式使用直连，不影响其他网络访问。",
            "Supports HTTP(S) and SOCKS proxy URLs; credentials are never written to project metadata.": "支持 HTTP(S) 与 SOCKS 代理；用户名和密码不会写入项目元数据。",
            "Proxy URL (for example http://127.0.0.1:7890 or socks5://127.0.0.1:1080)": "代理 URL（例如 http://127.0.0.1:7890 或 socks5://127.0.0.1:1080）",
            "Preparing video frames": "正在准备视频帧",
            "Sampling interval:": "采样间隔：",
            "Maximum accepted frames:": "最多保留帧数：",
            "Video download/import failed": "视频下载/导入失败",
            "Video import cancelled": "视频导入已取消",
            "Finding character identity clusters": "正在识别人物身份簇",
            "CCIP will group the filtered frames by likely character identity. You choose the target; the importer will not automatically assume that the largest cluster is correct.": "CCIP 会按可能的人物身份对筛选后的帧聚类。目标人物由你确认，程序不会自动假定最大簇就是正确人物。",
            "CCIP is unavailable, so target-character preselection cannot run.": "CCIP 当前不可用，因此无法在建项前筛选目标人物。",
            "The normal Character identity/review stages can still run later.": "之后仍可使用正常的人物一致性/人工审核步骤。",
            "CCIP could not produce stable pre-import clusters.": "CCIP 无法生成稳定的建项前人物聚类。",
            "Keep all filtered frames and continue to the normal Identity/Review stages?": "保留全部筛选帧，并在后续人物一致性/审核步骤中继续处理吗？",
            "Use cluster": "使用簇",
            "frames": "帧",
            "Representatives:": "代表帧：",
            "Skip pre-import identity filtering and rely on the later Character Identity/Review steps.": "跳过建项前人物筛选，依赖后续人物一致性/审核步骤。",
            "Video import cancelled before target-character selection": "在选择目标人物前取消了视频导入",
            "Keep every filtered frame, including other clusters and CCIP outliers?": "保留全部筛选帧，包括其他人物簇和 CCIP 离群帧吗？",
            "excluded when a target cluster is selected": "选择目标人物簇时会排除",
            "Tip: if the same character is split across outfits or extreme camera angles, use the representative filenames to inspect the candidates before choosing. You can also keep all frames and review later.": "提示：如果同一人物因换装或极端角度被拆成多个簇，可根据代表帧文件名先检查；也可以保留全部帧后续再审核。",
            "Change the exposure budget, strategy, or evaluation subject prompt.": "修改曝光预算、训练策略或评测主体提示词。",
            "Add unseen holdout images without copying files by hand.": "导入未参与训练的独立验证图片，无需手动复制文件。",
            "Updates preflight and future training without touching prepared data.": "只更新训练前检查和后续训练，不修改已准备的数据集。",
            "Switch between Quality, Fast, and Cached profiles.": "在质量优先、速度优先和缓存模式之间切换。",
            "Describe the subject used in Character evaluation prompts.": "设置人物 LoRA 评测提示词中的主体描述。",
            "Budget is unchanged.": "曝光预算没有变化。",
            "Exposure budget updated to": "曝光预算已更新为",
            "Training strategy is unchanged.": "训练策略没有变化。",
            "Training strategy updated to": "训练策略已更新为",
            "Evaluation subject prompt cannot be empty": "评测主体提示词不能为空",
            "Evaluation subject prompt is unchanged.": "评测主体提示词没有变化。",
            "Evaluation subject prompt updated.": "评测主体提示词已更新。",
            "Directory containing unseen validation images": "包含独立验证图片的目录",
            "Validation source directory does not exist:": "验证集来源目录不存在：",
            "That directory is already this project's validation directory.": "该目录已经是当前项目的验证集目录。",
            "Validation images must be independent holdouts and must not duplicate training images.": "验证图片必须是独立 holdout，不能与训练图片重复。",
            "Check for training overlap and import these images?": "检查是否与训练集重合并导入这些图片吗？",
            "Validation import blocked:": "验证集导入已阻止：",
            "image(s) exactly overlap the training set": "张图片与训练集完全重复",
            "Validation import complete": "验证集导入完成",
            "No supported images were found under": "没有找到支持的图片：",
            "Image exposures": "图片曝光次数",
            "Training strategy": "训练策略",
            "Proxy URL must use one of": "代理 URL 必须使用以下协议之一：",
            "and include a host": "并包含主机地址",
            "Proxy port must be a valid number between 1 and 65535": "代理端口必须是 1-65535 之间的有效数字",
            "Video import needs these executables on PATH:": "视频导入需要 PATH 中存在以下程序：",
            "Frame interval must be at least 1 second": "抽帧间隔至少为 1 秒",
            "Maximum frame count must be at least 1": "最大帧数至少为 1",
            "Perceptual-hash distance cannot be negative": "感知哈希距离不能为负数",
            "ffmpeg produced no frames from the selected video": "ffmpeg 没有从所选视频中抽取到任何帧",
            "All sampled video frames were rejected as blurry, duplicate, or badly exposed": "所有采样帧都因模糊、重复或曝光异常被剔除",
            "Video file does not exist:": "视频文件不存在：",
            "yt-dlp could not download the video": "yt-dlp 无法下载视频",
            "yt-dlp finished without producing a video file": "yt-dlp 结束后没有生成视频文件",
            "ffmpeg could not extract frames from the video": "ffmpeg 无法从视频抽取帧",
            "Optional GPU lease hook": "可选 GPU 独占钩子",
            "CUDA available": "CUDA 可用",
            "Compute capability": "计算能力",
            "sm_70 build": "sm_70 构建支持",
            "FP16 SDPA forward/backward": "FP16 SDPA 前向/反向",
            "sd-scripts entrypoint": "sd-scripts 入口",
            "sd-scripts pinned commit": "sd-scripts 固定提交",
            "Validated environment record": "已验证环境记录",
            "ONNX Runtime CUDA provider": "ONNX Runtime CUDA Provider",
            "imgutils/tagger backend": "imgutils/Tagger 后端",
            "WD EVA02-Large Tagger v3 cache": "WD EVA02-Large Tagger v3 缓存",
            "Registered base paths": "已注册底模路径",
            "Projects path writable": "项目目录可写",
            "Free disk": "可用磁盘空间",
            "not configured; ensure exclusive GPU access before training": "未配置；训练前请确保 GPU 独占使用",
            "configured with shell-free command arrays": "已使用无 Shell 的命令数组配置",
            "invalid command configuration": "命令配置无效",
            "cached": "已缓存",
            "not cached yet; first caption run may download it": "尚未缓存；首次生成标签时可能需要下载",
            "registered": "个已注册",
            "no base models registered": "没有注册底模",
        }.items(),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )
)


def _config_path() -> Path:
    override = os.environ.get("LORA_PIPELINE_UI_CONFIG")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".config" / "lora-pipeline" / "ui.json"


def normalize_language(value: str | None) -> str:
    if not value:
        return DEFAULT_LANGUAGE
    normalized = value.strip().replace("_", "-").casefold()
    if normalized in {"zh", "zh-cn", "zh-hans", "cn", "chinese", "简体中文", "中文"}:
        return "zh-CN"
    if normalized in {"en", "en-us", "en-gb", "english"}:
        return "en"
    return DEFAULT_LANGUAGE


def load_saved_language() -> str | None:
    env_value = os.environ.get("LORA_PIPELINE_LANG")
    if env_value:
        return normalize_language(env_value)
    path = _config_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    value = payload.get("language") if isinstance(payload, dict) else None
    return normalize_language(str(value)) if value else None


def save_language(language: str) -> None:
    path = _config_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps({"language": normalize_language(language)}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except OSError:
        # UI preferences must never prevent the trainer from starting.
        pass


def set_language(language: str) -> str:
    global _LANGUAGE
    _LANGUAGE = normalize_language(language)
    return _LANGUAGE


def get_language() -> str:
    return _LANGUAGE


def choose_language_first_run() -> str:
    saved = load_saved_language()
    if saved:
        return set_language(saved)

    # Avoid blocking automated/non-interactive callers. Real SSH terminals are TTYs.
    if not sys.stdin.isatty():
        return set_language("en")

    print("\n界面语言 / Interface language")
    print("  1. 简体中文（默认）")
    print("  2. English")
    try:
        raw = input("请选择 / Choose [1]: ").strip()
    except (EOFError, KeyboardInterrupt):
        raw = "1"
    language = "en" if raw == "2" else "zh-CN"
    save_language(language)
    return set_language(language)


def translate(value: str) -> str:
    if _LANGUAGE != "zh-CN" or not value:
        return value
    exact = _EXACT_ZH.get(value)
    if exact is not None:
        return exact
    result = value
    for english, chinese in _REPLACEMENTS_ZH:
        if english in result:
            result = result.replace(english, chinese)

    # A few compact dynamic forms occur frequently in tables and prompts.
    result = re.sub(r"(\d+) project\(s\)", r"\1 个项目", result)
    result = re.sub(r"(\d+) enabled base model\(s\)", r"\1 个已启用底模", result)
    result = re.sub(r"(\d+) image\(s\)", r"\1 张图片", result)
    result = re.sub(r"(\d+) run\(s\)", r"\1 次运行", result)
    result = re.sub(r"(\d+) checkpoint\(s\)", r"\1 个 Checkpoint", result)
    result = re.sub(r"Project: ([^\n]+)", r"项目：\1", result)
    result = re.sub(r"Base: ", "底模：", result)
    result = re.sub(r"Trigger: ", "触发词：", result)
    result = re.sub(r"Budget: ", "预算：", result)
    result = re.sub(r"Recommended: ", "建议：", result)
    result = re.sub(r"Latest run: ", "最近训练：", result)
    result = re.sub(r"current: ", "当前：", result)
    result = re.sub(r"Source: ", "来源：", result)
    result = re.sub(r"Network: ", "网络：", result)
    result = re.sub(r" via ", "，经由 ", result)
    result = re.sub(r"Found \[bold\](\d+)\[/bold\]", r"找到 [bold]\1[/bold]", result)
    return result


def _translate_visible(value: Any) -> Any:
    return translate(value) if isinstance(value, str) else value


def install_rich_hooks() -> None:
    """Translate existing Rich UI at render time without forking business logic."""

    global _HOOKS_INSTALLED
    if _HOOKS_INSTALLED:
        return
    _HOOKS_INSTALLED = True

    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
    from rich.table import Table

    original_console_print = Console.print
    original_table_init = Table.__init__
    original_add_column = Table.add_column
    original_add_row = Table.add_row
    original_panel_init = Panel.__init__

    def console_print(self: Console, *objects: Any, **kwargs: Any) -> None:
        original_console_print(self, *(_translate_visible(obj) for obj in objects), **kwargs)

    def table_init(self: Table, *args: Any, **kwargs: Any) -> None:
        if isinstance(kwargs.get("title"), str):
            kwargs["title"] = translate(kwargs["title"])
        original_table_init(self, *args, **kwargs)

    def add_column(self: Table, header: Any = "", *args: Any, **kwargs: Any) -> Any:
        return original_add_column(self, _translate_visible(header), *args, **kwargs)

    def add_row(self: Table, *renderables: Any, **kwargs: Any) -> None:
        original_add_row(self, *(_translate_visible(value) for value in renderables), **kwargs)

    def panel_init(self: Panel, renderable: Any, *args: Any, **kwargs: Any) -> None:
        renderable = _translate_visible(renderable)
        if isinstance(kwargs.get("title"), str):
            kwargs["title"] = translate(kwargs["title"])
        if isinstance(kwargs.get("subtitle"), str):
            kwargs["subtitle"] = translate(kwargs["subtitle"])
        original_panel_init(self, renderable, *args, **kwargs)

    Console.print = console_print  # type: ignore[assignment]
    Table.__init__ = table_init  # type: ignore[assignment]
    Table.add_column = add_column  # type: ignore[assignment]
    Table.add_row = add_row  # type: ignore[assignment]
    Panel.__init__ = panel_init  # type: ignore[assignment]

    def patch_prompt(cls: type[Any]) -> None:
        original: Callable[..., Any] = cls.ask

        def ask(inner_cls: type[Any], prompt: str, *args: Any, **kwargs: Any) -> Any:
            return original(translate(prompt), *args, **kwargs)

        cls.ask = classmethod(ask)  # type: ignore[method-assign]

    for prompt_cls in (Prompt, Confirm, IntPrompt, FloatPrompt):
        patch_prompt(prompt_cls)


def initialize_interactive() -> str:
    language = choose_language_first_run()
    install_rich_hooks()
    return language
