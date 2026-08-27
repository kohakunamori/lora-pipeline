# Web UI 使用说明

LoRA Pipeline Web 是现有四区模型的浏览器前端：

- 数据集 Dataset
- 训练配置 Training Config
- 训练状态 Training Status
- 训练结果 Training Results

它不会复制训练逻辑。Dataset、ProjectState、TrainingConfig、Run、checkpoint 和评测产物仍由现有 Python 后端负责。

## 启动

在 NAS 仓库目录：

```bash
./lora web
```

默认监听：

```text
127.0.0.1:7860
```

推荐从桌面电脑通过 SSH tunnel 访问：

```bash
ssh -L 7860:127.0.0.1:7860 user@nas
```

然后浏览器打开：

```text
http://127.0.0.1:7860
```

也可以使用安装后的入口：

```bash
lora-pipeline-web
```

## 局域网直接访问

Web v1 暂时没有账号登录层，因此默认拒绝监听非 loopback 地址。

如果确定 NAS 所在网络可信，可以显式开启：

```bash
./lora web --host 0.0.0.0 --port 7860 --allow-lan
```

然后访问：

```text
http://NAS_IP:7860
```

更推荐 SSH tunnel；不要把 Web v1 直接映射到公网。

所有写操作都带进程级 CSRF token；媒体与结果文件读取也会检查路径不能逃出对应 Dataset/Run 目录。

## 数据集

Web 数据集页适合图片级人工审核：

- 按 Source 打开图片墙
- lazy-load 缩略图
- 直接修改每张图片的 Tag
- 多选图片并批量排除
- 多选恢复
- 多选永久删除
- 启用/停用 Source
- 删除整个 Source
- 删除整个 Dataset

永久删除只影响 Dataset 持有的副本：

- 不删除最初导入的图片目录
- 不删除本地原视频
- 不删除已经冻结到 Project/Run 的数据
- 不删除历史权重和训练结果

日常清洗优先使用“排除”；确认不再需要时才永久删除。

当前 Web v1 先覆盖已有 Dataset 的可视化管理。需要新增本地视频/YouTube Source 并进行交互式 CCIP 人物簇选择时，仍使用 `./lora`。后续 Web 视频导入会拆成异步阶段：抽帧 -> 人物簇缩略图 -> 浏览器选择目标人物 -> 写入 Source。

## 训练配置

Web 可以：

- 查看 Training Config
- 新建 Training Config
- 修改底模
- 修改 Trigger
- 修改 strategy
- 修改 `images_seen`
- 设置/恢复默认之外的 rank、alpha、UNet LR

Dataset 与 Config 仍然互不绑定。只有点击“开始一次新训练”时才同时冻结两个 snapshot。

工作流中的少用高级开关（dedup/identity/caption/review 等）Web v1 会沿用 Config 已保存值；完整高级编辑仍可在 CLI 使用。

## 训练状态

Web 状态页可以选择：

```text
Dataset + Training Config
```

然后：

1. 可选执行安全自动排除；
2. 冻结 Dataset snapshot；
3. 冻结 Config snapshot；
4. 创建内部 training-run Project workspace；
5. 启动独立 CLI worker。

worker 使用：

```text
python -m pipeline.cli run ...
```

并从冻结后的 Project preferences 构造参数。因此浏览器关闭不会停止训练，也不会因为后来修改 Dataset/Config 而改变正在运行的 Run。

状态详情显示：

- Dataset / Config snapshot hash
- 当前内部 step
- 各 step 状态和 attempts
- 最近错误
- `web-worker.log` 尾部

“继续 / 恢复训练”会再次调用同一个冻结 workspace；如果已有 Web worker PID 仍存活，会拒绝重复启动。

## 训练结果

结果区列出已完成的 Run，并提供：

- checkpoint 数量
- sample 数量
- promoted/best 状态
- 权重文件入口
- sample 图片墙
- contact sheet 图片墙

Web v1 的结果区以浏览/下载已有产物为主。Screening / Full Evaluation / Promote 的复杂人工选择仍可从 CLI 的“训练结果”区执行，后续再迁移到网页交互。

## 设计原则

Web UI 不创建第二套数据库，也不改训练后端：

```text
Browser
   |
Web routes
   |
DatasetWorkspace / TrainingConfig / ProjectState
   |
sd-scripts / DeepGHS / video pipeline
```

因此 CLI 和 Web 可以交替使用同一份状态，不需要迁移已有 Dataset、Run 或权重。
