# Web UI 使用说明

LoRA Pipeline Web 是现有四区模型的浏览器前端：

- 数据集 Dataset
- 训练配置 Training Config
- 训练状态 Training Status
- 训练结果 Training Results

它不会创建第二套数据库，也不会复制训练逻辑。CLI 与 Web 直接操作相同的 DatasetWorkspace、TrainingConfig、ProjectState、Run、checkpoint 和评测产物。

## 启动

在 NAS 仓库目录：

```bash
./lora web
```

默认监听：

```text
127.0.0.1:7860
```

推荐从桌面电脑建立 SSH tunnel：

```bash
ssh -L 7860:127.0.0.1:7860 user@nas
```

然后浏览器打开：

```text
http://127.0.0.1:7860
```

安装为 Python package 后也可以：

```bash
lora-pipeline-web
```

## 局域网直接访问与认证

默认只监听 loopback，因此通过 SSH tunnel 使用时可以不配置登录 token。

如果希望直接从可信局域网访问，Web 默认要求显式允许 LAN，并配置 access token：

```bash
export LORA_WEB_TOKEN='换成你自己的长随机字符串'
./lora web --host 0.0.0.0 --port 7860 --allow-lan
```

也可以直接传入：

```bash
./lora web --host 0.0.0.0 --port 7860 --allow-lan --token '你的长随机字符串'
```

然后访问：

```text
http://NAS_IP:7860
```

浏览器会先显示登录页。登录成功后使用 `HttpOnly`、`SameSite=Strict` Cookie 保存会话 token；服务端使用恒定时间比较校验 token。状态修改请求还会额外检查 CSRF token 和同源 `Origin`。

如果没有 token，非 loopback 监听会被拒绝。只有明确指定：

```bash
--unsafe-no-auth
```

才允许无认证 LAN 模式；不推荐这样做。

如果局域网本身不可信，仍应使用 SSH tunnel，或在前面配置 HTTPS reverse proxy。不要把这个服务直接映射到公网。

Dataset、Job 和 Run 文件读取也会校验路径，不能通过 `../` 逃出允许目录。

## 数据集

Web 可以：

- 创建 Dataset
- 从 NAS 图片目录导入新的独立 Source
- 从本地视频路径导入 Source
- 从在线视频 / YouTube URL 导入 Source
- 按 Source 打开图片墙
- lazy-load 图片
- 直接修改每张图片的 Tag
- 多选图片批量排除 / 恢复
- 多选永久删除
- 启用 / 停用 Source
- 删除整个 Source
- 删除整个 Dataset
- 对整个 Dataset 或单一 Source 后台运行自动 Tag

永久删除只影响 Dataset 持有的副本：

- 不删除最初导入的图片目录
- 不删除本地原视频
- 不删除已经冻结到 Project / Run 的数据
- 不删除历史权重和训练结果

日常清洗优先使用“排除”；确认不再需要时才永久删除。

删除整个 Source 时，Web 与 CLI 一样要求再次输入完整 Source ID；删除整个 Dataset 则要求输入 Dataset 名称。

### 视频导入

Web 视频导入不是把 CLI 的交互塞进一个阻塞 HTTP 请求，而是使用持久化 Web Job：

```text
提交本地视频路径 / URL
        ↓
后台抽帧
        ↓
HDR -> SDR（需要时）
        ↓
邻近时间点清晰帧择优
        ↓
DeepGHS 人物 / 头部检测
        ↓
4K 原帧人物 crop
        ↓
CCIP 人物聚类
        ↓
Job 状态：awaiting_identity
        ↓
浏览器显示各人物簇代表图
        ↓
人工点击目标人物簇
        ↓
构图平衡 / 最终 crop
        ↓
写入新的 Dataset Source
```

不会因为某个簇最大就自动假定它是目标人物。

YouTube 路径继续复用已有：

- proxy 模式
- 自定义代理
- cookies.txt
- yt-dlp 兼容逻辑

本地视频仍完全绕过 yt-dlp / Cookie / PO Token。

## Web Jobs

长任务保存在：

```text
web/jobs/
```

每个任务都有 JSON 状态、独立日志和后台 worker。当前用于：

- `video_prepare`
- `video_finalize`
- `dataset_tag`
- `train`
- `evaluate`

因此关闭浏览器、刷新页面、重启 Web 服务或 SSH 客户端断开，不会因为原 HTTP 请求消失而停止已经启动的 detached worker。

V100 只有一张卡，所以 Web 会串行保护 GPU 型任务：已有 GPU Job 运行时，不再启动第二个冲突任务。

## 训练配置

Web 可以：

- 查看 Training Config
- 新建 Training Config
- 修改底模
- 修改 Trigger
- 修改 strategy
- 修改 `images_seen`
- 设置或恢复默认之外的 rank、alpha、UNet LR

Dataset 与 Config 仍然互不绑定。只有开始一次训练时才同时冻结两个 snapshot。

较少修改的 dedup / identity / caption / review 等工作流开关继续读取 Training Config 已保存值；完整高级编辑仍可使用 CLI。Web 修改核心训练字段时不会丢掉这些未显示的高级 overrides。

## 训练状态

状态页选择：

```text
Dataset + Training Config
```

系统随后：

1. 可选自动排除损坏文件和完全重复副本；
2. 冻结 Dataset snapshot；
3. 冻结 Config snapshot；
4. 创建内部 training-run Project workspace；
5. 创建持久化 `train` Web Job；
6. 后台 worker 调用现有 `run_remaining()` / sd-scripts 流程。

后续修改 Dataset 或 Training Config 都不会改变已经创建的 Run。

状态详情显示：

- Dataset / Config snapshot hash
- 当前内部 step
- 每个 step 的状态和 attempts
- 最近错误
- 对应 Web Jobs
- Job 独立日志

“继续 / 恢复训练”仍然使用原先冻结的 workspace，并复用现有 interrupted-run resume 逻辑。

## 训练结果

结果区提供：

- checkpoint 列表
- `.safetensors` 权重下载
- sample 图片墙
- contact sheet 图片墙
- promoted / best 状态
- Screening 评测
- Full Evaluation

Screening / Full 都作为后台 `evaluate` Job 运行，不阻塞浏览器。Full 必须人工选择 1–2 个 finalist checkpoint。

权重文件采用流式响应，不会为了下载一个大型 `.safetensors` 先把整个文件读进 Web 进程内存。

Web v1 暂时仍把“Promote / 设为 best.safetensors”的最终人工确认留在 CLI 的“训练结果”区，避免第一版把所有危险结果操作一次性搬入浏览器。

## 架构

```text
Browser
   |
stdlib HTTP routes
   |
DatasetWorkspace / TrainingConfig / ProjectState
   |
Persistent Web Jobs
   |
sd-scripts / DeepGHS / video pipeline
```

没有 React / Vite / Node 构建链，也没有新增 Web framework 运行依赖。CLI 和 Web 可以交替使用同一份状态，不需要迁移已有 Dataset、Run 或权重。
