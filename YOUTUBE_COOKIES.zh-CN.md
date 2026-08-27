# YouTube Cookie 认证（无图形 NAS）

当 YouTube 返回：

```text
Sign in to confirm you’re not a bot
```

这通常是 YouTube 对当前出口 IP 的反机器人校验。仅切换代理不一定能解决；`yt-dlp` 官方支持通过 Netscape/Mozilla 格式的 `cookies.txt` 传递已登录会话。

本项目假定 NAS 没有图形浏览器，因此**不提供 `--cookies-from-browser`**。交互界面只支持：

- 不使用 Cookie；
- 选择 NAS 上的 `cookies.txt`；
- 自动发现固定 Cookie 文件。

## 推荐导出方式

1. 在桌面电脑上打开一个新的无痕/隐私窗口。
2. 登录 YouTube。
3. 在同一个窗口、同一个标签页访问：

```text
https://www.youtube.com/robots.txt
```

4. 使用可信的 cookies.txt 导出工具，仅导出 YouTube Cookie；格式必须是 Netscape/Mozilla cookies.txt。
5. 导出后立即关闭整个无痕/隐私窗口，不再使用这个会话浏览 YouTube，避免 Cookie 被轮换。
6. 将文件安全复制到 NAS，例如：

```text
~/.config/lora-pipeline/youtube-cookies.txt
```

7. 建议限制文件权限为仅当前 NAS 用户可读写（0600）。

Cookie 相当于登录凭据：不要提交到 Git、不要发送给其他人、不要放进 `projects/`、不要贴进 Issue/日志。

## 自动发现

交互界面按以下顺序查找 Cookie 文件：

1. 环境变量 `LORA_VIDEO_COOKIES` 指定的文件；
2. `~/.config/lora-pipeline/youtube-cookies.txt`。

如果检测到文件，视频导入会优先提供“使用已检测到的 cookies.txt”。完整路径和 Cookie 内容不会写入 `project.yaml`；项目只记录使用了 Cookie 文件及其文件名。

## 下载失败恢复

如果 `yt-dlp` 检测到类似：

```text
Sign in to confirm you’re not a bot
Use --cookies ...
```

交互恢复菜单会默认选择“选择或更换 cookies.txt”。你也可以修改代理或使用当前设置重试。

## 安全与账号风险

`yt-dlp` 官方文档提醒：用 YouTube 账号 Cookie 下载存在账号被临时或永久限制的风险。只在确有需要时启用 Cookie，不要进行高频批量下载；如对账号安全敏感，优先使用专门/次要账号。

YouTube 会轮换账号 Cookie。如果之前可用的 Cookie 失效，按上述流程重新导出一个新的文件即可。
