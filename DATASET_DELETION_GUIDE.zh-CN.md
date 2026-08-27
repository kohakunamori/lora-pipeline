# Dataset 删除与排除

Dataset 工作区支持三个层级的永久删除：

1. 删除整个 Dataset。
2. 删除 Dataset 中的一个完整 Source。
3. 删除某个 Source 中选定的部分图片。

入口：

```text
./lora
→ 数据集
→ 打开数据集
→ 删除数据 / 来源 / 图片
```

## 排除和删除的区别

“排除”是日常清洗的默认操作：图片文件仍保留在 Dataset 中，只是不进入新的训练快照，并且可以恢复。

“永久删除”会真正删除 Dataset 工作区中的文件副本。单图删除同时删除同名 `.txt` Tag 文件，并清理对应 exclusion 记录。

## 删除不会影响什么

Dataset 导入会把素材复制/派生到 Dataset 工作区，因此永久删除不会删除：

- 最初导入的外部图片目录；
- 本地原视频；
- 在线视频来源本身；
- 已经冻结到 `projects/*/raw/` 的训练快照；
- 已有 Training Run、checkpoint、LoRA 权重、示例图片和评测结果；
- 独立的 Training Config。

## 删除 Source

删除 Source 会删除该 Source 在 Dataset 中的全部图片和 Tag 副本，并移除 Source 元数据与相关 exclusion 记录。

如果该 Source 曾经派生出 smart-crop Source，派生 Source 不会级联删除，因为它已经拥有自己独立的图片副本。界面会在删除前明确列出仍会保留的派生 Source。

删除 Source 需要二次确认，并输入 Source ID。

## 删除部分图片

图片删除界面按页显示文件、排除状态和 Tag。可使用：

```text
1,3-5,18
```

一次永久删除多张图片。删除前会显示选择预览并再次确认。

如果只是认为图片不适合当前训练，优先使用“排除”，而不是永久删除。

## 删除整个 Dataset

删除整个 Dataset 会移除 `datasets/<name>/` 工作区，包括 Source、副本、Tag、review/cache 等 Dataset 自身数据。

已有训练快照和结果不受影响。操作需要二次确认，并准确输入 Dataset 名称。
