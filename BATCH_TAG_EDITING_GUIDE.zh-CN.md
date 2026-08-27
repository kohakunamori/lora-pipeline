# 数据集批量 Tag 编辑

LoRA Pipeline 的 Dataset Workspace 现在支持对多张图片的 caption/tag 列表执行批量修改。

支持三种操作：

- 添加到首部：把指定 Tag 移到每条 caption 的开头；已经存在的同义 Tag 会先移除，避免重复。
- 添加到尾部：把指定 Tag 移到每条 caption 的末尾；已经存在的同义 Tag 会先移除，避免重复。
- 删除指定 Tag：从选中的 caption 中删除匹配 Tag。

Tag 匹配沿用现有规范化规则，忽略大小写，并把下划线和空格视为等价。例如：

```text
blue_hair
blue hair
Blue Hair
```

会被视为同一个 Tag。

## 终端交互

进入：

```text
管理数据集 -> 打开数据集 -> 人工修改 Tag
```

可以选择：

```text
修改单张图片
批量添加到 Tag 首部
批量添加到 Tag 尾部
批量删除指定 Tag
```

批量模式支持：

- 对全部当前可训练图片操作；
- 按编号或范围选择，例如 `1,3-5`。

执行前会显示操作、图片数量和 Tag，并要求确认。

## Web UI

打开某个 Dataset Source 的图片墙后，勾选多张图片，在“批量操作”区域输入 Tag，然后选择：

```text
Tag 添加到首部
Tag 添加到尾部
删除指定 Tag
```

原有的批量恢复、排除和永久删除功能保持不变。

## 数据安全

批量 Tag 操作只修改 Dataset Workspace 中的 `.txt` caption，不修改：

- 原始导入目录；
- `projects/*/raw/`；
- 已冻结的训练 Run；
- 历史 checkpoint 或评测结果。

创建新的训练 Run 时，当前 Dataset caption 会像之前一样进入不可变 snapshot，并参与 caption hash / dataset snapshot hash。
