---
name: detect-data-prep-pretrain
description: 客户/新增视频数据处理：5fps 截帧 → 现有 YOLO 模型预标注（label 0→1）→ 与原数据列表合并 → 送入训练工程做预训练。
alwaysApply: false
---

# 检测数据准备 · 预训练送训

当用户提到「视频截帧」「YOLO 预标注」「label 换类」「合并数据列表」「送预训练」时使用本 Skill。

## 固定规则（已确认，勿擅自改）

| 步骤 | 规则 |
|------|------|
| 截帧 | 视频按 **5 fps** 抽帧 |
| 标注格式 | **YOLO**（每图对应同名 `.txt`，行：`class cx cy w h`，归一化 0–1） |
| 预标注 | 用**现有检测模型**对截帧推理出框 |
| 类别映射 | 模型输出的 **`label=0` 全部改成 `label=1`**；其它类若出现，先停并询问，不要默默丢弃或改写 |
| 合并 | 新 `images/` + `labels/` 与**原训练数据列表**合并后送训 |
| 训练 | 合并结果进入训练工程，走**预训练**配置（非从零乱训） |

配置模板：`configs/detect_data_prep.example.yaml`（复制为本地 `configs/detect_data_prep.yaml` 再改路径）。

## 输入 / 输出约定

### 输入

- 原始视频目录：`raw_videos/`（或配置里的 `paths.raw_videos`）
- 预标注权重：现有 YOLO 检测模型（`paths.pretrained_det_weights`）
- 原数据列表：已有 train list（如 `data/lists/train.txt`，一行一个 image 路径或 `image label` 对，以工程实际为准）
- 原数据集根：已有 `images/` + `labels/`（或 list 指向的路径）

### 输出目录（建议）

```text
data/prep/<case_id>/
  frames/                 # 5fps 截帧 jpg/png
  labels_raw/             # 模型原始 YOLO txt（含 class 0）
  labels/                 # 映射后 YOLO txt（0→1）
  lists/
    new_only.txt          # 仅本批新数据
    train_merged.txt      # 原列表 ∪ 本批
  manifest.json           # 帧数、映射统计、合并统计
```

最终送训使用：`lists/train_merged.txt` + 对应 images/labels 根路径。

## 执行流程（必须按序）

### 1. 读配置并建目录

```bash
cp configs/detect_data_prep.example.yaml configs/detect_data_prep.yaml
# 编辑 paths / case_id / model 相关字段
mkdir -p "data/prep/<case_id>"/{frames,labels_raw,labels,lists}
```

若缺少视频路径、权重路径或原 `train` 列表 → **停止并询问**，不要编造路径。

### 2. 视频截帧（5 fps）

对每个视频：

- 帧率固定 **5 fps**（配置项 `extract.fps: 5`，禁止改成「按关键帧/全帧」除非用户明确要求）
- 文件名建议：`<video_stem>_f<frame_idx:06d>.jpg`，保证全局唯一
- 同步写入帧与源视频的对应关系到 `manifest.json`

示例（ffmpeg）：

```bash
ffmpeg -i "<video>" -vf fps=5 -q:v 2 "data/prep/<case_id>/frames/<video_stem>_f%06d.jpg"
```

截帧后统计：视频数、总帧数；帧数为 0 则失败退出。

### 3. 现有模型 YOLO 预标注

- 对 `frames/` 全量推理，输出 YOLO txt 到 `labels_raw/`
- **一张图一个 txt**；无框时写空文件（或按工程约定跳过——以训练工程为准，需在 manifest 注明）
- 保留推理置信度阈值配置：`infer.conf`（默认见 yaml）；改阈值需在 manifest 记录

优先调用工程内已有推理脚本；若无，再写临时脚本，但输出必须是标准 YOLO txt。

### 4. 类别映射：`0 → 1`

对 `labels_raw/` 每个文件：

1. 读每一行 `cls cx cy w h`
2. 若 `cls == 0` → 写成 `1`
3. 若 `cls == 1` → 保持 `1`（并在 manifest 记 `already_one` 计数）
4. 若 `cls` 为其它值 → **整批暂停**，列出文件与类别，等人确认映射表后再继续
5. 写出到 `labels/`（不要覆盖 `labels_raw/`）

校验：

- `labels/` 与 `frames/` 文件名一一对应（除扩展名）
- 映射后 **不得再出现 class 0**（除非用户改规则）
- 在 `manifest.json` 记录：`mapped_0_to_1` 行数、空框图数量、异常类数量

### 5. 与原数据列表合并

1. 生成本批列表 `lists/new_only.txt`（每行指向本批 image；若工程要 image+label 双路径，按原列表格式对齐）
2. 读取原列表 `paths.original_train_list`
3. 合并去重 → `lists/train_merged.txt`
4. 检查：原列表行数 + 新列表行数 − 重复数 = 合并后行数
5. 抽查若干行：image 存在、对应 label 存在、label 无 class 0

**禁止**直接改写原列表文件；只写 `train_merged.txt`（或用户指定的新 list 路径）。

### 6. 送入训练工程（预训练）

1. 确认训练工程数据配置指向：
   - 合并后的 list：`train_merged.txt`
   - 本批 labels 已挂到训练可读路径（拷贝/软链到统一 `images`/`labels` 或 list 使用绝对路径——与现网工程一致）
2. 使用**预训练**启动方式（加载 `paths.pretrain_checkpoint` 或工程默认 pretrained）
3. 给出将要执行的训练命令草稿，**先展示再执行**；限制性上网环境下提醒在训练服务器上跑

示例占位命令（按实际工程替换）：

```bash
# 在训练服务器 conda 环境中
python train.py \
  --data data/prep/<case_id>/lists/train_merged.txt \
  --weights <pretrain_checkpoint> \
  --cfg <pretrain_config>
```

### 7. 交付物检查清单

完成前必须核对：

- [ ] 截帧 fps = 5
- [ ] `labels_raw/` 保留原始预测
- [ ] `labels/` 中原 0 类已全部为 1，且无未确认的其它类
- [ ] `new_only.txt` / `train_merged.txt` 已生成且行数对得上
- [ ] 未覆盖原训练列表
- [ ] `manifest.json` 含帧数、映射统计、合并统计
- [ ] 训练命令指向合并列表 + 预训练权重

## 停止条件（NEED_HUMAN）

出现以下情况立即停止并提问：

1. 预标注结果出现 **非 0/1** 的 class
2. 原数据列表格式与本批 list 格式不一致（无法安全合并）
3. 帧路径与 label 大量缺失（>1%）
4. 用户未提供预训练权重或训练入口脚本
5. 用户要求改 fps / 改映射（规则变更需先改 yaml 并确认）

## Agent 行为约束

- 只执行本流水线；不要顺带改模型结构、后处理或 NPU 转换
- 类别映射以配置为准：`label_map: {0: 1}`；不要「智能推测」业务类定义
- 所有统计写入 `manifest.json`，便于回溯
- 内网训练服务器命令与本地预处理命令分开说明

## 相关文件

- 配置模板：`configs/detect_data_prep.example.yaml`
- 本 Skill：`.cursor/skills/detect-data-prep-pretrain/SKILL.md`
