# 量化校准数据生成

## `generate_cache_state()`

从 **大量音频（典型 N=1000+）** 批量生成 first / mid ONNX 量化校准 npy。

### 每个音频的处理流程

```
音频 i
  ├─ Host STFT → first spec_in
  ├─ first ONNX（零 cache 输入）→ 14 个 output state
  └─ mid ONNX × mid_frames
       ├─ 第 0 帧初始 state = first 输出（与其它音频无关）
       └─ 第 1..9 帧 state = 上一 mid 帧输出
```

**要点**：每条音频的 mid 初始 state **只来自该音频的 first 输出**，文件之间不传递 state。

### 音频列表来源（优先级）

1. `--quan_list`：文本文件，每行一个 wav 路径（推荐 1000+ 条）
2. `--input_dir` 为目录：自动扫描 `*.wav/*.flac/*.mp3/*.ogg`
3. `--input_dir` 为单文件：N=1（调试）

### 运行

```bash
cd stream_scnet

# 1000+ 文件列表
PYTHONPATH=. python3 SCNetStreamAudioConv1dChunk.py \
  --quan_list /path/to/audio_list.txt \
  --onnx_dir ./onnx \
  --quan_output_path ./quan_npy \
  --mid_frames 10 \
  --quan_seed 42
```

`--start_sec < 0`（默认）：每个文件随机截取片段；固定起点用 `--start_sec 30`。

### 输出 npy（N = 成功处理的文件数）

| 文件 | 形状 | 说明 |
|------|------|------|
| `first_input.npy` | `(N*2, 2049, 3, 2)` | first spec_in，N 条样本 |
| `first_cache_*.npy` | `(N, …)` | first 11 个零 cache 输入 |
| `mid_input.npy` | `(N*mid_frames*2, 2049, 3, 2)` | mid spec_in |
| `mid_{state}.npy` | `(N*mid_frames, …)` | mid 推理前 14 个 state |
| `mid_output.npy` | `(N*mid_frames*4, 2049, 3, 2)` | mid chunk_output |
| `meta.json` | — | 文件列表、起点、N、mid_frames |

示例：`N=1000`，`mid_frames=10` → first 轴长 1000，mid 轴长 10000。

### 所需音频长度

每条至少 `(4 + mid_frames * 3) * 1024` 样本（默认 mid_frames=10 → 34816 样本 ≈ 0.79s@44.1kHz），实际建议 ≥ 2s 以便随机截取。
