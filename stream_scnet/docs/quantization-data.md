# 量化校准数据生成

## `generate_cache_state()`

**目的**：为 first / mid ONNX 量化准备校准 npy，参考 `denoise.py` 的保存方式。

**特点**：

- **仅 ORT 推理**（first 一次 + mid 连续 N 帧）
- Host 只做 STFT；`permute/reshape` 在模型 `forward` 内，STFT 输出 `(2, 2049, 3, 2)` 直接喂 ONNX
- **随机起点**：`--start_sec < 0` 时从整段音频随机截取（`--quan_seed`）
- **固定起点**：`--start_sec >= 0` 时使用指定秒数
- 保存 **first 输入**（spec + 11 caches）+ **mid 连续 10 帧**（默认）输入/状态/输出

## 前置条件

```text
onnx/scnet_first.onnx
onnx/scnet_mid.onnx
```

需与当前 `(2, 2049, 3, 2)` 输入形状一致；可先运行 `test_onnx_inference()` 导出。

## 运行

```bash
cd stream_scnet
# 随机起点，保存 10 帧 mid
PYTHONPATH=. python3 -c \
  "from SCNetStreamAudioConv1dChunk import generate_cache_state; generate_cache_state()" \
  --input_dir /path/to/audio.wav \
  --onnx_dir ./onnx \
  --quan_output_path ./quan_npy \
  --mid_frames 10 \
  --quan_seed 42

# 固定起点 30s
PYTHONPATH=. python3 SCNetStreamAudioConv1dChunk.py \
  --input_dir /path/to/audio.wav \
  --start_sec 30 \
  --mid_frames 10
```

## 输出文件

目录：`--quan_output_path`（默认 `./quan_npy`）

### First 图（各 1 条样本）

| 文件 | 形状 | 说明 |
|------|------|------|
| `first_input.npy` | `(2, 2049, 3, 2)` | first spec_in |
| `first_cache_band0.npy` … `first_cache_fus2.npy` | 11 个 cache | 零初始化 cache |
| `meta.json` | — | 起点、帧数、随机种子等 |

### Mid 图（默认 10 帧，denoise 式 axis-0 打包）

| 文件 | 形状（N=10） | 说明 |
|------|-------------|------|
| `mid_input.npy` | `(N*2, 2049, 3, 2)` | 推理前 spec_in |
| `mid_cache_*.npy` / `mid_skip*.npy` | `(N, …)` | 推理前 14 个 state |
| `mid_output.npy` | `(N*4, 2049, 3, 2)` | chunk_output，量化后校验 |

## 所需音频长度

```text
min_samples = (4 + mid_frames * 3) * hop_size
            = (4 + 10 * 3) * 1024 = 34816  # 默认 mid_frames=10
```

随机起点时，音频总长度须 ≥ `min_samples`。

## 与 denoise.py 的对应

| denoise.py | SCNet |
|------------|-------|
| 每帧 `inp_spec` | `mid_input.npy` 每帧 2 行 |
| 每帧 state | `mid_{state}.npy` |
| `output.npy` | `mid_output.npy` |
| — | **新增** `first_input.npy` + `first_cache_*.npy` |

## 后续量化建议

1. 多段音频、不同 `--quan_seed` 多次生成，合并校准集
2. first 与 mid 分别用对应 npy 做量化
3. 用 `mid_output.npy` 对比量化图输出
