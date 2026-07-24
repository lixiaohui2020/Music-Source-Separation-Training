# 量化校准数据生成

## `generate_cache_state()`

**目的**：为 ONNX 量化准备校准 npy，参考 `denoise.py` 的 `run_onnx` 打包方式。

**特点**：

- **仅使用 ONNX Runtime** 做网络推理（first 热身 + mid 循环）
- Host 侧只做 STFT，不做 OLA ISTFT
- 从音频中按 `--start_sec` / `--duration_sec` 截取片段
- 每个 mid 帧：**推理前**保存输入与 state，**推理后**保存 `chunk_output`

## 前置条件

```text
onnx/scnet_first.onnx
onnx/scnet_mid.onnx
```

若不存在，先运行 `test_onnx_inference()` 导出。

## 运行

```bash
cd stream_scnet
PYTHONPATH=. python3 -c \
  "from SCNetStreamAudioConv1dChunk import generate_cache_state; generate_cache_state()" \
  --input_dir /path/to/audio.wav \
  --onnx_dir ./onnx \
  --quan_output_path ./quan_npy \
  --start_sec 30 \
  --duration_sec 10
```

或在 `SCNetStreamAudioConv1dChunk.py` 的 `if __name__ == '__main__'` 中调用 `generate_cache_state()`。

## 输出文件

目录：`--quan_output_path`（默认 `./quan_npy`）

| 文件 | 形状（N = mid 帧数） | 说明 |
|------|---------------------|------|
| `input.npy` | `(N, 4, 2049, 3)` | mid 输入 spec_in |
| `cache_band0.npy` | `(N, 64, 616, 2)` | 推理前 state |
| `cache_band1.npy` | `(N, 128, 186, 2)` | |
| `cache_band2.npy` | `(N, 64, 57, 2)` | |
| `cache_h1.npy` | `(N, 57, 64)` | |
| `cache_c1.npy` | `(N, 57, 64)` | |
| `cache_h2.npy` | `(N, 57, 128)` | |
| `cache_c2.npy` | `(N, 57, 128)` | |
| `cache_conv.npy` | `(N*57, 128, 6)` | 首维无 batch |
| `cache_fus0.npy` | `(N, 128, 57, 2)` | |
| `cache_fus1.npy` | `(N, 256, 186, 2)` | |
| `cache_fus2.npy` | `(N, 128, 616, 2)` | |
| `skip0.npy` | `(N, 64, 616, 3)` | |
| `skip1.npy` | `(N, 128, 186, 3)` | |
| `skip2.npy` | `(N, 64, 57, 3)` | |
| `output.npy` | `(N*4, 2049, 3, 2)` | mid chunk_output，用于量化后校验 |

## 与 denoise.py 的对应关系

| denoise.py | SCNet `generate_cache_state` |
|------------|------------------------------|
| 每帧保存 `inp_spec` | 每 mid 帧保存 `input.npy` 一行 |
| 每帧保存各 state | 每 mid 帧保存 14 个 state npy |
| `session.run` 后更新 state | first/mid ORT 更新 `ort_state` |
| `output.npy` 存网络输出 | `output.npy` 存 `chunk_output` |

## mid 帧数计算

```python
n_mid = int((L_ext - hop_size * 4) / (3 * hop_size))
# L_ext = len(pad_mix) + 2*hop + 3*hop
```

2 秒 44.1kHz 测试音频约 `n_mid=30`。

## 后续量化建议

1. 用多段音频、不同 `--start_sec` 生成多份 npy 或合并
2. 量化工具读取 `input.npy` + 各 state npy 作为校准输入
3. 量化后用 `output.npy` 对比量化图输出误差
