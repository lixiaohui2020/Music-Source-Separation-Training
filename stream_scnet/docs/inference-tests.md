# 推理测试函数说明

入口文件：`SCNetStreamAudioConv1dChunk.py`

## 公共参数 `parse_args()`

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input_dir` | 输入 wav 路径 | 项目内示例路径 |
| `--checkpoint_path` | 模型 checkpoint | 项目内示例路径 |
| `--instruments` | 分离源名称 | `accompaniment vocals` |
| `--save_dir` | 非流式输出目录 | `./output/...` |
| `--onnx_dir` | ONNX 模型目录 | `./onnx` |
| `--quan_output_path` | 量化 npy 输出目录 | `./quan_npy` |
| `--start_sec` | 量化数据起始秒 | `0.0` |
| `--duration_sec` | 量化数据时长（秒，≤0 到末尾） | `10.0` |

---

## `test_full_scnet()`

**目的**：PyTorch 非流式 vs 流式全链路对齐，并保存分离 wav。

**流程**：

1. `load_model` + `load_audio`
2. padding（hop + 3 帧对齐）
3. 非流式：`wave_ref = scnet(pad_mix[None])`
4. 流式：Host STFT → `forward_1st_frame` + mid 循环 → OLA ISTFT
5. EOS：`pad_mix_ext` 多补 3 hop 静音，for 循环内 flush
6. 保存 wav 到 `save_dir`（非流式）和 `save_dir_stream`（流式）

**运行**：

```bash
cd stream_scnet
PYTHONPATH=. python3 SCNetStreamAudioConv1dChunk.py \
  --checkpoint_path <checkpoint.th> \
  --input_dir <input.wav>
```

---

## `test_onnx_inference()`

**目的**：导出 first/mid ONNX，ORT 流式推理与 PyTorch 流式、非流式 SCNet 三方对齐。

**流程**：

1. 同 `test_full_scnet` 加载 checkpoint 与音频
2. 内联 `export_onnx()` 生成 `onnx/scnet_first.onnx`、`onnx/scnet_mid.onnx`
3. `run_stream(use_ort=False)`：PyTorch 流式参考
4. `run_stream(use_ort=True)`：ORT first + mid + Host STFT/OLA
5. 打印 max/mean diff

**运行**：

```bash
PYTHONPATH=. python3 -c \
  "from SCNetStreamAudioConv1dChunk import test_onnx_inference; test_onnx_inference()" \
  --checkpoint_path <checkpoint.th> \
  --input_dir <input.wav>
```

**典型输出**：

```
torch vs ref:  max≈5e-5
ort  vs ref:   max≈5e-5
ort  vs torch: max≈2e-7
```

---

## 其他测试函数（历史/单元）

| 函数 | 说明 |
|------|------|
| `test_scnet()` | 短随机音频，流式 vs 非流式（含 `forward_last_frame`） |
| `test_scnet_nostft()` | 无 STFT 频域流式测试 |
| `test_scnet_stft_stream()` | STFT 流式单元测试 |

部署路径以 `test_full_scnet` / `test_onnx_inference` 为准（mid + 3hop EOS，无 last）。
