# SCNet 流式 ONNX 文档索引

本目录记录 `stream_scnet` 模块的流式推理、ONNX 导出与量化校准相关文档。

## 文档列表

| 文档 | 说明 |
|------|------|
| [architecture.md](./architecture.md) | 整体架构：Host STFT / NPU ONNX / Host OLA ISTFT |
| [onnx-export.md](./onnx-export.md) | first / mid ONNX 导出、onnxsim 简化、张量命名 |
| [inference-tests.md](./inference-tests.md) | `test_full_scnet` / `test_onnx_inference` 使用说明 |
| [quantization-data.md](./quantization-data.md) | `generate_cache_state` 量化校准 npy 生成 |
| [CHANGELOG.md](./CHANGELOG.md) | 按日期记录的代码变更 |

## 核心文件

| 文件 | 职责 |
|------|------|
| `SCNetStreamAudioConv1dChunk.py` | 模型定义 + 流式/非流式/ONNX 测试入口 |
| `onnx_common.py` | ONNX Wrapper、state 命名、`export_onnx()` |
| `export_onnx_first.py` | 独立导出 `scnet_first.onnx` |
| `export_onnx_mid.py` | 独立导出 `scnet_mid.onnx` |
| `export_onnx_last.py` | 可选 last 图（部署路径未使用） |
| `check_onnx_ort.py` | 频域 PyTorch vs ORT 对齐检查 |
| `requirements-onnx.txt` | ONNX 相关依赖 |

## 快速开始

```bash
cd stream_scnet
pip install -r requirements-onnx.txt soundfile accelerate julius tqdm

# 1. 非流式 + PyTorch 流式对齐，保存 wav
PYTHONPATH=. python3 SCNetStreamAudioConv1dChunk.py \
  --checkpoint_path <checkpoint.th> \
  --input_dir <input.wav>

# 2. 导出 ONNX + ORT 全链路对齐验证
PYTHONPATH=. python3 -c "from SCNetStreamAudioConv1dChunk import test_onnx_inference; test_onnx_inference()" \
  --checkpoint_path <checkpoint.th> \
  --input_dir <input.wav>

# 3. 生成量化校准 npy（需已有 onnx/scnet_first.onnx、scnet_mid.onnx）
PYTHONPATH=. python3 -c "from SCNetStreamAudioConv1dChunk import generate_cache_state; generate_cache_state()" \
  --input_dir <input.wav> \
  --quan_output_path ./quan_npy \
  --start_sec 30 --duration_sec 10
```

## 分支

相关开发分支：`cursor/scnet-streaming-stft-005f`
