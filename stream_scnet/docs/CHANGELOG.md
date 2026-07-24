# 变更记录

## 2026-07-24

分支：`cursor/scnet-streaming-stft-005f`

### 新增

- **`test_onnx_inference()`**  
  内联导出 first/mid ONNX（onnxsim），ORT 全流式推理，对比 PyTorch 流式 / 非流式 SCNet。

- **`generate_cache_state()`**  
  ORT-only 推理，按 denoise 风格导出量化校准 npy（input、14 个 state、output）。

- **`parse_args()` 扩展**  
  新增 `--onnx_dir`、`--quan_output_path`、`--start_sec`、`--duration_sec`。

- **文档目录 `docs/`**  
  架构、ONNX 导出、推理测试、量化数据、本 CHANGELOG。

### 修改

- **`test_onnx_inference()`**  
  由随机权重改为 `load_model` + `load_audio`，与 `test_full_scnet()` 一致。

- **`test_full_scnet()` EOS**  
  将 +3 hop 静音 flush 并入 mid for 循环，移除单独 `forward_last_frame` / EOS 分支。

- **`test_full_scnet()` 流式路径**  
  STFT + `SCNetStreamNoSTFT` + OLA ISTFT 内联实现；padding 移出 `SCNet.forward`。

### 验证结果（参考）

| 测试 | max diff |
|------|----------|
| PyTorch 流式 vs 非流式 | ~3e-7 |
| ORT vs PyTorch 流式 | ~2.4e-7 |
| 流式 vs 非流式（含 EOS） | ~5e-5 |

### 提交记录

```
badbe26 Add generate_cache_state for ONNX quantization npy dump
5cc6650 Use load_model and load_audio in test_onnx_inference
68a5161 Add test_onnx_inference for full first+mid ONNX streaming pipeline
45ea94a Fold EOS silence flush into mid for-loop in test_full_scnet
0478461 Flush lookahead in test_full_scnet with mid + zero spec
```

### 涉及文件

- `SCNetStreamAudioConv1dChunk.py`（主变更）
- `onnx_common.py`（export_onnx + onnxsim，此前已合入）
- `export_onnx_first.py` / `export_onnx_mid.py`
- `requirements-onnx.txt`
