# SCNet 流式推理架构

## 数据流

```
波形 (Host)
  │
  ├─ STFT (Host, center=False) ──► spec (1, 4, 2049, 3)
  │
  ├─ scnet_first.onnx (NPU/ORT)  ──► 初始化 14 个 state
  │
  ├─ scnet_mid.onnx  × N 次 (NPU/ORT) ──► chunk_output (4, 2049, 3, 2)
  │
  └─ OLA ISTFT (Host) ──► 分离波形
```

- **STFT / OLA ISTFT 不在 ONNX 内**，便于 NPU 只跑网络部分。
- 每次 mid 处理 **3 帧**（hop=1024，即 3072 样本/chunk）。
- **lookahead=3**：first 帧预热 cache；mid 输出相对输入延迟 3 帧。

## 两张部署图

| 图 | PyTorch 对应 | 输入 | 输出 |
|----|-------------|------|------|
| `scnet_first.onnx` | `forward_1st_frame` | `spec_in` + 11 caches | 14 states |
| `scnet_mid.onnx` | `forward` (steady) | `spec_in` + 14 states | `spec_out` + 14 states |

`scnet_last.onnx` 可选，当前部署路径 **不使用** last 图。

## EOS 处理（无 last 图）

在 `pad_mix` 末尾追加：

- **+2 hop**：补偿 STFT `center=True` 右端效应（流式 STFT 用 `center=False` 时仍保留与训练对齐的 padding 习惯）
- **+3 hop 静音**：多跑一轮 mid，冲掉 lookahead 缓冲

```python
pad_mix_ext = F.pad(pad_mix, (0, 2 * hop_size + 3 * hop_size))
```

## Padding 规则（调用方负责）

`SCNet.forward` 不再内部 padding，由 `test_full_scnet` / `test_onnx_inference` / `generate_cache_state` 统一处理：

1. hop 对齐，且 STFT 帧数为奇数
2. 总帧数 `t_frames % 3 == 0`（3 帧 chunk 对齐）

## 对齐结果（参考）

| 对比 | max diff | 说明 |
|------|----------|------|
| PyTorch 流式 vs 非流式 | ~3e-7 | 主体一致 |
| ORT 流式 vs PyTorch 流式 | ~2.4e-7 | ONNX 数值一致 |
| 流式 vs 非流式（含 EOS flush） | ~5e-5 | EOS 近似 flush 引入 |
