# ONNX 导出说明

## 依赖

```bash
pip install -r requirements-onnx.txt
# torch>=2.0.1, onnx>=1.14.0, onnxruntime>=1.16.0, onnxsim>=0.4.33
```

## 导出方式

### 方式 A：独立脚本（随机权重演示）

```bash
cd stream_scnet
PYTHONPATH=. python3 export_onnx_first.py   # → onnx/scnet_first.onnx
PYTHONPATH=. python3 export_onnx_mid.py      # → onnx/scnet_mid.onnx
PYTHONPATH=. python3 export_onnx_last.py     # → onnx/scnet_last.onnx（可选）
```

### 方式 B：`test_onnx_inference()` 内联导出（真实 checkpoint）

加载 checkpoint 后，在函数内调用 `export_onnx()`，输出到 `onnx/scnet_first.onnx`、`onnx/scnet_mid.onnx`。

## `onnx_common.py` 要点

### State 命名（固定顺序）

```python
STATE_NAMES = (
    "cache_band0", "cache_band1", "cache_band2",
    "cache_h1", "cache_c1", "cache_h2", "cache_c2", "cache_conv",
    "cache_fus0", "cache_fus1", "cache_fus2",
    "skip0", "skip1", "skip2",
)
```

- first 输入：`spec_in` + 11 caches（`CACHE_NAMES`）
- first 输出：14 个 `{name}_out`
- mid 输入：`spec_in` + 14 states
- mid 输出：`spec_out` + 14 个 `{name}_out`

### Wrapper

| 类 | 作用 |
|----|------|
| `FirstONNXWrapper` | 包装 `forward_1st_frame`，输出扁平 state |
| `MidONNXWrapper` | 包装 `forward`，输入/输出含 skip deque |
| `LastONNXWrapper` | 包装 `forward_last_frame`（部署未用） |

### `export_onnx()` 流程

1. `torch.onnx.export` → `*.raw.onnx`
2. 检查无 STFT/FFT 节点
3. `onnxsim.simplify()` 简化并校验
4. 保存最终 `*.onnx`

## 张量形状（固定）

| 张量 | 形状 |
|------|------|
| `spec_in` | `(1, 4, 2049, 3)` |
| `spec_out` / chunk_output | `(4, 2049, 3, 2)` |
| caches | 见 `onnx_common.make_example_state()` |

## 频域对齐检查

```bash
PYTHONPATH=. python3 check_onnx_ort.py
```

仅在频域对比 PyTorch 与 ORT，不涉及 STFT/ISTFT。
