# Streaming SCNet

完整流式 SCNet：`STFT(pad_mode=constant)` → `SCNetStreamNoSTFT` → `ISTFT`，与离线 `SCNet` 对齐。

## 参数

- `n_fft=4096`, `hop=1024`, `center=True`, `pad_mode='constant'`, `normalized=True`, Hann 窗
- Separation `lookahead=3` → 按 **3 帧**一块喂入

## 用法

```python
from models.scnet.stream import SCNet, StreamingSCNet, offline_scnet

offline = SCNet()          # 或 load_state_dict 后
streamer = StreamingSCNet.from_offline(offline)

mix = ...                  # (2, L) stereo
y_ref = offline_scnet(mix, offline)          # (S, 2, L)
y_stream = streamer.process_waveform(mix)    # (S, 2, L) 对齐
```

## 测试

```bash
python3 -m models.scnet.stream.test_stream_scnet
```

覆盖：

1. nostft 频谱域流式对齐  
2. STFT constant 帧对齐  
3. STFT + nostft 频谱对齐  
4. 完整波形端到端对齐  
