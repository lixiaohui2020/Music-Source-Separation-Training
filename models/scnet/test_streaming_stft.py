"""
Verify streaming STFT/ISTFT alignment with SCNet offline STFT/ISTFT.

Reference pattern (test_scnet_nostft): streaming inference runs on spectrogram
domain without internal STFT. This test wraps that path with StreamingSTFT /
StreamingISTFT and checks batch vs stream parity.
"""

import torch
import torch.nn.functional as F

from models.scnet.streaming_stft import (
    StreamingISTFT,
    StreamingSTFT,
    scnet_output_to_spectrogram,
    scnet_pre_pad,
    spectrogram_to_scnet_input,
)


def offline_stft_istft(x: torch.Tensor, hop: int = 1024) -> torch.Tensor:
    """SCNet-compatible STFT -> ISTFT roundtrip (identity on waveform)."""
    n_fft = 4096
    window = torch.ones(n_fft)
    x_pad, padding = scnet_pre_pad(x, hop)
    spec = torch.stft(
        x_pad.reshape(-1, x_pad.shape[-1]),
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=True,
        normalized=True,
        return_complex=True,
    )
    y = torch.istft(
        spec,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=True,
        normalized=True,
    )
    return y.reshape(x.shape[0], -1)[:, : x.shape[-1]]


def streaming_stft_istft(x: torch.Tensor, hop: int = 1024) -> torch.Tensor:
    """Streaming wrapper with the same parameters as offline_stft_istft."""
    n_fft = 4096
    window = torch.ones(n_fft)
    x_pad, _ = scnet_pre_pad(x, hop)

    stft = StreamingSTFT(n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window)
    istft = StreamingISTFT(n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window)

    parts = []
    pos = 0
    while pos < x_pad.shape[-1]:
        chunk = x_pad[:, pos : pos + hop]
        if chunk.shape[-1] < hop:
            break
        is_last = pos + hop >= x_pad.shape[-1]
        spec = stft.forward(chunk, flush=is_last)
        if spec.shape[-1] > 0:
            y = istft.forward(spec, flush=is_last, length=x_pad.shape[-1] if is_last else None)
            if y.shape[-1] > 0:
                parts.append(y)
        pos += hop

    return torch.cat(parts, dim=-1)[:, : x.shape[-1]]


def test_stft_spec_alignment():
    """Streaming STFT spectrogram must match batch STFT exactly."""
    n_fft, hop = 4096, 1024
    window = torch.ones(n_fft)
    for length in [5000, 10000, 44100, 2048, 3000]:
        stft = StreamingSTFT(n_fft=n_fft, hop_length=hop, window=window)
        x = torch.randn(2, length)
        x_pad, _ = scnet_pre_pad(x, hop)
        spec_batch = torch.stft(
            x_pad.reshape(-1, x_pad.shape[-1]),
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            center=True,
            normalized=True,
            return_complex=True,
        )
        parts = []
        pos = 0
        while pos < x_pad.shape[-1]:
            chunk = x_pad[:, pos : pos + hop]
            is_last = pos + hop >= x_pad.shape[-1]
            parts.append(stft.forward(chunk, flush=is_last))
            pos += hop
        spec_stream = torch.cat([p for p in parts if p.shape[-1] > 0], dim=-1)
        err = (spec_batch - spec_stream).abs().max().item()
        assert err < 1e-5, f"STFT spec mismatch at L={length}: {err}"


def test_waveform_alignment():
    for length in [5000, 10000, 44100, 2048, 3000, 12345]:
        x = torch.randn(2, length)
        y_off = offline_stft_istft(x)
        y_str = streaming_stft_istft(x)
        err = (y_off - y_str).abs().max().item()
        assert err < 1e-5, f"Waveform mismatch at L={length}: {err}"


def test_scnet_layout_conversion():
    """Layout helpers round-trip the spectrogram tensor shape."""
    b, c, t = 1, 2, 8
    f = 4096 // 2 + 1
    spec = torch.randn(b * c, f, t, dtype=torch.complex64)
    x = spectrogram_to_scnet_input(spec, audio_channels=c)
    assert x.shape == (b, 4, f, t)
    back = scnet_output_to_spectrogram(x, batch_size=b, n_sources=1, audio_channels=c)
    assert back.shape == (b * c, f, t)


if __name__ == "__main__":
    test_stft_spec_alignment()
    test_waveform_alignment()
    test_scnet_layout_conversion()
    print("All streaming STFT/ISTFT alignment tests passed.")
