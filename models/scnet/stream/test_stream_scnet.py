"""Alignment tests for streaming SCNet (STFT + nostft + ISTFT)."""

from __future__ import annotations

import torch

from models.scnet.stream import (
    SCNet,
    SCNetNoISTFT,
    SCNetNoStft,
    SCNetStreamNoSTFT,
    StreamingSCNet,
    convert_state_dict,
    init_stream_nostft_state,
    offline_scnet,
    stream_align_pad,
)
from models.scnet.stream.pipeline import complex_to_scnet_input, stft_frames_from_buffer


def test_nostft_spec_alignment():
    """Spectrogram-domain stream vs offline (test_scnet_nostft part 1)."""
    torch.manual_seed(42)
    b, c, f, t = 1, 4, 2049, 18
    x = torch.randn(b, c, f, t)

    offline = SCNetNoStft()
    offline.eval()
    with torch.no_grad():
        y_ref = offline(x)

    stream = SCNetStreamNoSTFT()
    convert_state_dict(offline, stream)
    stream.eval()
    state = init_stream_nostft_state()

    outs = []
    with torch.no_grad():
        _, *state = stream.forward_1st_frame(x[..., 0:3], *state)
        for i in range(t // 3 - 1):
            chunk, *state = stream(x[..., (i + 1) * 3 : (i + 2) * 3], *state)
            outs.append(chunk)
        chunk, *state = stream.forward_last_frame(None, *state)
        outs.append(chunk)
    y_stream = torch.cat(outs, dim=-2)

    min_t = min(y_ref.shape[-2], y_stream.shape[-2])
    err = (y_stream[:, :, :min_t] - y_ref[:, :, :min_t]).abs().max().item()
    print(f"[nostft] max abs err = {err:.6e}")
    assert err < 1e-4, err


def test_stft_frame_alignment():
    """Streaming constant-pad STFT frames match offline torch.stft."""
    torch.manual_seed(42)
    hop, n_fft = 1024, 4096
    for length in [8000, 16000, 22050]:
        mix = torch.randn(2, length)
        pad_mix, _ = stream_align_pad(mix, hop=hop, n_fft=n_fft)
        window = torch.hann_window(n_fft)

        off = torch.stft(
            pad_mix,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            center=True,
            pad_mode="constant",
            normalized=True,
            return_complex=True,
        )

        # left + signal + right constant pad, center=False
        buf = torch.nn.functional.pad(pad_mix, (n_fft // 2, n_fft // 2))
        stream = stft_frames_from_buffer(buf, window, n_fft, hop)
        err = (stream - off).abs().max().item()
        print(f"[stft L={length}] T={off.shape[-1]} err={err:.6e}")
        assert err < 1e-5, err


def test_stft_plus_nostft_alignment():
    """STFT(constant) + streaming nostft vs offline SCNetNoISTFT."""
    torch.manual_seed(42)
    hop, n_fft, length = 1024, 4096, 16000
    mix = torch.randn(2, length)
    pad_mix, _ = stream_align_pad(mix, hop=hop, n_fft=n_fft)

    offline = SCNetNoISTFT()
    offline.eval()
    with torch.no_grad():
        y_ref = offline(pad_mix.unsqueeze(0))

    stream = SCNetStreamNoSTFT()
    convert_state_dict(offline, stream)
    stream.eval()
    window = stream.stft_window
    state = init_stream_nostft_state()

    # Same buffer as offline center=True constant pad
    buf = torch.nn.functional.pad(pad_mix, (n_fft // 2, n_fft // 2))
    all_frames = stft_frames_from_buffer(buf, window, n_fft, hop)
    x_all = complex_to_scnet_input(all_frames)
    assert x_all.shape[-1] % 3 == 0

    outs = []
    with torch.no_grad():
        _, *state = stream.forward_1st_frame(x_all[..., 0:3], *state)
        for i in range(1, x_all.shape[-1] // 3):
            out, *state = stream(x_all[..., i * 3 : (i + 1) * 3], *state)
            outs.append(out)
        out, *state = stream.forward_last_frame(None, *state)
        outs.append(out)

    y_stream = torch.cat(outs, dim=-2)
    min_t = min(y_ref.shape[-2], y_stream.shape[-2])
    err = (y_stream[:, :, :min_t] - y_ref[:, :, :min_t]).abs().max().item()
    print(f"[stft+nostft] T_ref={y_ref.shape[-2]} T_stream={y_stream.shape[-2]} err={err:.6e}")
    assert err < 1e-4, err


def test_full_scnet_waveform_alignment():
    """Full streaming SCNet vs offline SCNet waveforms (same align-pad)."""
    torch.manual_seed(0)
    for length in [8000, 16000, 22050]:
        mix = torch.randn(2, length)
        pad_mix, _ = stream_align_pad(mix)

        offline = SCNet()
        offline.eval()
        # Offline and stream must see the same padded mixture.
        y_ref = offline_scnet(pad_mix, offline)[:, :, :length]

        streamer = StreamingSCNet.from_offline(offline)
        y_stream = streamer.process_waveform(mix)

        err = (y_stream - y_ref).abs().max().item()
        print(f"[full L={length}] shape={tuple(y_stream.shape)} err={err:.6e}")
        assert err < 1e-4, f"L={length} err={err}"


if __name__ == "__main__":
    test_nostft_spec_alignment()
    test_stft_frame_alignment()
    test_stft_plus_nostft_alignment()
    test_full_scnet_waveform_alignment()
    print("All streaming SCNet alignment tests passed.")
