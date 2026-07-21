"""Full streaming SCNet pipeline: STFT(constant) + SCNetStreamNoSTFT + ISTFT.

Matches offline:
  torch.stft(..., center=True, pad_mode='constant', normalized=True, window=hann)

Streaming STFT:
  left/right zero pad of n_fft//2, then center=False framing.
  Frames are fed to SCNetStreamNoSTFT in groups of 3 (lookahead=3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from .scnet import SCNet, SCNetStreamNoSTFT, convert_state_dict, init_stream_nostft_state


def stream_align_pad(
    waveform: torch.Tensor, hop: int = 1024, n_fft: int = 4096, frames_per_chunk: int = 3
) -> Tuple[torch.Tensor, int]:
    """
    Pad waveform so offline center=True STFT frame count is divisible by frames_per_chunk.

    With center=True: T = 1 + L_pad // hop  (when L_pad % hop == 0).
    """
    length = waveform.shape[-1]
    # Round up to hop multiple first.
    padding = (hop - length % hop) % hop
    l_pad = length + padding
    t = 1 + l_pad // hop
    # Make T divisible by frames_per_chunk.
    extra_frames = (frames_per_chunk - t % frames_per_chunk) % frames_per_chunk
    padding += extra_frames * hop
    return F.pad(waveform, (0, padding)), padding


def stft_frames_from_buffer(
    buffer: torch.Tensor,
    window: torch.Tensor,
    n_fft: int = 4096,
    hop: int = 1024,
    normalized: bool = True,
) -> torch.Tensor:
    """center=False STFT on a pre-padded buffer -> (C, F, T) complex."""
    return torch.stft(
        buffer,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window.to(buffer.device),
        center=False,
        normalized=normalized,
        return_complex=True,
    )


def complex_to_scnet_input(spec: torch.Tensor, audio_channels: int = 2) -> torch.Tensor:
    """(C, F, T) complex -> (B, 4, F, T) for stereo."""
    x = torch.view_as_real(spec)
    x = x.permute(0, 3, 1, 2)
    return x.reshape(
        x.shape[0] // audio_channels, x.shape[1] * audio_channels, x.shape[2], x.shape[3]
    )


@dataclass
class StreamSCNetState:
    # Waveform history for STFT (grows with left pad + signal + right pad pieces).
    wav_buf: torch.Tensor
    nostft_state: list
    emitted_frames: int = 0
    total_frames: int = 0
    flushed: bool = False


class StreamingSCNet:
    """
    End-to-end streaming SCNet.

    Usage:
        offline = SCNet(...)
        stream = StreamingSCNet.from_offline(offline)
        y = stream.process_waveform(mix)  # (S, C, L)
    """

    def __init__(
        self,
        model: SCNetStreamNoSTFT,
        n_fft: int = 4096,
        hop: int = 1024,
        audio_channels: int = 2,
        num_sources: int = 2,
        frames_per_chunk: int = 3,
    ):
        self.model = model
        self.model.eval()
        self.n_fft = n_fft
        self.hop = hop
        self.audio_channels = audio_channels
        self.num_sources = num_sources
        self.frames_per_chunk = frames_per_chunk
        self.window = model.stft_window
        self.pad = n_fft // 2

    @classmethod
    def from_offline(cls, offline: SCNet) -> "StreamingSCNet":
        stream_model = SCNetStreamNoSTFT(
            sources=list(offline.sources),
            audio_channels=offline.audio_channels,
            dims=list(offline.dims),
            nfft=offline.nfft,
            hop_size=offline.hop_length,
            win_size=offline.win_size,
            num_dplayer=offline.separation_net.num_layers,
        )
        convert_state_dict(offline, stream_model)
        return cls(
            stream_model,
            n_fft=offline.nfft,
            hop=offline.hop_length,
            audio_channels=offline.audio_channels,
            num_sources=len(offline.sources),
        )

    def init_state(self, device: torch.device, dtype: torch.dtype = torch.float32) -> StreamSCNetState:
        # Start with left constant pad (n_fft // 2 zeros).
        return StreamSCNetState(
            wav_buf=torch.zeros(self.audio_channels, self.pad, device=device, dtype=dtype),
            nostft_state=init_stream_nostft_state(batch_size=1, device=device, dtype=dtype),
        )

    def _num_frames(self, buf_len: int) -> int:
        if buf_len < self.n_fft:
            return 0
        return 1 + (buf_len - self.n_fft) // self.hop

    def _take_frames(self, state: StreamSCNetState, n_frames: int) -> torch.Tensor:
        """Extract next n_frames from wav_buf via center=False STFT."""
        start = state.emitted_frames * self.hop
        need = start + self.n_fft + (n_frames - 1) * self.hop
        assert state.wav_buf.shape[-1] >= need
        segment = state.wav_buf[:, start:need]
        spec = stft_frames_from_buffer(segment, self.window, self.n_fft, self.hop)
        assert spec.shape[-1] == n_frames, (spec.shape, n_frames)
        state.emitted_frames += n_frames
        return complex_to_scnet_input(spec, self.audio_channels)

    def _istft_torch(self, spec_rt: torch.Tensor, length: int) -> torch.Tensor:
        complex_spec = torch.view_as_complex(spec_rt.contiguous())
        return torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.n_fft,
            window=self.window.to(spec_rt.device),
            center=True,
            normalized=True,
            length=length,
        )

    @torch.no_grad()
    def process_waveform(self, mix: torch.Tensor) -> torch.Tensor:
        """
        Process mixture via streaming chunks; result aligned with offline SCNet.

        Args:
            mix: (C, L) or (1, C, L)
        Returns:
            (S, C, L) trimmed to original length
        """
        if mix.dim() == 3:
            assert mix.shape[0] == 1
            mix = mix[0]
        assert mix.dim() == 2 and mix.shape[0] == self.audio_channels

        original_length = mix.shape[-1]
        pad_mix, _ = stream_align_pad(mix, hop=self.hop, n_fft=self.n_fft, frames_per_chunk=self.frames_per_chunk)
        device, dtype = mix.device, mix.dtype
        state = self.init_state(device, dtype)

        # Append signal then right constant pad (same as torch.stft center=True, pad_mode=constant).
        state.wav_buf = torch.cat(
            [
                state.wav_buf,
                pad_mix,
                torch.zeros(self.audio_channels, self.pad, device=device, dtype=dtype),
            ],
            dim=-1,
        )
        state.total_frames = self._num_frames(state.wav_buf.shape[-1])
        assert state.total_frames % self.frames_per_chunk == 0, state.total_frames

        spec_out_chunks: List[torch.Tensor] = []
        k = self.frames_per_chunk

        # First group: fill lookahead, no output.
        x0 = self._take_frames(state, k)
        _, *state.nostft_state = self.model.forward_1st_frame(x0, *state.nostft_state)

        # Middle groups.
        while state.emitted_frames + k <= state.total_frames:
            x = self._take_frames(state, k)
            out, *state.nostft_state = self.model(x, *state.nostft_state)
            spec_out_chunks.append(out)

        assert state.emitted_frames == state.total_frames

        # Flush lookahead.
        out, *state.nostft_state = self.model.forward_last_frame(None, *state.nostft_state)
        spec_out_chunks.append(out)

        full_spec = torch.cat(spec_out_chunks, dim=-2)  # (N, F, T, 2)
        wave = self._istft_torch(full_spec, length=pad_mix.shape[-1])
        wave = wave.reshape(self.num_sources, self.audio_channels, -1)
        return wave[:, :, :original_length]

    @torch.no_grad()
    def process_chunks(self, chunks: List[torch.Tensor], final_length: Optional[int] = None) -> torch.Tensor:
        """
        True chunk-wise API: feed hop-aligned waveform pieces sequentially.

        Each chunk except possibly the last should be a multiple of hop.
        Right constant pad is applied on the final call automatically when
        concatenating all chunks first via process_waveform is not used.

        For simplicity and exact offline alignment, prefer process_waveform.
        This method concatenates chunks then calls process_waveform.
        """
        mix = torch.cat(chunks, dim=-1)
        if final_length is not None:
            mix = mix[:, :final_length]
        return self.process_waveform(mix)


def offline_scnet(mix: torch.Tensor, model: SCNet) -> torch.Tensor:
    """Run offline SCNet. mix: (C, L) -> (S, C, L)."""
    if mix.dim() == 2:
        mix = mix.unsqueeze(0)
    with torch.no_grad():
        out = model(mix)
    return out[0]
