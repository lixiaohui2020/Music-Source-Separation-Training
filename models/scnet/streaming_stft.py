"""
Streaming STFT / ISTFT helpers aligned with SCNet offline processing.

Parameters match SCNet defaults:
    n_fft=4096, hop_length=1024, win_length=4096,
    center=True, normalized=True, rectangular window (torch.ones).

The offline path in SCNet is equivalent to:
    1. SCNet pre-pad (hop alignment + even frame count)
    2. torch.stft(..., center=True, normalized=True)
    3. model inference in spectrogram domain
    4. torch.istft(..., center=True, normalized=True)
    5. trim trailing pre-pad

Streaming uses manual center padding + center=False for STFT so partial buffers
work, and incremental torch.istft with delta emission for ISTFT.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


def scnet_pre_pad(x: torch.Tensor, hop_length: int = 1024) -> tuple[torch.Tensor, int]:
    """Same tail padding as SCNet.forward before STFT."""
    padding = hop_length - x.shape[-1] % hop_length
    if (x.shape[-1] + padding) // hop_length % 2 == 0:
        padding += hop_length
    return F.pad(x, (0, padding)), padding


def _extend_for_reflect(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Extend 1-D signal until F.pad(..., reflect) is valid (len > pad)."""
    out = x
    while out.shape[-1] <= pad:
        if out.shape[-1] == 1:
            out = out.repeat(2)
        else:
            out = torch.cat([out, out.flip(-1)[..., 1:]], dim=-1)
    return out


def center_reflect_pad(x: torch.Tensor, n_fft: int, right: bool = True) -> torch.Tensor:
    """Build the center-padded signal that torch.stft(..., center=True) uses internally."""
    pad = n_fft // 2
    if x.shape[-1] > pad:
        left = F.pad(x.unsqueeze(0), (pad, 0), mode="reflect").squeeze(0)[..., :pad]
        if not right:
            return torch.cat([left, x], dim=-1)
        right_part = F.pad(x.unsqueeze(0), (0, pad), mode="reflect").squeeze(0)[..., -pad:]
        return torch.cat([left, x, right_part], dim=-1)

    ext = _extend_for_reflect(x, pad)
    left = F.pad(ext.unsqueeze(0), (pad, 0), mode="reflect").squeeze(0)[..., :pad]
    if not right:
        return torch.cat([left, x], dim=-1)
    right_part = F.pad(ext.unsqueeze(0), (0, pad), mode="reflect").squeeze(0)[..., -pad:]
    return torch.cat([left, x, right_part], dim=-1)


@dataclass
class _STFTState:
    raw: torch.Tensor
    n_frames: int = 0
    flushed: bool = False


class StreamingSTFT:
    """
    Incremental STFT aligned with torch.stft(center=True, normalized=True).

    Feed hop-sized (or larger) chunks. The first frame appears after at least
    pad + 1 new samples are available (2049 samples for n_fft=4096).
    Call with flush=True on the last chunk to append right reflect padding.
    """

    def __init__(
        self,
        n_fft: int = 4096,
        hop_length: int = 1024,
        win_length: int = 4096,
        normalized: bool = True,
        window: Optional[torch.Tensor] = None,
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.normalized = normalized
        self.pad = n_fft // 2
        self.window = window if window is not None else torch.ones(win_length)
        self._state = _STFTState(raw=torch.zeros(0))

    def reset(self) -> None:
        self._state = _STFTState(raw=torch.zeros(0))

    def _stft(self, x: torch.Tensor) -> torch.Tensor:
        kwargs = dict(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(x.device),
            center=False,
            normalized=self.normalized,
        )
        return torch.stft(x, **kwargs, return_complex=True)

    def forward(self, chunk: torch.Tensor, flush: bool = False) -> torch.Tensor:
        """
        Args:
            chunk: (N, L) or (L,) waveform; N = batch * channels flattened like SCNet.
            flush: True on the final chunk to add right reflect padding.

        Returns:
            (N, F, T_new) complex spectrogram frames newly available.
        """
        if chunk.dim() == 1:
            chunk = chunk.unsqueeze(0)
        st = self._state
        st.raw = torch.cat([st.raw.to(chunk.device), chunk.reshape(chunk.shape[0], -1)], dim=-1)
        if flush:
            st.flushed = True

        n = st.raw.shape[0]
        specs = []
        max_frames = 0
        for i in range(n):
            raw_i = st.raw[i]
            if (not st.flushed) and raw_i.shape[-1] <= self.pad:
                continue
            padded = (
                center_reflect_pad(raw_i, self.n_fft, right=True)
                if st.flushed
                else torch.cat([center_reflect_pad(raw_i, self.n_fft, right=False)[..., : self.pad], raw_i], dim=-1)
            )
            if padded.shape[-1] < self.n_fft:
                continue
            spec = self._stft(padded)
            specs.append(spec)
            max_frames = max(max_frames, spec.shape[-1])

        if not specs:
            return torch.zeros(
                n, self.n_fft // 2 + 1, 0, dtype=torch.complex64, device=chunk.device
            )

        new_count = max_frames - st.n_frames
        if new_count <= 0:
            return torch.zeros(
                n, self.n_fft // 2 + 1, 0, dtype=torch.complex64, device=chunk.device
            )

        out = torch.stack(
            [spec[..., st.n_frames : st.n_frames + new_count] for spec in specs], dim=0
        )
        st.n_frames += new_count
        return out


@dataclass
class _ISTFTState:
    spec: Optional[torch.Tensor] = None
    emitted: int = 0


class StreamingISTFT:
    """
    Incremental ISTFT aligned with torch.istft(center=True, normalized=True).

    Each call may emit up to hop_length * T_new samples. Pass length= on flush
    to match offline reconstruction length (SCNet padded length).
    """

    def __init__(
        self,
        n_fft: int = 4096,
        hop_length: int = 1024,
        win_length: int = 4096,
        normalized: bool = True,
        window: Optional[torch.Tensor] = None,
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.normalized = normalized
        self.window = window if window is not None else torch.ones(win_length)
        self._state = _ISTFTState()

    def reset(self) -> None:
        self._state = _ISTFTState()

    def forward(
        self,
        spec_chunk: torch.Tensor,
        flush: bool = False,
        length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            spec_chunk: (N, F, T_new) complex spectrogram.
            flush: True on the final chunk.
            length: padded waveform length expected by offline SCNet ISTFT.

        Returns:
            (N, L_new) newly reconstructed samples.
        """
        if spec_chunk.numel() == 0:
            n = spec_chunk.shape[0] if spec_chunk.dim() > 0 else 1
            return torch.zeros(n, 0, device=spec_chunk.device)

        st = self._state
        flat = spec_chunk.reshape(spec_chunk.shape[0], spec_chunk.shape[1], -1)
        if st.spec is None:
            st.spec = flat
        else:
            st.spec = torch.cat([st.spec, flat], dim=-1)

        kwargs = dict(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(spec_chunk.device),
            center=True,
            normalized=self.normalized,
        )
        if flush and length is not None:
            kwargs["length"] = length

        y = torch.istft(st.spec, **kwargs)
        if y.dim() == 1:
            delta = y[st.emitted :]
        else:
            delta = y[:, st.emitted :]
        st.emitted = y.shape[-1]
        return delta


def spectrogram_to_scnet_input(spec: torch.Tensor, audio_channels: int = 2) -> torch.Tensor:
    """Convert streaming STFT output to SCNet nostft input layout (B, C, Fr, T)."""
    # spec: (B*C, F, T) complex -> (B, 4, Fr, T) for stereo with dims[0]=4
    x = torch.view_as_real(spec)
    x = x.permute(0, 3, 1, 2)
    b = spec.shape[0] // audio_channels
    return x.reshape(b, x.shape[1] * audio_channels, x.shape[2], x.shape[3])


def scnet_output_to_spectrogram(
    x: torch.Tensor, batch_size: int, n_sources: int, audio_channels: int = 2
) -> torch.Tensor:
    """Convert SCNet nostft output back to (B*S*C, F, T) complex spectrogram."""
    # x: (B, n, Fr, T) where n = dims[0] * num_sources
    b = batch_size
    n = x.shape[1] // n_sources
    x = x.view(b, n_sources, n, x.shape[2], x.shape[3])
    x = x.reshape(-1, 2, x.shape[3], x.shape[4]).permute(0, 2, 3, 1)
    return torch.view_as_complex(x.contiguous())
