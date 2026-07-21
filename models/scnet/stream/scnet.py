"""Offline and streaming SCNet variants used for stream/non-stream alignment.

STFT settings match the user's stream_scnet code:
  n_fft=4096, hop=1024, center=True, pad_mode='constant',
  normalized=True, hann window.
"""

from __future__ import annotations

import math
from collections import deque
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .separation import SeparationNet, SeparationNetStream


class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()


class ConvolutionModule(nn.Module):
    def __init__(self, channels, depth=2, compress=4, kernel=3):
        super().__init__()
        assert kernel % 2 == 1
        self.depth = abs(depth)
        hidden_size = int(channels / compress)
        self.layers = nn.ModuleList()
        for _ in range(self.depth):
            padding = kernel // 2
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(channels, hidden_size * 2, (kernel, 1), padding=(padding, 0)),
                    nn.BatchNorm2d(hidden_size * 2),
                    nn.GLU(1),
                    nn.Conv2d(
                        hidden_size, hidden_size, (kernel, 1), padding=(padding, 0), groups=hidden_size
                    ),
                    nn.BatchNorm2d(hidden_size),
                    Swish(),
                    nn.Conv2d(hidden_size, channels, (1, 1)),
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


class FusionLayer(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
        fpad = kernel_size[0] // 2
        self.pad = nn.ConstantPad2d((kernel_size[1] - 1, 0, 0, 0), 0.0)
        self.conv = nn.Conv2d(channels * 2, channels * 2, kernel_size, stride=stride, padding=(fpad, 0))

    def forward(self, x, skip=None):
        if skip is not None:
            x = x + skip
        x = x.repeat(1, 2, 1, 1)
        x = self.pad(x)
        x = self.conv(x)
        return F.glu(x, dim=1)


class FusionLayerStream(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
        fpad = kernel_size[0] // 2
        self.conv = nn.Conv2d(channels * 2, channels * 2, kernel_size, stride=stride, padding=(fpad, 0))

    def forward(self, x, cache_in, skip=None):
        if skip is not None:
            x = x + skip
        x = x.repeat(1, 2, 1, 1)
        x = torch.cat([cache_in, x], dim=-1)
        cache_out = x[..., -2:]
        x = self.conv(x)
        return F.glu(x, dim=1), cache_out


class SDlayer(nn.Module):
    def __init__(self, channels_in, channels_out, band_configs):
        super().__init__()
        self.convs = nn.ModuleList()
        self.strides = []
        self.kernels = []
        for config in band_configs.values():
            self.convs.append(
                nn.Conv2d(
                    channels_in,
                    channels_out,
                    (config["kernel"], 1),
                    (config["stride"], 1),
                    (0, 0),
                )
            )
            self.strides.append(config["stride"])
            self.kernels.append(config["kernel"])
        self.SR_low = band_configs["low"]["SR"]
        self.SR_mid = band_configs["mid"]["SR"]

    def forward(self, x):
        _, _, fr, _ = x.shape
        splits = [
            (0, math.ceil(fr * self.SR_low)),
            (math.ceil(fr * self.SR_low), math.ceil(fr * (self.SR_low + self.SR_mid))),
            (math.ceil(fr * (self.SR_low + self.SR_mid)), fr),
        ]
        outputs = []
        original_lengths = []
        for conv, stride, kernel, (start, end) in zip(self.convs, self.strides, self.kernels, splits):
            extracted = x[:, :, start:end, :]
            original_lengths.append(end - start)
            current_length = extracted.shape[2]
            if stride == 1:
                total_padding = kernel - stride
            else:
                total_padding = (stride - current_length % stride) % stride
            pad_left = total_padding // 2
            pad_right = total_padding - pad_left
            padded = F.pad(extracted, (0, 0, pad_left, pad_right))
            outputs.append(conv(padded))
        return outputs, original_lengths


class SUlayer(nn.Module):
    def __init__(self, channels_in, channels_out, band_configs):
        super().__init__()
        self.convtrs = nn.ModuleList(
            [
                nn.ConvTranspose2d(channels_in, channels_out, [config["kernel"], 1], [config["stride"], 1])
                for config in band_configs.values()
            ]
        )

    def forward(self, x, lengths, origin_lengths):
        splits = [
            (0, lengths[0]),
            (lengths[0], lengths[0] + lengths[1]),
            (lengths[0] + lengths[1], None),
        ]
        outputs = []
        for idx, (convtr, (start, end)) in enumerate(zip(self.convtrs, splits)):
            out = convtr(x[:, :, start:end, :])
            dist = abs(origin_lengths[idx] - out.shape[2]) // 2
            outputs.append(out[:, :, dist : dist + origin_lengths[idx], :])
        return torch.cat(outputs, dim=2)


class SDblock(nn.Module):
    def __init__(self, channels_in, channels_out, band_configs=None, conv_config=None, depths=None, kernel_size=3):
        super().__init__()
        band_configs = band_configs or {}
        conv_config = conv_config or {}
        depths = depths or [3, 2, 1]
        self.SDlayer = SDlayer(channels_in, channels_out, band_configs)
        self.conv_modules = nn.ModuleList(
            [ConvolutionModule(channels_out, depth, **conv_config) for depth in depths]
        )
        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        fpad = kernel_size[0] // 2
        self.pad = nn.ConstantPad2d((kernel_size[1] - 1, 0, 0, 0), 0.0)
        self.globalconv = nn.Conv2d(channels_out, channels_out, kernel_size, (1, 1), (fpad, 0))

    def forward(self, x):
        bands, original_lengths = self.SDlayer(x)
        conv_bands = [F.gelu(conv(band)) for conv, band in zip(self.conv_modules, bands)]
        lengths = [band.size(-2) for band in conv_bands]
        full_band = torch.cat(conv_bands, dim=2)
        output = self.globalconv(self.pad(full_band))
        return output, full_band, lengths, original_lengths


class SDblockStream(nn.Module):
    def __init__(self, channels_in, channels_out, band_configs=None, conv_config=None, depths=None, kernel_size=3):
        super().__init__()
        band_configs = band_configs or {}
        conv_config = conv_config or {}
        depths = depths or [3, 2, 1]
        self.SDlayer = SDlayer(channels_in, channels_out, band_configs)
        self.conv_modules = nn.ModuleList(
            [ConvolutionModule(channels_out, depth, **conv_config) for depth in depths]
        )
        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        fpad = kernel_size[0] // 2
        self.globalconv = nn.Conv2d(channels_out, channels_out, kernel_size, (1, 1), (fpad, 0))

    def forward(self, x, band_input):
        bands, original_lengths = self.SDlayer(x)
        conv_bands = [F.gelu(conv(band)) for conv, band in zip(self.conv_modules, bands)]
        lengths = [band.size(-2) for band in conv_bands]
        full_band = torch.cat(conv_bands, dim=2)
        tmp_input = torch.cat([band_input, full_band], dim=-1)
        output = self.globalconv(tmp_input)
        return output, full_band, lengths, original_lengths, tmp_input[..., -2:]


def _default_kwargs():
    return dict(
        sources=["accompaniment", "vocals"],
        audio_channels=2,
        dims=[4, 64, 128, 64],
        nfft=4096,
        hop_size=1024,
        win_size=4096,
        normalized=True,
        band_SR=[0.175, 0.392, 0.433],
        band_stride=[1, 4, 16],
        band_kernel=[3, 4, 16],
        conv_depths=[3, 2, 1],
        compress=4,
        conv_kernel=3,
        num_dplayer=2,
        expand=1,
    )


class _SCNetBase(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        cfg = _default_kwargs()
        cfg.update(kwargs)
        self.sources = cfg["sources"]
        self.audio_channels = cfg["audio_channels"]
        self.dims = cfg["dims"]
        self.nfft = cfg["nfft"]
        self.hop_length = cfg["hop_size"]
        self.win_size = cfg["win_size"]
        band_keys = ["low", "mid", "high"]
        self.band_configs = {
            band_keys[i]: {
                "SR": cfg["band_SR"][i],
                "stride": cfg["band_stride"][i],
                "kernel": cfg["band_kernel"][i],
            }
            for i in range(3)
        }
        self.conv_config = {"compress": cfg["compress"], "kernel": cfg["conv_kernel"]}
        self.register_buffer("stft_window", torch.hann_window(cfg["nfft"]), persistent=False)
        self.stft_config = {
            "n_fft": cfg["nfft"],
            "hop_length": cfg["hop_size"],
            "win_length": cfg["win_size"],
            "center": True,
            "normalized": True,
        }
        self._build_codec(cfg)

    def _build_codec(self, cfg):
        raise NotImplementedError

    def waveform_to_spec(self, x: torch.Tensor) -> torch.Tensor:
        """(B*C or C, L) / (B, C, L) -> (B, 4, F, T) real spectrogram layout."""
        if x.dim() == 3:
            b, c, length = x.shape
            flat = x.reshape(b * c, length)
        else:
            flat = x
            b = flat.shape[0] // self.audio_channels
        spec = torch.stft(
            flat,
            **self.stft_config,
            pad_mode="constant",
            window=self.stft_window.to(flat.device),
            return_complex=True,
        )
        spec = torch.view_as_real(spec)
        return spec.permute(0, 3, 1, 2).reshape(
            b, spec.shape[3] * self.audio_channels, spec.shape[1], spec.shape[2]
        )

    def spec_to_waveform(self, x: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        """(N, F, T, 2) -> (N, L) with same STFT config."""
        complex_spec = torch.view_as_complex(x.contiguous())
        kwargs = dict(self.stft_config)
        kwargs["window"] = self.stft_window.to(x.device)
        if length is not None:
            kwargs["length"] = length
        return torch.istft(complex_spec, **kwargs)


class SCNet(_SCNetBase):
    """Full offline SCNet: waveform -> STFT -> model -> ISTFT -> waveform."""

    def _build_codec(self, cfg):
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for index in range(len(self.dims) - 1):
            self.encoder.append(
                SDblock(
                    channels_in=self.dims[index],
                    channels_out=self.dims[index + 1],
                    band_configs=self.band_configs,
                    conv_config=self.conv_config,
                    depths=cfg["conv_depths"],
                )
            )
            self.decoder.insert(
                0,
                nn.Sequential(
                    FusionLayer(channels=self.dims[index + 1]),
                    SUlayer(
                        channels_in=self.dims[index + 1],
                        channels_out=self.dims[index]
                        if index != 0
                        else self.dims[index] * len(self.sources),
                        band_configs=self.band_configs,
                    ),
                ),
            )
        self.separation_net = SeparationNet(
            channels=self.dims[-1], expand=cfg["expand"], num_layers=cfg["num_dplayer"]
        )

    def forward_spec(self, x: torch.Tensor) -> torch.Tensor:
        """Spectrogram in/out: (B, 4, F, T) -> (N, F, T, 2)."""
        b, _, fr, t = x.shape
        save_skip = deque()
        save_lengths = deque()
        save_original_lengths = deque()
        for sd_layer in self.encoder:
            x, skip, lengths, original_lengths = sd_layer(x)
            save_skip.append(skip)
            save_lengths.append(lengths)
            save_original_lengths.append(original_lengths)
        x = self.separation_net(x)
        for fusion_layer, su_layer in self.decoder:
            x = fusion_layer(x, save_skip.pop())
            x = su_layer(x, save_lengths.pop(), save_original_lengths.pop())
        n = self.dims[0]
        x = x.view(b, n, -1, fr, t)
        return x.reshape(-1, 2, fr, t).permute(0, 2, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        if x.dim() == 2:
            x = x.unsqueeze(0)
        b = x.shape[0]
        spec = self.waveform_to_spec(x)
        out_spec = self.forward_spec(spec)
        wave = self.spec_to_waveform(out_spec, length=length)
        return wave.reshape(b, len(self.sources), self.audio_channels, -1)


class SCNetNoStft(_SCNetBase):
    """Offline SCNet without STFT/ISTFT (spectrogram domain)."""

    def _build_codec(self, cfg):
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for index in range(len(self.dims) - 1):
            self.encoder.append(
                SDblock(
                    channels_in=self.dims[index],
                    channels_out=self.dims[index + 1],
                    band_configs=self.band_configs,
                    conv_config=self.conv_config,
                    depths=cfg["conv_depths"],
                )
            )
            self.decoder.insert(
                0,
                nn.Sequential(
                    FusionLayer(channels=self.dims[index + 1]),
                    SUlayer(
                        channels_in=self.dims[index + 1],
                        channels_out=self.dims[index]
                        if index != 0
                        else self.dims[index] * len(self.sources),
                        band_configs=self.band_configs,
                    ),
                ),
            )
        self.separation_net = SeparationNet(
            channels=self.dims[-1], expand=cfg["expand"], num_layers=cfg["num_dplayer"]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, fr, t = x.shape
        save_skip = deque()
        save_lengths = deque()
        save_original_lengths = deque()
        for sd_layer in self.encoder:
            x, skip, lengths, original_lengths = sd_layer(x)
            save_skip.append(skip)
            save_lengths.append(lengths)
            save_original_lengths.append(original_lengths)
        x = self.separation_net(x)
        for fusion_layer, su_layer in self.decoder:
            x = fusion_layer(x, save_skip.pop())
            x = su_layer(x, save_lengths.pop(), save_original_lengths.pop())
        n = self.dims[0]
        x = x.view(b, n, -1, fr, t)
        return x.reshape(-1, 2, fr, t).permute(0, 2, 3, 1)


class SCNetNoISTFT(SCNet):
    """Offline SCNet with STFT but without ISTFT (returns spectrogram)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        spec = self.waveform_to_spec(x)
        return self.forward_spec(spec)


class SCNetStreamNoSTFT(_SCNetBase):
    """Streaming SCNet in spectrogram domain (3-frame chunks, lookahead=3)."""

    # Fixed band lengths for n_fft=4096 / F=2049 with default band configs.
    _SAVE_LENGTHS = [[359, 201, 56], [108, 61, 17], [33, 19, 5]]
    _SAVE_ORIGIN_LENGTHS = [[359, 803, 887], [108, 242, 266], [33, 73, 80]]

    def _build_codec(self, cfg):
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for index in range(len(self.dims) - 1):
            self.encoder.append(
                SDblockStream(
                    channels_in=self.dims[index],
                    channels_out=self.dims[index + 1],
                    band_configs=self.band_configs,
                    conv_config=self.conv_config,
                    depths=cfg["conv_depths"],
                )
            )
            self.decoder.insert(
                0,
                nn.Sequential(
                    FusionLayerStream(channels=self.dims[index + 1]),
                    SUlayer(
                        channels_in=self.dims[index + 1],
                        channels_out=self.dims[index]
                        if index != 0
                        else self.dims[index] * len(self.sources),
                        band_configs=self.band_configs,
                    ),
                ),
            )
        self.separation_net = SeparationNetStream(
            channels=self.dims[-1], expand=cfg["expand"], num_layers=cfg["num_dplayer"]
        )

    def _run_encoder(self, x, cache_band0, cache_band1, cache_band2):
        save_skip = deque()
        save_lengths = deque()
        save_original_lengths = deque()
        caches_in = [cache_band0, cache_band1, cache_band2]
        caches_out = []
        for icnt, sd_layer in enumerate(self.encoder):
            x, skip, lengths, original_lengths, cache_out = sd_layer(x, caches_in[icnt])
            caches_out.append(cache_out)
            save_skip.append(skip)
            save_lengths.append(lengths)
            save_original_lengths.append(original_lengths)
        return x, caches_out[0], caches_out[1], caches_out[2], save_skip, save_lengths, save_original_lengths

    def _run_decoder(self, x, save_skip, save_lengths, save_original_lengths, cache_fus0, cache_fus1, cache_fus2):
        caches_in = [cache_fus0, cache_fus1, cache_fus2]
        caches_out = []
        for icnt, (fusion_layer, su_layer) in enumerate(self.decoder):
            x, cache_out = fusion_layer(x, caches_in[icnt], save_skip.pop())
            caches_out.append(cache_out)
            x = su_layer(x, save_lengths.pop(), save_original_lengths.pop())
        return x, caches_out[0], caches_out[1], caches_out[2]

    def forward_1st_frame(
        self,
        x,
        cache_band0_in,
        cache_band1_in,
        cache_band2_in,
        cache_h1,
        cache_c1,
        cache_h2,
        cache_c2,
        cache_conv,
        cache_fus0_in,
        cache_fus1_in,
        cache_fus2_in,
        save_skip_in,
    ):
        x, cache_band0_out, cache_band1_out, cache_band2_out, save_skip, _, _ = self._run_encoder(
            x, cache_band0_in, cache_band1_in, cache_band2_in
        )
        (
            x,
            new_cache_h1,
            new_cache_c1,
            new_cache_h2,
            new_cache_c2,
            new_cache_conv,
        ) = self.separation_net.forward_1st_frame(x, cache_h1, cache_c1, cache_h2, cache_c2, cache_conv)
        return (
            None,
            cache_band0_out,
            cache_band1_out,
            cache_band2_out,
            new_cache_h1,
            new_cache_c1,
            new_cache_h2,
            new_cache_c2,
            new_cache_conv,
            cache_fus0_in,
            cache_fus1_in,
            cache_fus2_in,
            save_skip,
        )

    def forward(
        self,
        x,
        cache_band0_in,
        cache_band1_in,
        cache_band2_in,
        cache_h1,
        cache_c1,
        cache_h2,
        cache_c2,
        cache_conv,
        cache_fus0_in,
        cache_fus1_in,
        cache_fus2_in,
        save_skip_in,
    ):
        b, _, fr, t = x.shape
        x, cache_band0_out, cache_band1_out, cache_band2_out, save_skip, save_lengths, save_original_lengths = (
            self._run_encoder(x, cache_band0_in, cache_band1_in, cache_band2_in)
        )
        x, new_h1, new_c1, new_h2, new_c2, new_cache_conv = self.separation_net(
            x, cache_h1, cache_c1, cache_h2, cache_c2, cache_conv
        )
        x, cache_fus0_out, cache_fus1_out, cache_fus2_out = self._run_decoder(
            x, save_skip_in, save_lengths, save_original_lengths, cache_fus0_in, cache_fus1_in, cache_fus2_in
        )
        n = self.dims[0]
        x = x.view(b, n, -1, fr, t).reshape(-1, 2, fr, t).permute(0, 2, 3, 1)
        return (
            x,
            cache_band0_out,
            cache_band1_out,
            cache_band2_out,
            new_h1,
            new_c1,
            new_h2,
            new_c2,
            new_cache_conv,
            cache_fus0_out,
            cache_fus1_out,
            cache_fus2_out,
            save_skip,
        )

    def forward_last_frame(
        self,
        x,
        cache_band0_in,
        cache_band1_in,
        cache_band2_in,
        cache_h1,
        cache_c1,
        cache_h2,
        cache_c2,
        cache_conv,
        cache_fus0_in,
        cache_fus1_in,
        cache_fus2_in,
        save_skip_in,
    ):
        b, fr = 1, self.nfft // 2 + 1
        save_lengths = deque(self._SAVE_LENGTHS)
        save_original_lengths = deque(self._SAVE_ORIGIN_LENGTHS)
        x, new_h1, new_c1, new_h2, new_c2, new_cache_conv = self.separation_net.forward_last_frame(
            x, cache_h1, cache_c1, cache_h2, cache_c2, cache_conv
        )
        x, cache_fus0_out, cache_fus1_out, cache_fus2_out = self._run_decoder(
            x, save_skip_in, save_lengths, save_original_lengths, cache_fus0_in, cache_fus1_in, cache_fus2_in
        )
        t = x.shape[-1]
        n = self.dims[0]
        x = x.view(b, n, -1, fr, t).reshape(-1, 2, fr, t).permute(0, 2, 3, 1)
        return (
            x,
            cache_band0_in,
            cache_band1_in,
            cache_band2_in,
            new_h1,
            new_c1,
            new_h2,
            new_c2,
            new_cache_conv,
            cache_fus0_out,
            cache_fus1_out,
            cache_fus2_out,
            None,
        )


def convert_state_dict(src_net: nn.Module, dst_net: nn.Module) -> None:
    src = src_net.state_dict()
    dst = dst_net.state_dict()
    for key, value in src.items():
        if key in dst and dst[key].shape == value.shape:
            dst[key] = value
    dst_net.load_state_dict(dst)


def init_stream_nostft_state(
    batch_size: int = 1,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> List[torch.Tensor]:
    """Caches for dims=[4,64,128,64], F=2049, lookahead=3, num_layers=2."""
    device = device or torch.device("cpu")
    z = lambda *shape: torch.zeros(*shape, device=device, dtype=dtype)
    return [
        z(batch_size, 64, 616, 2),   # cache_band0
        z(batch_size, 128, 186, 2),  # cache_band1
        z(batch_size, 64, 57, 2),    # cache_band2
        z(1, 57, 64),                # h1
        z(1, 57, 64),                # c1
        z(1, 57, 128),               # h2
        z(1, 57, 128),               # c2
        z(57, 128, 6),               # conv lookahead cache (F, C, 2*lookahead)
        z(batch_size, 128, 57, 2),   # fus0
        z(batch_size, 256, 186, 2),  # fus1
        z(batch_size, 128, 616, 2),  # fus2
        None,                        # save_skip
    ]
