import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
# from separationStreamAudioChunk import SeparationNet
from separationStreamConv1dAudioChunk import SeparationNetStream, SeparationNet
# from separationband import SeparationNet
# from separationStreamSTFTGRU import SeparationNet
# from separationStreamSTFTLSTMTimeUniConv1d import SeparationNet
import typing as tp
import math
# import onnx
# import onnxsim
import numpy as np
# import parser
# import argparse
# import soundfile as sf
# from scnet.utils import load_model, convert_audio


class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()


class ConvolutionModule(nn.Module):
    """
    Convolution Module in SD block.

    Args:
        channels (int): input/output channels.
        depth (int): number of layers in the residual branch. Each layer has its own
        compress (float): amount of channel compression.
        kernel (int): kernel size for the convolutions.
        """

    def __init__(self, channels, depth=2, compress=4, kernel=3):
        super().__init__()
        assert kernel % 2 == 1
        self.depth = abs(depth)
        hidden_size = int(channels / compress)
        # norm = lambda d: nn.GroupNorm(1, d)
        norm = lambda d: nn.BatchNorm2d(d)
        self.layers = nn.ModuleList([])
        # for _ in range(self.depth):
        #     padding = (kernel // 2)
        #     mods = [
        #         norm(channels),
        #         nn.Conv1d(channels, hidden_size*2, kernel, padding = padding),
        #         nn.GLU(1),
        #         nn.Conv1d(hidden_size, hidden_size, kernel, padding = padding, groups = hidden_size),
        #         norm(hidden_size),
        #         Swish(),
        #         nn.Conv1d(hidden_size, channels, 1),
        #     ]
        #     layer = nn.Sequential(*mods)
        #     self.layers.append(layer)
        for _ in range(self.depth):
            padding = (kernel // 2)
            mods = [
                nn.Conv2d(channels, hidden_size * 2, (kernel, 1), padding=(padding, 0)),
                norm(hidden_size * 2),
                nn.GLU(1),
                nn.Conv2d(hidden_size, hidden_size, (kernel, 1), padding=(padding, 0), groups=hidden_size),
                norm(hidden_size),
                Swish(),
                nn.Conv2d(hidden_size, channels, (1, 1)),
            ]
            layer = nn.Sequential(*mods)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


class FusionLayer(nn.Module):
    """
    A FusionLayer within the decoder.

    Args:
    - channels (int): Number of input channels.
    - kernel_size (int, optional): Kernel size for the convolutional layer, defaults to 3.
    - stride (int, optional): Stride for the convolutional layer, defaults to 1.
    - padding (int, optional): Padding for the convolutional layer, defaults to 1.
    """

    def __init__(self, channels, kernel_size=3, stride=1, padding=1):
        super(FusionLayer, self).__init__()
        lookahead = 0
        kernel_size = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        )
        stride = (
            (stride, stride) if isinstance(stride, int) else tuple(stride)
        )

        fpad = kernel_size[1] // 2
        pad = (kernel_size[1] - 1 - lookahead, lookahead, 0, 0)
        self.pad = nn.ConstantPad2d(pad, 0.0)
        self.conv = nn.Conv2d(channels * 2, channels * 2, kernel_size, stride=stride, padding=(fpad, 0))

    def forward(self, x, skip=None):
        if skip is not None:
            x += skip
        x = x.repeat(1, 2, 1, 1)
        x = self.pad(x)
        x = self.conv(x)
        x = F.glu(x, dim=1)
        return x


class FusionLayerStream(nn.Module):
    """
    A FusionLayer within the decoder.

    Args:
    - channels (int): Number of input channels.
    - kernel_size (int, optional): Kernel size for the convolutional layer, defaults to 3.
    - stride (int, optional): Stride for the convolutional layer, defaults to 1.
    - padding (int, optional): Padding for the convolutional layer, defaults to 1.
    """

    def __init__(self, channels, kernel_size=3, stride=1, padding=1):
        super(FusionLayerStream, self).__init__()
        lookahead = 0
        kernel_size = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        )
        stride = (
            (stride, stride) if isinstance(stride, int) else tuple(stride)
        )

        fpad = kernel_size[1] // 2
        pad = (kernel_size[1] - 1 - lookahead, lookahead, 0, 0)
        self.pad = nn.ConstantPad2d(pad, 0.0)
        self.conv = nn.Conv2d(channels * 2, channels * 2, kernel_size, stride=stride, padding=(fpad, 0))

    def forward(self, x, cache_in, skip=None):
        if skip is not None:
            x += skip
        x = x.repeat(1, 2, 1, 1)
        x = torch.cat([cache_in, x], dim=-1)
        cache_out = x[..., -2:]
        x = self.conv(x)
        x = F.glu(x, dim=1)
        return x, cache_out


class SDlayer(nn.Module):
    """
    Implements a Sparse Down-sample Layer for processing different frequency bands separately.

    Args:
    - channels_in (int): Input channel count.
    - channels_out (int): Output channel count.
    - band_configs (dict): A dictionary containing configuration for each frequency band.
                           Keys are 'low', 'mid', 'high' for each band, and values are
                           dictionaries with keys 'SR', 'stride', and 'kernel' for proportion,
                           stride, and kernel size, respectively.
    """

    def __init__(self, channels_in, channels_out, band_configs):
        super(SDlayer, self).__init__()

        # Initializing convolutional layers for each band
        self.convs = nn.ModuleList()
        self.strides = []
        self.kernels = []
        for config in band_configs.values():
            self.convs.append(
                nn.Conv2d(channels_in, channels_out, (config['kernel'], 1), (config['stride'], 1), (0, 0)))
            self.strides.append(config['stride'])
            self.kernels.append(config['kernel'])

        # Saving rate proportions for determining splits
        self.SR_low = band_configs['low']['SR']
        self.SR_mid = band_configs['mid']['SR']

    def forward(self, x):
        B, C, Fr, T = x.shape
        # Define splitting points based on sampling rates
        splits = [
            (0, math.ceil(Fr * self.SR_low)),
            (math.ceil(Fr * self.SR_low), math.ceil(Fr * (self.SR_low + self.SR_mid))),
            (math.ceil(Fr * (self.SR_low + self.SR_mid)), Fr)
        ]

        # Processing each band with the corresponding convolution
        outputs = []
        original_lengths = []
        for conv, stride, kernel, (start, end) in zip(self.convs, self.strides, self.kernels, splits):
            extracted = x[:, :, start:end, :]
            original_lengths.append(end - start)
            current_length = extracted.shape[2]

            # padding
            if stride == 1:
                total_padding = kernel - stride
            else:
                total_padding = (stride - current_length % stride) % stride
            pad_left = total_padding // 2
            pad_right = total_padding - pad_left

            padded = F.pad(extracted, (0, 0, pad_left, pad_right))

            output = conv(padded)
            outputs.append(output)

        return outputs, original_lengths


class SUlayer(nn.Module):
    """
    Implements a Sparse Up-sample Layer in decoder.

    Args:
    - channels_in: The number of input channels.
    - channels_out: The number of output channels.
    - convtr_configs: Dictionary containing the configurations for transposed convolutions.
    """

    def __init__(self, channels_in, channels_out, band_configs):
        super(SUlayer, self).__init__()

        # Initializing convolutional layers for each band
        self.convtrs = nn.ModuleList([
            nn.ConvTranspose2d(channels_in, channels_out, [config['kernel'], 1], [config['stride'], 1])
            for _, config in band_configs.items()
        ])

    def forward(self, x, lengths, origin_lengths):
        B, C, Fr, T = x.shape
        # Define splitting points based on input lengths
        splits = [
            (0, lengths[0]),
            (lengths[0], lengths[0] + lengths[1]),
            (lengths[0] + lengths[1], None)
        ]
        # Processing each band with the corresponding convolution
        outputs = []
        for idx, (convtr, (start, end)) in enumerate(zip(self.convtrs, splits)):
            out = convtr(x[:, :, start:end, :])
            # Calculate the distance to trim the output symmetrically to original length
            current_Fr_length = out.shape[2]
            dist = abs(origin_lengths[idx] - current_Fr_length) // 2

            # Trim the output to the original length symmetrically
            trimmed_out = out[:, :, dist:dist + origin_lengths[idx], :]

            outputs.append(trimmed_out)

        # Concatenate trimmed outputs along the frequency dimension to return the final tensor
        x = torch.cat(outputs, dim=2)

        return x


class SDblock(nn.Module):
    """
    Implements a simplified Sparse Down-sample block in encoder.

    Args:
    - channels_in (int): Number of input channels.
    - channels_out (int): Number of output channels.
    - band_config (dict): Configuration for the SDlayer specifying band splits and convolutions.
    - conv_config (dict): Configuration for convolution modules applied to each band.
    - depths (list of int): List specifying the convolution depths for low, mid, and high frequency bands.
    """

    def __init__(self, channels_in, channels_out, band_configs={}, conv_config={}, depths=[3, 2, 1], kernel_size=3):
        super(SDblock, self).__init__()
        self.SDlayer = SDlayer(channels_in, channels_out, band_configs)

        # Dynamically create convolution modules for each band based on depths
        self.conv_modules = nn.ModuleList([
            ConvolutionModule(channels_out, depth, **conv_config) for depth in depths
        ])
        # Set the kernel_size to an odd number.
        lookahead = 0
        kernel_size = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        )

        fpad = kernel_size[0] // 2
        pad = (kernel_size[1] - 1 - lookahead, lookahead, 0, 0)
        self.pad = nn.ConstantPad2d(pad, 0.0)
        self.globalconv = nn.Conv2d(channels_out, channels_out, kernel_size, (1, 1), (fpad, 0))

    def forward(self, x):
        bands, original_lengths = self.SDlayer(x)
        # B, C, f, T = band.shape
        # reshape(-1, band.shape[1], band.shape[3]) --> reshape(-1, band.shape[1], band.shape[2])
        # bands = [
        #     F.gelu(
        #         conv(band.permute(0, 2, 1, 3).reshape(-1, band.shape[1], band.shape[2]))
        #         .view(band.shape[0], band.shape[2], band.shape[1], band.shape[3])
        #         .permute(0, 2, 1, 3)
        #     )
        #     for conv, band in zip(self.conv_modules, bands)
        #
        # ]
        conv_bands = []
        for conv, band in zip(self.conv_modules, bands):
            conv_bands.append(F.gelu(conv(band)))
        # bands = [
        #     F.gelu(conv(band))
        #     for conv, band in zip(self.conv_modules, bands)
        # ]
        # for conv, band in zip(self.conv_modules, bands):
        #     conv_out = F.gelu(conv(band))
        #     bands.append(conv_out)

        lengths = [band.size(-2) for band in conv_bands]
        full_band = torch.cat(conv_bands, dim=2)
        skip = full_band

        full_band = self.pad(full_band)
        output = self.globalconv(full_band)

        return output, skip, lengths, original_lengths


class SDblockStream(nn.Module):
    """
    Implements a simplified Sparse Down-sample block in encoder.

    Args:
    - channels_in (int): Number of input channels.
    - channels_out (int): Number of output channels.
    - band_config (dict): Configuration for the SDlayer specifying band splits and convolutions.
    - conv_config (dict): Configuration for convolution modules applied to each band.
    - depths (list of int): List specifying the convolution depths for low, mid, and high frequency bands.
    """

    def __init__(self, channels_in, channels_out, band_configs={}, conv_config={}, depths=[3, 2, 1], kernel_size=3):
        super(SDblockStream, self).__init__()
        self.SDlayer = SDlayer(channels_in, channels_out, band_configs)

        # Dynamically create convolution modules for each band based on depths
        self.conv_modules = nn.ModuleList([
            ConvolutionModule(channels_out, depth, **conv_config) for depth in depths
        ])
        # Set the kernel_size to an odd number.
        lookahead = 0
        kernel_size = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        )

        fpad = kernel_size[0] // 2
        pad = (kernel_size[1] - 1 - lookahead, lookahead, 0, 0)
        self.pad = nn.ConstantPad2d(pad, 0.0)
        self.globalconv = nn.Conv2d(channels_out, channels_out, kernel_size, (1, 1), (fpad, 0))

    def forward(self, x, band_input):
        bands, original_lengths = self.SDlayer(x)
        # B, C, f, T = band.shape
        # reshape(-1, band.shape[1], band.shape[3]) --> reshape(-1, band.shape[1], band.shape[2])
        # bands = [
        #     F.gelu(
        #         conv(band.permute(0, 2, 1, 3).reshape(-1, band.shape[1], band.shape[2]))
        #         .view(band.shape[0], band.shape[2], band.shape[1], band.shape[3])
        #         .permute(0, 2, 1, 3)
        #     )
        #     for conv, band in zip(self.conv_modules, bands)
        #
        # ]
        conv_bands = []
        for conv, band in zip(self.conv_modules, bands):
            conv_bands.append(F.gelu(conv(band)))
        # bands=[
        #     F.gelu(conv(band))
        #     for conv, band in zip(self.conv_modules, bands)
        # ]
        # for conv, band in zip(self.conv_modules, bands):
        #     conv_out = F.gelu(conv(band))
        #     bands.append(conv_out)

        lengths = [band.size(-2) for band in conv_bands]
        full_band = torch.cat(conv_bands, dim=2)
        skip = full_band

        tmp_input = torch.cat([band_input, full_band], dim=-1)

        output = self.globalconv(tmp_input)

        return output, skip, lengths, original_lengths, tmp_input[..., -2:]


class SCNet(nn.Module):
    """
    The implementation of SCNet: Sparse Compression Network for Music Source Separation. Paper: https://arxiv.org/abs/2401.13276.pdf

    Args:
    - sources (List[str]): List of sources to be separated.
    - audio_channels (int): Number of audio channels.
    - nfft (int): Number of FFTs to determine the frequency dimension of the input.
    - hop_size (int): Hop size for the STFT.
    - win_size (int): Window size for STFT.
    - normalized (bool): Whether to normalize the STFT.
    - dims (List[int]): List of channel dimensions for each block.
    - band_SR (List[float]): The proportion of each frequency band.
    - band_stride (List[int]): The down-sampling ratio of each frequency band.
    - band_kernel (List[int]): The kernel sizes for down-sampling convolution in each frequency band
    - conv_depths (List[int]): List specifying the number of convolution modules in each SD block.
    - compress (int): Compression factor for convolution module.
    - conv_kernel (int): Kernel size for convolution layer in convolution module.
    - num_dplayer (int): Number of dual-path layers.
    - expand (int): Expansion factor in the dual-path RNN, default is 1.

    """

    def __init__(self,
                 sources=['drums', 'bass', 'other', 'vocals'],
                 audio_channels=2,
                 # Main structure
                 dims=[4, 64, 128, 64],  # dims = [4, 64, 128, 256] in SCNet-large
                 # STFT
                 nfft=4096,
                 hop_size=1024,
                 win_size=4096,
                 normalized=True,
                 # SD/SU layer
                 band_SR=[0.175, 0.392, 0.433],
                 band_stride=[1, 4, 16],
                 band_kernel=[3, 4, 16],
                 # Convolution Module
                 conv_depths=[3, 2, 1],
                 compress=4,
                 conv_kernel=3,
                 # Dual-path RNN
                 num_dplayer=2,
                 expand=1,
                 ):
        super().__init__()
        self.sources = sources
        self.audio_channels = audio_channels
        self.dims = dims
        band_keys = ['low', 'mid', 'high']
        self.band_configs = {band_keys[i]: {'SR': band_SR[i], 'stride': band_stride[i], 'kernel': band_kernel[i]} for i
                             in range(len(band_keys))}
        self.hop_length = hop_size
        self.conv_config = {
            'compress': compress,
            'kernel': conv_kernel,
        }
        self.register_buffer("stft_window", torch.hann_window(nfft), persistent=False)
        self.register_buffer("stft_out_window", torch.hann_window(nfft), persistent=False)

        self.stft_config = {
            'n_fft': nfft,
            'hop_length': hop_size,
            'win_length': win_size,
            'center': True,
            'normalized': True
        }

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for index in range(len(dims) - 1):
            enc = SDblock(
                channels_in=dims[index],
                channels_out=dims[index + 1],
                band_configs=self.band_configs,
                conv_config=self.conv_config,
                depths=conv_depths
            )
            self.encoder.append(enc)

            dec = nn.Sequential(
                FusionLayer(channels=dims[index + 1]),
                SUlayer(
                    channels_in=dims[index + 1],
                    channels_out=dims[index] if index != 0 else dims[index] * len(sources),
                    band_configs=self.band_configs,
                )
            )
            self.decoder.insert(0, dec)

        self.separation_net = SeparationNet(
            channels=dims[-1],
            expand=expand,
            num_layers=num_dplayer,
        )

    def forward(self, x):
        # B, C, L = x.shape

        # STFT
        L = x.shape[-1]
        x = x.reshape(-1, L)
        x = torch.stft(x, **self.stft_config, pad_mode='constant', window=self.stft_window, return_complex=True)
        x = torch.view_as_real(x)
        x = x.permute(0, 3, 1, 2).reshape(x.shape[0] // self.audio_channels, x.shape[3] * self.audio_channels,
                                          x.shape[1], x.shape[2])

        B, C, Fr, T = x.shape
        # # # mean = x.mean(dim=(1, 2, 3), keepdim=True)
        # # # std = x.std(dim=(1, 2, 3), keepdim=True)
        # # # x = (x - mean) / (1e-5 + std)
        # #
        save_skip = deque()
        save_lengths = deque()
        save_original_lengths = deque()
        # # encoder
        for sd_layer in self.encoder:
            x, skip, lengths, original_lengths = sd_layer(x)
            save_skip.append(skip)
            save_lengths.append(lengths)
            save_original_lengths.append(original_lengths)

        # # separation
        x = self.separation_net(x)
        #
        # # decoder
        for fusion_layer, su_layer in self.decoder:
            x = fusion_layer(x, save_skip.pop())
            x = su_layer(x, save_lengths.pop(), save_original_lengths.pop())

        # # output
        n = self.dims[0]
        x = x.view(B, n, -1, Fr, T)
        # x = x * std[:, None] + mean[:, None]
        x = x.reshape(-1, 2, Fr, T).permute(0, 2, 3, 1)
        x = torch.view_as_complex(x.contiguous())
        # x = torch.istft(x, **self.stft_config, window=self.stft_out_window.to(x.device).to(torch.float16))

        x = torch.istft(x, **self.stft_config, window=self.stft_out_window.to(x.device))
        x = x.reshape(B, len(self.sources), self.audio_channels, -1)
        return x

    def forward_stft(self, x):
        # B, C, L = x.shape

        # STFT
        L = x.shape[-1]
        x = x.reshape(-1, L)
        x = torch.stft(x, **self.stft_config, pad_mode='constant', window=self.stft_window, return_complex=True)
        x = torch.view_as_real(x)
        x = x.permute(0, 3, 1, 2).reshape(x.shape[0] // self.audio_channels, x.shape[3] * self.audio_channels,
                                          x.shape[1], x.shape[2])
        return x

class SCNetNoISTFT(nn.Module):
    """
    The implementation of SCNet: Sparse Compression Network for Music Source Separation. Paper: https://arxiv.org/abs/2401.13276.pdf

    Args:
    - sources (List[str]): List of sources to be separated.
    - audio_channels (int): Number of audio channels.
    - nfft (int): Number of FFTs to determine the frequency dimension of the input.
    - hop_size (int): Hop size for the STFT.
    - win_size (int): Window size for STFT.
    - normalized (bool): Whether to normalize the STFT.
    - dims (List[int]): List of channel dimensions for each block.
    - band_SR (List[float]): The proportion of each frequency band.
    - band_stride (List[int]): The down-sampling ratio of each frequency band.
    - band_kernel (List[int]): The kernel sizes for down-sampling convolution in each frequency band
    - conv_depths (List[int]): List specifying the number of convolution modules in each SD block.
    - compress (int): Compression factor for convolution module.
    - conv_kernel (int): Kernel size for convolution layer in convolution module.
    - num_dplayer (int): Number of dual-path layers.
    - expand (int): Expansion factor in the dual-path RNN, default is 1.

    """

    def __init__(self,
                 sources=['drums', 'bass', 'other', 'vocals'],
                 audio_channels=2,
                 # Main structure
                 dims=[4, 64, 128, 64],  # dims = [4, 64, 128, 256] in SCNet-large
                 # STFT
                 nfft=4096,
                 hop_size=1024,
                 win_size=4096,
                 normalized=True,
                 # SD/SU layer
                 band_SR=[0.175, 0.392, 0.433],
                 band_stride=[1, 4, 16],
                 band_kernel=[3, 4, 16],
                 # Convolution Module
                 conv_depths=[3, 2, 1],
                 compress=4,
                 conv_kernel=3,
                 # Dual-path RNN
                 num_dplayer=2,
                 expand=1,
                 ):
        super().__init__()
        self.sources = sources
        self.audio_channels = audio_channels
        self.dims = dims
        band_keys = ['low', 'mid', 'high']
        self.band_configs = {band_keys[i]: {'SR': band_SR[i], 'stride': band_stride[i], 'kernel': band_kernel[i]} for i
                             in range(len(band_keys))}
        self.hop_length = hop_size
        self.conv_config = {
            'compress': compress,
            'kernel': conv_kernel,
        }
        self.register_buffer("stft_window", torch.hann_window(nfft), persistent=False)
        self.register_buffer("stft_out_window", torch.hann_window(nfft), persistent=False)

        self.stft_config = {
            'n_fft': nfft,
            'hop_length': hop_size,
            'win_length': win_size,
            'center': True,
            'normalized': True
        }

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for index in range(len(dims) - 1):
            enc = SDblock(
                channels_in=dims[index],
                channels_out=dims[index + 1],
                band_configs=self.band_configs,
                conv_config=self.conv_config,
                depths=conv_depths
            )
            self.encoder.append(enc)

            dec = nn.Sequential(
                FusionLayer(channels=dims[index + 1]),
                SUlayer(
                    channels_in=dims[index + 1],
                    channels_out=dims[index] if index != 0 else dims[index] * len(sources),
                    band_configs=self.band_configs,
                )
            )
            self.decoder.insert(0, dec)

        self.separation_net = SeparationNet(
            channels=dims[-1],
            expand=expand,
            num_layers=num_dplayer,
        )

    def forward(self, x):
        # B, C, L = x.shape

        # STFT
        L = x.shape[-1]
        x = x.reshape(-1, L)
        x = torch.stft(x, **self.stft_config, pad_mode='constant', window=self.stft_window, return_complex=True)
        x = torch.view_as_real(x)
        x = x.permute(0, 3, 1, 2).reshape(x.shape[0] // self.audio_channels, x.shape[3] * self.audio_channels,
                                          x.shape[1], x.shape[2])

        B, C, Fr, T = x.shape
        # # # mean = x.mean(dim=(1, 2, 3), keepdim=True)
        # # # std = x.std(dim=(1, 2, 3), keepdim=True)
        # # # x = (x - mean) / (1e-5 + std)
        # #
        save_skip = deque()
        save_lengths = deque()
        save_original_lengths = deque()
        # # encoder
        for sd_layer in self.encoder:
            x, skip, lengths, original_lengths = sd_layer(x)
            save_skip.append(skip)
            save_lengths.append(lengths)
            save_original_lengths.append(original_lengths)

        # # separation
        x = self.separation_net(x)
        #
        # # decoder
        for fusion_layer, su_layer in self.decoder:
            x = fusion_layer(x, save_skip.pop())
            x = su_layer(x, save_lengths.pop(), save_original_lengths.pop())

        # # output
        n = self.dims[0]
        x = x.view(B, n, -1, Fr, T)
        # x = x * std[:, None] + mean[:, None]
        x = x.reshape(-1, 2, Fr, T).permute(0, 2, 3, 1)
        # x = torch.view_as_complex(x.contiguous())
        # # x = torch.istft(x, **self.stft_config, window=self.stft_out_window.to(x.device).to(torch.float16))
        #
        # x = torch.istft(x, **self.stft_config, window=self.stft_out_window.to(x.device))
        # x = x.reshape(B, len(self.sources), self.audio_channels, -1)
        return x

    def forward_stft(self, x):
        # B, C, L = x.shape

        # STFT
        L = x.shape[-1]
        x = x.reshape(-1, L)
        x = torch.stft(x, **self.stft_config, pad_mode='constant', window=self.stft_window, return_complex=True)
        x = torch.view_as_real(x)
        x = x.permute(0, 3, 1, 2).reshape(x.shape[0] // self.audio_channels, x.shape[3] * self.audio_channels,
                                          x.shape[1], x.shape[2])
        return x

class SCNetNoStft(nn.Module):
    """
    The implementation of SCNet: Sparse Compression Network for Music Source Separation. Paper: https://arxiv.org/abs/2401.13276.pdf

    Args:
    - sources (List[str]): List of sources to be separated.
    - audio_channels (int): Number of audio channels.
    - nfft (int): Number of FFTs to determine the frequency dimension of the input.
    - hop_size (int): Hop size for the STFT.
    - win_size (int): Window size for STFT.
    - normalized (bool): Whether to normalize the STFT.
    - dims (List[int]): List of channel dimensions for each block.
    - band_SR (List[float]): The proportion of each frequency band.
    - band_stride (List[int]): The down-sampling ratio of each frequency band.
    - band_kernel (List[int]): The kernel sizes for down-sampling convolution in each frequency band
    - conv_depths (List[int]): List specifying the number of convolution modules in each SD block.
    - compress (int): Compression factor for convolution module.
    - conv_kernel (int): Kernel size for convolution layer in convolution module.
    - num_dplayer (int): Number of dual-path layers.
    - expand (int): Expansion factor in the dual-path RNN, default is 1.

    """

    def __init__(self,
                 sources=['drums', 'bass', 'other', 'vocals'],
                 audio_channels=2,
                 # Main structure
                 dims=[4, 64, 128, 64],  # dims = [4, 64, 128, 256] in SCNet-large
                 # STFT
                 nfft=4096,
                 hop_size=1024,
                 win_size=4096,
                 normalized=True,
                 # SD/SU layer
                 band_SR=[0.175, 0.392, 0.433],
                 band_stride=[1, 4, 16],
                 band_kernel=[3, 4, 16],
                 # Convolution Module
                 conv_depths=[3, 2, 1],
                 compress=4,
                 conv_kernel=3,
                 # Dual-path RNN
                 num_dplayer=2,
                 expand=1,
                 ):
        super().__init__()
        self.sources = sources
        self.audio_channels = audio_channels
        self.dims = dims
        band_keys = ['low', 'mid', 'high']
        self.band_configs = {band_keys[i]: {'SR': band_SR[i], 'stride': band_stride[i], 'kernel': band_kernel[i]} for i
                             in range(len(band_keys))}
        self.hop_length = hop_size
        self.conv_config = {
            'compress': compress,
            'kernel': conv_kernel,
        }
        self.register_buffer("stft_window", torch.hann_window(nfft), persistent=False)
        self.register_buffer("stft_out_window", torch.hann_window(nfft), persistent=False)

        self.stft_config = {
            'n_fft': nfft,
            'hop_length': hop_size,
            'win_length': win_size,
            'center': True,
            'normalized': True
        }

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for index in range(len(dims) - 1):
            enc = SDblock(
                channels_in=dims[index],
                channels_out=dims[index + 1],
                band_configs=self.band_configs,
                conv_config=self.conv_config,
                depths=conv_depths
            )
            self.encoder.append(enc)

            dec = nn.Sequential(
                FusionLayer(channels=dims[index + 1]),
                SUlayer(
                    channels_in=dims[index + 1],
                    channels_out=dims[index] if index != 0 else dims[index] * len(sources),
                    band_configs=self.band_configs,
                )
            )
            self.decoder.insert(0, dec)

        self.separation_net = SeparationNet(
            channels=dims[-1],
            expand=expand,
            num_layers=num_dplayer,
        )

    def forward(self, x):
        # B, C, L = x.shape


        # STFT
        # L = x.shape[-1]
        # x = x.reshape(-1, L)
        # x = torch.stft(x, **self.stft_config, pad_mode='constant', window=self.stft_window, return_complex=True)
        # x = torch.view_as_real(x)
        # x = x.permute(0, 3, 1, 2).reshape(x.shape[0] // self.audio_channels, x.shape[3] * self.audio_channels,
        #                                   x.shape[1], x.shape[2])

        B, C, Fr, T = x.shape
        # # # mean = x.mean(dim=(1, 2, 3), keepdim=True)
        # # # std = x.std(dim=(1, 2, 3), keepdim=True)
        # # # x = (x - mean) / (1e-5 + std)
        # #
        save_skip = deque()
        save_lengths = deque()
        save_original_lengths = deque()
        # # encoder
        for sd_layer in self.encoder:
            x, skip, lengths, original_lengths = sd_layer(x)
            save_skip.append(skip)
            save_lengths.append(lengths)
            save_original_lengths.append(original_lengths)

        # # separation
        x = self.separation_net(x)
        #
        # # decoder
        for fusion_layer, su_layer in self.decoder:
            x = fusion_layer(x, save_skip.pop())
            x = su_layer(x, save_lengths.pop(), save_original_lengths.pop())

        # # output
        n = self.dims[0]
        x = x.view(B, n, -1, Fr, T)
        # # x = x * std[:, None] + mean[:, None]
        x = x.reshape(-1, 2, Fr, T).permute(0, 2, 3, 1)
        # x = torch.view_as_complex(x.contiguous())
        # x = torch.istft(x, **self.stft_config, window=self.stft_out_window.to(x.device).to(torch.float16))

        # x = torch.istft(x, **self.stft_config, window=self.stft_out_window.to(x.device))
        # x = x.reshape(B, len(self.sources), self.audio_channels, -1)

        return x


class SCNetStream(nn.Module):
    """
    The implementation of SCNet: Sparse Compression Network for Music Source Separation. Paper: https://arxiv.org/abs/2401.13276.pdf

    Args:
    - sources (List[str]): List of sources to be separated.
    - audio_channels (int): Number of audio channels.
    - nfft (int): Number of FFTs to determine the frequency dimension of the input.
    - hop_size (int): Hop size for the STFT.
    - win_size (int): Window size for STFT.
    - normalized (bool): Whether to normalize the STFT.
    - dims (List[int]): List of channel dimensions for each block.
    - band_SR (List[float]): The proportion of each frequency band.
    - band_stride (List[int]): The down-sampling ratio of each frequency band.
    - band_kernel (List[int]): The kernel sizes for down-sampling convolution in each frequency band
    - conv_depths (List[int]): List specifying the number of convolution modules in each SD block.
    - compress (int): Compression factor for convolution module.
    - conv_kernel (int): Kernel size for convolution layer in convolution module.
    - num_dplayer (int): Number of dual-path layers.
    - expand (int): Expansion factor in the dual-path RNN, default is 1.

    """

    def __init__(self,
                 sources=['drums', 'bass', 'other', 'vocals'],
                 audio_channels=2,
                 # Main structure
                 dims=[4, 64, 128, 64],  # dims = [4, 64, 128, 256] in SCNet-large
                 # STFT
                 nfft=4096,
                 hop_size=1024,
                 win_size=4096,
                 normalized=True,
                 # SD/SU layer
                 band_SR=[0.175, 0.392, 0.433],
                 band_stride=[1, 4, 16],
                 band_kernel=[3, 4, 16],
                 # Convolution Module
                 conv_depths=[3, 2, 1],
                 compress=4,
                 conv_kernel=3,
                 # Dual-path RNN
                 num_dplayer=2,
                 expand=1,
                 ):
        super().__init__()
        self.sources = sources
        self.audio_channels = audio_channels
        self.dims = dims
        band_keys = ['low', 'mid', 'high']
        self.band_configs = {band_keys[i]: {'SR': band_SR[i], 'stride': band_stride[i], 'kernel': band_kernel[i]} for i
                             in range(len(band_keys))}
        self.hop_length = hop_size
        self.conv_config = {
            'compress': compress,
            'kernel': conv_kernel,
        }
        # stft parameter
        self.register_buffer("stft_window", torch.hann_window(nfft), persistent=False)
        self.register_buffer("stft_window_power", self.stft_window ** 2, persistent=False)
        rfft_real, rfft_imag = self.create_rfft_matrix(nfft)
        scale = 1.0 / math.sqrt(nfft)
        self.scale = scale
        self.register_buffer("rfft_real", rfft_real.T * scale, persistent=False)
        self.register_buffer("rfft_imag", rfft_imag.T * scale, persistent=False)

        self.stft_config = {
            'n_fft': nfft,
            'hop_length': hop_size,
            'win_length': win_size,
            'center': True,
            'normalized': True
        }

        irfft_real, irfft_imag = self.create_irfft_matrix(nfft)
        self.register_buffer("irfft_real", irfft_real.T * scale, persistent=False)
        self.register_buffer("irfft_imag", irfft_imag.T * scale, persistent=False)

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for index in range(len(dims) - 1):
            enc = SDblockStream(
                channels_in=dims[index],
                channels_out=dims[index + 1],
                band_configs=self.band_configs,
                conv_config=self.conv_config,
                depths=conv_depths
            )
            self.encoder.append(enc)

            dec = nn.Sequential(
                FusionLayerStream(channels=dims[index + 1]),
                SUlayer(
                    channels_in=dims[index + 1],
                    channels_out=dims[index] if index != 0 else dims[index] * len(sources),
                    band_configs=self.band_configs,
                )
            )
            self.decoder.insert(0, dec)

        self.separation_net = SeparationNetStream(
            channels=dims[-1],
            expand=expand,
            num_layers=num_dplayer,
        )

    def create_rfft_matrix(self, n_fft, dtype=torch.float32):
        """创建 RFFT 矩阵（实部和虚部分开）"""
        freq_size = int(n_fft / 2) + 1

        k = torch.arange(freq_size, dtype=torch.float64).unsqueeze(1)
        n = torch.arange(n_fft, dtype=torch.float64).unsqueeze(0)

        angle = -2.0 * math.pi * k * n / n_fft

        rfft_real = torch.cos(angle).to(dtype)
        rfft_imag = torch.sin(angle).to(dtype)

        return rfft_real, rfft_imag

    def create_irfft_matrix(self, n_fft, dtype=torch.float32):
        """创建 RFFT 矩阵（实部和虚部分开）"""
        freq_size = n_fft // 2 + 1

        k = torch.arange(freq_size, dtype=torch.float64).unsqueeze(0)
        n = torch.arange(n_fft, dtype=torch.float64).unsqueeze(1)

        angle = 2.0 * math.pi * k * n / n_fft

        irfft_real = torch.cos(angle).to(dtype)
        irfft_imag = torch.sin(angle).to(dtype)

        scale = torch.ones(freq_size, dtype=dtype)
        scale[1:-1] = 2.0
        irfft_real = irfft_real * scale.unsqueeze(0)
        irfft_imag = irfft_imag * scale.unsqueeze(0)

        return irfft_real, irfft_imag

    def forward(self, x, cache_stft_in, cache_band0_in, cache_band1_in, cache_band2_in,
                cache_h1, cache_c1, cache_h2, cache_c2, cache_conv, lookahead,
                cache_fus0_in, cache_fus1_in, cache_fus2_in, overlap_buffer_in, window_sum_in):
        # B, C, L = x.shape
        # B = x.shape[0]
        # In the initial padding, ensure that the number of frames after the STFT (the length of the T dimension) is even,
        # so that the RFFT operation can be used in the separation network.
        # padding = self.hop_length - x.shape[-1] % self.hop_length
        # if (x.shape[-1] + padding) // self.hop_length % 2 == 0:
        #     padding += self.hop_length
        # x = F.pad(x, (0, padding))

        # STFT
        x = torch.cat([cache_stft_in, x], dim=-1)
        cache_stft_out = x[:, self.hop_length:]
        frames_windowed = x * self.stft_window
        real_part = torch.matmul(frames_windowed, self.rfft_real)
        imag_part = torch.matmul(frames_windowed, self.rfft_imag)

        x = torch.cat([real_part.unsqueeze(-1), imag_part.unsqueeze(-1)], dim=-1)
        x = x.unsqueeze(-2)
        # x = torch.view_as_real(x)
        x = x.permute(0, 3, 1, 2).reshape(x.shape[0] // self.audio_channels, x.shape[3] * self.audio_channels,
                                          x.shape[1], x.shape[2])

        B, C, Fr, T = x.shape
        save_skip = deque()
        save_lengths = deque()
        save_original_lengths = deque()
        # encoder
        for icnt, sd_layer in enumerate(self.encoder):
            if icnt == 0:
                x, skip, lengths, original_lengths, cache_band0_out = sd_layer(x, cache_band0_in)
            elif icnt == 1:
                x, skip, lengths, original_lengths, cache_band1_out = sd_layer(x, cache_band1_in)
            elif icnt == 2:
                x, skip, lengths, original_lengths, cache_band2_out = sd_layer(x, cache_band2_in)
            save_skip.append(skip)
            save_lengths.append(lengths)
            save_original_lengths.append(original_lengths)

        # # separation
        x, new_cache_h1, new_cache_c1, new_cache_h2, new_cache_c2, new_cache_conv, lookahead = self.separation_net(x, cache_h1, cache_c1, cache_h2, cache_c2, cache_conv, lookahead)
        if x is None:
            return (None, cache_stft_out, cache_band0_out, cache_band1_out, cache_band2_out,
                    new_cache_h1, new_cache_c1, new_cache_h2, new_cache_c2, new_cache_conv, lookahead,
                    cache_fus0_in, cache_fus1_in, cache_fus2_in, overlap_buffer_in, window_sum_in)
        #
        # decoder
        icnt = 0
        for fusion_layer, su_layer in self.decoder:
            if icnt == 0:
                x, cache_fus0_out = fusion_layer(x, cache_fus0_in, save_skip.pop())
            elif icnt == 1:
                x, cache_fus1_out = fusion_layer(x, cache_fus1_in, save_skip.pop())
            elif icnt == 2:
                x, cache_fus2_out = fusion_layer(x, cache_fus2_in, save_skip.pop())
            x = su_layer(x, save_lengths.pop(), save_original_lengths.pop())
            icnt = icnt + 1

        # # output
        n = self.dims[0]
        x = x.view(B, n, -1, Fr, T)
        x = x.reshape(-1, 2, Fr, T).permute(0, 2, 3, 1)
        x = x.squeeze(2)
        time_frame = torch.matmul(x[..., 0], self.irfft_real) - torch.matmul(x[..., 1], self.irfft_imag)
        window_frame = time_frame * self.stft_window
        overlap_buffer_out = overlap_buffer_in + window_frame
        window_sum_out = window_sum_in + self.stft_window_power
        window_norm = window_sum_out[..., :self.hop_length]
        # window_norm = torch.where(window_norm > 1e-8, window_norm, torch.ones_like(window_norm))
        output = overlap_buffer_out[..., :self.hop_length].clone()
        x = output / (window_norm + 1e-10)
        overlap_buffer_out = torch.cat([overlap_buffer_out[..., self.hop_length:],
                                        torch.zeros([overlap_buffer_in.shape[0], self.hop_length])], dim=-1)
        window_sum_out = torch.cat([window_sum_out[..., self.hop_length:],
                                    torch.zeros([window_sum_in.shape[0], self.hop_length])], dim=-1)
        x = x.reshape(B, len(self.sources), self.audio_channels, -1)
        # cache_stft_out = cache_stft_in
        # overlap_buffer_out, window_sum_out = overlap_buffer_in, window_sum_in
        # x = x[:, :, :, :-padding]
        return (x, cache_stft_out, cache_band0_out, cache_band1_out, cache_band2_out,
                new_cache_h1, new_cache_c1, new_cache_h2, new_cache_c2, new_cache_conv, lookahead,
                cache_fus0_out, cache_fus1_out, cache_fus2_out,
                overlap_buffer_out, window_sum_out)

    def forward_stft_istft(self, x, cache_stft_in, cache_band0_in, cache_band1_in, cache_band2_in,
                cache_fus0_in, cache_fus1_in, cache_fus2_in, overlap_buffer_in, window_sum_in):
        # B, C, L = x.shape

        # STFT
        x_nstft = torch.cat([cache_stft_in, x], dim=-1)
        cache_stft_out = x_nstft[:, self.hop_length:]
        frames_windowed_nstft = x_nstft * self.stft_window
        real_part_nstft = torch.matmul(frames_windowed_nstft, self.rfft_real)
        imag_part_nstft = torch.matmul(frames_windowed_nstft, self.rfft_imag)
        x_nstft = torch.cat([real_part_nstft.unsqueeze(-1), imag_part_nstft.unsqueeze(-1)], dim=-1)
        x_nstft = x_nstft.unsqueeze(-2)
        # x = torch.view_as_real(x)

        L = x.shape[-1]
        x = x.reshape(-1, L)
        # x = torch.stft(x, **self.stft_config, pad_mode='constant', window=self.stft_window, return_complex=True)
        x = torch.view_as_real(x)

        x = x.permute(0, 3, 1, 2).reshape(x.shape[0] // self.audio_channels, x.shape[3] * self.audio_channels,
                                          x.shape[1], x.shape[2])

        B, C, Fr, T = x.shape
        # # mean = x.mean(dim=(1, 2, 3), keepdim=True)
        # # std = x.std(dim=(1, 2, 3), keepdim=True)
        # # x = (x - mean) / (1e-5 + std)
        #
        save_skip = deque()
        save_lengths = deque()
        save_original_lengths = deque()
        # encoder
        for icnt, sd_layer in enumerate(self.encoder):
            if icnt == 0:
                x, skip, lengths, original_lengths, cache_band0_out = sd_layer(x, cache_band0_in)
            elif icnt == 1:
                x, skip, lengths, original_lengths, cache_band1_out = sd_layer(x, cache_band1_in)
            elif icnt == 2:
                x, skip, lengths, original_lengths, cache_band2_out = sd_layer(x, cache_band2_in)
            save_skip.append(skip)
            save_lengths.append(lengths)
            save_original_lengths.append(original_lengths)

        # # separation
        x = self.separation_net(x)
        #
        # decoder
        icnt = 0
        for fusion_layer, su_layer in self.decoder:
            if icnt == 0:
                x, cache_fus0_out = fusion_layer(x, cache_fus0_in, save_skip.pop())
            elif icnt == 1:
                x, cache_fus1_out = fusion_layer(x, cache_fus1_in, save_skip.pop())
            elif icnt == 2:
                x, cache_fus2_out = fusion_layer(x, cache_fus2_in, save_skip.pop())
            x = su_layer(x, save_lengths.pop(), save_original_lengths.pop())
            icnt = icnt + 1

        # # output
        n = self.dims[0]
        x = x.view(B, n, -1, Fr, T)
        # x = x * std[:, None] + mean[:, None]
        x = x.reshape(-1, 2, Fr, T).permute(0, 2, 3, 1)
        x = x.squeeze(2)
        time_frame = torch.matmul(x[..., 0], self.irfft_real) - torch.matmul(x[..., 1], self.irfft_imag)
        window_frame = time_frame * self.stft_window
        overlap_buffer_out = overlap_buffer_in + window_frame
        window_sum_out = window_sum_in + self.stft_window_power
        window_norm = window_sum_out[..., :self.hop_length]
        # window_norm = torch.where(window_norm > 1e-8, window_norm, torch.ones_like(window_norm))
        output = overlap_buffer_out[..., :self.hop_length].clone()
        x = output / (window_norm + 1e-10)
        # x = output / window_norm
        overlap_buffer_out = torch.cat([overlap_buffer_out[..., self.hop_length:],
                                        torch.zeros([overlap_buffer_in.shape[0], self.hop_length])], dim=-1)
        window_sum_out = torch.cat([window_sum_out[..., self.hop_length:],
                                    torch.zeros([window_sum_in.shape[0], self.hop_length])], dim=-1)
        x = x.reshape(B, len(self.sources), self.audio_channels, -1)
        # cache_stft_out = cache_stft_in
        # overlap_buffer_out, window_sum_out = overlap_buffer_in, window_sum_in
        # x = x[:, :, :, :-padding]
        return (x, cache_stft_out, cache_band0_out, cache_band1_out, cache_band2_out,
                cache_fus0_out, cache_fus1_out, cache_fus2_out, overlap_buffer_out, window_sum_out)

    def forward_stft(self, x, x_chunk, cache_stft_in):
        L = x.shape[-1]
        x = x.reshape(-1, L)
        stft_x = torch.stft(x, **self.stft_config, pad_mode='constant', window=self.stft_window, return_complex=True)
        real_stft_x = torch.view_as_real(stft_x)
        output_non_stream = real_stft_x.permute(0, 3, 1, 2).reshape(real_stft_x.shape[0] // self.audio_channels, real_stft_x.shape[3] * self.audio_channels,
                                          real_stft_x.shape[1], real_stft_x.shape[2])

        x_nstft = torch.cat([cache_stft_in, x_chunk], dim=-1)
        cache_stft_out = x_nstft[:, self.hop_length:]
        frames_windowed_nstft = x_nstft * self.stft_window
        real_part_nstft = torch.matmul(frames_windowed_nstft, self.rfft_real)
        real_out = torch.matmul(x_nstft, self.stft_window[:, None]*self.rfft_real)

        imag_part_nstft = torch.matmul(frames_windowed_nstft, self.rfft_imag)
        x_nstft = torch.cat([real_part_nstft.unsqueeze(-1), imag_part_nstft.unsqueeze(-1)], dim=-1)
        x_nstft = x_nstft.unsqueeze(-2)

        return output_non_stream, x_nstft, cache_stft_out

class SCNetStreamNoISTFT(nn.Module):
    """
    The implementation of SCNet: Sparse Compression Network for Music Source Separation. Paper: https://arxiv.org/abs/2401.13276.pdf

    Args:
    - sources (List[str]): List of sources to be separated.
    - audio_channels (int): Number of audio channels.
    - nfft (int): Number of FFTs to determine the frequency dimension of the input.
    - hop_size (int): Hop size for the STFT.
    - win_size (int): Window size for STFT.
    - normalized (bool): Whether to normalize the STFT.
    - dims (List[int]): List of channel dimensions for each block.
    - band_SR (List[float]): The proportion of each frequency band.
    - band_stride (List[int]): The down-sampling ratio of each frequency band.
    - band_kernel (List[int]): The kernel sizes for down-sampling convolution in each frequency band
    - conv_depths (List[int]): List specifying the number of convolution modules in each SD block.
    - compress (int): Compression factor for convolution module.
    - conv_kernel (int): Kernel size for convolution layer in convolution module.
    - num_dplayer (int): Number of dual-path layers.
    - expand (int): Expansion factor in the dual-path RNN, default is 1.

    """

    def __init__(self,
                 sources=['drums', 'bass', 'other', 'vocals'],
                 audio_channels=2,
                 # Main structure
                 dims=[4, 64, 128, 64],  # dims = [4, 64, 128, 256] in SCNet-large
                 # STFT
                 nfft=4096,
                 hop_size=1024,
                 win_size=4096,
                 normalized=True,
                 # SD/SU layer
                 band_SR=[0.175, 0.392, 0.433],
                 band_stride=[1, 4, 16],
                 band_kernel=[3, 4, 16],
                 # Convolution Module
                 conv_depths=[3, 2, 1],
                 compress=4,
                 conv_kernel=3,
                 # Dual-path RNN
                 num_dplayer=2,
                 expand=1,
                 ):
        super().__init__()
        self.sources = sources
        self.audio_channels = audio_channels
        self.dims = dims
        band_keys = ['low', 'mid', 'high']
        self.band_configs = {band_keys[i]: {'SR': band_SR[i], 'stride': band_stride[i], 'kernel': band_kernel[i]} for i
                             in range(len(band_keys))}
        self.hop_length = hop_size
        self.conv_config = {
            'compress': compress,
            'kernel': conv_kernel,
        }
        # stft parameter
        self.register_buffer("stft_window", torch.hann_window(nfft), persistent=False)
        self.register_buffer("stft_window_power", self.stft_window ** 2, persistent=False)
        rfft_real, rfft_imag = self.create_rfft_matrix(nfft)
        scale = 1.0 / math.sqrt(nfft)
        self.scale = scale
        self.register_buffer("rfft_real", rfft_real.T * scale, persistent=False)
        self.register_buffer("rfft_imag", rfft_imag.T * scale, persistent=False)

        self.stft_config = {
            'n_fft': nfft,
            'hop_length': hop_size,
            'win_length': win_size,
            'center': True,
            'normalized': True
        }

        irfft_real, irfft_imag = self.create_irfft_matrix(nfft)
        self.register_buffer("irfft_real", irfft_real.T * scale, persistent=False)
        self.register_buffer("irfft_imag", irfft_imag.T * scale, persistent=False)

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for index in range(len(dims) - 1):
            enc = SDblockStream(
                channels_in=dims[index],
                channels_out=dims[index + 1],
                band_configs=self.band_configs,
                conv_config=self.conv_config,
                depths=conv_depths
            )
            self.encoder.append(enc)

            dec = nn.Sequential(
                FusionLayerStream(channels=dims[index + 1]),
                SUlayer(
                    channels_in=dims[index + 1],
                    channels_out=dims[index] if index != 0 else dims[index] * len(sources),
                    band_configs=self.band_configs,
                )
            )
            self.decoder.insert(0, dec)

        self.separation_net = SeparationNetStream(
            channels=dims[-1],
            expand=expand,
            num_layers=num_dplayer,
        )

    def create_rfft_matrix(self, n_fft, dtype=torch.float32):
        """创建 RFFT 矩阵（实部和虚部分开）"""
        freq_size = int(n_fft / 2) + 1

        k = torch.arange(freq_size, dtype=torch.float64).unsqueeze(1)
        n = torch.arange(n_fft, dtype=torch.float64).unsqueeze(0)

        angle = -2.0 * math.pi * k * n / n_fft

        rfft_real = torch.cos(angle).to(dtype)
        rfft_imag = torch.sin(angle).to(dtype)

        return rfft_real, rfft_imag

    def create_irfft_matrix(self, n_fft, dtype=torch.float32):
        """创建 RFFT 矩阵（实部和虚部分开）"""
        freq_size = n_fft // 2 + 1

        k = torch.arange(freq_size, dtype=torch.float64).unsqueeze(0)
        n = torch.arange(n_fft, dtype=torch.float64).unsqueeze(1)

        angle = 2.0 * math.pi * k * n / n_fft

        irfft_real = torch.cos(angle).to(dtype)
        irfft_imag = torch.sin(angle).to(dtype)

        scale = torch.ones(freq_size, dtype=dtype)
        scale[1:-1] = 2.0
        irfft_real = irfft_real * scale.unsqueeze(0)
        irfft_imag = irfft_imag * scale.unsqueeze(0)

        return irfft_real, irfft_imag

    def forward_1st_frame(self, x, cache_stft, cache_band0_in, cache_band1_in, cache_band2_in,
                cache_h1, cache_c1, cache_h2, cache_c2, cache_conv,
                cache_fus0_in, cache_fus1_in, cache_fus2_in, save_skip_in):
        C = x.shape
        full_input = torch.cat([cache_stft, x], dim=-1)
        cache_stft_out = full_input[:, -3*self.hop_length:]
        frame = F.unfold(full_input.view(C, 1, 1, -1),
                         [1, self.win_length],
                         stride=(1, self.hop_length))
        frame = frame.permute(2, 0, 1)
        frame_win = frame * self.stft_window
        real_part = torch.matmul(frame_win, self.rfft_real)
        imag_part = torch.matmul(frame_win, self.rfft_imag)

        x = torch.cat([real_part.unsqueeze(-1), imag_part.unsqueeze(-1)], dim=-1)
        x = x.unsqueeze(-2)
        # x = torch.view_as_real(x)
        x = x.permute(0, 3, 1, 2).reshape(x.shape[0] // self.audio_channels, x.shape[3] * self.audio_channels,
                                          x.shape[1], x.shape[2])

        save_skip = deque()
        save_lengths = deque()
        save_original_lengths = deque()
        # encoder
        for icnt, sd_layer in enumerate(self.encoder):
            if icnt == 0:
                x, skip, lengths, original_lengths, cache_band0_out = sd_layer(x, cache_band0_in)
            elif icnt == 1:
                x, skip, lengths, original_lengths, cache_band1_out = sd_layer(x, cache_band1_in)
            elif icnt == 2:
                x, skip, lengths, original_lengths, cache_band2_out = sd_layer(x, cache_band2_in)
            save_skip.append(skip)
            save_lengths.append(lengths)
            save_original_lengths.append(original_lengths)

        # # separation
        (x, new_cache_h1, new_cache_c1, new_cache_h2, new_cache_c2,
         new_cache_conv) = self.separation_net.forward_1st_frame(x, cache_h1, cache_c1, cache_h2, cache_c2, cache_conv)

        return (None, cache_stft_out, cache_band0_out, cache_band1_out, cache_band2_out,
                new_cache_h1, new_cache_c1, new_cache_h2, new_cache_c2, new_cache_conv,
                cache_fus0_in, cache_fus1_in, cache_fus2_in, save_skip)

    def forward(self, x, cache_stft_in, cache_band0_in, cache_band1_in, cache_band2_in,
                cache_h1, cache_c1, cache_h2, cache_c2, cache_conv, lookahead,
                cache_fus0_in, cache_fus1_in, cache_fus2_in, overlap_buffer_in, window_sum_in):
        # STFT
        x = torch.cat([cache_stft_in, x], dim=-1)
        cache_stft_out = x[:, self.hop_length:]
        frames_windowed = x * self.stft_window
        real_part = torch.matmul(frames_windowed, self.rfft_real)
        imag_part = torch.matmul(frames_windowed, self.rfft_imag)

        x = torch.cat([real_part.unsqueeze(-1), imag_part.unsqueeze(-1)], dim=-1)
        x = x.unsqueeze(-2)
        # x = torch.view_as_real(x)
        x = x.permute(0, 3, 1, 2).reshape(x.shape[0] // self.audio_channels, x.shape[3] * self.audio_channels,
                                          x.shape[1], x.shape[2])

        B, C, Fr, T = x.shape
        save_skip = deque()
        save_lengths = deque()
        save_original_lengths = deque()
        # encoder
        for icnt, sd_layer in enumerate(self.encoder):
            if icnt == 0:
                x, skip, lengths, original_lengths, cache_band0_out = sd_layer(x, cache_band0_in)
            elif icnt == 1:
                x, skip, lengths, original_lengths, cache_band1_out = sd_layer(x, cache_band1_in)
            elif icnt == 2:
                x, skip, lengths, original_lengths, cache_band2_out = sd_layer(x, cache_band2_in)
            save_skip.append(skip)
            save_lengths.append(lengths)
            save_original_lengths.append(original_lengths)

        # # separation
        x, new_cache_h1, new_cache_c1, new_cache_h2, new_cache_c2, new_cache_conv, lookahead = self.separation_net(x, cache_h1, cache_c1, cache_h2, cache_c2, cache_conv, lookahead)
        #
        # decoder
        icnt = 0
        for fusion_layer, su_layer in self.decoder:
            if icnt == 0:
                x, cache_fus0_out = fusion_layer(x, cache_fus0_in, save_skip.pop())
            elif icnt == 1:
                x, cache_fus1_out = fusion_layer(x, cache_fus1_in, save_skip.pop())
            elif icnt == 2:
                x, cache_fus2_out = fusion_layer(x, cache_fus2_in, save_skip.pop())
            x = su_layer(x, save_lengths.pop(), save_original_lengths.pop())
            icnt = icnt + 1

        # # output
        n = self.dims[0]
        x = x.view(B, n, -1, Fr, T)
        x = x.reshape(-1, 2, Fr, T).permute(0, 2, 3, 1)
        # x = x.squeeze(2)
        # time_frame = torch.matmul(x[..., 0], self.irfft_real) - torch.matmul(x[..., 1], self.irfft_imag)
        # window_frame = time_frame * self.stft_window
        # overlap_buffer_out = overlap_buffer_in + window_frame
        # window_sum_out = window_sum_in + self.stft_window_power
        # window_norm = window_sum_out[..., :self.hop_length]
        # # window_norm = torch.where(window_norm > 1e-8, window_norm, torch.ones_like(window_norm))
        # output = overlap_buffer_out[..., :self.hop_length].clone()
        # x = output / (window_norm + 1e-10)
        # overlap_buffer_out = torch.cat([overlap_buffer_out[..., self.hop_length:],
        #                                 torch.zeros([overlap_buffer_in.shape[0], self.hop_length])], dim=-1)
        # window_sum_out = torch.cat([window_sum_out[..., self.hop_length:],
        #                             torch.zeros([window_sum_in.shape[0], self.hop_length])], dim=-1)
        # x = x.reshape(B, len(self.sources), self.audio_channels, -1)
        return (x, cache_stft_out, cache_band0_out, cache_band1_out, cache_band2_out,
                new_cache_h1, new_cache_c1, new_cache_h2, new_cache_c2, new_cache_conv, lookahead,
                cache_fus0_out, cache_fus1_out, cache_fus2_out)
                # overlap_buffer_out, window_sum_out)

    def forward_stft_istft(self, x, cache_stft_in, cache_band0_in, cache_band1_in, cache_band2_in,
                cache_fus0_in, cache_fus1_in, cache_fus2_in, overlap_buffer_in, window_sum_in):
        # B, C, L = x.shape

        # STFT
        x_nstft = torch.cat([cache_stft_in, x], dim=-1)
        cache_stft_out = x_nstft[:, self.hop_length:]
        frames_windowed_nstft = x_nstft * self.stft_window
        real_part_nstft = torch.matmul(frames_windowed_nstft, self.rfft_real)
        imag_part_nstft = torch.matmul(frames_windowed_nstft, self.rfft_imag)
        x_nstft = torch.cat([real_part_nstft.unsqueeze(-1), imag_part_nstft.unsqueeze(-1)], dim=-1)
        x_nstft = x_nstft.unsqueeze(-2)
        # x = torch.view_as_real(x)

        L = x.shape[-1]
        x = x.reshape(-1, L)
        # x = torch.stft(x, **self.stft_config, pad_mode='constant', window=self.stft_window, return_complex=True)
        x = torch.view_as_real(x)

        x = x.permute(0, 3, 1, 2).reshape(x.shape[0] // self.audio_channels, x.shape[3] * self.audio_channels,
                                          x.shape[1], x.shape[2])

        B, C, Fr, T = x.shape
        # # mean = x.mean(dim=(1, 2, 3), keepdim=True)
        # # std = x.std(dim=(1, 2, 3), keepdim=True)
        # # x = (x - mean) / (1e-5 + std)
        #
        save_skip = deque()
        save_lengths = deque()
        save_original_lengths = deque()
        # encoder
        for icnt, sd_layer in enumerate(self.encoder):
            if icnt == 0:
                x, skip, lengths, original_lengths, cache_band0_out = sd_layer(x, cache_band0_in)
            elif icnt == 1:
                x, skip, lengths, original_lengths, cache_band1_out = sd_layer(x, cache_band1_in)
            elif icnt == 2:
                x, skip, lengths, original_lengths, cache_band2_out = sd_layer(x, cache_band2_in)
            save_skip.append(skip)
            save_lengths.append(lengths)
            save_original_lengths.append(original_lengths)

        # # separation
        x = self.separation_net(x)
        #
        # decoder
        icnt = 0
        for fusion_layer, su_layer in self.decoder:
            if icnt == 0:
                x, cache_fus0_out = fusion_layer(x, cache_fus0_in, save_skip.pop())
            elif icnt == 1:
                x, cache_fus1_out = fusion_layer(x, cache_fus1_in, save_skip.pop())
            elif icnt == 2:
                x, cache_fus2_out = fusion_layer(x, cache_fus2_in, save_skip.pop())
            x = su_layer(x, save_lengths.pop(), save_original_lengths.pop())
            icnt = icnt + 1

        # # output
        n = self.dims[0]
        x = x.view(B, n, -1, Fr, T)
        # x = x * std[:, None] + mean[:, None]
        x = x.reshape(-1, 2, Fr, T).permute(0, 2, 3, 1)
        x = x.squeeze(2)
        time_frame = torch.matmul(x[..., 0], self.irfft_real) - torch.matmul(x[..., 1], self.irfft_imag)
        window_frame = time_frame * self.stft_window
        overlap_buffer_out = overlap_buffer_in + window_frame
        window_sum_out = window_sum_in + self.stft_window_power
        window_norm = window_sum_out[..., :self.hop_length]
        # window_norm = torch.where(window_norm > 1e-8, window_norm, torch.ones_like(window_norm))
        output = overlap_buffer_out[..., :self.hop_length].clone()
        x = output / (window_norm + 1e-10)
        # x = output / window_norm
        overlap_buffer_out = torch.cat([overlap_buffer_out[..., self.hop_length:],
                                        torch.zeros([overlap_buffer_in.shape[0], self.hop_length])], dim=-1)
        window_sum_out = torch.cat([window_sum_out[..., self.hop_length:],
                                    torch.zeros([window_sum_in.shape[0], self.hop_length])], dim=-1)
        x = x.reshape(B, len(self.sources), self.audio_channels, -1)
        # cache_stft_out = cache_stft_in
        # overlap_buffer_out, window_sum_out = overlap_buffer_in, window_sum_in
        # x = x[:, :, :, :-padding]
        return (x, cache_stft_out, cache_band0_out, cache_band1_out, cache_band2_out,
                cache_fus0_out, cache_fus1_out, cache_fus2_out, overlap_buffer_out, window_sum_out)



class SCNetStreamNoSTFT(nn.Module):
    """
    The implementation of SCNet: Sparse Compression Network for Music Source Separation. Paper: https://arxiv.org/abs/2401.13276.pdf

    Args:
    - sources (List[str]): List of sources to be separated.
    - audio_channels (int): Number of audio channels.
    - nfft (int): Number of FFTs to determine the frequency dimension of the input.
    - hop_size (int): Hop size for the STFT.
    - win_size (int): Window size for STFT.
    - normalized (bool): Whether to normalize the STFT.
    - dims (List[int]): List of channel dimensions for each block.
    - band_SR (List[float]): The proportion of each frequency band.
    - band_stride (List[int]): The down-sampling ratio of each frequency band.
    - band_kernel (List[int]): The kernel sizes for down-sampling convolution in each frequency band
    - conv_depths (List[int]): List specifying the number of convolution modules in each SD block.
    - compress (int): Compression factor for convolution module.
    - conv_kernel (int): Kernel size for convolution layer in convolution module.
    - num_dplayer (int): Number of dual-path layers.
    - expand (int): Expansion factor in the dual-path RNN, default is 1.

    """

    def __init__(self,
                 sources=['drums', 'bass', 'other', 'vocals'],
                 audio_channels=2,
                 # Main structure
                 dims=[4, 64, 128, 64],  # dims = [4, 64, 128, 256] in SCNet-large
                 # STFT
                 nfft=4096,
                 hop_size=1024,
                 win_size=4096,
                 normalized=True,
                 # SD/SU layer
                 band_SR=[0.175, 0.392, 0.433],
                 band_stride=[1, 4, 16],
                 band_kernel=[3, 4, 16],
                 # Convolution Module
                 conv_depths=[3, 2, 1],
                 compress=4,
                 conv_kernel=3,
                 # Dual-path RNN
                 num_dplayer=2,
                 expand=1,
                 ):
        super().__init__()
        self.sources = sources
        self.audio_channels = audio_channels
        self.dims = dims
        band_keys = ['low', 'mid', 'high']
        self.band_configs = {band_keys[i]: {'SR': band_SR[i], 'stride': band_stride[i], 'kernel': band_kernel[i]} for i
                             in range(len(band_keys))}
        self.hop_length = hop_size
        self.conv_config = {
            'compress': compress,
            'kernel': conv_kernel,
        }
        # stft parameter
        self.register_buffer("stft_window", torch.hann_window(nfft), persistent=False)
        self.register_buffer("stft_window_power", self.stft_window ** 2, persistent=False)
        rfft_real, rfft_imag = self.create_rfft_matrix(nfft)
        scale = 1.0 / math.sqrt(nfft)
        self.scale = scale
        self.register_buffer("rfft_real", rfft_real.T * scale, persistent=False)
        self.register_buffer("rfft_imag", rfft_imag.T * scale, persistent=False)

        self.stft_config = {
            'n_fft': nfft,
            'hop_length': hop_size,
            'win_length': win_size,
            'center': True,
            'normalized': True
        }

        irfft_real, irfft_imag = self.create_irfft_matrix(nfft)
        self.register_buffer("irfft_real", irfft_real.T * scale, persistent=False)
        self.register_buffer("irfft_imag", irfft_imag.T * scale, persistent=False)

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for index in range(len(dims) - 1):
            enc = SDblockStream(
                channels_in=dims[index],
                channels_out=dims[index + 1],
                band_configs=self.band_configs,
                conv_config=self.conv_config,
                depths=conv_depths
            )
            self.encoder.append(enc)

            dec = nn.Sequential(
                FusionLayerStream(channels=dims[index + 1]),
                SUlayer(
                    channels_in=dims[index + 1],
                    channels_out=dims[index] if index != 0 else dims[index] * len(sources),
                    band_configs=self.band_configs,
                )
            )
            self.decoder.insert(0, dec)

        self.separation_net = SeparationNetStream(
            channels=dims[-1],
            expand=expand,
            num_layers=num_dplayer,
        )

    def create_rfft_matrix(self, n_fft, dtype=torch.float32):
        """创建 RFFT 矩阵（实部和虚部分开）"""
        freq_size = int(n_fft / 2) + 1

        k = torch.arange(freq_size, dtype=torch.float64).unsqueeze(1)
        n = torch.arange(n_fft, dtype=torch.float64).unsqueeze(0)

        angle = -2.0 * math.pi * k * n / n_fft

        rfft_real = torch.cos(angle).to(dtype)
        rfft_imag = torch.sin(angle).to(dtype)

        return rfft_real, rfft_imag

    def create_irfft_matrix(self, n_fft, dtype=torch.float32):
        """创建 RFFT 矩阵（实部和虚部分开）"""
        freq_size = n_fft // 2 + 1

        k = torch.arange(freq_size, dtype=torch.float64).unsqueeze(0)
        n = torch.arange(n_fft, dtype=torch.float64).unsqueeze(1)

        angle = 2.0 * math.pi * k * n / n_fft

        irfft_real = torch.cos(angle).to(dtype)
        irfft_imag = torch.sin(angle).to(dtype)

        scale = torch.ones(freq_size, dtype=dtype)
        scale[1:-1] = 2.0
        irfft_real = irfft_real * scale.unsqueeze(0)
        irfft_imag = irfft_imag * scale.unsqueeze(0)

        return irfft_real, irfft_imag

    def forward_1st_frame(self, x, cache_band0_in, cache_band1_in, cache_band2_in,
                cache_h1, cache_c1, cache_h2, cache_c2, cache_conv,
                cache_fus0_in, cache_fus1_in, cache_fus2_in, save_skip_in):
        B, C, Fr, T = x.shape
        save_skip = deque()
        save_lengths = deque()
        save_original_lengths = deque()
        # encoder
        for icnt, sd_layer in enumerate(self.encoder):
            if icnt == 0:
                x, skip, lengths, original_lengths, cache_band0_out = sd_layer(x, cache_band0_in)
            elif icnt == 1:
                x, skip, lengths, original_lengths, cache_band1_out = sd_layer(x, cache_band1_in)
            elif icnt == 2:
                x, skip, lengths, original_lengths, cache_band2_out = sd_layer(x, cache_band2_in)
            save_skip.append(skip)
            save_lengths.append(lengths)
            save_original_lengths.append(original_lengths)

        # # separation
        (x, new_cache_h1, new_cache_c1, new_cache_h2, new_cache_c2,
         new_cache_conv) = self.separation_net.forward_1st_frame(x, cache_h1, cache_c1, cache_h2, cache_c2, cache_conv)

        return (None, cache_band0_out, cache_band1_out, cache_band2_out,
                new_cache_h1, new_cache_c1, new_cache_h2, new_cache_c2, new_cache_conv,
                cache_fus0_in, cache_fus1_in, cache_fus2_in, save_skip)

    def forward(self, x, cache_band0_in, cache_band1_in, cache_band2_in,
                cache_h1, cache_c1, cache_h2, cache_c2, cache_conv,
                cache_fus0_in, cache_fus1_in, cache_fus2_in, save_skip_in):
        # STFT
        # x = torch.cat([cache_stft_in, x], dim=-1)
        # cache_stft_out = x[:, self.hop_length:]
        # frames_windowed = x * self.stft_window
        # real_part = torch.matmul(frames_windowed, self.rfft_real)
        # imag_part = torch.matmul(frames_windowed, self.rfft_imag)
        #
        # x = torch.cat([real_part.unsqueeze(-1), imag_part.unsqueeze(-1)], dim=-1)
        # x = x.unsqueeze(-2)
        # # x = torch.view_as_real(x)
        # x = x.permute(0, 3, 1, 2).reshape(x.shape[0] // self.audio_channels, x.shape[3] * self.audio_channels,
        #                                   x.shape[1], x.shape[2])

        B, C, Fr, T = x.shape
        save_skip = deque()
        save_lengths = deque()
        save_original_lengths = deque()
        # encoder
        for icnt, sd_layer in enumerate(self.encoder):
            if icnt == 0:
                x, skip, lengths, original_lengths, cache_band0_out = sd_layer(x, cache_band0_in)
            elif icnt == 1:
                x, skip, lengths, original_lengths, cache_band1_out = sd_layer(x, cache_band1_in)
            elif icnt == 2:
                x, skip, lengths, original_lengths, cache_band2_out = sd_layer(x, cache_band2_in)
            save_skip.append(skip)
            save_lengths.append(lengths)
            save_original_lengths.append(original_lengths)

        # # separation
        x, new_cache_h1, new_cache_c1, new_cache_h2, new_cache_c2, new_cache_conv = self.separation_net(x, cache_h1, cache_c1, cache_h2, cache_c2, cache_conv)
        # decoder
        icnt = 0
        for fusion_layer, su_layer in self.decoder:
            if icnt == 0:
                x, cache_fus0_out = fusion_layer(x, cache_fus0_in, save_skip_in.pop())
            elif icnt == 1:
                x, cache_fus1_out = fusion_layer(x, cache_fus1_in, save_skip_in.pop())
            elif icnt == 2:
                x, cache_fus2_out = fusion_layer(x, cache_fus2_in, save_skip_in.pop())
            x = su_layer(x, save_lengths.pop(), save_original_lengths.pop())
            icnt = icnt + 1

        # # output
        n = self.dims[0]
        x = x.view(B, n, -1, Fr, T)
        x = x.reshape(-1, 2, Fr, T).permute(0, 2, 3, 1)
        # x = x.squeeze(2)
        # time_frame = torch.matmul(x[..., 0], self.irfft_real) - torch.matmul(x[..., 1], self.irfft_imag)
        # window_frame = time_frame * self.stft_window
        # overlap_buffer_out = overlap_buffer_in + window_frame
        # window_sum_out = window_sum_in + self.stft_window_power
        # window_norm = window_sum_out[..., :self.hop_length]
        # # window_norm = torch.where(window_norm > 1e-8, window_norm, torch.ones_like(window_norm))
        # output = overlap_buffer_out[..., :self.hop_length].clone()
        # x = output / (window_norm + 1e-10)
        # overlap_buffer_out = torch.cat([overlap_buffer_out[..., self.hop_length:],
        #                                 torch.zeros([overlap_buffer_in.shape[0], self.hop_length])], dim=-1)
        # window_sum_out = torch.cat([window_sum_out[..., self.hop_length:],
        #                             torch.zeros([window_sum_in.shape[0], self.hop_length])], dim=-1)
        # x = x.reshape(B, len(self.sources), self.audio_channels, -1)
        # cache_stft_out = cache_stft_in
        # overlap_buffer_out, window_sum_out = overlap_buffer_in, window_sum_in
        # x = x[:, :, :, :-padding]
        return (x,  cache_band0_out, cache_band1_out, cache_band2_out,
                new_cache_h1, new_cache_c1, new_cache_h2, new_cache_c2, new_cache_conv,
                cache_fus0_out, cache_fus1_out, cache_fus2_out, save_skip)

    def forward_last_frame(self, x, cache_band0_in, cache_band1_in, cache_band2_in,
                cache_h1, cache_c1, cache_h2, cache_c2, cache_conv,
                cache_fus0_in, cache_fus1_in, cache_fus2_in, save_skip_in):
        B, Fr = 1, 2049
        save_lengths = deque()
        save_lengths.append([359, 201, 56])
        save_lengths.append([108, 61, 17])
        save_lengths.append([33, 19, 5])
        save_original_lengths = deque()
        save_original_lengths.append([359, 803, 887])
        save_original_lengths.append([108, 242, 266])
        save_original_lengths.append([33, 73, 80])

        (x, new_cache_h1, new_cache_c1,
         new_cache_h2, new_cache_c2, new_cache_conv) = self.separation_net.forward_last_frame(x, cache_h1, cache_c1, cache_h2, cache_c2, cache_conv)
        # decoder
        icnt = 0
        for fusion_layer, su_layer in self.decoder:
            if icnt == 0:
                x, cache_fus0_out = fusion_layer(x, cache_fus0_in, save_skip_in.pop())
            elif icnt == 1:
                x, cache_fus1_out = fusion_layer(x, cache_fus1_in, save_skip_in.pop())
            elif icnt == 2:
                x, cache_fus2_out = fusion_layer(x, cache_fus2_in, save_skip_in.pop())
            x = su_layer(x, save_lengths.pop(), save_original_lengths.pop())
            icnt = icnt + 1

        # # output
        T = x.shape[-1]
        n = self.dims[0]
        x = x.view(B, n, -1, Fr, T)
        x = x.reshape(-1, 2, Fr, T).permute(0, 2, 3, 1)

        return (x, cache_band0_in, cache_band1_in, cache_band2_in,
                new_cache_h1, new_cache_c1, new_cache_h2, new_cache_c2, new_cache_conv,
                cache_fus0_out, cache_fus1_out, cache_fus2_out, None)



def convert_state_dict(src_net, dst_net):
    state_dict = src_net.state_dict()
    new_state_dict = dst_net.state_dict()
    for k, v in state_dict.items():
        if k in state_dict.keys():
            new_state_dict[k] = v
        else:
            print(k, v)
    dst_net.load_state_dict(new_state_dict)



def test_scnet_nostft():
    B, C, Freq, T = 1, 4, 2049, 18
    torch.manual_seed(42)
    x = torch.randn([B, C, Freq, T], dtype=torch.float32)

    scnet = SCNetNoStft(sources=['accompaniment', 'vocals'])
    scnet.eval()
    with torch.no_grad():
        output = scnet(x)

    scnet_stream = SCNetStreamNoSTFT(sources=['accompaniment', 'vocals'])
    convert_state_dict(scnet, scnet_stream)

    scnet_stream.eval()
    cache_band0_state = torch.zeros([1, 64, 616, 2], dtype=torch.float32)
    cache_band1_state = torch.zeros([1, 128, 186, 2], dtype=torch.float32)
    cache_band2_state = torch.zeros([1, 64, 57, 2], dtype=torch.float32)
    cache_fus0_state = torch.zeros([1, 128, 57, 2], dtype=torch.float32)
    cache_fus1_state = torch.zeros([1, 256, 186, 2], dtype=torch.float32)
    cache_fus2_state = torch.zeros([1, 128, 616, 2], dtype=torch.float32)
    cache_h1, cache_c1 = torch.zeros([1, 57, 64], dtype=torch.float32), torch.zeros([1, 57, 64], dtype=torch.float32)
    cache_h2, cache_c2 = torch.zeros([1, 57, 128], dtype=torch.float32), torch.zeros([1, 57, 128], dtype=torch.float32)
    cache_conv = torch.zeros([57, 128, 6], dtype=torch.float32)
    save_skip = None
    state = [cache_band0_state, cache_band1_state, cache_band2_state,
             cache_h1, cache_c1, cache_h2, cache_c2, cache_conv,
             cache_fus0_state, cache_fus1_state, cache_fus2_state, save_skip]

    stream_output = []
    with torch.no_grad():
        input_chunk = x[..., 0:3]
        _, *state = scnet_stream.forward_1st_frame(input_chunk, *state)
        for i in range(int(T/3)-1):
            input_chunk = x[..., (i+1)*3:(i+2)*3]
            chunk_output, *state = scnet_stream(input_chunk, *state)
            stream_output.append(chunk_output)
        chunk_output, *state = scnet_stream.forward_last_frame(None, *state)
        stream_output.append(chunk_output)
    stream_output = torch.cat(stream_output, dim=-2)

    print(f"Correct output shape: {output.shape}, Reference shape: {stream_output.shape}")
    # 比较修正后的输出
    min_len = min(stream_output.shape[-2], output.shape[-2])
    diff_correct = (stream_output[:, :, :min_len, :] - output[:, :, :min_len, :]).abs()
    max_diff_0 = diff_correct[:, :, :, :].max().item()
    mean_diff_0 = diff_correct[:, :, :, :].mean().item()
    print(f"Correct vs Ref: max diff = {max_diff_0:.6e}, mean diff = {mean_diff_0:.6e}")

    # ------------------------------------------------------------------
    # Part2: 在 nostft 流式基础上加 STFT / ISTFT，与完整 SCNet 波形对齐
    # STFT: center=True, pad_mode='constant'  <=>  左右各补 n_fft//2 零 + center=False
    # ------------------------------------------------------------------
    hop_size, n_fft, T = 1024, 4096, 16000
    torch.manual_seed(42)
    input = torch.randn([2, T], dtype=torch.float32)

    # 使 center=True 的帧数 T_frames = 1 + L/hop 能被 3 整除（lookahead=3）
    padding = (hop_size - input.shape[-1] % hop_size) % hop_size
    l_tmp = input.shape[-1] + padding
    t_frames = 1 + l_tmp // hop_size
    padding += ((3 - t_frames % 3) % 3) * hop_size
    pad_input = F.pad(input, (0, padding))
    L = pad_input.shape[-1]

    # 离线完整 SCNet（内部 STFT + 模型 + ISTFT）
    scnet = SCNet(sources=['accompaniment', 'vocals'])
    scnet.eval()
    with torch.no_grad():
        wave_ref = scnet(pad_input)

    # 频谱域对照（可选）：SCNetNoISTFT
    scnet_spec = SCNetNoISTFT(sources=['accompaniment', 'vocals'])
    convert_state_dict(scnet, scnet_spec)
    scnet_spec.eval()
    with torch.no_grad():
        spec_ref = scnet_spec(pad_input)

    scnet_stream = SCNetStreamNoSTFT(sources=['accompaniment', 'vocals'])
    convert_state_dict(scnet, scnet_stream)
    scnet_stream.eval()

    cache_band0_state = torch.zeros([1, 64, 616, 2], dtype=torch.float32)
    cache_band1_state = torch.zeros([1, 128, 186, 2], dtype=torch.float32)
    cache_band2_state = torch.zeros([1, 64, 57, 2], dtype=torch.float32)
    cache_fus0_state = torch.zeros([1, 128, 57, 2], dtype=torch.float32)
    cache_fus1_state = torch.zeros([1, 256, 186, 2], dtype=torch.float32)
    cache_fus2_state = torch.zeros([1, 128, 616, 2], dtype=torch.float32)
    cache_h1, cache_c1 = torch.zeros([1, 57, 64], dtype=torch.float32), torch.zeros([1, 57, 64], dtype=torch.float32)
    cache_h2, cache_c2 = torch.zeros([1, 57, 128], dtype=torch.float32), torch.zeros([1, 57, 128], dtype=torch.float32)
    cache_conv = torch.zeros([57, 128, 6], dtype=torch.float32)
    save_skip = None
    state = [cache_band0_state, cache_band1_state, cache_band2_state,
             cache_h1, cache_c1, cache_h2, cache_c2, cache_conv,
             cache_fus0_state, cache_fus1_state, cache_fus2_state, save_skip]

    # 右端 constant pad（与 torch.stft center=True, pad_mode='constant' 一致）
    # 左端用 cache_stft_0 = zeros(2*hop) 模拟
    pad_input_ext = F.pad(pad_input, (0, 2 * hop_size))
    L_ext = pad_input_ext.shape[-1]
    cache_stft_0 = torch.zeros([2, 2 * hop_size], dtype=torch.float32)

    stream_output = []
    wave_chunks = []
    # 流式 OLA ISTFT：只用小型 overlap 状态 + torch.fft.irfft 批量反变换
    # 不再使用 irfft_real/irfft_imag 大矩阵，也不再逐帧 Python 循环 matmul
    n_ola = len(scnet.sources) * scnet.audio_channels  # 4
    overlap_buffer = torch.zeros([n_ola, n_fft], dtype=torch.float32)
    window_sum = torch.zeros([n_fft], dtype=torch.float32)
    win = scnet_stream.stft_window
    win_pow = win * win
    fft_scale = n_fft ** 0.5  # 对齐 normalized=True 的 ISTFT 幅度

    def push_ola_istft(spec_rt):
        """
        高效流式 ISTFT。
        spec_rt: (N, F, T_new, 2) → (N, T_new * hop)
        - irfft 对 T_new 帧一次批处理
        - fold 做 overlap-add
        - 状态仅保留 (N, n_fft) / (n_fft,) 的尾部缓冲
        """
        nonlocal overlap_buffer, window_sum
        n, _, t_new, _ = spec_rt.shape
        if t_new == 0:
            return torch.zeros(n, 0, dtype=spec_rt.dtype, device=spec_rt.device)

        # (N, T, F) complex → (N, T, n_fft)
        spec_c = torch.view_as_complex(spec_rt.permute(0, 2, 1, 3).contiguous())
        frames = torch.fft.irfft(spec_c, n=n_fft, dim=-1).mul_(fft_scale)
        frames.mul_(win)  # (N, T, n_fft)

        ola_len = n_fft + (t_new - 1) * hop_size
        # fold: (N, n_fft, T) → (N, 1, ola_len)
        frames_nct = frames.transpose(1, 2).contiguous()
        ola = F.fold(
            frames_nct,
            output_size=(1, ola_len),
            kernel_size=(1, n_fft),
            stride=(1, hop_size),
        ).reshape(n, ola_len)
        ola[:, :n_fft] += overlap_buffer

        w_frames = win_pow.view(1, n_fft, 1).expand(1, n_fft, t_new)
        wola = F.fold(
            w_frames,
            output_size=(1, ola_len),
            kernel_size=(1, n_fft),
            stride=(1, hop_size),
        ).reshape(ola_len)
        wola[:n_fft] += window_sum

        out_len = t_new * hop_size
        out = ola[:, :out_len] / (wola[:out_len] + 1e-10)

        # 尾部回写为下一包的 overlap 状态（长度仍为 n_fft）
        overlap_buffer = F.pad(ola[:, out_len:], (0, hop_size))
        window_sum = F.pad(wola[out_len:], (0, hop_size))
        return out

    stft_cfg = dict(scnet_stream.stft_config)
    stft_cfg['center'] = False  # 左右 pad 已手工加上
    with torch.no_grad():
        # 首包: 左零填(2hop) + 4hop 音频 -> 3 帧，只填 lookahead（尚无模型输出）
        input_chunk = pad_input_ext[..., 0:hop_size * 4]
        stft_x = torch.stft(
            torch.cat([cache_stft_0, input_chunk], dim=-1),
            **stft_cfg,
            window=scnet_stream.stft_window,
            return_complex=True,
        )
        x = torch.view_as_real(stft_x)
        x = x.permute(0, 3, 1, 2).reshape(
            x.shape[0] // 2, x.shape[3] * 2, x.shape[1], x.shape[2]
        )
        _, *state = scnet_stream.forward_1st_frame(x, *state)
        cache_stft_0 = input_chunk[:, -3 * hop_size:]

        # 中间包: 每次 3hop -> 3 帧；模型一有输出就立刻 OLA ISTFT
        n_mid = int((L_ext - hop_size * 4) / (3 * hop_size))
        for i in range(n_mid):
            start = (i * 3 + 4) * hop_size
            end = (i * 3 + 7) * hop_size
            input_chunk = pad_input_ext[..., start:end]
            stft_x = torch.stft(
                torch.cat([cache_stft_0, input_chunk], dim=-1),
                **stft_cfg,
                window=scnet_stream.stft_window,
                return_complex=True,
            )
            x = torch.view_as_real(stft_x)
            x = x.permute(0, 3, 1, 2).reshape(
                x.shape[0] // 2, x.shape[3] * 2, x.shape[1], x.shape[2]
            )
            cache_stft_0 = input_chunk
            chunk_output, *state = scnet_stream(x, *state)
            stream_output.append(chunk_output)
            wave_chunks.append(push_ola_istft(chunk_output))

        # flush lookahead，并立刻 ISTFT
        chunk_output, *state = scnet_stream.forward_last_frame(None, *state)
        stream_output.append(chunk_output)
        wave_chunks.append(push_ola_istft(chunk_output))

        # center=True 还需要再吐 1 个 hop，才能凑齐 length=L（见右端 pad）
        wave_chunks.append(
            overlap_buffer[:, :hop_size] / (window_sum[:hop_size] + 1e-10)
        )

    stream_output = torch.cat(stream_output, dim=-2)  # (N, F, T, 2)

    # OLA 时间轴含左 pad，去掉 n_fft//2 后取 L（对齐 torch.istft center=True, length=L）
    wave_ola = torch.cat(wave_chunks, dim=-1)
    pad = n_fft // 2
    wave_stream = wave_ola[:, pad:pad + L].reshape(wave_ref.shape)

    print(f"[STFT+nostft] spec_ref={tuple(spec_ref.shape)}, stream={tuple(stream_output.shape)}")
    min_len = min(stream_output.shape[-2], spec_ref.shape[-2])
    diff_spec = (stream_output[:, :, :min_len, :] - spec_ref[:, :, :min_len, :]).abs()
    print(f"[STFT+nostft] max diff = {diff_spec.max().item():.6e}, mean diff = {diff_spec.mean().item():.6e}")

    print(f"[STFT+nostft+OLA_ISTFT] wave_ref={tuple(wave_ref.shape)}, stream={tuple(wave_stream.shape)}")
    diff_wave = (wave_stream - wave_ref).abs()
    print(f"[STFT+nostft+OLA_ISTFT] max diff = {diff_wave.max().item():.6e}, mean diff = {diff_wave.mean().item():.6e}")


if __name__ == '__main__':
    test_scnet_nostft()
