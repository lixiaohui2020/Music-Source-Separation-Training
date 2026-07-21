"""Streaming Dual-Path separation network (lookahead-aware)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.modules.rnn import LSTM


class ConvConversion(nn.Module):
    def __init__(self, channels, inverse):
        super().__init__()
        self.inverse = inverse
        self.channels = channels
        self.conv1 = nn.Conv2d(channels // 2, channels // 2, (3, 1), stride=(1, 1), padding=(1, 0))
        self.conv2 = nn.Conv2d(channels, channels // 2, (3, 1), stride=(1, 1), padding=(1, 0))
        self.bn = nn.BatchNorm2d(channels // 2)

    def forward(self, x):
        if self.inverse:
            x = self.conv2(x.float())
            return self.bn(x)
        original_x = x.float()
        x = self.bn(self.conv1(original_x))
        return torch.cat([original_x, x], dim=1)


class DualPathRNN(nn.Module):
    def __init__(self, d_model, expand, bidirectional=True, lookahead=0):
        super().__init__()
        self.d_model = d_model
        self.hidden_size = d_model * expand
        self.bidirectional = bidirectional
        self.lstm_layers = nn.ModuleList(
            [
                LSTM(d_model, self.hidden_size, num_layers=1, bidirectional=True, batch_first=True),
                LSTM(d_model, self.hidden_size, num_layers=1, bidirectional=False, batch_first=True),
            ]
        )
        if lookahead > 0:
            self.time_conv = nn.Conv1d(
                self.hidden_size, self.hidden_size, kernel_size=2 * lookahead + 1, padding=lookahead
            )
        else:
            self.time_conv = nn.Identity()
        self.linear_layers = nn.ModuleList(
            [nn.Linear(self.hidden_size * 2, d_model), nn.Linear(self.hidden_size, d_model)]
        )
        self.norm_layers = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])

    def forward(self, x):
        b, c, f, t = x.shape
        original_x = x
        x = x.transpose(1, 3).contiguous().view(b * t, f, c)
        x = self.norm_layers[0](x)
        x, _ = self.lstm_layers[0](x)
        x = self.linear_layers[0](x)
        x = x.view(b, t, f, c).transpose(1, 3)
        x = x + original_x

        original_x = x
        x = x.transpose(1, 2).contiguous().view(b * f, c, t)
        x = self.time_conv(x)
        x = x.transpose(1, 2)
        x = self.norm_layers[1](x)
        x, _ = self.lstm_layers[1](x)
        x = self.linear_layers[1](x)
        x = x.transpose(1, 2).contiguous().view(b, f, c, t).transpose(1, 2)
        return x + original_x


class DualPathRNNStream(nn.Module):
    def __init__(self, d_model, expand, bidirectional=True, lookahead=0):
        super().__init__()
        self.d_model = d_model
        self.hidden_size = d_model * expand
        self.bidirectional = bidirectional
        self.lookahead = lookahead
        self.lstm_layers = nn.ModuleList(
            [
                LSTM(d_model, self.hidden_size, num_layers=1, bidirectional=True, batch_first=True),
                LSTM(d_model, self.hidden_size, num_layers=1, bidirectional=False, batch_first=True),
            ]
        )
        if lookahead > 0:
            self.time_conv = nn.Conv1d(
                self.hidden_size, self.hidden_size, kernel_size=2 * lookahead + 1, padding=0
            )
        else:
            self.time_conv = nn.Identity()
        self.linear_layers = nn.ModuleList(
            [nn.Linear(self.hidden_size * 2, d_model), nn.Linear(self.hidden_size, d_model)]
        )
        self.norm_layers = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])

    def _freq_path(self, x):
        b, c, f, t = x.shape
        original_x = x
        x = x.transpose(1, 3).contiguous().view(b * t, f, c)
        x = self.norm_layers[0](x)
        x, _ = self.lstm_layers[0](x)
        x = self.linear_layers[0](x)
        x = x.view(b, t, f, c).transpose(1, 3)
        return x + original_x

    def forward_1st_frame(self, x, h, c, time_conv_cache):
        b, c_dim, f, t = x.shape
        x = self._freq_path(x)
        original_x = x
        x = x.transpose(1, 2).contiguous().view(b * f, c_dim, t)
        if self.lookahead > 0:
            x = torch.cat((time_conv_cache, x), dim=2)
            return None, h, c, x[..., self.lookahead :]
        x = self.time_conv(x).transpose(1, 2)
        x = self.norm_layers[1](x)
        x, (new_h, new_c) = self.lstm_layers[1](x, (h, c))
        x = self.linear_layers[1](x)
        x = x.transpose(1, 2).contiguous().view(b, f, c_dim, t).transpose(1, 2)
        return x + original_x, new_h, new_c, None

    def forward(self, x, h, c, time_conv_cache):
        b, c_dim, f, t = x.shape
        x = self._freq_path(x)
        original_x = x
        x = x.transpose(1, 2).contiguous().view(b * f, c_dim, t)
        if self.lookahead > 0:
            x = torch.cat((time_conv_cache, x), dim=2)
            original_x = x[..., self.lookahead : self.lookahead * 2].view(b, f, c_dim, t).transpose(1, 2)
            new_cache = x[..., self.lookahead :]
            x = self.time_conv(x).transpose(1, 2)
            x = self.norm_layers[1](x)
            x, (new_h, new_c) = self.lstm_layers[1](x, (h, c))
            x = self.linear_layers[1](x)
            x = x.transpose(1, 2).contiguous().view(b, f, c_dim, t).transpose(1, 2)
            return x + original_x, new_h, new_c, new_cache
        x = self.time_conv(x).transpose(1, 2)
        x = self.norm_layers[1](x)
        x, (new_h, new_c) = self.lstm_layers[1](x, (h, c))
        x = self.linear_layers[1](x)
        x = x.transpose(1, 2).contiguous().view(b, f, c_dim, t).transpose(1, 2)
        return x + original_x, new_h, new_c, None

    def forward_last_frame(self, x, h, c, time_conv_cache):
        b, c_dim, f, t = 1, time_conv_cache.shape[1], time_conv_cache.shape[0], 3
        pad_x = torch.zeros(b * f, c_dim, 3, device=time_conv_cache.device, dtype=time_conv_cache.dtype)
        x = torch.cat((time_conv_cache, pad_x), dim=2)
        original_x = x[..., self.lookahead : self.lookahead * 2].view(b, f, c_dim, t).transpose(1, 2)
        new_cache = x[..., self.lookahead :]
        x = self.time_conv(x).transpose(1, 2)
        x = self.norm_layers[1](x)
        x, (new_h, new_c) = self.lstm_layers[1](x, (h, c))
        x = self.linear_layers[1](x)
        x = x.transpose(1, 2).contiguous().view(b, f, c_dim, t).transpose(1, 2)
        return x + original_x, new_h, new_c, new_cache


class SeparationNet(nn.Module):
    def __init__(self, channels, expand=1, num_layers=2, lookahead=3):
        super().__init__()
        self.num_layers = num_layers
        self.dp_modules = nn.ModuleList(
            [
                DualPathRNN(
                    channels * (2 if i % 2 == 1 else 1),
                    expand,
                    lookahead=lookahead if i == num_layers - 1 else 0,
                )
                for i in range(num_layers)
            ]
        )
        self.feature_conversion = nn.ModuleList(
            [ConvConversion(channels * 2, inverse=(i % 2 == 1)) for i in range(num_layers)]
        )

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.dp_modules[i](x)
            x = self.feature_conversion[i](x)
        return x


class SeparationNetStream(nn.Module):
    def __init__(self, channels, expand=1, num_layers=2, lookahead=3):
        super().__init__()
        self.num_layers = num_layers
        self.dp_modules = nn.ModuleList(
            [
                DualPathRNNStream(
                    channels * (2 if i % 2 == 1 else 1),
                    expand,
                    lookahead=lookahead if i == num_layers - 1 else 0,
                )
                for i in range(num_layers)
            ]
        )
        self.feature_conversion = nn.ModuleList(
            [ConvConversion(channels * 2, inverse=(i % 2 == 1)) for i in range(num_layers)]
        )

    def forward_1st_frame(self, x, cache_h1, cache_c1, cache_h2, cache_c2, time_conv_cache):
        x, new_h1, new_c1, _ = self.dp_modules[0](x, cache_h1, cache_c1, None)
        x = self.feature_conversion[0](x)
        x, new_h2, new_c2, new_cache = self.dp_modules[1].forward_1st_frame(
            x, cache_h2, cache_c2, time_conv_cache
        )
        return x, new_h1, new_c1, new_h2, new_c2, new_cache

    def forward(self, x, cache_h1, cache_c1, cache_h2, cache_c2, time_conv_cache):
        x, new_h1, new_c1, _ = self.dp_modules[0](x, cache_h1, cache_c1, None)
        x = self.feature_conversion[0](x)
        x, new_h2, new_c2, new_cache = self.dp_modules[1](x, cache_h2, cache_c2, time_conv_cache)
        x = self.feature_conversion[1](x)
        return x, new_h1, new_c1, new_h2, new_c2, new_cache

    def forward_last_frame(self, x, cache_h1, cache_c1, cache_h2, cache_c2, time_conv_cache):
        x, new_h2, new_c2, new_cache = self.dp_modules[1].forward_last_frame(
            x, cache_h2, cache_c2, time_conv_cache
        )
        x = self.feature_conversion[1](x)
        return x, cache_h1, cache_c1, new_h2, new_c2, new_cache
