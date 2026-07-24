import torch
import torch.nn as nn
from torch.nn.modules.rnn import LSTM
from torch.nn.modules.rnn import GRU


class FeatureConversion(nn.Module):
    """
    Integrates into the adjacent Dual-Path layer.

    Args:
        channels (int): Number of input channels.
        inverse (bool): If True, uses ifft; otherwise, uses rfft.
    """

    def __init__(self, channels, inverse):
        super().__init__()
        self.inverse = inverse
        self.channels = channels

    def forward(self, x):
        # B, C, F, T = x.shape
        if self.inverse:
            x = x.float()
            x_r = x[:, :self.channels // 2, :, :]
            x_i = x[:, self.channels // 2:, :, :]
            x = torch.complex(x_r, x_i)
            x = torch.fft.irfft(x, dim=3, norm="ortho")
        else:
            x = x.float()
            x = torch.fft.rfft(x, dim=3, norm="ortho")
            x_real = x.real
            x_imag = x.imag
            x = torch.cat([x_real, x_imag], dim=1)
        return x


class ConvConversion(nn.Module):
    """
    Integrates into the adjacent Dual-Path layer.

    Args:
        channels (int): Number of input channels.
        inverse (bool): If True, uses ifft; otherwise, uses rfft.
    """

    def __init__(self, channels, inverse):
        super().__init__()
        self.inverse = inverse
        self.channels = channels
        self.conv1 = nn.Conv2d(channels // 2, channels // 2, (3, 1), stride=(1, 1), padding=(1, 0))
        self.conv2 = nn.Conv2d(channels, channels // 2, (3, 1), stride=(1, 1), padding=(1, 0))
        self.bn = nn.BatchNorm2d(channels // 2)

    def forward(self, x):
        # B, C, F, T = x.shape
        if self.inverse:
            x = x.float()
            x = self.conv2(x)
            x = self.bn(x)
        else:
            original_x = x.float()
            x = x.float()
            x = self.conv1(x)
            x = self.bn(x)
            x = torch.cat([original_x, x], dim=1)
        return x


# class DualPathRNNStream(nn.Module):
#     """
#     Dual-Path RNN in Separation Network.
#
#     Args:
#         d_model (int): The number of expected features in the input (input_size).
#         expand (int): Expansion factor used to calculate the hidden_size of LSTM.
#         bidirectional (bool): If True, becomes a bidirectional LSTM.
#     """
#
#     def __init__(self, d_model, expand, bidirectional=True, lookahead=0):
#         super(DualPathRNNStream, self).__init__()
#
#         self.d_model = d_model
#         self.hidden_size = d_model * expand
#         self.bidirectional = bidirectional
#         # Initialize LSTM layers and normalization layers
#
#         self.lstm_layers = nn.ModuleList([self._init_lstm_layer(self.d_model, self.hidden_size, self.bidirectional),
#                                           self._init_lstm_layer(self.d_model, self.hidden_size, False)])
#         # self.lstm_layers = nn.ModuleList([self._init_lstm_layer(self.d_model, self.hidden_size) for _ in range(2)])
#         # self.linear_layers = nn.ModuleList([nn.Linear(self.hidden_size*2, self.d_model) for _ in range(2)])
#         if lookahead > 0:
#             # self.pad = nn.ConstantPad1d((lookahead,lookahead), 0)
#             self.time_conv = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=2 * lookahead + 1,
#                                        padding=0)  ## padding set 0 for stream
#         else:
#             # self.pad = nn.Identity()
#             self.time_conv = nn.Identity()
#
#         self.linear_layers = nn.ModuleList([nn.Linear(self.hidden_size * 2, self.d_model),
#                                             nn.Linear(self.hidden_size, self.d_model)])
#
#         self.norm_layers = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
#
#     def _init_lstm_layer(self, d_model, hidden_size, bidirectional):
#         return LSTM(d_model, hidden_size, num_layers=1, bidirectional=bidirectional, batch_first=True)
#
#     def forward(self, x, time_conv_cache, h, c, look_head):
#         B, C, F, T = x.shape
#
#         # Process dual-path rnn
#         original_x = x
#         # Frequency-path
#         x = x.transpose(1, 3).contiguous().view(B * T, F, C)
#         x = self.norm_layers[0](x)
#         x, _ = self.lstm_layers[0](x)
#         x = self.linear_layers[0](x)
#         x = x.view(B, T, F, C).transpose(1, 3)
#         x = x + original_x
#
#         original_x = x
#         # # Time-path
#         x = x.transpose(1, 2).contiguous().view(B * F, C, T)
#         if look_head < 3:
#             look_head = look_head + 1
#             x = torch.cat((time_conv_cache, x), dim=2)
#             return None, x, h, c, look_head
#         else:
#             look_head = 3
#             x = torch.cat((time_conv_cache, x), dim=2)
#             original_x = x[..., look_head:look_head+1].view(B, F, C, T).transpose(1, 2)
#             new_time_conv_cache = x[...,1:]
#             x = self.time_conv(x)
#             x = x.transpose(1, 2)
#             x = self.norm_layers[1](x)
#             x, (new_h, new_c) = self.lstm_layers[1](x, (h, c))
#             x = self.linear_layers[1](x)
#             x = x.transpose(1, 2).contiguous().view(B, F, C, T).transpose(1, 2)
#             x = x + original_x
#             # new_time_conv_cache = None
#             # new_h, new_c = None, None
#             return x, new_time_conv_cache, new_h, new_c, look_head

class DualPathRNNStream(nn.Module):
    """
    Dual-Path RNN in Separation Network.

    Args:
        d_model (int): The number of expected features in the input (input_size).
        expand (int): Expansion factor used to calculate the hidden_size of LSTM.
        bidirectional (bool): If True, becomes a bidirectional LSTM.
    """

    def __init__(self, d_model, expand, bidirectional=True, lookahead=0):
        super(DualPathRNNStream, self).__init__()

        self.d_model = d_model
        self.hidden_size = d_model * expand
        self.bidirectional = bidirectional
        # Initialize LSTM layers and normalization layers

        self.lstm_layers = nn.ModuleList([self._init_lstm_layer(self.d_model, self.hidden_size, self.bidirectional),
                                          self._init_lstm_layer(self.d_model, self.hidden_size, False)])
        # self.lstm_layers = nn.ModuleList([self._init_lstm_layer(self.d_model, self.hidden_size) for _ in range(2)])
        # self.linear_layers = nn.ModuleList([nn.Linear(self.hidden_size*2, self.d_model) for _ in range(2)])
        self.lookahead = lookahead
        if lookahead > 0:
            # self.pad = nn.ConstantPad1d((lookahead,lookahead), 0)
            self.time_conv = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=2 * lookahead + 1,
                                       padding=0)  ## padding set 0 for stream
        else:
            # self.pad = nn.Identity()
            self.time_conv = nn.Identity()

        self.linear_layers = nn.ModuleList([nn.Linear(self.hidden_size * 2, self.d_model),
                                            nn.Linear(self.hidden_size, self.d_model)])

        self.norm_layers = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])

    def _init_lstm_layer(self, d_model, hidden_size, bidirectional):
        return LSTM(d_model, hidden_size, num_layers=1, bidirectional=bidirectional, batch_first=True)

    def forward_1st_frame(self, x, h, c, time_conv_cache):
        B, C, F, T = x.shape

        # Process dual-path rnn
        original_x = x
        # Frequency-path
        x = x.transpose(1, 3).contiguous().view(B * T, F, C)
        x = self.norm_layers[0](x)
        x, _ = self.lstm_layers[0](x)
        x = self.linear_layers[0](x)
        x = x.view(B, T, F, C).transpose(1, 3)
        x = x + original_x

        original_x = x
        # # Time-path
        x = x.transpose(1, 2).contiguous().view(B * F, C, T)

        if self.lookahead > 0:
            x = torch.cat((time_conv_cache, x), dim=2)
            # original_x = x[..., lookahead:lookahead + 1].view(B, F, C, T).transpose(1, 2)
            new_time_conv_cache = x[..., self.lookahead:]
            return None, h, c, new_time_conv_cache
        else:
            new_time_conv_cache = None
            x = self.time_conv(x)
            x = x.transpose(1, 2)
            x = self.norm_layers[1](x)
            x, (new_h, new_c) = self.lstm_layers[1](x, (h, c))
            x = self.linear_layers[1](x)
            x = x.transpose(1, 2).contiguous().view(B, F, C, T).transpose(1, 2)
            x = x + original_x
        return x, new_h, new_c, new_time_conv_cache

    def forward(self, x, h, c, time_conv_cache):
        B, C, F, T = x.shape

        # Process dual-path rnn
        original_x = x
        # Frequency-path
        x = x.transpose(1, 3).contiguous().view(B * T, F, C)
        x = self.norm_layers[0](x)
        x, _ = self.lstm_layers[0](x)
        x = self.linear_layers[0](x)
        x = x.view(B, T, F, C).transpose(1, 3)
        x = x + original_x

        original_x = x
        # # Time-path
        x = x.transpose(1, 2).contiguous().view(B * F, C, T)

        if self.lookahead > 0:
            x = torch.cat((time_conv_cache, x), dim=2)
            original_x = x[..., self.lookahead:self.lookahead*2].view(B, F, C, T).transpose(1, 2)
            new_time_conv_cache = x[...,self.lookahead:]
            x = self.time_conv(x)
            x = x.transpose(1, 2)
            x = self.norm_layers[1](x)
            x, (new_h, new_c) = self.lstm_layers[1](x, (h, c))
            x = self.linear_layers[1](x)
            x = x.transpose(1, 2).contiguous().view(B, F, C, T).transpose(1, 2)
            x = x + original_x[...,]
        else:
            new_time_conv_cache = None
            x = self.time_conv(x)
            x = x.transpose(1, 2)
            x = self.norm_layers[1](x)
            x, (new_h, new_c) = self.lstm_layers[1](x, (h, c))
            x = self.linear_layers[1](x)
            x = x.transpose(1, 2).contiguous().view(B, F, C, T).transpose(1, 2)
            x = x + original_x
        return x, new_h, new_c, new_time_conv_cache

    def forward_last_frame(self, x, h, c, time_conv_cache):
        # B, C, F, T = x.shape

        # # Process dual-path rnn
        # original_x = x
        # # Frequency-path
        # x = x.transpose(1, 3).contiguous().view(B * T, F, C)
        # x = self.norm_layers[0](x)
        # x, _ = self.lstm_layers[0](x)
        # x = self.linear_layers[0](x)
        # x = x.view(B, T, F, C).transpose(1, 3)
        # x = x + original_x

        original_x = x
        # # Time-path
        # x = x.transpose(1, 2).contiguous().view(B * F, C, T)
        B, C, F, T = 1, time_conv_cache.shape[1], time_conv_cache.shape[0], 3
        pad_x = torch.zeros([B * F, C, 3])
        x = torch.cat((time_conv_cache, pad_x), dim=2)
        original_x = x[..., self.lookahead:self.lookahead * 2].view(B, F, C, T).transpose(1, 2)
        new_time_conv_cache = x[..., self.lookahead:]
        x = self.time_conv(x)
        x = x.transpose(1, 2)
        x = self.norm_layers[1](x)
        x, (new_h, new_c) = self.lstm_layers[1](x, (h, c))
        x = self.linear_layers[1](x)
        x = x.transpose(1, 2).contiguous().view(B, F, C, T).transpose(1, 2)
        x = x + original_x
        return x, new_h, new_c, new_time_conv_cache

class DualPathRNN(nn.Module):
    """
    Dual-Path RNN in Separation Network.

    Args:
        d_model (int): The number of expected features in the input (input_size).
        expand (int): Expansion factor used to calculate the hidden_size of LSTM.
        bidirectional (bool): If True, becomes a bidirectional LSTM.
    """

    def __init__(self, d_model, expand, bidirectional=True, lookahead=0):
        super(DualPathRNN, self).__init__()

        self.d_model = d_model
        self.hidden_size = d_model * expand
        self.bidirectional = bidirectional
        # Initialize LSTM layers and normalization layers

        self.lstm_layers = nn.ModuleList([self._init_lstm_layer(self.d_model, self.hidden_size, self.bidirectional),
                                          self._init_lstm_layer(self.d_model, self.hidden_size, False)])
        # self.lstm_layers = nn.ModuleList([self._init_lstm_layer(self.d_model, self.hidden_size) for _ in range(2)])
        # self.linear_layers = nn.ModuleList([nn.Linear(self.hidden_size*2, self.d_model) for _ in range(2)])
        if lookahead > 0:
            # self.pad = nn.ConstantPad1d((lookahead,lookahead), 0)
            self.time_conv = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=2 * lookahead + 1,
                                       padding=lookahead)
        else:
            # self.pad = nn.Identity()
            self.time_conv = nn.Identity()

        self.linear_layers = nn.ModuleList([nn.Linear(self.hidden_size * 2, self.d_model),
                                            nn.Linear(self.hidden_size, self.d_model)])

        self.norm_layers = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])

    def _init_lstm_layer(self, d_model, hidden_size, bidirectional):
        return LSTM(d_model, hidden_size, num_layers=1, bidirectional=bidirectional, batch_first=True)

    def forward(self, x):
        B, C, F, T = x.shape

        # # Process dual-path rnn
        original_x = x
        # Frequency-path
        x = x.transpose(1, 3).contiguous().view(B * T, F, C)
        x = self.norm_layers[0](x)
        x, _ = self.lstm_layers[0](x)
        x = self.linear_layers[0](x)
        x = x.view(B, T, F, C).transpose(1, 3)
        x = x + original_x

        original_x = x
        # Time-path
        x = x.transpose(1, 2).contiguous().view(B * F, C, T)
        x = self.time_conv(x)
        x = x.transpose(1, 2)
        x = self.norm_layers[1](x)
        x, _ = self.lstm_layers[1](x)
        x = self.linear_layers[1](x)
        x = x.transpose(1, 2).contiguous().view(B, F, C, T).transpose(1, 2)
        x = x + original_x

        return x

class SeparationNet(nn.Module):
    """
    Implements a simplified Sparse Down-sample block in an encoder architecture.

    Args:
    - channels (int): Number input channels.
    - expand (int): Expansion factor used to calculate the hidden_size of LSTM.
    - num_layers (int): Number of dual-path layers.
    """

    def __init__(self, channels, expand=1, num_layers=6, lookahead=3):
        super(SeparationNet, self).__init__()

        self.num_layers = num_layers

        self.dp_modules = nn.ModuleList([
            DualPathRNN(channels * (2 if i % 2 == 1 else 1), expand,
                        lookahead=3 if i == num_layers - 1 else 0) for i in range(num_layers)
        ])

        self.feature_conversion = nn.ModuleList([
            ConvConversion(channels * 2, inverse=False if i % 2 == 0 else True) for i in range(num_layers)
        ])

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.dp_modules[i](x)
            x = self.feature_conversion[i](x)
        return x

class SeparationNetStream(nn.Module):
    """
    Implements a simplified Sparse Down-sample block in an encoder architecture.

    Args:
    - channels (int): Number input channels.
    - expand (int): Expansion factor used to calculate the hidden_size of LSTM.
    - num_layers (int): Number of dual-path layers.
    """

    def __init__(self, channels, expand=1, num_layers=6, lookahead=3):
        super(SeparationNetStream, self).__init__()

        self.num_layers = num_layers

        self.dp_modules = nn.ModuleList([
            DualPathRNNStream(channels * (2 if i % 2 == 1 else 1), expand,
                        lookahead=3 if i == num_layers - 1 else 0) for i in range(num_layers)
        ])

        self.feature_conversion = nn.ModuleList([
            ConvConversion(channels * 2, inverse=False if i % 2 == 0 else True) for i in range(num_layers)
        ])

    def forward_1st_frame(self, x, cache_h1, cache_c1, cache_h2, cache_c2, time_conv_cache):
        x, new_cache_h1, new_cache_c1, _ = self.dp_modules[0](x, cache_h1, cache_c1, None)
        x = self.feature_conversion[0](x)
        x, new_cache_h2, new_cache_c2, new_time_conv_cache = self.dp_modules[1].forward_1st_frame(x, cache_h2, cache_c2, time_conv_cache)
        return x, new_cache_h1, new_cache_c1, new_cache_h2, new_cache_c2, new_time_conv_cache

    def forward(self, x, cache_h1, cache_c1, cache_h2, cache_c2, time_conv_cache):
        x, new_cache_h1, new_cache_c1, _ = self.dp_modules[0](x, cache_h1, cache_c1, None)
        x = self.feature_conversion[0](x)
        x, new_cache_h2, new_cache_c2, new_time_conv_cache = self.dp_modules[1](x, cache_h2, cache_c2, time_conv_cache)
        x = self.feature_conversion[1](x)
        return x,new_cache_h1, new_cache_c1, new_cache_h2, new_cache_c2, new_time_conv_cache

    def forward_last_frame(self, x, cache_h1, cache_c1, cache_h2, cache_c2, time_conv_cache):
        x, new_cache_h2, new_cache_c2, new_time_conv_cache = self.dp_modules[1].forward_last_frame(x, cache_h2, cache_c2, time_conv_cache)
        x = self.feature_conversion[1](x)
        return x, cache_h1, cache_c1, new_cache_h2, new_cache_c2, new_time_conv_cache

def convert_state_dict(src_net, dst_net):
    state_dict = src_net.state_dict()
    new_state_dict = dst_net.state_dict()
    for k, v in state_dict.items():
        if k in state_dict.keys():
            new_state_dict[k] = v
    dst_net.load_state_dict(new_state_dict)

def test_dupthrnn():
    torch.manual_seed(42)
    x = torch.randn([1, 64, 57, 432], dtype=torch.float32)

    rnn = DualPathRNN(64, 1, bidirectional=True, lookahead=3)
    rnn.eval()
    with torch.no_grad():
        output = rnn(x)
    total_params = 0
    for name, p in rnn.named_parameters():
        # if p.requires_grad:
        total_params += p.numel()
        # print(f"{name} {str(list(p.shape))}  ")
    print('scnet param num:', total_params)
    rnn_stream = DualPathRNNStream(64, 1, bidirectional=True, lookahead=3)
    convert_state_dict(rnn, rnn_stream)
    rnn_stream.eval()

    time_conv_cache = torch.zeros([1 * 57, 64, 3], dtype=torch.float32)
    h, c = torch.zeros([1, 57, 64], dtype=torch.float32), torch.zeros([1, 57, 64], dtype=torch.float32)
    chunk_size = x.shape[-1]
    stream_output = []
    look_ahead = 0
    state = [time_conv_cache, h, c, look_ahead]
    for i in range(chunk_size):
        with torch.no_grad():
            input_chunk = x[..., i :i + 1]
            chunk_output, *state = rnn_stream(input_chunk, *state)
            if chunk_output is not None:
             stream_output.append(chunk_output)

    stream_output = torch.cat(stream_output, dim=-1)

    print(f"Correct output shape: {output.shape}, Reference shape: {stream_output.shape}")
    # 比较修正后的输出
    min_len = min(stream_output.shape[-1], output.shape[-1])
    diff_correct = (stream_output[..., 0:] - output[..., :min_len]).abs()
    max_diff_0 = diff_correct.max().item()
    mean_diff_0 = diff_correct.mean().item()
    # max_diff_1 = diff_correct[:, :, :, 1].max().item()
    # mean_diff_1 = diff_correct[:, :, :, 1].mean().item()
    print(f"Correct vs Ref: max diff = {max_diff_0:.6e}, mean diff = {mean_diff_0:.6e}")

def test_separationNet():
    torch.manual_seed(42)
    x = torch.randn([1, 64, 57, 9], dtype=torch.float32)

    net = SeparationNet(64, 1, num_layers=2)
    net.eval()
    with torch.no_grad():
        output = net(x)

    stream_net = SeparationNetStream(64, 1, num_layers=2)
    convert_state_dict(net, stream_net)
    stream_net.eval()

    cache_h1, cache_c1 = torch.zeros([1, 57, 64], dtype=torch.float32), torch.zeros([1, 57, 64], dtype=torch.float32)
    cache_h2, cache_c2 = torch.zeros([1, 57, 128], dtype=torch.float32), torch.zeros([1, 57, 128], dtype=torch.float32)
    cache_conv = torch.zeros([57, 128, 6], dtype=torch.float32)
    state=[cache_h1, cache_c1, cache_h2, cache_c2, cache_conv]
    stream_output = []
    with torch.no_grad():
        x_chunk = x[..., 0:3]
        _, *state = stream_net.forward_1st_frame(x_chunk, *state)
        for i in range(int(x.shape[-1]/3)-1):
            x_chunk = x[..., (i+1)*3:(i+2)*3]
            output_chunk, *state = stream_net(x_chunk, *state)
            stream_output.append(output_chunk)
        # x_chunk = torch.zeros(x_chunk.shape, dtype=torch.float32)
        output_chunk, *state = stream_net.forward_last_frame(None, *state)
        stream_output.append(output_chunk)
        stream_output = torch.cat(stream_output, dim=-1)

    print(f"Correct output shape: {output.shape}, Reference shape: {stream_output.shape}")
    # 比较修正后的输出
    min_len = min(stream_output.shape[-1], output.shape[-1])
    diff_correct = (stream_output[..., :min_len] - output[..., :min_len]).abs()
    max_diff_0 = diff_correct.max().item()
    mean_diff_0 = diff_correct.mean().item()
    print(f"Correct vs Ref: max diff = {max_diff_0:.6e}, mean diff = {mean_diff_0:.6e}")


if __name__ == '__main__':
    # test_dupthrnn()
    test_separationNet()
