"""ONNX wrappers and fixed-shape state helpers for SCNetStreamNoSTFT.

The three exported graphs operate only on real-valued spectrogram tensors.
STFT and OLA ISTFT intentionally remain outside the ONNX/NPU graphs.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

from SCNetStreamAudioConv1dChunk import SCNetNoStft, SCNetStreamNoSTFT, convert_state_dict


# State ordering is stable across first/mid/last ONNX files.
STATE_NAMES = (
    "cache_band0",
    "cache_band1",
    "cache_band2",
    "cache_h1",
    "cache_c1",
    "cache_h2",
    "cache_c2",
    "cache_conv",
    "cache_fus0",
    "cache_fus1",
    "cache_fus2",
    "skip0",
    "skip1",
    "skip2",
)
CACHE_NAMES = STATE_NAMES[:11]
FIRST_INPUT_NAMES = ("spec_in", *CACHE_NAMES)
FIRST_OUTPUT_NAMES = tuple(f"{name}_out" for name in STATE_NAMES)
MID_INPUT_NAMES = ("spec_in", *STATE_NAMES)
MID_OUTPUT_NAMES = ("spec_out", *FIRST_OUTPUT_NAMES)
LAST_INPUT_NAMES = STATE_NAMES
LAST_OUTPUT_NAMES = MID_OUTPUT_NAMES


def make_stream_model(device: torch.device = torch.device("cpu")) -> SCNetStreamNoSTFT:
    """Create the exact model pair used by test_scnet_nostft and copy weights."""
    # Each standalone exporter must use exactly the same weights for ORT parity.
    # Replace this random demo model with your checkpoint-loaded SCNet model for deployment.
    torch.manual_seed(42)
    offline = SCNetNoStft(sources=["accompaniment", "vocals"]).to(device).eval()
    stream = SCNetStreamNoSTFT(sources=["accompaniment", "vocals"]).to(device).eval()
    convert_state_dict(offline, stream)
    return stream


def make_example_state(
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, ...]:
    """Return tensors in STATE_NAMES order for F=2049 and a 3-frame chunk."""
    z = lambda *shape: torch.zeros(*shape, dtype=dtype, device=device)
    return (
        z(1, 64, 616, 2),
        z(1, 128, 186, 2),
        z(1, 64, 57, 2),
        z(1, 57, 64),
        z(1, 57, 64),
        z(1, 57, 128),
        z(1, 57, 128),
        z(57, 128, 6),
        z(1, 128, 57, 2),
        z(1, 256, 186, 2),
        z(1, 128, 616, 2),
        z(1, 64, 616, 3),
        z(1, 128, 186, 3),
        z(1, 64, 57, 3),
    )


def _as_deque(skip0: torch.Tensor, skip1: torch.Tensor, skip2: torch.Tensor) -> deque:
    # The original model pops skip2, then skip1, then skip0.
    return deque((skip0, skip1, skip2))


def _state_from_first(result) -> Tuple[torch.Tensor, ...]:
    (
        _,
        band0,
        band1,
        band2,
        h1,
        c1,
        h2,
        c2,
        conv,
        fus0,
        fus1,
        fus2,
        skips,
    ) = result
    skip0, skip1, skip2 = tuple(skips)
    return (band0, band1, band2, h1, c1, h2, c2, conv, fus0, fus1, fus2, skip0, skip1, skip2)


def _state_from_mid_or_last(result) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
    (
        spec_out,
        band0,
        band1,
        band2,
        h1,
        c1,
        h2,
        c2,
        conv,
        fus0,
        fus1,
        fus2,
        skips,
    ) = result
    if skips is None:
        # last does not need a following invocation; retain inputs in the wrapper.
        skip0 = skip1 = skip2 = None
    else:
        skip0, skip1, skip2 = tuple(skips)
    return spec_out, (band0, band1, band2, h1, c1, h2, c2, conv, fus0, fus1, fus2, skip0, skip1, skip2)


class FirstONNXWrapper(nn.Module):
    """First 3-frame graph: consumes a spectrogram, initializes lookahead/cache state."""

    def __init__(self, model: SCNetStreamNoSTFT):
        super().__init__()
        self.model = model

    def forward(self, spec_in, *cache_in):
        result = self.model.forward_1st_frame(spec_in, *cache_in, None)
        return _state_from_first(result)


class MidONNXWrapper(nn.Module):
    """Steady-state 3-frame graph: spectrogram + all states -> spectrogram + states."""

    def __init__(self, model: SCNetStreamNoSTFT):
        super().__init__()
        self.model = model

    def forward(self, spec_in, *state_in):
        *cache_in, skip0, skip1, skip2 = state_in
        result = self.model(spec_in, *cache_in, _as_deque(skip0, skip1, skip2))
        spec_out, state_out = _state_from_mid_or_last(result)
        return (spec_out, *state_out)


class LastONNXWrapper(nn.Module):
    """Flush graph: no new spectrogram; drains the final 3 lookahead frames."""

    def __init__(self, model: SCNetStreamNoSTFT):
        super().__init__()
        self.model = model

    def forward(self, *state_in):
        *cache_in, skip0, skip1, skip2 = state_in
        result = self.model.forward_last_frame(
            None, *cache_in, _as_deque(skip0, skip1, skip2)
        )
        spec_out, state_out = _state_from_mid_or_last(result)
        # Last does not create new skips. Keep the input values as harmless state outputs
        # so all NPU graph signatures have the same state layout.
        state_out = (*state_out[:11], skip0, skip1, skip2)
        return (spec_out, *state_out)


def _assert_no_fft_nodes(graph, path: Path) -> None:
    import onnx

    onnx.checker.check_model(graph)
    banned = {"STFT", "DFT", "FFT", "RFFT", "IRFFT"}
    found = sorted({node.op_type for node in graph.graph.node if node.op_type.upper() in banned})
    if found:
        raise RuntimeError(f"{path.name} unexpectedly contains FFT/STFT nodes: {found}")


def export_onnx(
    wrapper: nn.Module,
    args: Tuple[torch.Tensor, ...],
    path: Path,
    input_names,
    output_names,
    opset_version: int = 17,
    simplify: bool = True,
) -> None:
    """Export a static-shape ONNX model, optionally simplify with onnxsim, and verify no FFT/STFT nodes."""
    import onnx
    import onnxsim

    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper.eval()
    raw_path = path.with_name(f"{path.stem}.raw{path.suffix}")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            args,
            str(raw_path),
            input_names=list(input_names),
            output_names=list(output_names),
            opset_version=opset_version,
            do_constant_folding=True,
            dynamo=False,
        )
    graph = onnx.load(str(raw_path))
    _assert_no_fft_nodes(graph, path)

    if simplify:
        input_data = {
            name: tensor.detach().cpu().numpy()
            for name, tensor in zip(input_names, args)
        }
        graph, check_ok = onnxsim.simplify(graph, input_data=input_data)
        if not check_ok:
            raise RuntimeError(f"onnxsim verification failed for {path.name}")
        onnx.save(graph, str(path))
        raw_path.unlink(missing_ok=True)
    else:
        raw_path.replace(path)

    _assert_no_fft_nodes(graph, path)

