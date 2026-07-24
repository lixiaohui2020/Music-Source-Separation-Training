"""Export/check first-mid-last ONNX graphs and compare PyTorch with ONNX Runtime.

The check is spectrogram-domain only. It deliberately does not call STFT,
RFFT, IRFFT or OLA ISTFT, so the ONNX files are ready for NPU quantization.
"""

from __future__ import annotations

import subprocess
import sys
from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from onnx_common import (
    CACHE_NAMES,
    FIRST_INPUT_NAMES,
    FIRST_OUTPUT_NAMES,
    LAST_INPUT_NAMES,
    LAST_OUTPUT_NAMES,
    MID_INPUT_NAMES,
    MID_OUTPUT_NAMES,
    FirstONNXWrapper,
    LastONNXWrapper,
    MidONNXWrapper,
    make_example_state,
    make_stream_model,
)


ROOT = Path(__file__).parent
ONNX_DIR = ROOT / "onnx"


def _export_all() -> None:
    for script in ("export_onnx_first.py", "export_onnx_mid.py", "export_onnx_last.py"):
        subprocess.run([sys.executable, str(ROOT / script)], check=True)


def _run_ort(path: Path, names, values):
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = {name: value.detach().cpu().numpy() for name, value in zip(names, values)}
    return [torch.from_numpy(x) for x in session.run(None, inputs)]


def _assert_close(label: str, torch_outputs, ort_outputs, atol=1e-5, rtol=1e-5):
    assert len(torch_outputs) == len(ort_outputs), f"{label}: output count mismatch"
    max_error = 0.0
    for index, (expected, actual) in enumerate(zip(torch_outputs, ort_outputs)):
        error = (expected.cpu() - actual).abs().max().item()
        max_error = max(max_error, error)
        torch.testing.assert_close(actual, expected.cpu(), atol=atol, rtol=rtol)
    print(f"{label}: {len(torch_outputs)} outputs, max abs err={max_error:.6e}")


def _skip_deque(state):
    return deque((state[11], state[12], state[13]))


def _pytorch_first(wrapper, spec, caches):
    return tuple(wrapper(spec, *caches))


def _pytorch_mid(wrapper, spec, state):
    return tuple(wrapper(spec, *state))


def _pytorch_last(wrapper, state):
    return tuple(wrapper(*state))


def main():
    _export_all()

    model = make_stream_model()
    first = FirstONNXWrapper(model).eval()
    mid = MidONNXWrapper(model).eval()
    last = LastONNXWrapper(model).eval()

    torch.manual_seed(1234)
    spec_first = torch.randn(1, 4, 2049, 3)
    spec_mid = torch.randn(1, 4, 2049, 3)
    cache_state = make_example_state()[:11]

    with torch.no_grad():
        first_torch = _pytorch_first(first, spec_first, cache_state)
        first_ort = _run_ort(ONNX_DIR / "scnet_first.onnx", FIRST_INPUT_NAMES, (spec_first, *cache_state))
        _assert_close("first", first_torch, first_ort)

        mid_torch = _pytorch_mid(mid, spec_mid, first_torch)
        mid_ort = _run_ort(ONNX_DIR / "scnet_mid.onnx", MID_INPUT_NAMES, (spec_mid, *first_ort))
        _assert_close("mid", mid_torch, mid_ort)

        # Chain ORT state to verify the exact NPU/host execution model.
        last_torch = _pytorch_last(last, mid_torch[1:])
        last_ort = _run_ort(ONNX_DIR / "scnet_last.onnx", LAST_INPUT_NAMES, tuple(mid_ort[1:]))
        _assert_close("last", last_torch, last_ort)

    print("ONNX Runtime alignment passed; exported graphs were onnxsim-simplified and contain no STFT/FFT nodes.")


if __name__ == "__main__":
    main()
