"""Export the steady-state 3-frame, spectrogram-only SCNet ONNX graph."""

from pathlib import Path

import torch

from onnx_common import (
    MID_INPUT_NAMES,
    MID_OUTPUT_NAMES,
    MidONNXWrapper,
    export_onnx,
    make_example_state,
    make_stream_model,
)


def main():
    model = make_stream_model()
    wrapper = MidONNXWrapper(model)
    spec_in = torch.randn(1, 4, 2049, 3)
    path = Path(__file__).parent / "onnx" / "scnet_mid.onnx"
    export_onnx(wrapper, (spec_in, *make_example_state()), path, MID_INPUT_NAMES, MID_OUTPUT_NAMES)
    print(f"Exported and simplified {path}")


if __name__ == "__main__":
    main()
