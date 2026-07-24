"""Export the first 3-frame, spectrogram-only SCNet ONNX graph."""

from pathlib import Path

import torch

from onnx_common import (
    FIRST_INPUT_NAMES,
    FIRST_OUTPUT_NAMES,
    FirstONNXWrapper,
    export_onnx,
    make_example_state,
    make_stream_model,
)


def main():
    model = make_stream_model()
    wrapper = FirstONNXWrapper(model)
    spec_in = torch.randn(1, 4, 2049, 3)
    cache_in = make_example_state()[:11]
    path = Path(__file__).parent / "onnx" / "scnet_first.onnx"
    export_onnx(wrapper, (spec_in, *cache_in), path, FIRST_INPUT_NAMES, FIRST_OUTPUT_NAMES)
    print(f"Exported and simplified {path}")


if __name__ == "__main__":
    main()
