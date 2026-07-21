"""Export the lookahead-flush, spectrogram-only SCNet ONNX graph."""

from pathlib import Path

from onnx_common import (
    LAST_INPUT_NAMES,
    LAST_OUTPUT_NAMES,
    LastONNXWrapper,
    export_onnx,
    make_example_state,
    make_stream_model,
)


def main():
    model = make_stream_model()
    wrapper = LastONNXWrapper(model)
    path = Path(__file__).parent / "onnx" / "scnet_last.onnx"
    export_onnx(wrapper, make_example_state(), path, LAST_INPUT_NAMES, LAST_OUTPUT_NAMES)
    print(f"Exported {path}")


if __name__ == "__main__":
    main()
