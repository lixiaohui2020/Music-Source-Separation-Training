"""Run the existing STFT + nostft + OLA_ISTFT alignment test.

The implementation deliberately stays in SCNetStreamAudioConv1dChunk.py.
Keeping this thin entry point avoids changing the test_scnet_nostft logic.
"""

from SCNetStreamAudioConv1dChunk import test_scnet_nostft


if __name__ == "__main__":
    test_scnet_nostft()
