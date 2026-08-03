"""Run SCNet streaming alignment tests."""

from SCNetStreamAudioConv1dChunk import test_scnet, test_scnet_nostft


if __name__ == "__main__":
    test_scnet()
    # test_scnet_nostft()
