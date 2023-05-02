"""This module implements a dummy COVACapture. It does nothing."""

import numpy as np

from cova.pipeline.pipeline import COVACapture


class DummyCapture(COVACapture):
    def __init__(self, stream):
        pass

    def capture(self) -> None:
        return np.zeros((100, 100, 3))

    def epilogue(self) -> None:
        pass
