"""This module implements a simple COVACapturer using OpenCV's VideoCapture class."""

from typing import Tuple

import cv2

from cova.pipeline.pipeline import COVACapture


class VideoCapture(COVACapture):
    """Class implementing decoding as a COVACapture pipeline stage using OpenCV"""
    def __init__(self, stream: str, frameskip: int = 0, resize: Tuple[int, int] = None):
        self.cap = cv2.VideoCapture(stream)
        self.frameskip = frameskip + 1
        self.resize = tuple(resize)
        assert(self.cap.isOpened())

    def capture(self) -> None:
        """Captures next frame from stream."""
        ret, frame = self.cap.read()
        self.cap.set(1, self.cap.get(cv2.CAP_PROP_POS_FRAMES)+self.frameskip)
        if ret and self.resize:
            frame = cv2.resize(frame, self.resize)
        return ret, frame

    def epilogue(self) -> None:
        pass
