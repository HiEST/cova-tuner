"""This module implements a simple COVACapturer using OpenCV's VideoCapture class."""

import cv2

from edge_autotune.pipeline.pipeline import COVACapture


class VideoCapture(COVACapture):
    def __init__(self, stream):
        self.cap = cv2.VideoCapture(stream)

    def capture(self) -> None:
        """Captures next frame from stream."""
        return self.cap.read()

    def epilogue(self) -> None:
        pass