"""This module implements a COVAFilter that filters static frames out.
Uses default parameters for motion detection."""

import numpy as np
from typing import List

from edge_autotune.pipeline.pipeline import COVAFilter
from edge_autotune.motion.motion_detector import BackgroundCV, MotionDetector


class FilterStatic(COVAFilter):
    """Class implementing a simple COVAFilter using motion detection."""

    def __init__(self, warmup: int = 0):
        self.detector = MotionDetector(BackgroundCV())
        self.warmup = warmup
        self.processed_frames = 0

    def filter(self, img: np.ndarray) -> List[np.ndarray]:
        """Filters the input image using motion detection.

        Args:
            img (np.ndarray): current frame

        Returns:
            [list[np.ndarray]]: list of regions in the input img containing movement.
        """
        boxes, _ = self.detector.detect(img)
        self.processed_frames += 1
        if self.processed_frames < self.warmup:
            return []
        return boxes

    def epilogue(self):
        pass
