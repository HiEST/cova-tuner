"""This module implements a COVAFilter that filters static frames out.
Uses default parameters for motion detection."""


from edge_autotune.pipeline.pipeline import COVAFilter
from edge_autotune.motion.motion_detector import BackgroundCV, MotionDetector


class FilterStatic(COVAFilter):
    def __init__(self):
        self.detector = MotionDetector(BackgroundCV)

    def filter(self, img):
        boxes, _ = self.detector.detect()
        return boxes

    def epilogue(self):
        pass