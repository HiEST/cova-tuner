"""This module implements a COVAFilter that filters static frames out.
Uses default parameters for motion detection."""


from edge_autotune.pipeline.pipeline import COVAFilter
from edge_autotune.motion.motion_detector import BackgroundCV, MotionDetector


class FilterStatic(COVAFilter):
    def __init__(self, warmup: int = 0):
        self.detector = MotionDetector(BackgroundCV())
        self.warmup = warmup
        self.processed_frames = 0

    def filter(self, img):
        boxes, _ = self.detector.detect(img)
        self.processed_frames += 1
        if self.processed_frames < self.warmup:
            return []
        return boxes

    def epilogue(self):
        pass
