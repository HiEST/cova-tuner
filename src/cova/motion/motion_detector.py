#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements MotionDetetor and Background classes, and auxiliary methods."""

from abc import ABC, abstractmethod
from enum import Enum
from logging import BASIC_FORMAT
from typing import Optional, TypeAlias

import cv2
import imutils
import numpy as np
from numpy.typing import NDArray

BoundingBox = list[int]  # [xmin, ymin, xmax, ymax]
Frame: TypeAlias = NDArray[np.uint8]  # 3D array of shape (height, width, 3)


class BackgroundMethod(Enum):
    FIRST = 1
    AVERAGE = 2
    PREVIOUS = 3
    ACUM_MEAN = 4
    ACUM_MEDIAN = 5
    MOG2 = 6
    HYBRID = 7
    KNN = 8


class Background(ABC):
    """This is an abstract class for background models.

    The goal of a Background object is to infer the background of a scene from a set of consecutive images.
    A robust background is a requirement to get an effective motion detector.
    If the creation of the background fails, regions might be falsely flagged as changed and viceversa.

    Attributes:
        background: latest grayscale computed background.
        background_color: color version of the latest computed background. Useful for debugging purposes.
        frameskip: number of frames to skip for the next average background.
        take: number of frames taken into account for the next average.
        use_last: number of background averages to use for next average background.
        last_avgs: list of last computed backgrounds.
        last_frame: list of last frames for next background computation.
        skipped: count for number of frames skipped since last taken.
        frozen: when True, the background is frozen and new updates are discarded and have no effect on the background.
                Useful when the current background can be considered optimal.
    """

    def freeze(self):
        self.frozen = True

    def unfreeze(self):
        self.frozen = False

    @abstractmethod
    def update(self, frame: Frame) -> Frame:
        pass


class BackgroundCV(Background):
    """This class models the background from a scene using OpenCV's implementation of MOG2 or KNN."""

    model: cv2.BackgroundSubtractor
    background_blur: Frame
    mask: Frame
    kernel: np.ndarray
    recompute_background: bool

    def __init__(self, method: BackgroundMethod = BackgroundMethod.MOG2):
        if method == BackgroundMethod.MOG2:
            self.model = cv2.createBackgroundSubtractorMOG2()
        elif method == BackgroundMethod.KNN:
            self.model = cv2.createBackgroundSubtractorKNN()
        else:
            raise ValueError("Incorrect BGS method for BackgroundCV.")

        self.kernel = np.ones((2, 2), np.uint8)
        self.updated_background = False

    def update(self, frame: Frame) -> Frame:
        self.mask = self.model.apply(frame)
        self.mask = cv2.dilate(self.mask, self.kernel, iterations=2)
        self.recompute_background = True
        return self.mask

    def getBackgroundImage(self) -> Frame:
        if self.recompute_background:
            background_color = self.model.getBackgroundImage()
            self.background_blur = GaussianBlur(background_color)
            self.recompute_background = False

        return self.background_blur


class BackgroundHybrid(Background):
    """This class models the background from a scene using a hybrid approach.

    The hybrid approach uses a combination of MOG2 and a simple average of the last frames.
    """

    model: cv2.BackgroundSubtractorMOG2
    background: Frame
    frameskip: int
    skipped: int
    initialized: bool
    delta_threshold: int

    def __init__(self, delta_threshold: int = 25, frameskip: int = 0):
        self.model = cv2.createBackgroundSubtractorMOG2()
        self.kernel = np.ones((5, 5), np.uint8)
        self.frameskip = frameskip
        self.skipped = 0
        self.initialized = False
        self.delta_threshold = delta_threshold

    def update(self, frame: Frame) -> Frame:
        if self.frameskip > 1:
            self.skipped += 1
            if self.skipped < self.frameskip:
                if not self.initialized:
                    self.background = GaussianBlur(frame)
                    background_color = frame.copy()
                    self.model.apply(frame)
                    self.initialized = True
                return self.background
            self.skipped = 0

        self.model.apply(frame)

        background_color = self.model.getBackgroundImage()
        self.background = GaussianBlur(background_color)

        self.current_gray = GaussianBlur(frame)
        self.current_delta = cv2.absdiff(self.background, self.current_gray)
        self.current_mask = cv2.threshold(
            self.current_delta, self.delta_threshold, 255, cv2.THRESH_BINARY
        )[1]

        self.current_threshold = cv2.dilate(
            self.current_mask, self.kernel, iterations=2
        )

        return self.current_threshold


class BackgroundSimple(Background):
    """This class models the background from a scene using a simple method based on average pixel color over time."""

    method: BackgroundMethod
    background: Frame
    prev_background: Frame
    frozen: bool
    kernel: np.ndarray
    last_avgs: list[Frame]
    last_frames: list[Frame]
    take: int
    use_last: int
    skipped: int
    frameskip: int
    initialized: bool

    def __init__(
        self,
        method: BackgroundMethod,
        take: int = 10,
        use_last: int = 15,
        frameskip: int = 1,
    ):
        """Initialize background model.

        If method is FIRST, only the first frame will be considered for the background
        and no average will be computed.

        Args:
            method: BackgroundMethod. Method to use for background computation.
            take: int. Number of frames to take into account for the next average background.
            use_last: int. Number of background averages to use for next average background.
            frameskip: number of frames to skip for the next average background.
        """
        self.method = method

        self.use_last = use_last
        self.take = take
        self.skipped = 0
        self.frameskip = frameskip

        self.last_avgs = []
        self.last_frames = []

        self.frozen = method == BackgroundMethod.FIRST
        self.kernel = np.ones((5, 5), np.uint8)
        self.initialized = False

    def update(self, frame: Frame) -> Frame:
        """Update background with new frame.

        Args:
            :param frame: 3D numpy array containing a frame
        """
        if not self.initialized:
            self.background = GaussianBlur(frame)
            self.background_color = frame.copy()
            self.last_frames.append(self.background_color)
            self.prev_background = self.background
            self.initialized = True
            return self.background

        if self.frozen:
            return self.background

        if self.method == BackgroundMethod.PREVIOUS:
            prev_background = self.prev_background
            self.prev_background = self.background

            self.background = GaussianBlur(frame)
            self.background_color = frame.copy()

            return prev_background

        self.skipped += 1

        if self.method in [BackgroundMethod.ACUM_MEAN, BackgroundMethod.ACUM_MEAN]:
            self.last_frames.append(frame.copy())
            if len(self.last_frames) > self.use_last:
                self.last_frames = self.last_frames[1:]
            if self.method == BackgroundMethod.ACUM_MEAN:
                self.background_color = np.mean(self.last_frames, axis=0).astype(
                    np.uint8
                )
            else:
                self.background_color = np.median(self.last_frames, axis=0).astype(
                    np.uint8
                )
            self.background = GaussianBlur(self.background_color)
            return self.background

        # skip this frame for the average
        elif self.skipped <= self.frameskip:
            return self.background

        # count this frame for the average
        else:
            self.last_frames.append(frame.copy())
            self.skipped = 0

        # if gathered enough frames, compute new average background
        if len(self.last_frames) >= self.take:
            # avg_last = np.mean(self.last_frames, axis=0)
            avg_last = np.median(self.last_frames, axis=0)
            avg_last = avg_last.astype(np.uint8)
            self.last_frames = [avg_last.copy()]

            self.last_avgs.append(avg_last)

            # If we have enough averages, drop the oldest
            if len(self.last_avgs) > self.use_last:
                self.last_avgs = self.last_avgs[1:]

            # Compute average background from all averages
            # avg_background = np.mean(self.last_avgs, axis=0)
            avg_background = np.median(self.last_avgs, axis=0)
            avg_background = avg_background.astype(np.uint8)

            self.background_color = avg_background
            self.background = GaussianBlur(avg_background)

        return self.background


class MotionDetector:
    """This class detects motion in a scene and computes regions of interest."""

    background: Background
    delta_threshold: int
    min_area_contour: int
    roi_size: tuple[int, int]
    merge_rois: bool

    def __init__(
        self,
        background: Background,
        delta_threshold: int = 25,
        min_area_contour: int = 500,
        roi_size: tuple[int, int] = (1, 1),
        merge_rois: bool = True,
    ):
        """Initialize MotionDetector with Background object.

        Args:
        background (Background): Background object to obtain background of the scene.
        delta_threshold (int, optional): threshold for delta over frame difference. Defaults to 25.
        min_area_contour (int, optional): Minimum area (in pixels) for a contour to be considered as actual "motion" and for region proposal. Defaults to 500.
        roi_size (tuple, optional): Minimum size of a region. If a contour is smaller, the bounding box of the region is expanded. Defaults to (1,1).
        merge_rois (bool, optional): Merge overlapping regions. Defaults to True.
        """

        self.background = background

        self.merge_rois = merge_rois
        self.delta_threshold = delta_threshold
        self.min_area_contour = min_area_contour

        self.roi_size = roi_size

    def detect(self, frame: Frame):
        """Update background and detect movement in the input frame with respect to the scene.

        The process for motion detection is as follows:
            - 1. First, we update the background image with the current frame.
            - 2. Then, we compute GaussianBlur of the current frame.
            - 3. Compute the absolute difference between the current frame and the background.
            - 4. We apply a cv2.THRESH_BINARY operation to the absolute difference to filter out small (in intensity) differences.
            - 5. Then, we dilate the thresholded image to fill in holes.
            - 6. Finally, we use cv2.findContours on the thresholded image to find the regions of the changed regions. Contours smaller than min_area are ignored.

        Note: Dilation operation (opposite to erosion). A pixel element is '1'
        if at least one pixel under the kernel is '1'. So it increases
        the white region in the image or size of foreground object increases.

        Args:
            frame (np.array): Input frame in which detect movement.

        Returns:
            tuple: Regions where movement is detected [xmin, ymin, xmax, ymax], and the area in pixels of each region.
        """

        frame_mask = self.background.update(frame)

        cnts = cv2.findContours(
            frame_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        cnts = imutils.grab_contours(cnts)

        boxes = []
        areas = []

        for c in cnts:
            contourArea = cv2.contourArea(c)
            if cv2.contourArea(c) < self.min_area_contour:
                continue

            (x, y, w, h) = cv2.boundingRect(c)
            boxes.append([x, y, x + w, y + h])
            areas.append(contourArea)

        if self.merge_rois and len(boxes) >= 1:
            max_height, max_width, _ = frame.shape
            boxes = propose_rois(
                boxes,
                roi_width=self.roi_size[0],
                roi_height=self.roi_size[1],
                max_width=max_width,
                max_height=max_height,
            )

        return boxes, areas


def non_max_suppression_fast(boxes: NDArray, overlapThresh: float = 0.35) -> NDArray:
    """Non-maximum suppression algorithm.

    Extracted from https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    Version based on Malisiewicz et al. solution.

    Args:
        boxes (list): List of boxes to apply non-maximum suppression.
        overlapThresh (float, optional): Threshold for overlapping boxes. Defaults to 0.35.

    Returns:
        list: List of boxes after non-maximum suppression.
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return np.array([])
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(
            idxs,
            np.concatenate((np.array([last]), np.where(overlap > overlapThresh)[0])),
        )
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


def compute_iou(boxA: list, boxB: list):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def merge_all_boxes(boxes):
    minX = np.min(boxes[:, 0])
    minY = np.min(boxes[:, 1])
    maxX = np.max(boxes[:, 2])
    maxY = np.max(boxes[:, 3])

    return (minX, minY, maxX, maxY)


def merge_overlapping_boxes(
    boxes: list[BoundingBox], iou: float = 0.01
) -> list[BoundingBox]:
    while True:
        num_boxes = len(boxes)
        for i, box in enumerate(boxes):
            intersections = [
                compute_iou(box, box2) if j != i else 1 for j, box2 in enumerate(boxes)
            ]
            overlap = np.where(np.array(intersections) > iou)[0]
            not_overlap = np.where(np.array(intersections) <= iou)[0]

            if len(overlap) <= 1:
                continue

            overlapping = [boxes[idx] for idx in overlap]
            new_box = merge_all_boxes(np.array(overlapping))
            new_boxes = [boxes[idx] for idx in not_overlap]
            new_boxes.append(new_box)
            boxes = new_boxes
            break

        if num_boxes == len(boxes):
            break

    return boxes


def merge_near_boxes(
    boxes: list[BoundingBox], proximity: float = 1.05
) -> list[BoundingBox]:
    """Merge boxes that are near each other.

    Args:
        boxes (list): List of boxes to merge.
        proximity (float, optional): Proximity to merge boxes. Defaults to 1.05.

    Return:
        list[BoundingBox]: List of merged boxes.
    """
    new_boxes = [
        [
            max(0, int(box[0] * (proximity - 1))),
            max(0, int(box[1] * (proximity - 1))),
            min(1920, int(box[2] * proximity)),
            min(1080, int(box[3] * proximity)),
        ]
        for box in boxes
    ]

    merged_boxes = merge_overlapping_boxes(new_boxes, 0.5)
    return merged_boxes


def resize_if_smaller(
    box: BoundingBox, max_dims: tuple[int, int], min_size: tuple[int, int] = (32, 32)
) -> BoundingBox:
    """Resize box if it is smaller than min_size on any dimension.

    Args:
        box (BoundingBox): Bounding Box coordinates [left, top, right, bottom].
        max_dims (tuple): Maximum size of x, y dimensions to max out new box coordinates.
        min_size (tuple, optional): Resize box if it is smaller than min_size on any dimension. Defaults to (32,32).

    Returns:
        BoundingBox: Bounding box coordinates of the new resized box [left, top, right, bottom].
    """
    if box[2] - box[0] > min_size[0] and box[3] - box[1] > min_size[1]:
        return box

    box_size = [
        box[2] - box[0],
        box[3] - box[1],
    ]

    new_size = [
        max(min_size[0], box_size[0]),
        max(min_size[1], box_size[1]),
    ]

    center = [
        box[0] + (box[2] - box[0]) / 2,
        box[1] + (box[3] - box[1]) / 2,
    ]

    offset = [
        new_size[0] / 2,
        new_size[1] / 2,
    ]

    new_box = [
        int(center[0] - offset[0]),
        int(center[1] - offset[1]),
        int(center[0] + offset[0]),
        int(center[1] + offset[1]),
    ]

    if new_box[0] < 0:
        new_box[2] += abs(new_box[0])
        new_box[0] = 0
    if new_box[1] < 0:
        new_box[3] += abs(new_box[1])
        new_box[1] = 0
    if new_box[2] >= max_dims[0]:
        new_box[0] -= abs(new_box[2]) - (max_dims[0] + 1)
        new_box[2] = max_dims[0] - 1
    if new_box[3] >= max_dims[1]:
        new_box[1] -= abs(new_box[3]) - (max_dims[1] + 1)
        new_box[3] = max_dims[1] - 1

    new_box = [
        int(max(0, new_box[0])),
        int(max(0, new_box[1])),
        int(min(max_dims[0], new_box[2])),
        int(min(max_dims[1], new_box[3])),
    ]

    return new_box


def propose_rois(
    boxes: list[BoundingBox],
    roi_width: int,
    roi_height: int,
    max_width: int,
    max_height: int,
    roi_increment: Optional[float] = None,
    force_aspect: Optional[float] = None,
) -> list[BoundingBox]:
    """Propose regions of interest of size at least (roi_width, roi_height).
    Coordinates are guaranteed to be between (0, 0) and (max_width, max_height) and keep aspect ratio.

    Args:
        boxes (list[BoundingBox]): List of boxes [xmin, ymin, xmax, ymax]
        roi_width (int): Minimum width of the proposed RoIs.
        roi_height (int): Minimum height of the proposed RoIs.
        max_width (int): Maximum x coordinate of the proposed RoIs.
        max_height (int): Maximum y coordinate of the proposed RoIs.
        roi_increment (float, optional): If set, resizes the resulting RoIs. Defaults to None.
        force_aspect (float, optional): If set, forces the aspect ratio of the resulting RoIs. Defaults to None.

    Returns:
        list[BoundingBox]: list of regions proposed. Each is a tuple [xmin, ymin, xmax, ymax]
    """
    roi_proposals = []

    if force_aspect:
        roi_ar = force_aspect
    else:
        roi_ar = roi_width / roi_height

    for box in boxes:
        width = box[2] - box[0]
        height = box[3] - box[1]

        if width < roi_width and height < roi_height:
            new_roi = (roi_width, roi_height)

        elif width < height:
            # Resize with same aspect ratio the default roi
            aspect = width / height
            new_roi = (int(width * (roi_ar / aspect)), height)
        else:
            aspect = width / height
            new_roi = (
                width,
                int(height * (aspect / roi_ar)),
            )

        if roi_increment:
            new_roi = (
                int(new_roi[0] * roi_increment),
                int(new_roi[1] * roi_increment),
            )

        # Offset from the center of the box
        offset_x = new_roi[0] / 2
        offset_y = new_roi[1] / 2

        center = [box[0] + width / 2, box[1] + height / 2]

        box = [
            int(max(center[0] - offset_x, 0)),
            int(max(center[1] - offset_y, 0)),
            int(min(center[0] + offset_x, max_width)),
            int(min(center[1] + offset_y, max_height)),
        ]

        roi_proposals.append(box)

    roi_proposals = merge_overlapping_boxes(roi_proposals, 0.05)
    return roi_proposals


def GaussianBlur(frame: Frame) -> Frame:
    """Compute GaussianBlur on input frame.

    GaussianBlur is used to filter out small variations from frame to frame in the order of a few pixels,
    we apply Gaussian smoothing to average pixel intensities across a 21 x 21 region.
    This helps to smooth out (and hopefully remove) the high frequency noise.


    Args:
        frame (np.array): Input frame.

    Returns:
        np.array: Frame after being converted to grayscale and applying GaussianBlur to it.
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    return gray


def first_pass_bg(
    video: str,
    background: Background,
    output: Optional[str] = None,
) -> None:
    """First pass of the background subtraction algorithm.

    Args:
        video (str): Path to the video.
        background (Background): Background object.
        output (str, optional): Path to the output image. Defaults to None.
    """
    cap = cv2.VideoCapture(video)
    ret, frame = cap.read()

    while ret:
        background.update(frame)
        ret, frame = cap.read()
        background.freeze()

    # if output is not None:
    #     bg = background.background_color.copy()
    #     cv2.imwrite(output, bg)
