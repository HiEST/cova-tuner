import numpy as np

import imutils
import cv2

from utils.nms import non_max_suppression_fast
from utils.iou import compute_iou
from utils import constants

def GaussianBlur(frame):
# To filter out small variations from frame to frame 
# in the order of a few pixels, we apply Gaussian smoothing 
# to average pixel intensities across a 21 x 21 region.
# This helps to smooth out (and hopefully remove) the high frequency noise.

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    return gray  

def merge_all_boxes(boxes):
    minX = np.min(boxes[:,0])
    minY = np.min(boxes[:,1])
    maxX = np.max(boxes[:,2])
    maxY = np.max(boxes[:,3])

    return (minX, minY, maxX, maxY)

def merge_overlapping_boxes(boxes, iou=0.01):
    
    while True:
        num_boxes = len(boxes)
        for i, box in enumerate(boxes):
            intersections = [compute_iou(box, box2) if j != i else 1 for j, box2 in enumerate(boxes)]
            overlap = np.where(np.array(intersections) > iou)[0]
            not_overlap = np.where(np.array(intersections) <= iou)[0]
                
            if len(overlap) <= 1:
                continue

            for over in overlap:
                if over == i:
                    continue
            
            overlapping = [boxes[idx] for idx in overlap]
            new_box = merge_all_boxes(np.array(overlapping))
            new_boxes = [boxes[idx] for idx in not_overlap]
            new_boxes.append(new_box)
            boxes = np.array(new_boxes)
            break

        if num_boxes == len(boxes):
            break

    return boxes

def merge_near_boxes(boxes, proximity=1.05):
    new_boxes = [
        [
            max(0, int(box[0]*(proximity-1))),
            max(0, int(box[1]*(proximity-1))),
            min(1920, int(box[2]*proximity)),
            min(1080, int(box[3]*proximity))
        ]
        for box in boxes
    ]

    merged_boxes = merge_overlapping_boxes(np.array(new_boxes), 0.5)
    # print(f'boxes: {boxes}')
    # print(f'new_boxes: {new_boxes}')
    # print(f'merged_boxes: {merged_boxes}')

    return boxes
    return np.array(new_boxes)
    # return merged_boxes


def propose_rois(boxes, roi_width=256, roi_height=256, max_width=1920, max_height=1080, random_factor=1):
    roi_proposals = []
    boxes = np.array(boxes)

    
    if len(boxes) > 1:
        boxes = non_max_suppression_fast(boxes)

    # boxes = merge_near_boxes(boxes)

    for box in boxes:
        width = box[2] - box[0]
        height = box[3] - box[1]
        if width < constants.ROI_SIZE[0] and height < constants.ROI_SIZE[1]:
            new_roi = constants.ROI_SIZE

        elif width < height:
            # Resize with same aspect ratio as constants.ROI_SIZE
            aspect = width / height
            new_roi = [
                width * (constants.ROI_AR/aspect),
                height
            ]
        else:
            aspect = width / height
            new_roi = [
                width,
                height * (aspect/constants.ROI_AR),
            ]

        # Offset from the center of the box
        offset_x = new_roi[0]/2
        offset_y = new_roi[1]/2

        # Center of the box
        center = [
            box[0]+width/2,
            box[1]+height/2
        ]

        # Coordinates of the new box
        box = [
            int(max(center[0]  - offset_x, 0)),
            int(max(center[1]  - offset_y, 0)),
            int(min(center[0]  + offset_x, max_width)),
            int(min(center[1] + offset_y, max_height)),
        ]

        roi_proposals.append(box)

    if len(roi_proposals) > 1:
        roi_proposals = non_max_suppression_fast(np.array(roi_proposals))

    # print(f'before merging: {len(roi_proposals)}')
    roi_proposals = merge_overlapping_boxes(roi_proposals, 0.05)
    # print(f'after merging: {len(roi_proposals)}')
    return roi_proposals


class Background:
# no_average: bool. If True, only first frame will be considered
#             for the background and no average will be computed.
# skip: number of frames to skip for the next average background.
# take: number of frames taken into account for the next average.
# use_last: number of background averages to use for next average background.
    def __init__(self, no_average=False, skip=20, take=10, use_last=15):
        self.background = None
        self.background_color = None
        
        self.skip = skip
        self.use_last = use_last
        self.take = take

        # Vector to store last average of backgrounds
        self.last_avgs = []
        # Vector to store last frames for next average background
        self.last_frames = []
        self.skipped = 0

    def update(self, frame):
        if self.background is None:
            self.background = GaussianBlur(frame)
            self.background_color = frame.copy()
            return self.background
        
        self.skipped += 1

        # skip this frame for the average
        if self.skipped <= self.skip:
            return self.background
        
        # count this frame for the average
        else:
            self.last_frames.append(frame.copy())
            self.skipped = 0

        # if gathered enough frames, compute new average background
        if len(self.last_frames) >= self.take:

            avg_last = np.mean(self.last_frames, axis=0)
            avg_last = avg_last.astype(np.uint8)
            self.last_frames = [avg_last.copy()]
            
            self.last_avgs.append(avg_last)

            # If we have enough averages, drop the oldest
            if len(self.last_avgs) > self.use_last:
                self.last_avgs = self.last_avgs[1:]
            
            # Compute average background from all averages
            avg_background = np.mean(self.last_avgs, axis=0)
            avg_background = avg_background.astype(np.uint8)

            self.background_color = avg_background
            self.background = GaussianBlur(avg_background)
        
        return self.background

            

class MotionDetection:
# background: Background object
# delta_threshold: threshold for delta for the cv2.threshold operation.
# min_area_contour: is the minimum size (in pixels) for a region of an 
#    image to be considered actual “motion”. With this, we expect 
#    to further filter out regions that changed due to noise or 
#    changes in lighting conditions.

    def __init__(
        self, 
        background=None, 
        delta_threshold=25, 
        min_area_contour=500,
        merge_rois=True
    ):

        self.background = background

        self.merge_rois = merge_rois
        self.delta_threshold = delta_threshold
        self.min_area_contour = min_area_contour
        self.kernel = np.ones((5, 5), np.uint8)

        self.current_gray = None
        self.current_delta = None
        self.current_threshold = None

    def detect(self, frame):

        # 1. First, we update the background image
        background = self.background.update(frame)
        # 2. Then, we compute GaussianBlur of the current frame
        self.current_gray = GaussianBlur(frame)

        # 3. Compute the absolute difference between the current frame and the background
        self.current_delta = cv2.absdiff(background, self.current_gray)
        self.current_threshold = cv2.threshold(
                                    self.current_delta, 
                                    self.delta_threshold, 
                                    255, 
                                    cv2.THRESH_BINARY)[1]

        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image. We use a 5x5 kernel.
        # Note: Dilation operation (opposite to erosion). A pixel element is '1'
        #   if at least one pixel under the kernel is '1'. So it increases 
        #   the white region in the image or size of foreground object increases.
        
        self.current_threshold = cv2.dilate(self.current_threshold, self.kernel, iterations=2)
        cnts = cv2.findContours(
            self.current_threshold.copy(), 
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        cnts = imutils.grab_contours(cnts)
        
        boxes = []
        # loop over the contours
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < self.min_area_contour:
                continue

            (x, y, w, h) = cv2.boundingRect(c)
            boxes.append([x, y, x+w, y+h])


        if self.merge_rois and len(boxes) >= 1:
            boxes = propose_rois(boxes)

        return boxes
        
