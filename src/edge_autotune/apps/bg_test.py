#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import time
import cv2 as cv
import argparse
import imutils
import numpy as np
from utils.nms import non_max_suppression_fast
from utils.iou import compute_iou

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



def propose_rois(boxes, roi_width=0, roi_height=0, max_width=1920, max_height=1080, random_factor=1):
    roi_proposals = []
    boxes = np.array(boxes)

    
    roi_ar = roi_width / roi_height

    if len(boxes) > 1:
        boxes = non_max_suppression_fast(boxes)

    # boxes = merge_near_boxes(boxes)

    for box in boxes:
        width = box[2] - box[0]
        height = box[3] - box[1]
        if width < roi_width and height < roi_height:
            new_roi = (roi_width, roi_height)

        elif width < height:
            # Resize with same aspect ratio the default roi
            aspect = width / height
            new_roi = [
                width * (roi_ar/aspect),
                height
            ]
        else:
            aspect = width / height
            new_roi = [
                width,
                height * (aspect/roi_ar),
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



def compute_dilation(fgMask):
    kernel = np.ones((5, 5), np.uint8)
    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image. We use a 5x5 kernel.
    # Note: Dilation operation (opposite to erosion). A pixel element is '1'
    #   if at least one pixel under the kernel is '1'. So it increases 
    #   the white region in the image or size of foreground object increases.
    dilation = cv.dilate(fgMask, kernel, iterations=2)
    return dilation

def find_cnts(dilation):
    min_area_contour = 5000

    cnts = cv.findContours(
        dilation.copy(), 
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )
    
    cnts = imutils.grab_contours(cnts)
    
    boxes = []
    areas = []
    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        contourArea = cv.contourArea(c)
        if cv.contourArea(c) < min_area_contour:
            continue

        (x, y, w, h) = cv.boundingRect(c)
        boxes.append([x, y, x+w, y+h])
        areas.append(contourArea)

    if len(boxes) >= 1:
        max_height, max_width, _ = frame.shape
        boxes = propose_rois(boxes,
                             roi_width=16,
                             roi_height=16,
                             max_width=dilation.shape[0],
                             max_height=dilation.shape[1])


    return boxes



parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()
if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()
capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
if not capture.isOpened:
    print('Unable to open: ' + args.input)
    exit(0)

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    ts0 = time.time()
    fgMask = backSub.apply(frame)
    ts1 = time.time()
    
    # dil_ts0 = time.time()
    # dilation = compute_dilation(fgMask)
    # dil_ts1 = time.time()

    # cnt_ts0 = time.time()
    # boxes = find_cnts(fgMask)
    # cnt_ts1 = time.time()
    # for roi in boxes:
    #     cv.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 2))

    frame = cv.resize(frame, (800, 600))
    fgMask = cv.resize(fgMask, (800, 600))
    
    cv.rectangle(frame, (10, 2), (140,75), (255,255,255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    cv.putText(frame, f'Bg: {ts1-ts0:.3f} sec.', (15, 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    # cv.putText(frame, f'Dil: {dil_ts1-dil_ts0:.3f} sec.', (15, 45),
    #            cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    # cv.putText(frame, f'Cnts: {cnt_ts1-cnt_ts0:.3f} sec.', (15, 60),
    #            cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    
    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)
    cv.imshow('BG Image', cv.resize(backSub.getBackgroundImage(), (800, 600)))
    # cv.imshow('Dilation', cv.resize(dilation, (800, 600)))
    
    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
