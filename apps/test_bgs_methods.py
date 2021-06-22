#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import cv2 as cv
import argparse
import os
from pathlib import Path
import time
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm
import imutils

from edge_autotune.motion.motion_detector import non_max_suppression_fast, Background, BackgroundMethod, \
                                                GaussianBlur, merge_overlapping_boxes, resize_if_smaller
from edge_autotune.dnn.metrics import compute_area_match

parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--video', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
parser.add_argument('--gt', type=str, help='Path to ground-truth.')
parser.add_argument('--show', default=False, action='store_true', help='Show window with results.')
parser.add_argument('--hybrid', default=False, action='store_true', help='Use hybrid (MOG+Diff) method.')

args = parser.parse_args()
if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()
capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.video))

total_frames = capture.get(cv.CAP_PROP_FRAME_COUNT)
if total_frames < 1000:
    sys.exit()

# video_width = capture.get(cv.CAP_PROP_FRAME_WIDTH)
# video_height = capture.get(cv.CAP_PROP_FRAME_HEIGHT)

video_width = 1280
video_height = 720

backMean = Background(BackgroundMethod.ACUM_MEAN, use_last=10, skip=50)
backMOG = cv.createBackgroundSubtractorMOG2()


def draw_cnts(frame, mask, color, min_area=500):
    cnts = cv.findContours(
            mask.copy(), 
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE
        )

    cnts = imutils.grab_contours(cnts)
    boxes = []
    for c in cnts:
        contourArea = cv.contourArea(c)
        if cv.contourArea(c) < min_area:
            continue

        (x, y, w, h) = cv.boundingRect(c)
        box = [x, y, x+w, y+h]
        box = resize_if_smaller(box, min_size=(32,32), max_dims=frame.shape[:2])
        boxes.append(box)
    
    # boxes = non_max_suppression_fast(np.array(boxes))
    boxes = merge_overlapping_boxes(boxes)
    for box in boxes:
        x1, y1, x2, y2 = box
        cv.rectangle(frame, (x1, y1), (x2, y2), color, 1)

    return boxes

def get_mask(background, frame, kernel):
    current_gray = GaussianBlur(frame)
    current_delta = cv.absdiff(background, current_gray)
    current_threshold = cv.threshold(
                                current_delta, 
                                25, 
                                255, 
                                cv.THRESH_BINARY)[1]
    
    current_threshold = cv.dilate(current_threshold, kernel, iterations=2)
    return current_threshold


# def compute_iou(boxA: list, boxB: list):

# 	# determine the (x, y)-coordinates of the intersection rectangle
# 	xA = max(boxA[0], boxB[0])
# 	yA = max(boxA[1], boxB[1])
# 	xB = min(boxA[2], boxB[2])
# 	yB = min(boxA[3], boxB[3])

# 	# compute the area of intersection rectangle
# 	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	
#     # compute the area of both the prediction and ground-truth
# 	# rectangles
# 	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
# 	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	
#     # compute the intersection over union by taking the intersection
# 	# area and dividing it by the sum of prediction + ground-truth
# 	# areas - the intersection area
# 	iou = interArea / float(boxAArea + boxBArea - interArea)
	
#     # return the intersection over union value
# 	return iou, interArea


# def compute_area_match(boxes, gt_boxes):
#     boxes_areas = [(box[2]-box[0])*(box[3]-box[1]) for box in boxes]
#     gt_areas = [(box[2]-box[0])*(box[3]-box[1]) for box in gt_boxes]
#     boxes_area = sum(boxes_areas)
#     gt_area = sum(gt_areas)

#     results = []
#     for iou in [0.3, 0.5]:
#         intersection_area = 0
#         area_in_matches = 0
#         matches = 0
#         avg_iou = []
#         for gt_id, gt in enumerate(gt_boxes):
#             intersections = [compute_iou(gt, box) for box in boxes]
#             overlap_iou = [i[0] for i in intersections if i[0] > 0]
#             overlap_area = [i[1] for i in intersections]
#             # overlap = np.where(np.array(intersections[0]) > 0.01)[0]
#             # not_overlap = np.where(np.array(intersections[0]) <= 0.01)[0]

#             intersection_area += sum(overlap_area)
#             matches += len([i for i in overlap_iou if i > iou])
#             assert len([i for i in overlap_iou if i > iou]) <= 1
#             area_in_matches += sum([i[1] for i in intersections if i[0] > iou])
#             if len(overlap_iou):
#                 avg_iou.append(sum(overlap_iou)/len(overlap_iou))

#         results.append({
#             'iou': iou,
#             'intersection_area': intersection_area,
#             'matches': matches,
#             'area_in_matches': area_in_matches,
#             'avg_iou': 0 if not len(avg_iou) else sum(avg_iou)/len(avg_iou)
#         })

#     results = {
#         'gt_area': gt_area,
#         'boxes_area': boxes_area,
#         'results': [r for r in results]
#     }

#     return results


def read_virat(fn):
    annotations = pd.read_csv(fn, header=None, sep=' ', index_col=False)
    annotations.columns = ['object_id', 'object_duration', 'current_frame', 
                            'xmin', 'ymin', 'width', 'height', 'object_type']

    annotations = annotations[annotations.object_type > 0]
    annotations['xmax'] = annotations['xmin'] + annotations['width']
    annotations['ymax'] = annotations['ymin'] + annotations['height']
    object_labels = ['person', 'car', 'vehicle', 'object', 'bike']
    annotations['label'] = annotations['object_type'].apply(lambda obj: object_labels[obj-1])
    annotations = annotations[annotations.label != 'object']
    annotations = annotations[annotations.label != 'bike']
    return annotations


if not capture.isOpened():
    print('Unable to open: ' + args.video)
    exit(0)


annotations_fn = os.path.join(Path(args.video).parent, '../annotations', Path(args.video).stem + '.viratdata.objects.txt')
df = read_virat(annotations_fn)

kernel_mog = np.ones((2, 2), np.uint8)
kernel_mean = np.ones((5, 5), np.uint8)

avg_mog = 1
avg_mean = 1
avg_hybrid = 1
acum_mog = 0
acum_mean = 0
acum_hybrid = 0
frame_id = 0
bg_mog = None

results = []
columns = ['frame_id', 'num_objects', 'area_objects', 'method', 'proposed_area', 'proposed_regions',
            'iou', 'intersection_area', 'num_matches', 'num_misses', 'avg_iou_matches', 'avg_iou_misses', 'latency']
df_rois = []
columns_rois = ['frame_id', 'method', 'xmin', 'ymin', 'xmax', 'ymax', 'label']

pbar = tqdm(total=int(total_frames), desc=str(Path(args.video).stem))
while frame_id < total_frames:
    pbar.update(1)
    frame_objs = df[df.current_frame == frame_id][['xmin', 'ymin', 'xmax', 'ymax']].values

    if not len(frame_objs):
        frame_id += 1
        continue
    # frame_objs = merge_overlapping_boxes(frame_objs)
    
    ret, frame = capture.read()
    if not ret:
        break
    frame = cv.resize(frame, (video_width, video_height))
    
    timings = {}

    t0 = time.time()
    fgMask = backSub.apply(frame)
    fgMask_dilated = cv.dilate(fgMask, kernel_mog, iterations=2)
    tmog = time.time() - t0
    acum_mog += tmog
    timings['mog'] = tmog

    t0 = time.time()
    bg = backMean.update(frame)
    mask = get_mask(bg, frame, kernel_mean)
    tmean = time.time() - t0
    acum_mean += tmean
    timings['mean'] = tmean

    t0 = time.time()
    if frame_id % 5 == 0:
        fgMaskMOG = backMOG.apply(frame)

    if frame_id == 250:
            avg_hybrid = 1
            acum_hybrid = 0

    if frame_id >= 250 and frame_id % 50 == 0:
        bg_MOG = backMOG.getBackgroundImage()
        bg_MOG = GaussianBlur(bg_MOG)
    elif frame_id < 250:
        bg_MOG = bg
    
    mask_hybrid = get_mask(bg_MOG, frame, kernel_mean)
    thybrid = time.time() - t0
    acum_hybrid += thybrid
    timings['hybrid'] = thybrid

    proposed_boxes = {}
    proposed_boxes['mog'] = draw_cnts(frame, fgMask_dilated, color=(0, 255, 0))
    proposed_boxes['mean'] = draw_cnts(frame, mask, (255, 0, 0))
    proposed_boxes['hybrid'] = draw_cnts(frame, mask_hybrid, (0, 0, 255))

    for method, boxes in proposed_boxes.items():
        for box in boxes:
            df_rois.append([
                frame_id,
                method,
                box[0]/float(video_width),
                box[1]/float(video_height),
                box[2]/float(video_width),
                box[3]/float(video_height),
            ])
            # assert all([b >= 0 and b <= 1 for b in df_rois[-1][2:]])

    for obj_id, obj in df[df.current_frame == frame_id].iterrows():
        (xmin, ymin, xmax, ymax, label) = obj[['xmin', 'ymin', 'xmax', 'ymax', 'label']].values
        df_rois.append([
            frame_id,
            'gt',
            xmin/float(video_width),
            ymin/float(video_height),
            xmax/float(video_width),
            ymax/float(video_height),
            label,
        ])
        # assert all([b >= 0 and b <= 1 for b in df_rois[-1][2:]])


    results_frame = {}
    for method, boxes in proposed_boxes.items():
        if len(boxes) == 0:
            continue
        results_frame[method] = compute_area_match(proposed_boxes[method], frame_objs)

    for method, r in results_frame.items():
        for r_iou in r['results']:
            results.append([
                frame_id,
                len(frame_objs), # num_objects
                r['gt_area'], # area_objects
                method,
                r['boxes_area'], # proposed_area
                len(proposed_boxes[method]), # proposed_regions
                r_iou['iou'], # iou threshold
                r_iou['intersection_area'], # intersection_area
                r_iou['matches'], # num_matches
                r_iou['misses'],
                r_iou['avg_iou_matches'],
                r_iou['avg_iou_misses'],
                timings[method]])
    
    frame_id += 1
    if frame_id % 30 == 0:
        avg_mog = acum_mog / frame_id
        avg_mean = acum_mean / frame_id
        avg_hybrid = acum_hybrid / frame_id

    if frame_id % 100 == 0:
        print(f'avg_mog: {avg_mog*1000:.2f} ms.')
        print(f'avg_mean: {avg_mean*1000:.2f} ms.')
        print(f'avg_hybrid: {avg_hybrid*1000:.2f} ms.')
        
    if args.show:
        cv.rectangle(frame, (10, 2), (300,90), (255,255,255), -1)
        cv.putText(frame, f'{int(frame_id)}/{int(total_frames)}', (15, 15),
                cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
        cv.putText(frame, f'MOG2: {1/avg_mog:.2f} fps ({avg_mog*1000:.2f}ms.)', (15, 30),
                cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
        cv.putText(frame, f'Mean: {1/avg_mean:.2f} fps ({avg_mean*1000:.2f}ms.)', (15, 45),
                cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
        cv.putText(frame, f'Hybrid: {1/avg_hybrid:.2f} fps ({avg_hybrid*1000:.2f}ms.)', (15, 60),
                cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

        cv.imshow('FG Mask', fgMask)
        cv.imshow('Background Mean', backMean.background_color)
        cv.imshow('Background MOG', backSub.getBackgroundImage())
        cv.imshow('Background Hybrid', backSub.getBackgroundImage())
        cv.imshow('Dilated MOG', fgMask_dilated)
        cv.imshow('Dilated Mean', mask)
        cv.imshow('Dilated Hybrid', mask_hybrid)
        cv.imshow('Dilated Hybrid', mask_hybrid)
        cv.imshow('Frame', frame)
        
        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break

df = pd.DataFrame(results, columns=columns)
df.to_csv(f'{Path(args.video).stem}_perf.csv', index=False, sep=',')

df_rois = pd.DataFrame(df_rois, columns=columns_rois)
df_rois.to_csv(f'{Path(args.video).stem}_rois.csv', index=False, sep=',')
