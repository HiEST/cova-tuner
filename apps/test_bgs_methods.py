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
parser.add_argument('-v', '--video', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
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

video_width = capture.get(cv.CAP_PROP_FRAME_WIDTH)
video_height = capture.get(cv.CAP_PROP_FRAME_HEIGHT)

resize_bg_width = 1280
resize_bg_height = 720

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


if not capture.isOpened():
    print('Unable to open: ' + args.video)
    exit(0)


# annotations_fn = os.path.join(Path(args.video).parent, '../annotations', Path(args.video).stem + '.viratdata.objects.txt')
# df = read_virat(annotations_fn)
df = pd.read_csv(os.path.join('annotations', f'{Path(args.video).stem}.no-static.csv'))
df = df[df['static_object'] == False].copy().reset_index(drop=True)
frames_with_objects = df['frame_id']

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

# pbar = tqdm(total=int(total_frames), desc=str(Path(args.video).stem))
for frame_id in range(int(total_frames)):
    # pbar.update(1)
    
    ret, frame = capture.read()
    if not ret:
        break
    frame = cv.resize(frame, (resize_bg_width, resize_bg_height))
    
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
        bg_MOG = bg.copy()
    
    try:
        mask_hybrid = get_mask(bg_MOG, frame, kernel_mean)
    except Exception as e:
        print(f'frame_id: {frame_id} - video: {Path(args.video).stem}')
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
                box[0]/float(resize_bg_width),
                box[1]/float(resize_bg_height),
                box[2]/float(resize_bg_width),
                box[3]/float(resize_bg_height),
            ])
            # assert all([b >= 0 and b <= 1 for b in df_rois[-1][2:]])

    frame_objs = df[df.frame_id == frame_id][['xmin', 'ymin', 'xmax', 'ymax']].values

    for obj_id, obj in df[df.frame_id == frame_id].iterrows():
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

    # if frame_id % 100 == 0:
    #     print(f'avg_mog: {avg_mog*1000:.2f} ms.')
    #     print(f'avg_mean: {avg_mean*1000:.2f} ms.')
    #     print(f'avg_hybrid: {avg_hybrid*1000:.2f} ms.')
        
    if args.show:
        if (frame_id+1) % 30 == 0:
            avg_mog = acum_mog / frame_id
            avg_mean = acum_mean / frame_id
            avg_hybrid = acum_hybrid / frame_id

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
df.to_csv(f'bgs/{Path(args.video).stem}_perf.csv', index=False, sep=',')

df_rois = pd.DataFrame(df_rois, columns=columns_rois)
df_rois.to_csv(f'bgs/{Path(args.video).stem}_rois.csv', index=False, sep=',')
