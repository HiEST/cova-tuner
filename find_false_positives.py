#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import pandas as pd

from utils.datasets import MSCOCO as label_map

COLORS = [
    (0, 255, 0),
    (0, 255, 255),
    (0, 0, 255)
]


def detect_false_positives(detections, min_score=0.5, max_frames=25, labels=None):
    pd.options.mode.chained_assignment = None  # default='warn'

    df = detections[detections.score >= min_score]
    if labels is not None:
        class_ids = [c['id'] for c in label_map.values() if c['name'] in labels]

        df = df[df['class_id'].isin(class_ids)]
        df['label'] = df['class_id'].apply(lambda x: label_map[x]['name'])
    else:
        df['label'] = df['class_id'].apply(lambda x: label_map[x]['name'])
        labels = df.label.unique()

    print(labels)
    df['FP'] = True
    for l in labels:
        df_ = df[df.label == l]
        frames_with_detections = df_.frame.unique()
        print(f'{len(frames_with_detections)} frames with detections of class {l}')

        print(f'False Positives for class {l}:')
        for i, frame_id in enumerate(frames_with_detections[:-1]):
            next_detection = frames_with_detections[i+1]
            if next_detection - frame_id <= max_frames:
                df.loc[((df.frame == frame_id) & (df.label == l)), 'FP'] = False
                df.loc[((df.frame == frame_id+1) & (df.label == l)), 'FP'] = False
            
    return df #[df.FP == True]

def main():
    args = argparse.ArgumentParser()
    args.add_argument("-v", "--video", default=None, help="path to the dir with videos")
    args.add_argument("-r", "--results", default=None, help="dir ground truth")
    args.add_argument("-t", "--threshold", default=0.5, type=float, help="score threshold")
    args.add_argument("-m", "--max-frames", default=25, type=int, help="score threshold")
    args.add_argument("-f", "--fps", default=100, type=float, help="play fps")
    args.add_argument("-b", "--background", default=None, help="path to the background file")
    args.add_argument("-l", "--labels", default=None, nargs='+', help="path to the background file")

    config = args.parse_args()

    results = pd.read_pickle('{}/{}.pkl'.format(config.results, Path(config.video).stem), "bz2")
    FP = detect_false_positives(
                results,
                min_score=config.threshold,
                max_frames=config.max_frames,
                labels=config.labels)
    print(f'Found {len(FP[FP.FP == True])} potential False Positives.')
    if len(FP[FP == False]) < len(FP[FP == True]):
        stop_on_true_positive = True
    else:
        stop_on_true_positive = False

    cap = cv2.VideoCapture(config.video)
    ret, frame = cap.read()

    max_boxes = 10
    min_score = config.threshold
    frame_id = 0

    frame_lat = 1.0 / config.fps
    last_frame = time.time()
    draw_FP = False
    while ret:
        detections = FP[FP.frame == frame_id]
        if len(detections) > 0:
            boxes_drawn = 0
            for i, det in detections.iterrows():
                score = det['score']
                if score > min_score and boxes_drawn < max_boxes:
                    boxes_drawn += 1
                    class_id = int(det['class_id'])
                    class_name = label_map[class_id]['name']
                    print(f'score for class {class_name}: {score} on frame {frame_id}')
                    (left, right, top, bottom) = det[['xmin', 'xmax', 'ymin', 'ymax']].values 

                    display_str = "{}: {}%".format(class_name, int(100 * score))

                    if det['FP']:
                        cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), COLORS[2], 2)
                        cv2.putText(frame, display_str, (int(left), int(top)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS[2], 2)
                        draw_FP = True
                    else:
                        cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), COLORS[0], 2)
                        cv2.putText(frame, display_str, (int(left), int(top)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS[0], 2)
                        if stop_on_true_positive:
                            draw_FP = True


        frame = cv2.resize(frame, (1280, 768))
        cv2.putText(frame, f'frame: {frame_id}', (10, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        cv2.imshow('Detections', frame)

        if draw_FP:
            cv2.waitKeyEx(0)
            draw_FP = False

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            sys.exit()

        while time.time() - last_frame < frame_lat:
            time.sleep(time.time() - last_frame)
        last_frame = time.time()

        ret, frame = cap.read()
        frame_id += 1


if __name__ == "__main__":
    main()
