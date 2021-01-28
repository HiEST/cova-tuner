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

def main():
    args = argparse.ArgumentParser()
    args.add_argument("-v", "--video", default=None, help="path to the video file")
    args.add_argument("-r", "--results", default=None, nargs='+', help="ground truth")
    args.add_argument("-t", "--threshold", default=0.5, type=float, help="score threshold")
    args.add_argument("-f", "--fps", default=25, type=float, help="play fps")

    config = args.parse_args()

    results = []
    for r in config.results:
        results.append(pd.read_pickle(r, "bz2"))

    print(len(results))
    cap = cv2.VideoCapture(config.video)
    ret, frame = cap.read()

    max_boxes = 10
    min_score = config.threshold
    frame_id = 0

    frame_lat = 1.0 / config.fps
    last_frame = time.time()
    while ret:
        for r_id, r in enumerate(results):
            detections = r[r.frame == frame_id]
            if len(detections) > 0:
                boxes_drawn = 0
                for i, det in detections.iterrows():
                    score = det['score']
                    if score > min_score and boxes_drawn < max_boxes:
                        boxes_drawn += 1
                        # if boxes[i] not in boxes_nms:
                        #     continue
                        class_id = int(det['class_id'])
                        class_name = label_map[class_id]['name']
                        print(f'score for class {class_name}: {score}')
                        (left, right, top, bottom) = det[['xmin', 'xmax', 'ymin', 'ymax']].values 

                        display_str = "{}: {}%".format(class_name, int(100 * score))

                        cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), COLORS[r_id], 2)
                        cv2.putText(frame, display_str, (int(left), int(top)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS[r_id], 2)


        frame = cv2.resize(frame, (1280, 768))
        cv2.putText(frame, f'frame: {frame_id}', (10, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        for r_id, _ in enumerate(results):
            cv2.putText(frame, f'Results {r_id}', (10, 30+15*r_id),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[r_id], 2)

        cv2.imshow('Detections', frame)

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
