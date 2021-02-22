#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
from pathlib import Path
import sys
import time

import cv2
import imutils
import numpy as np
import pandas as pd

sys.path.append('../')
from utils.datasets import MSCOCO as label_map

COLORS = [
    (0, 255, 0),
    (0, 255, 255),
    (0, 0, 255)
]

def main():
    args = argparse.ArgumentParser()
    args.add_argument("-v", "--videos", required=True, nargs='+', default=None, help="path to the video file")
    args.add_argument("-o", "--output", default=None, help="path save new detections.")
    args.add_argument("-d", "--detections", default=None, help="ground truth")
    args.add_argument("-t", "--threshold", default=0.5, type=float, help="score threshold")
    args.add_argument("-f", "--fps", default=25, type=float, help="play fps")
    args.add_argument("-c", "--classes", nargs='+', default=None, help="valid classes")


    config = args.parse_args()


    max_boxes = 10
    min_score = config.threshold
    frame_id = 0

    frame_lat = 1.0 / config.fps
    last_frame = time.time()

    TP = []
    FP = []
    for video_id, video in enumerate(config.videos):
        cap = cv2.VideoCapture(video)
        ret, frame = cap.read()

        results = pd.read_pickle('{}/{}.pkl'.format(config.detections, Path(video).stem), "bz2")
        while ret:
            detections = results[results.frame == frame_id]
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
                        if class_name not in config.classes:
                            continue
                        print(f'score for class {class_name}: {score}')
                        (left, right, top, bottom) = det[['xmin', 'xmax', 'ymin', 'ymax']].values 

                        display_str = "{}: {}%".format(class_name, int(100 * score))

                        cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), COLORS[0], 2)
                        cv2.putText(frame, display_str, (int(left), int(top)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS[0], 2)

                        obj = frame[int(top):int(bottom), int(left):int(right)].copy()
                        obj = imutils.resize(obj, width=600)
                        mini_frame = imutils.resize(frame, width=800)
                        cv2.imshow(f'Mini Frame', mini_frame)
                        cv2.imshow(f'{class_name}', obj)
                        k = cv2.waitKey(0)
                        if k == 9: # TAB
                            print(f'{class_name} was a false positive')
                            FP.append(det)
                        elif k == 32: # Space
                            print(f'{class_name} was a true positive')
                            TP.append(det)

                        cv2.destroyWindow(f'{class_name}')

            frame = cv2.resize(frame, (1280, 768))
            cv2.putText(frame, f'frame: {frame_id}', (10, 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            cv2.imshow('Detections', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
                sys.exit()

            while time.time() - last_frame < frame_lat:
                time.sleep(time.time() - last_frame)
            last_frame = time.time()

            ret, frame = cap.read()
            frame_id += 1

    TP = pd.DataFrame(TP, columns=results.columns)
    FP = pd.DataFrame(FP, columns=results.columns)

    print(TP)
    print(FP)

if __name__ == "__main__":
    main()
