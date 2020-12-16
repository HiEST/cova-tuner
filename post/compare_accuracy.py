#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import sys
import time

import cv2
import numpy as np
import pandas as pd

sys.path.append('../')
from utils.datasets import MSCOCO as label_map


# def compare_models_accuracy(detectA, detectB):
    # for each detection of A, 
    # check if there is intersection over union
    # with any detection of A 
    # if so, check if classes match (true positive)
    # or differ (false positive)


def draw_detections(frame, detections, min_score, max_boxes, color_box):
    if len(detections) == 0:
        return

    boxes_drawn = 0
    for i, det in detections.iterrows():
        score = det['score']
        if score > min_score and boxes_drawn < max_boxes:
            boxes_drawn += 1
            class_id = int(det['class_id'])
            class_name = label_map[str(class_id)]['name']
            print(f'score for class {class_name}: {score}')
            (left, right, top, bottom) = det[['xmin', 'xmax', 'ymin', 'ymax']].values 

            display_str = "{}: {}%".format(class_name, int(100 * score))

            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), color_box, 2)
            cv2.putText(frame, display_str, (int(left), int(top)+20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color_box, 2)



def main():
    args = argparse.ArgumentParser()
    args.add_argument("-v", "--video",
                      default=None,
                      type=str,
                      help="Path to the dataset to process.")
    args.add_argument("-t", "--threshold", default=0.5, type=float, help="score threshold")
    args.add_argument("-f", "--fps", default=25, type=float, help="play fps")
    args.add_argument("--frame-skipping", default=5, type=float, help="play fps")

    args.add_argument('modelA')
    args.add_argument('modelB')

    config = args.parse_args()

    modelA = pd.read_pickle(config.modelA, compression='bz2')
    modelB = pd.read_pickle(config.modelB, compression='bz2')

    print(modelA.head())
    print(modelB.head())

    frame_ids = modelA.frame.unique()

    # Open video and process each frame even if no detections
    if config.video is not None:
        cap = cv2.VideoCapture(config.video)
        ret, frame = cap.read()

        frame_ids = np.arange(0, max(frame_ids))
        frame_lat = 1.0 / config.fps
        last_frame = -1
        min_score = config.threshold
        max_boxes = 10

        colorA = (0, 255, 0)
        colorB = (0, 0, 255)

    for frame_id in frame_ids:
        detectA = modelA[modelA.frame == frame_id]
        detectB = modelB[modelB.frame == frame_id]

        # metrics = compare_models_accuracy(detectA, detectB)

        if config.video is not None:
            if frame_id % config.frame_skipping == 0:
                draw_detections(frame, detectA, min_score, max_boxes, colorA)
                draw_detections(frame, detectB, min_score, max_boxes, colorB)

                frame = cv2.resize(frame, (1280, 768))
                cv2.putText(frame, f'frame: {frame_id}', (frame.shape[0]-50, frame.shape[1]-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                cv2.imshow('Detections', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    sys.exit()

                while time.time() - last_frame < frame_lat:
                    time.sleep(time.time() - last_frame)
                last_frame = time.time()

            ret, frame = cap.read()
            assert ret

if __name__ == "__main__":
    main()
