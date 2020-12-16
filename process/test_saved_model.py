#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import trange

sys.path.append('../')
from utils.detector import run_detector


def main():
    # construct the argument parser and parse the arguments
    args = argparse.ArgumentParser()

    # Required
    args.add_argument("-v", "--video", default=None, required=True, help="path to the video file")
    args.add_argument("-m", "--model-dir", default=None, required=True, help="Directory containing saved model")
    # args.add_argument("-l", "--labels", default=None, required=True, help="Path to label_map.pbtxt")

    # Detection/Classification
    args.add_argument("--min-score", type=float, default=0.6, help="minimum score for detections")
    args.add_argument("--max-boxes", type=float, default=10, help="maximumim number of bounding boxes per frame")
    
    # Application control
    args.add_argument("--max-frames", type=int, default=0, help="maximum frames to process")
    args.add_argument("--skip-frames", type=int, default=0, help="number of frames to skip for each frame processed")

    # Save results
    args.add_argument("-o", "--output", default=None, help="Path to where results are saved in pickle+bz2 format")
    
    config = args.parse_args()

    max_boxes = config.max_boxes
    min_score = config.min_score

    detector = tf.saved_model.load(config.model_dir)

    if '.pkl' in config.output:
        pklfile = config.output
    else:
        pklfile = f'{output}/{Path(config.video).stem}.pkl'

    label_map = {
        1: {
            'name': 'car',
            'id': 1
        },
        2: {
            'name': 'person',
            'id': 2
        },
        3: {
            'name': 'traffic light',
            'id': 3
        }
    }

    cap = cv2.VideoCapture(config.video)
    ret, frame = cap.read()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_id = 0
    columns = ['frame', 'class_id', 'label', 'score', 'xmin', 'ymin', 'xmax', 'ymax']
    detections = []
    # while ret:
    for _ in trange(total_frames):
        if config.skip_frames > 0:
            if config.skip_frames > frames_skipped:
                frames_skipped += 1
                continue
            else:
                frames_skipped = 0

        results = run_detector(detector, frame) 
        boxes = results['detection_boxes'][0]
        scores = results['detection_scores'][0]
        class_ids = results['detection_classes'][0]

        # selected_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold=0.3)
        # selected_boxes = np.array(tf.gather(boxes, selected_indices))

        # boxes = selected_boxes
        # scores = np.array(tf.gather(scores, selected_indices))
        # class_ids = np.array(tf.gather(class_ids, selected_indices))
       
        for i in range(min(boxes.shape[0], max_boxes)):
            ymin, xmin, ymax, xmax = tuple(boxes[i])
                
            (left, right, top, bottom) = (
                xmin * frame.shape[1], 
                xmax * frame.shape[1],
                ymin * frame.shape[0], 
                ymax * frame.shape[0]
            )

            score = scores[i]
            class_id = int(class_ids[i])
            class_name = label_map[class_id]['name']
            detections.append([frame_id, class_id, class_name, score, int(left), int(top), int(right), int(bottom)])

        frame_id += 1
        if config.max_frames > 0 and frame_id >= config.max_frames:
            break

        ret, frame = cap.read()
        if not ret:
            break

    print(f'Finished after {frame_id} frames')
    if ret:
        print('Finished but still got more frames')

    detections = pd.DataFrame(detections, columns=columns)
    detections.to_pickle(pklfile, compression='bz2')


if __name__ == '__main__':
    main()
