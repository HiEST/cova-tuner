#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import cv2
import pandas as pd


COLORS = [
    (0, 255, 0),
    (0, 255, 255),
    (0, 0, 255)
]


def main():
    # construct the argument parser and parse the arguments
    args = argparse.ArgumentParser()

    # Required
    args.add_argument("-i", "--image", default=None, required=True, help="path to image or frame number")
    args.add_argument("-c", "--csv", default=None, required=True, nargs='+', help="path to the video file")

    args.add_argument("-v", "--video", default=None, type=str, help="path to the video file")

    # Detection/Classification
    args.add_argument("--min-score", type=float, default=0.6, help="minimum score for detections")
    args.add_argument("--max-boxes", type=float, default=10, help="maximumim number of bounding boxes per frame")
    
    # Save results
    args.add_argument("-o", "--output", default=None, help="Path to where annotated image is saved")
    
    config = args.parse_args()

    max_boxes = config.max_boxes
    min_score = config.min_score

    if config.video is not None:
        frame = int(config.image)
        cap = cv2.VideoCapture(config.video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, frame = cap.read()
        cap.release()
        img = frame.copy()
    else:
        img = cv2.imread(config.image)

    header = ['label', 'score', 'xmin', 'ymin', 'xmax', 'ymax']
    header_gt = ['label', 'xmin', 'ymin', 'xmax', 'ymax']
    for csv_id, csvfile in enumerate(config.csv):
        print(f'[{csv_id}] {csvfile}')
        csv = pd.read_csv(csvfile, header=None, sep=' ', index_col=False)
        if len(csv.columns) == len(header):
            csv.columns = header
        else:
            csv.columns = header_gt
        
        for _, det in csv.iterrows():
            if 'score' in det.keys():
                score = det['score']
            else:
                score = 1

            if score < min_score:
                continue

            xmin, ymin, xmax, ymax = det[['xmin', 'ymin', 'xmax', 'ymax']]
            label = det['label']
                
            label_str = '{} ({:.2f}%)'.format(label, score*100)
            cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), COLORS[csv_id], 2)
            cv2.putText(img, label_str, (int(xmin), int(ymin)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS[csv_id], 2)

    cv2.imwrite(config.output, img)


if __name__ == '__main__':
    main()
