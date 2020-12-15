#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

sys.path.append('../../')
from utils.datasets import MSCOCO as label_map


def main():
    args = argparse.ArgumentParser()
    args.add_argument("-v", "--video", default=None, type=str, help="Video file to annotate")
    args.add_argument("-d", "--detections", default=None, nargs='+', help="Pickle file with ground truth to annotate")
    args.add_argument("-t", "--threshold", default=0.5, type=float, help="score threshold")
    args.add_argument("--images", default="images/", type=str, help="path to save train/test images")
    args.add_argument("--csv", default="data/", type=str, help="path to save csv")
    args.add_argument("--validation", default=0.2, type=float, help="Ratio of images that are saved for validation")
    args.add_argument("-c", "--classes", default=None, nargs='+', help="Subset of classes to annotate. The rest are ignored")

    config = args.parse_args()

    if not os.path.exists(config.images):
        os.makedirs(config.images)
    os.makedirs('{}/train'.format(config.images), exist_ok=True)
    os.makedirs('{}/test'.format(config.images), exist_ok=True)
    os.makedirs('{}'.format(config.csv), exist_ok=True)

    results = []
    for r in config.detections:
        results.append(pd.read_pickle(r, "bz2"))

    cap = cv2.VideoCapture(config.video)
    ret, frame = cap.read()

    max_boxes = 10
    min_score = config.threshold
    frame_id = 0

    image_format = 'jpg'

    columns = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    annotations = []
    while ret:
        for r in results:
            detections = r[r.frame == frame_id]
            if len(detections) > 0:
                boxes_saved = 0
                for i, det in detections.iterrows():
                    score = det['score']
                    if score > min_score and boxes_saved < max_boxes:
                        class_name = label_map[str(int(det['class_id']))]['name']
                        if config.classes is not None and class_name not in config.classes:
                            continue

                        boxes_saved += 1
                        # Get bbox info + frame info
                        (xmin, xmax, ymin, ymax) = det[['xmin', 'xmax', 'ymin', 'ymax']].values 
                        (width, height, _) = frame.shape
                        # New class id: 0 is reserved for background
                        # class_id = det['class_id'] if config.classes is None else config.classes.index(class_name)+1
                        # Filename 
                        filename = '{}.{}'.format(frame_id, image_format)
                        annotations.append([filename, width, height, class_name, xmin, ymin, xmax, ymax])

                        cv2.imwrite('{}/{}'.format(config.images, filename), frame)

        if frame_id == 100:
            break


        ret, frame = cap.read()
        frame_id += 1

    df = pd.DataFrame(annotations, columns=columns)
    all_images = df.filename.unique()
    train_imgs, test_imgs = train_test_split(all_images, test_size=config.validation)

    train_dataset = df[df.filename.isin(train_imgs)]
    test_dataset = df[df.filename.isin(test_imgs)]

    print(train_dataset)
    print(test_dataset)
    for img in train_dataset.filename.unique():
        shutil.move('{}/{}'.format(config.images, img),
                    '{}/train/'.format(config.images))
    for img in test_dataset.filename.unique():
        shutil.move('{}/{}'.format(config.images, img),
                    '{}/test/'.format(config.images))

    train_dataset['filename'] = train_dataset['filename'].apply(lambda x: '{}/train/{}'.format(config.images, x))
    test_dataset['filename'] = test_dataset['filename'].apply(lambda x: '{}/test/{}'.format(config.images, x))
    train_dataset.to_csv('{}/train_label.csv'.format(config.csv), index=None)
    test_dataset.to_csv('{}/test_label.csv'.format(config.csv), index=None)

if __name__ == "__main__":
    main()
