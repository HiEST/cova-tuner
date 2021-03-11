# import the necessary packages
import argparse
from functools import partial
import json
import os
from pathlib import Path
import shlex
import subprocess as sp
import sys

import numpy as np
import pandas as pd
from PIL import Image
from six import BytesIO
from tqdm import tqdm

# Tensorflow
import tensorflow as tf
import tensorflow_hub as hub

from object_detection.builders import model_builder
from object_detection.utils import config_util, dataset_util

import cv2
import imutils

sys.path.append('../')
# Auxiliary functions
from utils.datasets import MSCOCO
from apps.evaluate_saved_model import load_model, load_pbtxt, inputs


def main():
    # construct the argument parser and parse the arguments
    args = argparse.ArgumentParser()

    args. add_argument("-d", "--dataset", default=None, help="Path to the dataset to evaluate.")
    args. add_argument("-o", "--output", default=None, help="Path to the output dir.")
    args. add_argument("--csv", default=None, help="Path to the output dir.")

    # Detection/Classification
    args.add_argument("-m", "--model", default=None, help="Model for image classification")
    args.add_argument("-l", "--label-map", default=None, help="Label map for the model")
    args.add_argument("--min-score", type=float, default=0, help="minimum score for detections")
    
    args.add_argument("--show", action="store_true", default=False, help="show detections")

    config = args.parse_args()
    min_score = config.min_score

    if config.label_map is None:
        label_map = MSCOCO
    else:
        label_map = load_pbtxt(config.label_map)

    detector = None
    if config.model is not None:
        detector = load_model(config.model)
    
    if config.csv is None:
        annotate(detector, label_map, config.dataset, config.output, config.min_score)
    else:
        annotate_from_csv(
                config.dataset, config.csv,
                label_map, config.output, config.min_score)


def annotate_from_csv(tfrecord, csv_file, label_map, output_dir, min_score=0.5):
    writer = tf.compat.v1.python_io.TFRecordWriter(f'{output_dir}/{Path(tfrecord).stem}.record')
    detections = pd.read_csv(csv_file)
    detections = detections[detections.score >= min_score]

    img_id = 0
    for img, img_shape, gt_label, gt_box in inputs(tfrecord): 
        encoded_img = tf.image.encode_jpeg(img).numpy()
        frame_dets = detections[detections['frame'] == img_id]
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        labels = []
        classes = []
        for row in frame_dets.iterrows():
            xmins.append(row['xmin'])
            xmaxs.append(row['xmax'])
            ymins.append(row['ymin'])
            ymaxs.append(row['ymax'])
            label = label_map[row['class_id']]['name']
            labels.append(label)
            classes.append(row['class_id'])

        tf_example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'image/height': dataset_util.int64_feature(height),
                        'image/width': dataset_util.int64_feature(width),
                        'image/encoded': dataset_util.bytes_feature(encoded_img),
                        'image/format': dataset_util.bytes_feature(b'jpg'),
                        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
                        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
                        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
                        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
                        'image/object/class/text': dataset_util.bytes_list_feature(labels),
                        'image/object/class/label': dataset_util.int64_list_feature(classes),
                }))

        writer.write(tf_example.SerializeToString())
        img_id += 1


def annotate(detector, label_map, tfrecord, output_dir, min_score=0.5):
    writer = tf.compat.v1.python_io.TFRecordWriter(f'{output_dir}/{Path(tfrecord).stem}.record')
    img_id = 0
    detections = []
    for img, img_shape, gt_label, gt_box in inputs(tfrecord): 
        encoded_img = tf.image.encode_jpeg(img).numpy()
        img = img.numpy()
        height, width, _ = img.shape 
        gt_label = gt_label.numpy()
        gt_box = gt_box.numpy()
        
        results = detector.detect(img) 
        boxes = results['detection_boxes'][0]
        scores = results['detection_scores'][0]
        class_ids = results['detection_classes'][0]

        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        labels = []
        classes = []
        for i in range(len(boxes)):
            if scores[i] >= min_score:
                ymin, xmin, ymax, xmax = tuple(boxes[i])
                (xmin, xmax, ymin, ymax) = (
                        int(xmin * img.shape[1]), 
                        int(xmax * img.shape[1]),
                        int(ymin * img.shape[0]), 
                        int(ymax * img.shape[0])
                    )
                det = [img_id, int(class_ids[i]), scores[i], 
                      xmin, ymin, xmax, ymax]
                detections.append(det)
                xmins.append(xmin/width)
                xmaxs.append(xmax/width)
                ymins.append(ymin/height)
                ymaxs.append(ymax/height)
                classes.append(int(class_ids[i]))
                labels.append(label_map[int(class_ids[i])]['name'].encode('utf8'))
                
        tf_example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'image/height': dataset_util.int64_feature(height),
                        'image/width': dataset_util.int64_feature(width),
                        'image/encoded': dataset_util.bytes_feature(encoded_img),
                        'image/format': dataset_util.bytes_feature(b'jpg'),
                        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
                        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
                        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
                        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
                        'image/object/class/text': dataset_util.bytes_list_feature(labels),
                        'image/object/class/label': dataset_util.int64_list_feature(classes),
                }))

        writer.write(tf_example.SerializeToString())

        img_id += 1

    columns = ['frame', 'class_id', 'score',
                'xmin', 'ymin', 'xmax', 'ymax']

    detections = pd.DataFrame(detections, columns=columns)
    detections.to_csv(f'{output_dir}/{Path(tfrecord).stem}_annotations.csv', sep=',', index=False)


if __name__ == "__main__":
    main()
