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

try:
    from object_detection.builders import model_builder
    from object_detection.utils import config_util, dataset_util
except Exception:
    print('No module object_detection.')
    pass

import cv2
import imutils

sys.path.append('../')
# Auxiliary functions
from utils.datasets import MSCOCO
from apps.evaluate_saved_model import load_model, load_pbtxt, inputs, evaluate


def main():
    # construct the argument parser and parse the arguments
    args = argparse.ArgumentParser()

    args.add_argument("-d", "--dataset", default=None, help="Path to the dataset to evaluate.")
    args.add_argument("-o", "--output", default=None, help="Path to the output dir.")
    args.add_argument("--csv", default=None, help="Path to the output dir.")

    # Detection/Classification
    args.add_argument("-m", "--model", default=None, help="Model for image classification")
    args.add_argument("-l", "--label-map", default=None, help="Label map for the model")
    args.add_argument("--min-score", type=float, default=0, help="minimum score for detections")
    args.add_argument("--sliding-window", action="store_true", default=False, help="Annotate using sliding window")
    args.add_argument("--roi-size", nargs='+', default=None, help="Size of the ROI to crop images")

    args.add_argument("--img-size", nargs='+', default=None, help="Size of the images in the output tfrecord")
    
    args.add_argument("--show", action="store_true", default=False, help="show detections")

    config = args.parse_args()
    min_score = config.min_score
    input_csv = config.csv

    img_size = None
    if config.img_size is not None:
        assert len(config.img_size) == 2
        img_size = [int(config.img_size[0]), int(config.img_size[1])]

    if config.label_map is None:
        label_map = MSCOCO
    else:
        label_map = load_pbtxt(config.label_map)

    detector = None
    if config.model is not None:
        detector = load_model(config.model)
   
    if config.sliding_window:
        input_csv = f'{config.output}/detections.csv'
        print(f'Writing detections to {input_csv}')
        detections, _ = evaluate(
                detector=detector, label_map=label_map,
                dataset=config.dataset,
                output_dir=config.output,
                min_score=min_score,
                debug=True,
                show=config.show,
                single_inference=False,
                roi_size=config.roi_size)

    if input_csv is not None:
        annotate_from_csv(
                tfrecord=config.dataset,
                csv_file=input_csv,
                label_map=label_map,
                output_dir=config.output,
                min_score=config.min_score,
                img_size=img_size)
    else:
        if detector is None:
            if min_score < 1:
                print(f'[ERROR] Model not specified and confidence threshold below 1 (groundtruth)')
                sys.exit()
            elif img_size is None:
                print(f'[ERROR] If model is not specified, specify img_size to resize.')
                sys.exit()
        
        annotate(
                detector=detector,
                label_map=label_map,
                tfrecord=config.dataset,
                output_dir=config.output,
                min_score=config.min_score,
                img_size=img_size)


def annotate_from_csv(tfrecord, csv_file, label_map, output_dir, img_size=None, min_score=0.5, skip_empty_imgs=True):
    record_path = f'{output_dir}/{Path(tfrecord).stem}.record'
    
    detections = pd.read_csv(csv_file)
    detections = detections[detections.score >= min_score]
    detections.to_csv(f'{output_dir}/{Path(tfrecord).stem}_annotations.csv', sep=',', index=False)

    # if os.path.isfile(record_path):
    #     return

    writer = tf.compat.v1.python_io.TFRecordWriter(record_path)

    img_id = 0
    for img, img_shape, gt_label, gt_box in inputs(tfrecord): 
        img = tf.squeeze(img)
        encoded_img = tf.image.encode_jpeg(img).numpy()
        
        # img_ = Image.open(BytesIO(encoded_img))
        # img_.save(f'{output_dir}/img_{img_id}-1.jpg')
        frame_dets = detections[detections['frame'] == img_id]
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        labels = []
        classes = []
        height, width, _ = img.shape 

        # import pdb; pdb.set_trace()
        for _, row in frame_dets.iterrows():
            ymins.append(row['ymin']/height)
            xmins.append(row['xmin']/width)
            ymaxs.append(row['ymax']/height)
            xmaxs.append(row['xmax']/width)
            label = label_map[row['class_id']]['name']
            labels.append(label.encode('utf-8'))
            classes.append(int(row['class_id']))

            if  any([coord < 0 or coord > 1 for coord in [ymins[-1], xmins[-1], ymaxs[-1], xmaxs[-1]]]):
                print(f'ymin: {ymins[-1]} - {row["ymin"]} - {height}')
                print(f'xmin: {xmins[-1]} - {row["xmin"]} - {width}')
                print(f'ymax: {ymaxs[-1]} - {row["ymax"]} - {height}')
                print(f'xmax: {xmaxs[-1]} - {row["xmax"]} - {width}')
                assert False

        if skip_empty_imgs and len(frame_dets) == 0:
            img_id += 1
            continue

        if img_size is not None:
            width, height = img_size
            img_ = Image.open(BytesIO(encoded_img))
            img_ = img_.resize(img_size)
            encoded_img = BytesIO()
            img_.save(encoded_img, 'JPEG')
            # import pdb; pdb.set_trace()
            encoded_img.seek(0)
            encoded_img = encoded_img.read()
            # with open(f'{output_dir}/img_{img_id}.jpg', 'wb') as f:
            #     f.write(encoded_img)

        # img_ = Image.open(BytesIO(encoded_img))
        # img_.save(f'{output_dir}/img_{img_id}-2.jpg')
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


def annotate(detector, label_map, tfrecord, output_dir, min_score=0.5, img_size=None):
    writer = tf.compat.v1.python_io.TFRecordWriter(f'{output_dir}/{Path(tfrecord).stem}.record')
    img_id = 0
    detections = []
    for img, img_shape, gt_label, gt_box in inputs(tfrecord): 
        if detector is not None:
            results = detector.detect(img) 

            boxes = results['detection_boxes'][0]
            scores = results['detection_scores'][0]
            class_ids = results['detection_classes'][0]
        else:
            boxes = gt_box.numpy()[0]
            class_ids = gt_label.numpy()[0]
            scores = [1 for _ in boxes]

        img = tf.squeeze(img)
        encoded_img = tf.image.encode_jpeg(img).numpy()
        img = img.numpy()
        height, width, _ = img.shape 
        
        if img_size is not None:
            width, height = img_size
            img_ = Image.open(BytesIO(encoded_img))
            img_ = img_.resize(img_size)
            encoded_img = BytesIO()
            img_.save(encoded_img, 'JPEG')
            encoded_img.seek(0)
            encoded_img = encoded_img.read()

        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        labels = []
        classes = []
        for i in range(len(boxes)):
            if scores[i] >= min_score:
                ymin, xmin, ymax, xmax = tuple(boxes[i])
                # (xmin, xmax, ymin, ymax) = (
                #         int(xmin * img.shape[1]), 
                #         int(xmax * img.shape[1]),
                #         int(ymin * img.shape[0]), 
                #         int(ymax * img.shape[0])
                #     )
                det = [img_id, int(class_ids[i]), scores[i], 
                      xmin, ymin, xmax, ymax]
                detections.append(det)
                xmins.append(xmin)
                xmaxs.append(xmax)
                ymins.append(ymin)
                ymaxs.append(ymax)
                # xmins.append(xmin/width)
                # xmaxs.append(xmax/width)
                # ymins.append(ymin/height)
                # ymaxs.append(ymax/height)
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
