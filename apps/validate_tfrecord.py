# import the necessary packages
import argparse
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image
from six import BytesIO
from tqdm import tqdm

# Tensorflow
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import cv2
import imutils

sys.path.append('../')
# Auxiliary functions
from utils.datasets import MSCOCO
from evaluate_saved_model import inputs, load_pbtxt


def main():
    # construct the argument parser and parse the arguments
    args = argparse.ArgumentParser()

    args. add_argument("-d", "--dataset", nargs='+', default=None, help="Path to the dataset to evaluate.")
    args.add_argument("-l", "--label-map", default=None, help="Label map for the model")
    args.add_argument("--min-score", type=float, default=0, help="minimum score for detections")
    args.add_argument("--csv", default=None, help="Get detections from csv instead of dataset")

    config = args.parse_args()
    min_score = config.min_score

    img_id = 0
    batch_size = 1

    # Checks
    all_same_resolution = True
    all_normalized = True

    prev_shape = None
    for images, shapes, gt_labels, gt_boxes in inputs(config.dataset): 
        images = images.numpy()
        shapes = shapes.numpy() 
        gt_labels = gt_labels.numpy()
        gt_boxes = gt_boxes.numpy()

        for batch_id in range(batch_size):
            img = images[batch_id]
            shapes = shapes[batch_id]
            gt_label = gt_labels[batch_id]
            gt_box = gt_boxes[batch_id]
            gt_scores = [1 for _ in gt_label]
        
            if prev_shape is None:
                prev_shape = shapes
                print(shapes)
            if shapes[0] != prev_shape[0] or shapes[1] != prev_shape[1]:
                print(f'[WARNING] Image {img_id} has resolution {shapes} and previous was {prev_shape}')
                all_same_resolution = False
                sys.exit()
            prev_shape = shapes

            if any([any([coord > 1 for coord in box])
                    for box in gt_box]):
                print(f'[ERROR] Image {img_id} contains bbox coordinates not normalized: {gt_box}')
                sys.exit()
                all_normalized = False
            img_id += 1

    if all_same_resolution and all_normalized:
        print('PASSED')
    else:
        print('ERROR Found')

if __name__ == "__main__":
    main()
