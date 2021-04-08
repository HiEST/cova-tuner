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

    if config.label_map is None:
        label_map = MSCOCO
    else:
        label_map = load_pbtxt(config.label_map)

    if config.csv is not None:
        detections = pd.read_csv(config.csv)

    img_id = 0
    batch_size = 1
    nms = False
    iou_threshold = 0.5
    # import pdb; pdb.set_trace()
    for images, shapes, gt_labels, gt_boxes in inputs(config.dataset): 
        print(f'img: {img_id}')
        images = images.numpy()
        gt_labels = gt_labels.numpy()
        gt_boxes = gt_boxes.numpy()

        for batch_id in range(batch_size):
            img = images[batch_id]
            gt_label = gt_labels[batch_id]
            gt_box = gt_boxes[batch_id]
            gt_scores = [1 for _ in gt_label]
        
            if config.csv is not None:
                height, width, _ = img.shape
                det_frame = detections[detections.frame == img_id]
                gt_label = [d['class_id'] for _,d in det_frame.iterrows()]
                gt_box = [[
                        d['ymin'],
                        d['xmin'],
                        d['ymax'],
                        d['xmax']]
                        for _,d in det_frame.iterrows()]

                for box in gt_box:
                    if any([c > 1 for c in box]):
                        print(box)
                if any([any([c > 1 for c in box]) for box in gt_box]):
                    gt_box = [
                        [box[0]/height,
                         box[1]/width,
                         box[2]/height,
                         box[3]/width]
                        for box in gt_box]
                gt_scores = [d['score'] for _,d in det_frame.iterrows()]

            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            while True:
                img = img_bgr.copy()
            
                if nms:
                    selected_indices = tf.image.non_max_suppression(
                        boxes=gt_box, scores=gt_scores, 
                        max_output_size=100,
                        iou_threshold=iou_threshold,
                        score_threshold=min_score)
                    boxes = tf.gather(gt_box, selected_indices).numpy()
                    labels = tf.gather(gt_label, selected_indices).numpy()
                    scores = tf.gather(gt_scores, selected_indices).numpy()
                else:
                    boxes = gt_box
                    labels = gt_label
                    scores = gt_scores

                for box, label, score in zip(boxes, labels, scores):
                    ymin, xmin, ymax, xmax = tuple(box)
                    if any([coord > 1 for coord in box]):
                        assert False
                        ymin, xmin, ymax, xmax = [int(coord) for coord in box]
                    else:
                        (xmin, xmax, ymin, ymax) = (
                                int(xmin * img.shape[1]), 
                                int(xmax * img.shape[1]),
                                int(ymin * img.shape[0]), 
                                int(ymax * img.shape[0])
                            )

                    cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 1)
                    cv2.putText(img, f'{label_map[int(label)]["name"]}: {score:.2f}%', (int(xmin), int(ymin)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    # print(f'{label}: ({xmin}, {ymin}), ({xmax}, {ymax})')

                cv2.imshow("Detections", img)

                key = cv2.waitKey(0) & 0xFF
                if key == ord("q"):
                    sys.exit()
                    break
                elif key == ord("a"):
                    min_score = min(min_score+0.05, 1)
                    print(f'min_score: {min_score}')
                elif key == ord("s"):
                    min_score = max(min_score-0.05, 0)
                    print(f'min_score: {min_score}')
                elif key == ord("z"):
                    iou_threshold = min(iou_threshold+0.05, 1)
                    print(f'iou_threshold: {iou_threshold}')
                elif key == ord("x"):
                    iou_threshold = min(iou_threshold-0.05, 1)
                    print(f'iou_threshold: {iou_threshold}')
                elif key == ord("n"):
                    nms = not nms
                elif key == ord("c"):
                    break
            else:
                break

            img_id += 1


if __name__ == "__main__":
    main()
