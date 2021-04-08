# import the necessary packages
import sys
import time
import random
import argparse
from os.path import isfile

import numpy as np
import pandas as pd

import imutils
import cv2

# Tensorflow
import tensorflow as tf
import tensorflow_hub as hub

# Auxiliary functions
sys.path.append('../')
from utils.motion_detection import MotionDetection, Background
from utils.detector import init_detector, run_detector, detect_and_draw, label_map
# from utils.capture_screen import CaptureScreen
from utils.constants import *
from utils.datasets import MSCOCO


COLORS = [
    (0, 255, 0),
    (0, 255, 255),
    (0, 0, 255)
]


def load_pbtxt(filename):
    label_map = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    label = ''
    for l in lines:
        if 'name' in l:
            label = l.split('"')[1]
        elif 'id' in l:
            class_id = int(l.split(':')[1])
            label_map[class_id] = {
                'name': label,
                'id': class_id
            }

    return label_map


def pause():
    cv2.waitKey(0)

def load_saved_model(model):
    return tf.saved_model.load(model)

def main():
    # construct the argument parser and parse the arguments
    args = argparse.ArgumentParser()

    # Required
    args.add_argument("-i", "--image", nargs='+', default=None, help="path to the video file")

    # Detection/Classification
    args.add_argument("-m", "--model", default="resnet", help="Model for image classification")
    args.add_argument("-l", "--label-map", default=None, help="Label map for the model")
    args.add_argument("--min-score", type=float, default=0.6, help="minimum score for detections")
    args.add_argument("--max-boxes", type=float, default=10, help="maximumim number of bounding boxes per frame")
    
    config = args.parse_args()

    max_boxes = config.max_boxes
    min_score = config.min_score

    detector = None
    saved_model = False
    if 'saved_model' in config.model:
        if config.label_map is None:
            label_map_model = MSCOCO
        else:
            label_map_model = load_pbtxt(config.label_map)
        saved_model = True
        detector = load_saved_model(config.model)
    else:
        label_map_model = MSCOCO
        detector = init_detector('Faster R-CNN Inception ResNet V2 1024x1024')
    print(f'Detector initialized.')

    for image in config.image:
        frame = cv2.imread(image)
        start_time = time.time()

        results = run_detector(detector, frame) 
        boxes = results['detection_boxes'][0]
        scores = results['detection_scores'][0]
        class_ids = results['detection_classes'][0]

        for i in range(min(boxes.shape[0], max_boxes)):
            
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            (left, right, top, bottom) = (
                xmin * frame.shape[1], 
                xmax * frame.shape[1],
                ymin * frame.shape[0], 
                ymax * frame.shape[0]
            )

            if scores[i] >= min_score:
                class_id = int(class_ids[i])
                display_str = "{}: {}%".format(label_map_model[class_id]['name'],
                                                int(100 * scores[i]))

                
                cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), 1)
                cv2.putText(frame, display_str, (int(left), int(top)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)

        end_time = time.time() 

        cv2.putText(frame, f'Infer: {end_time-start_time:.3f} sec.', (frame.shape[1]-120, frame.shape[0]-40),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

        frame = imutils.resize(frame, height=1024) 
        cv2.imshow("Detections", frame)
            

        key = cv2.waitKey(0)
    # if key == ord("q"):
    #     break


if __name__ == "__main__":
    # pr = cProfile.Profile()
    # pr.enable()
    main()
    # pr.disable()
    # ps = pstats.Stats(pr).print_stats()
    # pr.print_stats(sort='time')
    # pr.dump_stats('main.profile')
