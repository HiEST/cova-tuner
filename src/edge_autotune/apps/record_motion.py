#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import tensorflow as tf
import torch
import torchvision
from torchvision import transforms
from torchvision.models import mobilenet_v2, resnet18, resnet50, resnet101
from torchvision.models.detection import FasterRCNN, fasterrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import pandas as pd
 
sys.path.append('../')
from utils.datasets import IMAGENET, MSCOCO
from utils.detector import init_detector, run_detector
from utils.motion_detection import Background, MotionDetection
from utils.nms import non_max_suppression_fast
from dnn.utils import load_pbtxt
from evaluate_saved_model import load_model

class Recorder():
    def __init__(self, path, name, format_str='jpg', frame_skip=0):
        self.path = path
        self.name = name
        self.id = 0
        self.frame_skip = frame_skip
        self.format = format_str
        
        if not os.path.exists(path):
            os.makedirs(self.path)

    def record(self, img, force=False):
        if self.id >= 88:
            sys.exit()
        output = f'{self.path}/{self.name}_{self.id}.{self.format}'

        self.id += 1
        if self.frame_skip == 0 or (self.id % self.frame_skip) == 0 or force:
            cv2.imwrite(output, img)
        return True


def scale_roi(roi, factor, frame_shape):
    roi_width = roi[2] - roi[0]
    roi_height = roi[3] - roi[1]

    new_width = roi_width * factor
    new_height = roi_height * factor
    
    diff_width = new_width - roi_width
    diff_height = new_height - roi_height

    new_roi = [
        max(roi[0] - int(diff_width/2), 0),
        max(roi[1] - int(diff_height/2), 0),
        min(roi[2] + int(diff_width/2), frame_shape[1]),
        min(roi[3] + int(diff_height/2), frame_shape[0])
    ]
    return new_roi


def main():
    args = argparse.ArgumentParser()

    # App's I/O
    args.add_argument("-v", "--video", required=True, default=None, help="path to the video file")
    args.add_argument("-o", "--output", default="/tmp/images", help="path to the output dir")


    # Object Detection
    args.add_argument("-m", "--model", default=None, help="Model for object detection")
    args.add_argument("-l", "--label-map", default=None, help="pbtxt file for the label map")
    args.add_argument("-i", "--input-size", type=int, nargs='+', default=None, help="Model's input size")
    args.add_argument("-t", "--threshold", type=float, default=0.5, help="Confidence threshold to accept predictions")
    args.add_argument("--no-infer", help="Skip inference.", action="store_true")
    args.add_argument("--rgb", action="store_true", help="Convert images to RGB before inference")

    # Motion Detection
    args.add_argument("--first-pass-bg", action="store_true", help="Give a first pass to the video to get a more accuracte background")
    args.add_argument("--no-merge-rois", action="store_true", help="Don't merge ROIs on scene")
    args.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
    args.add_argument("--no-average", help="use always first frame as background.", action="store_true")
    args.add_argument("--min-roi-size", type=int, nargs='+', default=(1,1), help="Model's input size")
    args.add_argument("--scale-roi", type=float, default=1, help="Factor to scale ROI")
    args.add_argument("--save-bg", type=str, default=None, help="Save background as image to recover it later.")
    args.add_argument("--load-bg", type=str, default=None, help="Path to background to load.")

    # App control
    args.add_argument("--jump-to", type=int, default=0, help="Jumpt to frame")
    args.add_argument("--no-show", help="Do not show results.", action="store_true")
    args.add_argument("--debug", help="Show debug info.", action="store_true")

    args.add_argument("--frame-skip", type=int, default=0, help="Frame skipping")

    config = args.parse_args()

    background = Background(
        no_average=config.no_average,
        skip=10,
        take=10,
        use_last=15,
    )

    bgrec = Recorder(config.output, "background", frame_skip=20)
    compute_bg = False
    if config.first_pass_bg:
        cap = cv2.VideoCapture(config.video)
        ret, frame = cap.read()
        frame_id = 0
        while ret:
            background.update(frame)
            bgrec.record(background.background_color)
            if not config.no_show:
                bg = cv2.resize(background.background.copy(), (1280, 768))
                cv2.putText(bg, f'frame: {frame_id}', (10, 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.imshow('Current Background', bg)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    sys.exit()
            ret, frame = cap.read()
            frame_id += 1
        cv2.destroyAllWindows()
        cap.release()
        background.freeze() 

        if config.save_bg is not None:
            bg = background.background.copy()
            cv2.imwrite(config.save_bg, bg)
    elif config.load_bg is not None:
        bg = cv2.imread(config.load_bg, cv2.IMREAD_UNCHANGED)
        background.background = bg.copy()
        background.freeze()
    elif config.no_average:
        cap = cv2.VideoCapture(config.video)
        cap.set(1, config.jump_to)
        ret, frame = cap.read()
        background.update(frame)
        cap.release()
        background.freeze()
        bgrec.record(background.background_color, force=True)
    else:
        compute_bg = True

    input_size = config.input_size
    if input_size is None:
        input_size = (320,320)

    motionDetector = MotionDetection(
        background=background,
        min_area_contour=config.min_area,
        roi_size=config.min_roi_size,
        merge_rois=(not config.no_merge_rois),
    )

    if config.no_infer or config.model is None:
        no_infer = True
    else:
        no_infer = False
        detector = load_model(config.model)
        if config.label_map is None:
            label_map = MSCOCO
        else:
            label_map = load_pbtxt(config.label_map)

    cap = cv2.VideoCapture(config.video)
    cap.set(1, config.jump_to)
    ret, frame = cap.read()
    max_boxes = 10
    min_score = config.threshold
    frame_id = 0

    framerec = Recorder(config.output, "frame", frame_skip=config.frame_skip)
    detrec = Recorder(config.output, "detections", frame_skip=config.frame_skip)
    fullrec = Recorder(config.output, "fulldets", frame_skip=config.frame_skip)
    motionrec = Recorder(config.output, "motion", frame_skip=config.frame_skip)
    thrrec = Recorder(config.output, "threshold", frame_skip=config.frame_skip)
    deltarec = Recorder(config.output, "delta", frame_skip=config.frame_skip)
    grayrec = Recorder(config.output, "gray", frame_skip=config.frame_skip)
    recorders = [
        framerec,
        fullrec,
        detrec,
        motionrec,
        thrrec,
        deltarec,
        grayrec
    ]
    to_record = []

    df_full = []
    df_motion = []
    df_columns = ['frame', 'label', 'score']

    while ret:
        if compute_bg:
            background.update(frame)
            bgrec.record(background.background_color)

        if config.frame_skip > 0 and frame_id % config.frame_skip != 0:
            frame_id += 1
            ret, frame = cap.read()
            continue

        bg_ts0 = time.time()
        motion_boxes, areas = motionDetector.detect(frame)
        bg_ts1 = time.time()
        infer_ts = 0
            
        motion_frame = frame.copy()
        detect_frame = frame.copy()
        full_detect = frame.copy()

        if len(motion_boxes):
            original_frame = frame.copy()
            for roi_id, roi in enumerate(motion_boxes):
                if areas[roi_id] < 2*config.min_area:
                    continue

                roi = scale_roi(roi, config.scale_roi, frame.shape)
                cropped_roi = np.array(original_frame[roi[1]:roi[3], roi[0]:roi[2]])

                infer_ts0 = time.time()
                if no_infer:
                    boxes = []
                    scores = []
                    class_ids = []
                else:
                    if config.rgb:
                        cropped_roi = cv2.cvtColor(cropped_roi, cv2.COLOR_BGR2RGB)
                    preds = run_detector(detector, cropped_roi, input_size=input_size) 
                    boxes = preds['detection_boxes'][0]
                    scores = preds['detection_scores'][0]
                    class_ids = preds['detection_classes'][0]

                    selected_indices = tf.image.non_max_suppression(
                        boxes=boxes,
                        scores=scores,
                        max_output_size=10,
                        iou_threshold=0.5)
                    selected_boxes = np.array(tf.gather(boxes, selected_indices))
                    boxes = selected_boxes
                    scores = np.array(tf.gather(scores, selected_indices))
                    class_ids = np.array(tf.gather(class_ids, selected_indices))

                infer_ts1 = time.time()
                infer_ts += infer_ts1 - infer_ts0
                                   
                cv2.rectangle(motion_frame, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)
                cv2.putText(motion_frame, 'Motion Detected', (roi[0], roi[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.putText(motion_frame, str(areas[roi_id]), (roi[0], roi[3]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                for i in range(min(len(boxes), max_boxes)):
                    class_id = int(class_ids[i])
                    class_name = label_map[class_id]['name']
                    if scores[i] >= min_score:
                        ymin, xmin, ymax, xmax = tuple(boxes[i])
                        (left, right, top, bottom) = (roi[0] + xmin * cropped_roi.shape[1], roi[0] + xmax * cropped_roi.shape[1],
                                                      roi[1] + ymin * cropped_roi.shape[0], roi[1] + ymax * cropped_roi.shape[0])
                        
                        label = label_map[class_id]['name']

                        display_str = "{}: {:.2f}%".format(label, int(100 * scores[i]))
                        cv2.rectangle(detect_frame, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), 2)
                        cv2.putText(detect_frame, display_str, (int(left), int(top)-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        df_motion.append([frame_id, label, scores[i]])

            if no_infer:
                boxes = []
                scores = []
                class_ids = []
            else:
                full_detect_ = full_detect.copy()
                if config.rgb:
                    full_detect_ = cv2.cvtColor(full_detect_, cv2.COLOR_BGR2RGB)
                preds = run_detector(detector, full_detect_, input_size=input_size) 
                boxes = preds['detection_boxes'][0]
                scores = preds['detection_scores'][0]
                class_ids = preds['detection_classes'][0]

                selected_indices = tf.image.non_max_suppression(
                    boxes=boxes,
                    scores=scores,
                    max_output_size=10,
                    iou_threshold=0.5)
                selected_boxes = np.array(tf.gather(boxes, selected_indices))
                boxes = selected_boxes
                scores = np.array(tf.gather(scores, selected_indices))
                class_ids = np.array(tf.gather(class_ids, selected_indices))

                for i in range(min(len(boxes), max_boxes)):
                    class_id = int(class_ids[i])
                    class_name = label_map[class_id]['name']
                    if scores[i] >= min_score/2:
                        ymin, xmin, ymax, xmax = tuple(boxes[i])
                        (left, right, top, bottom) = (xmin * full_detect.shape[1],xmax * full_detect.shape[1],
                                                      ymin * full_detect.shape[0],ymax * full_detect.shape[0])
                        
                        label = label_map[class_id]['name']

                        display_str = "{}: {:.2f}%".format(label, int(100 * scores[i]))
                        cv2.rectangle(full_detect, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), 2)
                        cv2.putText(full_detect, display_str, (int(left), int(top)-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                        df_full.append([frame_id, label, scores[i]])


            threshold = motionDetector.current_threshold.copy()
            delta = motionDetector.current_delta.copy()
            gray = motionDetector.current_gray.copy()

            to_record = [
                frame.copy(),
                full_detect if not no_infer else None,
                detect_frame if not no_infer else None,
                motion_frame,
                threshold,
                delta,
                gray
            ]
            
            for rec, img in zip(recorders, to_record):
                if img is None:
                    continue
                
                if rec.id == 23:
                   import pdb; pdb.set_trace()
                img = cv2.resize(img, (640, 380))
                rec.record(img)

            if not config.no_show:
                if not no_infer:
                    frame = cv2.resize(detect_frame, (1280, 768))

                    cv2.rectangle(frame, (10, 2), (140,60), (255,255,255), -1)
                    cv2.putText(frame, str(cap.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
                    cv2.putText(frame, f'Bg: {bg_ts1-bg_ts0:.3f} sec.', (15, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
                    cv2.putText(frame, f'Infer: {infer_ts:.3f} sec.', (15, 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
                    cv2.imshow('Detections', frame)
                
                    full_detect = cv2.resize(full_detect, (800, 600))
                    detect_frame = cv2.resize(detect_frame, (800, 600))
                    cv2.imshow('Full Detect', full_detect)
                    cv2.imshow('Motion Detect', detect_frame)

                motion_frame = cv2.resize(motion_frame, (800, 600))
                threshold = cv2.resize(threshold, (800, 600))
                delta = cv2.resize(delta, (800, 600))
                gray = cv2.resize(gray, (800, 600))
                cv2.imshow('Motion ROIs', motion_frame)
                cv2.imshow('Threshold', threshold)
                cv2.imshow('Delta', delta)
                cv2.imshow('Gray', gray)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    sys.exit()

        ret, frame = cap.read()
        frame_id += 1

        df1 = pd.DataFrame(df_full, columns=df_columns)
        df2 = pd.DataFrame(df_motion, columns=df_columns)
        df1.to_csv(f'{config.output}/full.csv', sep=',', index=False)
        df2.to_csv(f'{config.output}/motion.csv', sep=',', index=False)


if __name__ == "__main__":
    main()
