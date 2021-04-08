#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from six import BytesIO
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange

import tensorflow as tf
from object_detection.utils import config_util, dataset_util

sys.path.append('../')
from dnn.utils import load_pbtxt
from utils.datasets import MSCOCO
from utils.motion_detection import Background, MotionDetection
from evaluate_saved_model import load_model


def annotate(
        detector,
        label_map,
        tfrecord,
        imgs_dir,
        min_score=0.5,
        img_size=None,
        motion_detection=True,
        min_area=1000,
        min_roi_size=(256,256), 
        show=True):

    if motion_detection:
        background = Background()
        bg = cv2.imread(f'{imgs_dir}/background.bmp')
        if bg.shape[2] == 3:
            bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        background.background = bg
        background.freeze()

        motionDetector = MotionDetection(
            background=background,
            min_area_contour=min_area,
            roi_size=min_roi_size,
            merge_rois=True,
        )

    writer = tf.compat.v1.python_io.TFRecordWriter(tfrecord)

    detections = []
    images = [str(img) for img in Path(imgs_dir).glob('*.jpg')]
    # import pdb; pdb.set_trace()
    for img_id, img_fn in enumerate(images): 
        
        img = cv2.imread(img_fn)
        height, width, _ = img.shape 
        if motion_detection:
            motion_boxes, areas = motionDetector.detect(img)
            if len(motion_boxes):
                xmin, ymin, xmax, ymax = [
                        min([box[0] for box in motion_boxes]),
                        min([box[1] for box in motion_boxes]),
                        max([box[2] for box in motion_boxes]),
                        max([box[3] for box in motion_boxes])
                ]

                roi_width = xmax - xmin
                roi_height = ymax - ymin
                if roi_width < roi_height:
                    diff = roi_height - roi_width
                    new_xmin = max(xmin - int(diff/2), 0)
                    new_xmax = min(xmax + int(diff/2), img.shape[1])
                    if new_xmin == 0: # we hit left border
                        left = int(diff/2) - (xmin-new_xmin)
                        new_xmax = min(new_xmax + left, img.shape[1])
                    elif new_xmax == img.shape[1]: # we hit left border
                        left = int(diff/2) - (new_xmax-xmax)
                        new_xmin = max(new_xmin - left, 0)
                    # if roi_width < roi_height * 0.99:
                    xmin = new_xmin
                    xmax = new_xmax

                elif roi_height < roi_width:
                    diff = roi_width - roi_height
                    new_ymin = max(ymin - int(diff/2), 0)
                    new_ymax = min(ymax + int(diff/2), img.shape[0])
                    if new_ymin == 0: # we hit top border
                        left = int(diff/2) - (ymin-new_ymin)
                        new_ymax = min(new_ymax + left, img.shape[0])
                    elif new_ymax == img.shape[0]: # we hit left border
                        left = int(diff/2) - (new_ymax-ymax)
                        new_ymin = max(new_ymin - left, 0)
                    ymin = new_ymin
                    ymax = new_ymax

                # import pdb; pdb.set_trace()
                roi = img[ymin:ymax, xmin:xmax].copy()
                input_img = tf.image.convert_image_dtype(roi, tf.uint8)[tf.newaxis, ...]
                input_tensor = input_img # tf.convert_to_tensor(input_img.numpy(), dtype=tf.float32)
                results = detector.detect(input_tensor) 

                boxes = results['detection_boxes'][0]
                scores = results['detection_scores'][0]
                class_ids = results['detection_classes'][0]
             
                selected_indices = tf.image.non_max_suppression(
                        boxes=boxes, scores=scores, 
                        max_output_size=100,
                        iou_threshold=0.5,
                    score_threshold=max(min_score, 0.05))
                boxes = tf.gather(boxes, selected_indices).numpy()
                scores = tf.gather(scores, selected_indices).numpy()
                class_ids = tf.gather(class_ids, selected_indices).numpy()
           
                boxes_img = []
                for box in boxes:
                    ymin_det, xmin_det, ymax_det, xmax_det = box
                    ymin_det, xmin_det, ymax_det, xmax_det = (
                            (ymin_det * roi.shape[0] + ymin)/height,
                            (xmin_det * roi.shape[1] + xmin)/width,
                            (ymax_det * roi.shape[0] + ymin)/height,
                            (xmax_det * roi.shape[1] + xmin)/width
                    )

                    box_img = [ymin_det, xmin_det, ymax_det, xmax_det]
                    if not all([c <= 1 for c in box_img]) and \
                            all([c >= 0 for c in box_img]):
                        import pdb; pdb.set_trace()

                    boxes_img.append(box_img)
                boxes = boxes_img
 
        else:
            input_img = tf.image.convert_image_dtype(img, tf.uint8)[tf.newaxis, ...]
            input_tensor = tf.convert_to_tensor(input_img.numpy(), dtype=tf.float32)
            results = detector.detect(input_tensor) 

            boxes = results['detection_boxes'][0]
            scores = results['detection_scores'][0]
            class_ids = results['detection_classes'][0]

            selected_indices = tf.image.non_max_suppression(
                    boxes=boxes, scores=scores, 
                    max_output_size=100,
                    iou_threshold=0.5,
                    score_threshold=max(min_score, 0.05))
            boxes = tf.gather(boxes, selected_indices).numpy()
            scores = tf.gather(scores, selected_indices).numpy()
            class_ids = tf.gather(class_ids, selected_indices).numpy()
        # if img_size is not None:
        #     width, height = img_size
        #     img = img.resize(img_size)

        encoded_img = BytesIO()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil.save(encoded_img, 'JPEG')
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
                det = [img_id, img_fn, int(class_ids[i]), scores[i], 
                      xmin, ymin, xmax, ymax]
                detections.append(det)
                xmins.append(xmin)
                xmaxs.append(xmax)
                ymins.append(ymin)
                ymaxs.append(ymax)
                classes.append(int(class_ids[i]))
                label = label_map[int(class_ids[i])]['name']
                labels.append(label.encode('utf8'))

                if show and scores[i] > 0.3:
                    xmin_abs = int(xmin*width)
                    xmax_abs = int(xmax*width)
                    ymin_abs = int(ymin*height)
                    ymax_abs = int(ymax*height)
                    cv2.rectangle(img, (xmin_abs, ymin_abs), (xmax_abs, ymax_abs), (255, 0, 0), 1)
                    cv2.putText(img, f'{label}: {scores[i]*100:.2f}%', (xmin_abs, ymin_abs-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            

        if show:
            cv2.imwrite(f'{imgs_dir}/{img_id}-box_{i}.jpg', img)

                
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

    columns = ['frame', 'filename', 'class_id', 'score',
                'xmin', 'ymin', 'xmax', 'ymax']

    detections = pd.DataFrame(detections, columns=columns)
    detections.to_csv(f'{imgs_dir}/{Path(tfrecord).stem}_annotations.csv', sep=',', index=False)


def extract_images(video, output_dir, frame_skip=25, min_area=1000, min_roi_size=(1,1), no_merge_rois=False, first_pass_bg=True, no_average=False, show=False, debug=False):

    background = Background(
        no_average=no_average,
        skip=5,
        take=100,
        use_last=10000,
    )

    # import pdb; pdb.set_trace()
    if first_pass_bg:
        cap = cv2.VideoCapture(video)
        ret, frame = cap.read()
        frame_id = 0
        while ret:
            background.update(frame)
            if show:
                bg = cv2.resize(background.background.copy(), (1280, 768))
                cv2.putText(bg, f'frame: {frame_id}', (10, 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.imshow('Current Background', bg)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    sys.exit()
            ret, frame = cap.read()
            frame_id += 1

        if show:
            cv2.destroyAllWindows()
        cap.release()
        background.freeze()

    motionDetector = MotionDetection(
        background=background,
        min_area_contour=min_area,
        roi_size=min_roi_size,
        merge_rois=(not no_merge_rois),
    )

    cap = cv2.VideoCapture(video)
    ret, frame = cap.read()
    frame_id = 0
    last_frame = 0
    frames_saved = 0

    prefix = Path(video).stem
    while ret:
        motion_boxes, areas = motionDetector.detect(frame)
        if len(motion_boxes):
            if frame_skip > 0 and (frame_id - last_frame) < frame_skip:
                frame_id += 1
                ret, frame = cap.read()
                continue

            last_frame = frame_id
            # print(f'[{prefix}] frame {frame_id}')
            frames_saved += 1
            cv2.imwrite(f'{output_dir}/{prefix}_{frame_id}.jpg', frame)

            if show:
                img = frame.copy()
                xmin, ymin, xmax, ymax = [
                        min([box[0] for box in motion_boxes]),
                        min([box[1] for box in motion_boxes]),
                        max([box[2] for box in motion_boxes]),
                        max([box[3] for box in motion_boxes])
                ]
                roi = img[ymin:ymax, xmin:xmax]
                cv2.imshow('Default ROI', roi)

                roi_width = xmax - xmin
                roi_height = ymax - ymin
                if roi_width < roi_height:
                    diff = roi_height - roi_width
                    new_xmin = max(xmin - int(diff/2), 0)
                    new_xmax = min(xmax + int(diff/2), img.shape[1])
                    if new_xmin == 0: # we hit left border
                        left = int(diff/2) - (xmin-new_xmin)
                        new_xmax = min(new_xmax + left, img.shape[1])
                    elif new_xmax == img.shape[1]: # we hit left border
                        left = int(diff/2) - (new_xmax-xmax)
                        new_xmin = max(new_xmin - left, 0)
                    # if roi_width < roi_height * 0.99:
                    xmin = new_xmin
                    xmax = new_xmax

                elif roi_height < roi_width:
                    diff = roi_width - roi_height
                    new_ymin = max(ymin - int(diff/2), 0)
                    new_ymax = min(ymax + int(diff/2), img.shape[0])
                    if new_ymin == 0: # we hit top border
                        left = int(diff/2) - (ymin-new_ymin)
                        new_ymax = min(new_ymax + left, img.shape[0])
                    elif new_ymax == img.shape[0]: # we hit left border
                        left = int(diff/2) - (new_ymax-ymax)
                        new_ymin = max(new_ymin - left, 0)
                    ymin = new_ymin
                    ymax = new_ymax

                roi = img[ymin:ymax, xmin:xmax]
                cv2.imshow('NEW ROI', roi)
                for roi_id, roi in enumerate(motion_boxes):
                    cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)
                    cv2.putText(frame, 'ROI', (roi[0], roi[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    cv2.putText(frame, str(areas[roi_id]), (roi[0], roi[3]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                cv2.rectangle(frame, (10, 2), (140,60), (255,255,255), -1)
                cv2.putText(frame, str(cap.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
                cv2.imshow('Detections', frame)
            
                cv2.imwrite(f'{output_dir}/{prefix}_{frame_id}-bbox.jpg', frame)

                threshold = motionDetector.current_threshold.copy()
                threshold = cv2.resize(threshold, (800, 600))
                cv2.imshow('Threshold', threshold)
                cv2.imwrite(f'{output_dir}/{prefix}_{frame_id}-threshold.jpg', threshold)

                if debug:
                    delta = motionDetector.current_delta.copy()
                    delta = cv2.resize(delta, (800, 600))
                    cv2.imshow('Delta', delta)

                    gray = motionDetector.current_gray.copy()
                    gray = cv2.resize(gray, (800, 600))
                    cv2.imshow('Gray', gray)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    sys.exit()

                time.sleep(1)
        ret, frame = cap.read()
        frame_id += 1

    print(f'Frames saved: {frames_saved}')
    
    # Save resulting background
    cv2.imwrite(f'{output_dir}/background.bmp', background.background.copy())
    cv2.imwrite(f'{output_dir}/background-color.bmp', background.background_color.copy())
    return frames_saved


def main():
    args = argparse.ArgumentParser()
    args.add_argument("-v", "--videos", nargs='+', default=None, help="path to the video file")
    args.add_argument("-f", "--frames-per-scene", default=0, type=int, help="play fps")
    # args.add_argument("-m", "--minutes-per-scene", default=0, type=float, help="play fps")
    args.add_argument("-o", "--output", required=True, type=str, help="output folder")

    # Annotation
    args.add_argument("-m", "--model", default=None, type=str, help="Model to use for annotation")
    args.add_argument("--ckpt-id", default=None, type=str, help="ckpt id")
    args.add_argument("--classes", required=True, nargs='+', help="classes to annotate")
    args.add_argument("-l", "--label-map", default=None, help="Label map for the model")
    args.add_argument("--min-score", default=0.01, type=float, help="min score for detections")

    # Motion Detection
    args.add_argument("--min-area", default=1000, type=int, help="min area to consider motion detection")
    args.add_argument("--frame-skip", default=10, type=int, help="Frame skipping")
    # args.add_argument("--min-roi-size", default=(1,1), type=int, help="min size for RoI proposal")
    args.add_argument("--first-pass-bg", action='store_true', default=False, help="Do a first pass to compute background")

    args.add_argument("--show", action='store_true', default=False, help="show results")

    config = args.parse_args()

    # label_map = generate_label_map(config.classes)
    if config.label_map is not None:
        label_map = load_pbtxt(config.label_map)
    else:
        label_map = MSCOCO

    if config.model is not None:
        detector = load_model(config.model, config.ckpt_id)

    # os.makedirs(config.output, exist_ok=True)
    for video in config.videos:
        # cam = [s for s in Path(video).stem.split('.') if 'cam' in s][0]
        output_dir = f'{config.output}/{Path(video).stem}'
        os.makedirs(output_dir, exist_ok=True)
        frames_saved = len([img for img in Path(output_dir).glob('*')])
        print(f'[{video}]Found {frames_saved} frames saved in {output_dir}.')
        if frames_saved == 0 or \
                not os.path.exists(f'{output_dir}/background.bmp'):
            frames_saved = extract_images(
                    video,
                    output_dir=output_dir,
                    frame_skip=config.frame_skip,
                    min_area=config.min_area,
                    first_pass_bg=config.first_pass_bg,
                    show=config.show)

        if frames_saved > 1:
            if os.path.exists(f'{output_dir}/train.record'):
                print(f'Found {output_dir}/train.record')
                print(f'Skipping annotation of {video}')
                continue

            print(f'Annotating {video}')
            assert os.path.exists(f'{output_dir}/background.bmp')
            annotate(
                    detector,
                    label_map,
                    min_score=config.min_score,
                    tfrecord=f'{output_dir}/train.record',
                    imgs_dir=output_dir)


if __name__ == "__main__":
    main()
