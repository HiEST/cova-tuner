#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import cv2
import imutils
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import trange

sys.path.append('../')
from dnn.tf_infer import run_detector


def detect_video(filename, detector, min_score=0.5, max_boxes=100, label_map=None, save_frames_to=None, resize=True, max_frames=0, skip_frames=0):

    if save_frames_to is not None:
        os.makedirs(save_frames_to, exist_ok=True)

    cap = cv2.VideoCapture(filename)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    detections = []
    columns = ['frame', 'class_id', 'score', 'xmin', 'ymin', 'xmax', 'ymax']
    if label_map is not None:
        columns.append('label')

    ret, frame = cap.read()

    frames_skipped = 0
    for frame_id in trange(total_frames):
        if skip_frames > 0:
            if skip_frames > frames_skipped:
                frames_skipped += 1
                continue
            else:
                frames_skipped = 0

        if resize:
            frame = imutils.resize(frame, detector.input_size)

        if save_frames_to is not None:
            img_filename = '{}/{}.jpg'.format(save_frames_to, frame_id)
            cv2.imwrite(img_filename, frame)

        boxes, scores, class_ids = run_detector(detector, frame, max_boxes=max_boxes)
        
        for i in range(min(boxes.shape[0], max_boxes)):
            score = scores[i]
            if score < min_score: 
                continue

            ymin, xmin, ymax, xmax = tuple(boxes[i])
                
            (left, right, top, bottom) = (
                xmin * frame.shape[1], 
                xmax * frame.shape[1],
                ymin * frame.shape[0], 
                ymax * frame.shape[0]
            )

            class_id = int(class_ids[i])
            row = [frame_id, class_id, score, int(left), int(top), int(right), int(bottom)]
            if label_map is not None:
                label = label_map[class_id]['name']
                row.append(label)
            detections.append(row)

        frame_id += 1
        if max_frames > 0 and frame_id >= max_frames:
            break

        ret, frame = cap.read()
        if not ret:
            break

    detections = pd.DataFrame(detections, columns=columns)
    return detections


def to_pickle(detections, filename):
    try:
        detections.to_pickle(filename, compression='bz2')
    except:
        return False

    return True


def read_pickle(filename):
    return pd.read_pickle(filename, compression='bz2')


def pickle_to_csv(pickle, frame_shape, classes, reshape=None, label_map=None):
    csv = []
    results = pd.read_pickle(pickle_path, "bz2")
    for frame_id in results.frame.unique():

        detections = results[results.frame == frame_id]

        # Resize image before writing it to disk
        original_width, original_height, _ = frame.shape
        frame = imutils.resize(frame, height=config.resize_width) 

        filename = '{}.{}'.format(frame_id+previous_frames, image_format)
        boxes_saved = 0
        for i, det in detections.iterrows():
            score = det['score']
            if score > min_score and boxes_saved < max_boxes:
                class_name = label_map[str(int(det['class_id']))]['name']
                if config.classes is not None and class_name not in config.classes:
                    continue

                boxes_saved += 1

                # Get bbox info + frame info and scale to new width
                (width, height, _) = frame.shape
                resize_width = width / original_width
                resize_height = height / original_height
                (xmin, xmax, ymin, ymax) = det[['xmin', 'xmax', 'ymin', 'ymax']].values 
                xmin = int(xmin * resize_width)
                xmax = int(xmax * resize_width)
                ymin = int(ymin * resize_height)
                ymax = int(ymax * resize_height)
                
                # New class id: 0 is reserved for background
                # class_id = det['class_id'] if config.classes is None else config.classes.index(class_name)+1
                # Filename 
                annotations.append([filename, width, height, class_name, xmin, ymin, xmax, ymax])


def save_frames(video, save_to, frames_to_save, new_height=None):
    os.makedirs(save_to, exist_ok=True)

    cap = cv2.VideoCapture(video)
    ret, frame = cap.read()

    frame_id = 0
    while ret:
        if frame_id in frames_to_save:
            if new_height is not None:
                frame = imutils.resize(frame, height=new_height)
            cv2.imwrite('{}/{}.jpg'.format(save_to, frame_id))

        frame_id += 1
        ret, frame = cap.read()


def annotate_video(filename,
                   groundtruth,
                   label_map,
                   images_dir,
                   data_dir,
                   val_ratio=0.2,
                   reshape=None,
                   min_score=0.5,
                   max_boxes=10,
                   img_format='jpg'):

    classes = [c['name'] for c in label_map.values()]

    cap = cv2.VideoCapture(filename)
    ret, frame = cap.read()
    width, height, _ = frame.shape

    if reshape is not None:
        reshape_width = reshape[0] / width
        reshape_height = reshape[1] / height
    else:
        reshape_width = 1
        reshape_height = 1

    gt_detections = pd.read_pickle(groundtruth, "bz2")
    annotations = []
    frame_id = 0
    while ret:
        detections = gt_detections[gt_detections.frame == frame_id]
        if len(detections) > 0:
            # Resize image before writing it to disk
            if reshape is not None:
                frame = imutils.resize(frame, width=reshape[0], height=reshape[1]) 

            filename = '{}.{}'.format(frame_id, img_format)
            boxes_saved = 0
            for i, det in detections.iterrows():
                score = det['score']
                if score > min_score and boxes_saved < max_boxes:
                    class_name = label_map[str(int(det['class_id']))]['name']
                    if class_name not in classes:
                        continue

                    boxes_saved += 1

                    # Get bbox info + frame info and scale to new width
                    (xmin, xmax, ymin, ymax) = det[['xmin', 'xmax', 'ymin', 'ymax']].values 
                    xmin = int(xmin * resize_width)
                    xmax = int(xmax * resize_width)
                    ymin = int(ymin * resize_height)
                    ymax = int(ymax * resize_height)
                    
                    annotations.append([filename, width, height, class_name, xmin, ymin, xmax, ymax])

            if boxes_saved > 0:
                cv2.imwrite('{}/{}'.format(images_dir, filename), frame)

        ret, frame = cap.read()
        frame_id += 1

    columns = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    df = pd.DataFrame(annotations, columns=columns)

    all_images = df.filename.unique()
    train_imgs, test_imgs = train_test_split(all_images, test_size=val_ratio)

    train_dataset = df[df.filename.isin(train_imgs)]
    test_dataset = df[df.filename.isin(test_imgs)]

    for img in train_dataset.filename.unique():
        shutil.move('{}/{}'.format(images_dir, img),
                    '{}/train/'.format(images_dir))
    for img in test_dataset.filename.unique():
        shutil.move('{}/{}'.format(images_dir, img),
                    '{}/test/'.format(images_dir))

    train_dataset['filename'] = train_dataset['filename'].apply(lambda x: '{}/train/{}'.format(images_dir, x))
    test_dataset['filename'] = test_dataset['filename'].apply(lambda x: '{}/test/{}'.format(images_dir, x))
    train_dataset.to_csv('{}/train_label.csv'.format(data_dir), index=None)
    test_dataset.to_csv('{}/test_label.csv'.format(data_dir), index=None)

    return train_dataset, test_dataset


def generate_label_map(classes):
    label_map = {
        i: {
            'name': c,
            'id': str(i)
        }
        for i, c in enumerate(classes)
    }

    return label_map


def configure_pipeline(template, output, checkpoint, data_dir, classes, batch_size): 
    # Generate label_map.pbtxt
    label_map_entries = [
        'item {\n'
            f'\tname: "{c}",\n'
            f'\tid: {i+1}\n'
        '}'#.format(c, i)
        for i, c in enumerate(classes)
    ]

    label_map = '\n'.join(label_map_entries)
    num_classes = len(classes)
    print(label_map)

    with open('data/label_map.pbtxt', 'w') as f:
        f.write(label_map)

    pipeline_params = {
        'NUM_CLASSES': str(num_classes),
        'LABEL_MAP': '{}/label_map.pbtxt'.format(data_dir),
        'BATCH_SIZE': str(batch_size),
        'CHECKPOINT': checkpoint,
        'TRAIN_TFRECORD': '{}/train.record'.format(data_dir),
        'EVAL_TFRECORD': '{}/test.record'.format(data_dir),
    }

    with open(template, 'r') as f:
        lines = f.readlines()

    with open(output, 'w') as output:
        for l in lines:
            match = [k for k in pipeline_params.keys() if k in l]
            if len(match) > 0:
                l = l.replace(match[0], pipeline_params[match[0]])

            output.write(l)


