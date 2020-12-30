#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import sys

import cv2
import imutils
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import trange

sys.path.append('../')
from dnn.tfinfer import run_detector
from utils.datasets import MSCOCO as mscoco


def detect_video(filename,
                 detector,
                 min_score=0.5,
                 max_boxes=100,
                 label_map=None,
                 save_frames_to=None,
                 resize=None,
                 max_frames=0,
                 skip_frames=0):

    if save_frames_to is not None:
        os.makedirs(save_frames_to, exist_ok=True)

    cap = cv2.VideoCapture(filename)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    detections = []
    columns = ['frame', 'class_id', 'score', 'xmin', 'ymin', 'xmax', 'ymax']
    if label_map is not None:
        columns.append('label')

    ret, frame = cap.read()
    if resize is not None:
        if len(resize) == 1:  # Only height is set, calculate width keeping aspect ratio
            width = (resize[0] / frame.shape[0]) * frame.shape[1]
            resize.append(width)

    frames_skipped = 0
    for frame_id in trange(total_frames, desc='Detecting video contents', file=sys.__stderr__):
        if skip_frames > 0:
            if skip_frames > frames_skipped:
                frames_skipped += 1
                continue
            else:
                frames_skipped = 0

        if resize is not None:
            frame = imutils.resize(frame, height=resize[0], width=resize[1])

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
                class_name = label_map[int(det['class_id'])]['name']
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
                   images_dir,
                   data_dir,
                   valid_classes,
                   label_map=None,
                   val_ratio=0.2,
                   reshape=None,
                   min_score=0.5,
                   max_boxes=10,
                   img_format='jpg'):

    if label_map is None:
        label_map = mscoco

    cap = cv2.VideoCapture(filename)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame = cap.read()
    width, height, _ = frame.shape

    if reshape is not None:
        new_height = reshape[0]
        if len(reshape) == 1:  # Only height is set, calculate width keeping aspect ratio
            new_width = (new_height / frame.shape[0]) * frame.shape[1]
            reshape.append(int(new_width))
        new_width = reshape[1]
        reshape_height = new_height / frame.shape[0]
        reshape_width = new_width / frame.shape[1]

        height = new_height
        width = new_width
    else:
        reshape_width = 1
        reshape_height = 1

    gt_detections = pd.read_pickle(groundtruth, "bz2")
    annotations = []
    frame_id = 0
    for frame_id in trange(total_frames, desc='Processing frames', file=sys.__stderr__):  #while ret:
        detections = gt_detections[gt_detections.frame == frame_id]
        if len(detections) > 0:
            # Resize image before writing it to disk
            if reshape is not None:
                frame = imutils.resize(frame, height=reshape[0], width=reshape[1]) 

            filename = '{}.{}'.format(frame_id, img_format)
            boxes_saved = 0
            for i, det in detections.iterrows():
                score = det['score']
                if score > min_score and boxes_saved < max_boxes:
                    class_name = label_map[int(det['class_id'])]['name']
                    if class_name not in valid_classes:
                        continue

                    boxes_saved += 1

                    # Get bbox info + frame info and scale to new width
                    (xmin, xmax, ymin, ymax) = det[['xmin', 'xmax', 'ymin', 'ymax']].values 
                    xmin = int(xmin * reshape_width)
                    xmax = int(xmax * reshape_width)
                    ymin = int(ymin * reshape_height)
                    ymax = int(ymax * reshape_height)
                    
                    assert xmin >= 0.0 and xmin <= width
                    assert xmax >= 0.0 and xmax <= width
                    assert ymin >= 0.0 and ymin <= height
                    assert ymax >= 0.0 and ymax <= height
                    annotations.append([filename, width, height, class_name, xmin, ymin, xmax, ymax])

            if boxes_saved > 0:
                cv2.imwrite('{}/{}'.format(images_dir, filename), frame)

        ret, frame = cap.read()
        # frame_id += 1
        if not ret:
            break

    columns = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    df = pd.DataFrame(annotations, columns=columns)

    all_images = df.filename.unique()
    train_imgs, test_imgs = train_test_split(all_images, test_size=val_ratio)

    train_dataset = df[df.filename.isin(train_imgs)]
    test_dataset = df[df.filename.isin(test_imgs)]

    # Create train/test subfolders
    os.makedirs('{}/train'.format(images_dir), exist_ok=True)
    os.makedirs('{}/test'.format(images_dir), exist_ok=True)
    for img in train_dataset.filename.unique():
        try:
            shutil.move('{}/{}'.format(images_dir, img),
                        '{}/train/{}'.format(images_dir, img))
        except:
            print('ERROR moving {}/{} to {}/train/'.format(images_dir, img, images_dir))
            raise
    for img in test_dataset.filename.unique():
        try:
            shutil.move('{}/{}'.format(images_dir, img),
                        '{}/test/{}'.format(images_dir, img))
        except:
            print('ERROR moving {}/{} to {}/train/'.format(images_dir, img, images_dir))
            raise

    # Commented out because the right full relative path is set during .record creation
    # train_dataset['filename'] = train_dataset['filename'].apply(lambda x: '{}/train/{}'.format(images_dir, x))
    # test_dataset['filename'] = test_dataset['filename'].apply(lambda x: '{}/test/{}'.format(images_dir, x))
    train_dataset.to_csv('{}/train_label.csv'.format(data_dir), index=None)
    test_dataset.to_csv('{}/test_label.csv'.format(data_dir), index=None)

    return train_dataset, test_dataset


def generate_label_map(classes):
    label_map = {
        i+1: {
            'name': c,
            'id': str(i+1)  # Non-background classes start at id 1
        }
        for i, c in enumerate(classes)
    }

    return label_map


def configure_pipeline(template, output, checkpoint, data_dir, model_dir, classes, batch_size): 
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

    with open('{}/label_map.pbtxt'.format(model_dir), 'w') as f:
        f.write(label_map)

    os.makedirs('{}/vis'.format(model_dir), exist_ok=True)
    pipeline_params = {
        'NUM_CLASSES': str(num_classes),
        'LABEL_MAP': '{}/label_map.pbtxt'.format(model_dir),
        'BATCH_SIZE': str(batch_size),
        'CHECKPOINT': checkpoint,
        'TRAIN_TFRECORD': '{}/train.record'.format(data_dir),
        'EVAL_TFRECORD': '{}/test.record'.format(data_dir),
        'VIS_EXPORT_DIR': '{}/vis'.format(model_dir)
    }

    with open(template, 'r') as f:
        lines = f.readlines()

    with open(output, 'w') as output:
        for l in lines:
            match = [k for k in pipeline_params.keys() if k in l]
            if len(match) > 0:
                l = l.replace(match[0], pipeline_params[match[0]])

            output.write(l)


def generate_detection_files(detections, output_dir, prefix, label_map=None, groundtruth=False, threshold=0.0):
    if label_map is None:
        label_map = mscoco
    if 'label' not in detections.columns:
        if label_map is None:
            return False
        detections['label'] = detections.class_id.apply(lambda x: label_map[x]['name'].replace(' ', '_'))
    else:
        detections['label'] = detections.label.apply(lambda x: x.replace(' ', '_'))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frames = detections.frame.unique()
    for frame in frames:
        frame_detections = detections[(detections.frame == frame) & (detections.score > threshold)]
        columns = ['label', 'score', 'xmin', 'ymin', 'xmax', 'ymax']
        if groundtruth:
            columns = ['label', 'xmin', 'ymin', 'xmax', 'ymax']

        frame_detections = frame_detections[columns]

        output_file = f'{output_dir}/{prefix}_{frame}.txt'
        with open(output_file, 'w') as f:
            frame_detections.to_csv(f, sep=' ', index=False, header=False)


    return True
