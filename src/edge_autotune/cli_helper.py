# -*- coding: utf-8 -*-

import logging
from typing import Tuple
import os

import cv2
import numpy as np
import tensorflow as tf
import tqdm

from edge_autotune.api import server, client
from edge_autotune.dnn import dataset, train
from edge_autotune.motion import motion_detector as motion

logger = logging.getLogger(__name__)


def _server(
    model: str,
    model_id: str,
    port: int = 6000):
    """Start annotation server. 

    Args:
        model (str): [description]
        model_id (str, optional): Path to dir containing saved_model or checkpoint from TensorFlow. Defaults to ''.
        port (int, optional): Port to listen to clients. Defaults to 6000.
    """
    if not model_id:
        model_id = 'default'
    print(f'server.start_server({model}, {model_id}, {port})')
    server.start_server(model, model_id, port)


def _capture(
    stream: str,
    output: str,
    server: str,
    port: int = 6000,
    valid_classes: str = None,
    disable_motion: bool = False,
    frame_skip: int = 25,
    min_score: float = 0.5,
    max_images: int = 1000,
    min_images: int = 100,
    min_area: int = 1000,
    timeout: int = 0,
    tmp_dir: str = '/tmp/',
    first_frame_background: bool = False,
    flush: bool = True
):
    """Capture and annotate images from stream and generate dataset.

    Args:
        stream (str): Input stream from which to capture images.
        output (str): Path to the output dataset. 
        server (str): Server's url.
        port (int, optional): Port to connect to the server. Defaults to 6000.
        valid_classes (str, optional): Comma-separated list of classes to detect. If None, all classes will be considered during annotation. Defaults to None
        disable_motion (bool, optional): Disable motion detection for the region proposal. Defaults to False.
        frame_skip (int, optional): Frame skipping value. Defaults to 25.
        min_score (float, optional): Minimum score to accept groundtruth model's predictions as valid. Defaults to 0.5.
        max_images (int, optional): Stop when maximum is reached. Defaults to 1000.
        min_images (int, optional): Prevents timeout to stop execution if the minimum of images has not been reached. 
        min_area (int, optional): Minimum area for countours to be considered as actual movement. Defaults to 1000.
        Used only if timeout > 0. Defaults to 0.
        timeout (int, optional): timeout for capture. When reached, capture stops unless there is a minimum of images enforced. Defaults to 0.
        tmp_dir (str, optional): Path to temporary directory where intermediate results are written. Defaults to '/tmp'.
        first_frame_background (bool, optional): If True, first frame of the stream is chosen as background and never changed. Defaults to False.
        flush (bool, optional): If True, new frames and annotations are written to the output TFRecord as soon as received. Defaults to True.
    """
    max_boxes = 100
    use_motion = not disable_motion
    background = None
    motionDetector = None

    if valid_classes:
        valid_classes = valid_classes.split(',')
    
    if use_motion:
        background = motion.Background(no_average=first_frame_background)
        motionDetector = motion.MotionDetector(
            background=background,
            min_area_contour=min_area,
            roi_size=(300, 300))

    print(f'capture: from {stream} to {output}. Connect to {server} (port={port}). Motion? {not disable_motion}')

    edge = client.EdgeClient(server, port)
    writer = tf.compat.v1.python_io.TFRecordWriter(output)

    cap = cv2.VideoCapture(stream)
    ret, frame = cap.read()

    annotated_dataset = {'images': [], 'annotations': []}

    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) / frame_skip
    progress_cap = tqdm.tqdm(range(int(num_frames)), unit=' decoded frames')
    progress_ann = tqdm.tqdm(range(max_images), desc='Images annotated', unit='saved imgs')
    while ret:
        if motionDetector:
            regions_proposed, _ = motionDetector.detect(frame)
        else:
            regions_proposed = [[0, 0, frame.shape[1]-1, frame.shape[0]-1]]

        frame_detections = []
        for roi_id, roi in enumerate(regions_proposed):
            cropped_roi = np.array(frame[roi[1]:roi[3], roi[0]:roi[2]])
            status_code, results = edge.post_infer(cropped_roi)
            if status_code != 200:
                raise Exception(f'Error with offloaded inference: {status_code}')

            boxes = results[0]['boxes']
            scores = results[0]['scores']
            class_ids = results[0]['class_ids']
            labels = results[0]['labels']

            for i in range(min(len(boxes), max_boxes)):
                class_id = int(class_ids[i])
                label = labels[i]
                score = scores[i]
                if valid_classes is not None and label not in valid_classes:
                    continue 
                if score >= min_score:
                    ymin, xmin, ymax, xmax = tuple(boxes[i])
                    (left, right, top, bottom) = (roi[0] + xmin * cropped_roi.shape[1], roi[0] + xmax * cropped_roi.shape[1],
                                                  roi[1] + ymin * cropped_roi.shape[0], roi[1] + ymax * cropped_roi.shape[0])
                    xmin, xmax, ymin, ymax = (left/frame.shape[1], right/frame.shape[1],
                                              top/frame.shape[0], bottom/frame.shape[0])
                    
                    frame_detections.append([class_id, label, xmin, xmax, ymin, ymax, score])

        if frame_detections:
            annotated_dataset['images'].append(frame)
            annotated_dataset['annotations'].append(frame_detections)

            if flush:
                print(f'Writting image with {len(frame_detections)} detections to {output}.')
                ret = dataset.add_example_to_record(writer, frame, frame_detections)

            progress_ann.update(1)

        progress_cap.update(1)
        if max_images > 0 and len(annotated_dataset['images']) >= max_images:
            break

        next_frame = cap.get(cv2.CAP_PROP_POS_FRAMES) + frame_skip
        cap.set(1, next_frame)
        ret, frame = cap.read()

    print(f'Annotated {len(annotated_dataset["images"])}/{max_images}')


def _tune(
    checkpoint: str,
    dataset: str,
    config: str,
    output: str,
    label_map: str,
    train_steps: int = 1000):
    """Start fine-tuning from base model's checkpoint.

    Args:
        checkpoint (str): Path to directory containing the checkpoint to use as base model.
        dataset (str): Path to the training dataset TFRecord file.
        config (str): Path to the pipeline.config file with the training config.
        output (str): Path to the output directory.
        train_steps (int, optional): Number of training steps. Defaults to 1000.
    """
    if not os.path.isfile(f'{checkpoint}.index'):
        raise Exception("Checkpoint not found.")

    train_datasets = dataset.split(',')
    train.train_loop_wrapper(
        pipeline_config=config,
        train_datasets=train_datasets,
        model_dir=output,
        base_model=checkpoint,
        label_map=label_map,
        num_train_steps=train_steps
    )

    train.export_trained_model(
        pipeline_config_path=config,
        trained_checkpoint_dir=output,
        output_dir=f'{output}/saved_model'
    )


def _autotune(
    checkpoint: str,
    dataset: str,
    port: int):
    """Start capture, annotation, and fine-tuning. 

    Args:
        model (str): Path to dir containing saved_model or checkpoint from TensorFlow.
        port (int): Port to listen to.
    """
    print('app')


def _deploy(
    stream: str,
    dataset: str,
    port: int):
    """Start inference on stream. 

    Args:
        stream (str): Input video stream (file or url).
        model (str): Path to dir containing saved_model or checkpoint from TensorFlow.
        port (int): Port to listen to.
    """
    print('app')


def _tool(
    tool: str,
    *kwars: str):
    """Start tool. 

    Args:
        stream (str): Input video stream (file or url).
        model (str): Path to dir containing saved_model or checkpoint from TensorFlow.
        port (int): Port to listen to.
    """
    print('app')

