# -*- coding: utf-8 -*-

import itertools
import logging
import math
from typing import Tuple
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm

#Prometheus
import prometheus_client as prom
import random
from threading import Thread

from flask import Flask, request
from flask_prometheus import monitor

from edge_autotune.api import server, client
from edge_autotune.dnn import dataset, infer
# from edge_autotune.dnn.infer import Model
from edge_autotune.motion import motion_detector as motion
from edge_autotune.motion import object_crop as crop

logger = logging.getLogger(__name__)

metric_bgs_lat=prom.Gauge('multicam_bgs_fps',"Background Subtraction latency in ms")
metric_inf_lat=prom.Gauge('multicam_inf_fps',"Inference latency in ms")
metric_dec_lat=prom.Gauge('multicam_dec_fps',"Decoding latency in ms")
metric_num_dets=prom.Gauge('multicam_num_objects', "Number of Objects Detected")
metric_num_rois=prom.Gauge('multicam_num_rois', "Number of RoIs proposed")
metric_fps=prom.Gauge('multicam_fps',"App's fps")
metric_lat_histo=prom.Histogram('multicam_latency', 'Latency in ms.')
SLO_metric_fps=prom.Gauge('multicam_SLO_fps',"App's latency SLO in fps")
app = Flask("multicam")

SLO_target=12


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
    # first_frame_background: bool = False,
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
        min_images (int, optional): Prevents timeout to stop execution if the minimum of images has not been reached. Used only if timeout > 0. Defaults to 0.
        min_area (int, optional): Minimum area for countours to be considered as actual movement. Defaults to 1000.
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
        background = motion.Background()
        motionDetector = motion.MotionDetector(
            background=background,
            min_area_contour=min_area,
            roi_size=(1, 1))

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
            print(labels)
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
    from edge_autotune.dnn import train
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
    model: str,
    label_map: str = None,
    valid_classes: Tuple[str] = None,
    min_score: float = 0.5,
    disable_motion: bool = False,
    min_area: int = 1000,
    frame_skip: int = 1,
    first_frame_background: bool = True,
    window_size: Tuple[int, int] = [1280, 720],
    save_to: str = None,
    debug: bool = False,
    model_is_classifier: bool = False, # FIXME: make it go through the cli pipeline.
    play_fps: int = 1, # FIXME: make it go through the cli pipeline.
    save_detections: str = '/tmp/detections.csv'
    ):
    """Start inference on stream using model.

    Args:
        stream (str): Input video stream (file or url).
        model (str): Path to dir containing saved_model or checkpoint from TensorFlow.
        label_map (str, optional): Path to pbtxt label_map file. Defaults to None.
        valid_classes (Tuple[str], optional): List of classes to draw in case of detection. Defaults to None.
        min_score (float, optional): Confidence threshold to draw detections. Defaults to 0.5.
        disable_motion (bool, optional): Disable motion detection for RoI proposal. Defaults to False.
        min_area (int, optional): Minimum area for countours to be considered as actual movement. Defaults to 1000.
        frame_skip (int, optional): Number of frames to skip between inferences. Defaults to 25.
        first_frame_background (bool, optional): If True, first frame of the stream is chosen as background and never changed. Defaults to False.
        window_size (Tuple[int,int], optional): Size of the window to show stream with detections [width, height]. Defaults to [1280,720]
        save_to (str, option): Record output stream, if set. Path to output video with detection results. Defaults to None.
        debug (bool, option): Show debug information, like current background. Defaults to None.
    """
    max_boxes = 100
    use_motion = not disable_motion
    background = None
    motionDetector = None

    frame_lat = 0 # 1.0 / play_fps

    if valid_classes:
        valid_classes = valid_classes.split(',')
    
    if True: #use_motion:
        background = motion.Background(method=motion.BackgroundMethod.FIRST)
        motionDetector = motion.MotionDetector(
            background=background,
            min_area_contour=min_area,
            roi_size=(300, 300))

    if save_to:
        fourcc=cv2.VideoWriter_fourcc('M','J','P','G')
        recorder = cv2.VideoWriter(save_to, fourcc, 25, window_size)

    if model_is_classifier:
        edge = infer.Classifier(label_map=label_map)
        edge.load_model(model)
    else:
        edge = infer.Model(
            model_dir=model,
            label_map=label_map,
            min_score=min_score,
            iou_threshold=0.3)
    cap = cv2.VideoCapture(stream)
    ts0_decode_frame = time.time()
    ret, frame = cap.read()
    ts1_decode_frame = time.time()
    total_decoding_time = ts1_decode_frame-ts0_decode_frame
    # start_frame = 0
    next_frame = 45
    df = []
    df_stats = []
    while ret:
        ts0_frame = time.time()
        ts0_bg = time.time()
        if motionDetector:
            regions_proposed, _ = motionDetector.detect(frame)
            if not use_motion and len(regions_proposed):  # FIXME: Add option to use motion to only skip inferences but process full frames.
                regions_proposed = [[0, 0, frame.shape[1]-1, frame.shape[0]-1]]
        else:
            regions_proposed = [[0, 0, frame.shape[1]-1, frame.shape[0]-1]]
        ts1_bg = time.time()
        total_time_bg = ts1_bg-ts0_bg

        frame_rgb = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
        
        num_detections = 0
        total_time_infer = 0
        total_decode_infer = 0
        for roi in regions_proposed:
            ts0_infer = time.time()
            cropped_roi = np.array(frame_rgb[roi[1]:roi[3], roi[0]:roi[2]])
            results = edge.run([cropped_roi])
            ts1_infer = time.time()
            total_time_infer += (ts1_infer-ts0_infer)
            
            ts0_decode_infer = time.time()
            boxes = results[0]['boxes']
            scores = results[0]['scores']
            labels = results[0]['labels']

            for i in range(min(len(boxes), max_boxes)):
                label = labels[i]
                score = scores[i]
                # import pdb; pdb.set_trace()
                if valid_classes is not None and label not in valid_classes:
                    continue 
                if score >= min_score:
                    ymin, xmin, ymax, xmax = tuple(boxes[i])
                    (left, right, top, bottom) = (roi[0] + xmin * cropped_roi.shape[1], roi[0] + xmax * cropped_roi.shape[1],
                                                  roi[1] + ymin * cropped_roi.shape[0], roi[1] + ymax * cropped_roi.shape[0])
                    xmin, xmax, ymin, ymax = (left/frame.shape[1], right/frame.shape[1],
                                              top/frame.shape[0], bottom/frame.shape[0])
                    
                    ts1_decode_infer = time.time()
                    total_decode_infer += ts1_decode_infer-ts0_decode_infer

                    if score >= 0.5:
                        display_str = f'{label} ({score*100:.2f}%)'
                        cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), 2)
                        cv2.putText(frame, display_str, (int(left), int(top)-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    num_detections += 1

                    columns = ['cam', 'frame', 'label', 'score', 'xmin', 'ymin', 'xmax', 'ymax']
                    df.append([
                        0,
                        next_frame,
                        label,
                        score,
                        left,
                        top,
                        right,
                        bottom,
                    ])

                    ts0_decode_infer = time.time()

                elif model_is_classifier:
                    ymin, xmin, ymax, xmax = tuple(boxes[i])
                    (left, right, top, bottom) = (roi[0] + xmin * cropped_roi.shape[1], roi[0] + xmax * cropped_roi.shape[1],
                                                    roi[1] + ymin * cropped_roi.shape[0], roi[1] + ymax * cropped_roi.shape[0])
                    xmin, xmax, ymin, ymax = (left/frame.shape[1], right/frame.shape[1],
                                                top/frame.shape[0], bottom/frame.shape[0])

                    display_str = f'NOT {label} ({(1-score)*100:.2f}%)'
                    cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
                    cv2.putText(frame, display_str, (int(left), int(top)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ts1_frame = time.time()
        total_frame_time = ts1_frame - ts0_frame
        df_stats.append([
            next_frame,
            total_frame_time,
            total_decoding_time,
            total_time_bg,    
            total_time_infer,
            total_decode_infer,
            len(regions_proposed),
        ])
        
        frame = cv2.resize(frame, window_size)

        if debug:
            cv2.rectangle(frame, (10, 2), (180,100), (255,255,255), -1)
            cv2.putText(frame, str(cap.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
            cv2.putText(frame, f'Bg: {total_time_bg:.3f} sec.', (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
            cv2.putText(frame, f'Infer: {total_time_infer:.3f} sec.', (15, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
            cv2.putText(frame, f'Motion: {1 if not use_motion else len(regions_proposed)} regions.', (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
            cv2.putText(frame, f'Detection: {num_detections} objects.', (15, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
            if use_motion:
                threshold = motionDetector.current_threshold.copy()
                delta = motionDetector.current_delta.copy()
                gray = motionDetector.current_gray.copy()
                bg_color = background.background_color.copy()

                threshold = cv2.resize(threshold, window_size)
                delta = cv2.resize(delta, window_size)
                gray = cv2.resize(gray, window_size)
                bg_color = cv2.resize(bg_color, window_size)
                cv2.imshow('Background Color', bg_color)
                cv2.imshow('Threshold', threshold)
                cv2.imshow('Delta', delta)
                cv2.imshow('Gray', gray)

        if save_to:
            recorder.write(frame)

        # cv2.imshow(stream, frame)
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord("q"):
            # break

        # ret, frame = cap.read()
        next_frame = next_frame + frame_skip
        cap.set(1, next_frame)

        ts0_decode_frame = time.time()
        ret, frame = cap.read()
        ts1_decode_frame = time.time()
        total_decoding_time = ts1_decode_frame-ts0_decode_frame

        # if next_frame == (24*25-10):
        #     cv2.imwrite('/tmp/frame.png', frame)
            
    # cv2.destroyAllWindows()
    if save_to:
        recorder.release()

    columns = ['cam', 'frame', 'label', 'score', 'xmin', 'ymin', 'xmax', 'ymax']
    df = pd.DataFrame(df, columns=columns)
    df.to_csv(save_detections, sep=',', index=False)

    columns_stats = ['frame', 'total', 'decoding_frame', 'motion', 'inference', 'decoding_inference', 'regions']
    df_stats = pd.DataFrame(df_stats, columns=columns_stats)
    df_stats.to_csv('/tmp/stats.csv', sep=',', index=False)


def _deploy_multi_cam(
    streams: Tuple[str],
    model: str,
    label_map: str = None,
    valid_classes: Tuple[str] = None,
    min_score: float = 0.5,
    disable_motion: bool = False,
    min_area: int = 1000,
    roi_size: Tuple[int,int] = (1,1),
    frame_skip: int = 1,
    first_frame_background: bool = False,
    window_size: Tuple[int, int] = [1280, 720],
    save_to: str = None,
    debug: bool = False,
    replicate_multi: bool = False,
    save_detections: str = '/tmp/detections.csv'):
    """Start inference on stream using model.

    Args:
        stream (Tuple[str]): List of input video stream (file or url).
        model (str): Path to dir containing saved_model or checkpoint from TensorFlow.
        label_map (str, optional): Path to pbtxt label_map file. Defaults to None.
        valid_classes (Tuple[str], optional): List of classes to draw in case of detection. Defaults to None.
        min_score (float, optional): Confidence threshold to draw detections. Defaults to 0.5.
        disable_motion (bool, optional): Disable motion detection for RoI proposal. Defaults to False.
        min_area (int, optional): Minimum area for countours to be considered as actual movement. Defaults to 1000.
        roi_size (Tuple[int,int]):
        frame_skip (int, optional): Number of frames to skip between inferences. Defaults to 25.
        first_frame_background (bool, optional): If True, first frame of the stream is chosen as background and never changed. Defaults to False.
        window_size (Tuple[int,int], optional): Size of the window to show stream with detections [width, height]. Defaults to [1280,720]
        save_to (str, option): Record output stream, if set. Path to output video with detection results. Defaults to None.
        debug (bool, option): Show debug information, like current background. Defaults to None.
        save_detections (str, optional)
    """
    max_boxes = 100
    use_motion = not disable_motion
    background = None
    motionDetector = None
    no_show = not debug

    frame_shift = 100

    if valid_classes:
        valid_classes = valid_classes.split(',')

    if save_to:
        fourcc=cv2.VideoWriter_fourcc('M','J','P','G')
        recorder = cv2.VideoWriter(save_to, fourcc, 25, window_size)

    edge = infer.Model(
        model_dir=model,
        label_map=label_map,
        min_score=min_score,
        iou_threshold=0.3)

    num_streams = len(streams) 
    cam_grid_w = math.ceil(math.sqrt(num_streams))
    cam_grid_h = math.ceil(num_streams / cam_grid_w)

    if replicate_multi:
        cap = [cv2.VideoCapture(streams[0])]
    else:
        cap = [cv2.VideoCapture(v) for v in streams]
    
    frame_limit = cap[0].get(cv.CAP_PROP_FRAME_COUNT)

    if use_motion:
        if replicate_multi:
            background = [motion.Background(method=motion.BackgroundMethod.FIRST)]
        else:
            background = [motion.Background(method=motion.BackgroundMethod.FIRST) for _ in range(num_streams)]
        motionDetector = [
            motion.MotionDetector(
                background=b,
                min_area_contour=min_area,
                roi_size=roi_size)
            for b in background]


    if frame_shift:
        # Set background using first frame of the first stream
        for i, c in enumerate(cap):
            ret, frame = c.read()
            motionDetector[i].background.update(frame)
            c.set(1, i*frame_shift)

    merged_frame = None

    processed_frames = 0
    next_frame = 1
    update_metrics_interval = 100
    
    df = [] 
    df_stats = []
    while True:
        if next_frame >= frame_limit:
            break
        
        ts0_frame = time.time()
        next_frame = next_frame + frame_skip
        
        if replicate_multi:
            t0_decoding = time.time()
            ret, frame = cap[0].read()
            cap[0].set(1, next_frame)
            t1_decoding = time.time()

            rets = [ret]
            frames = [frame.copy() for _ in range(num_streams)]

            total_decoding_time = t1_decoding - t0_decoding
            total_decoding_time *= num_streams
        else:
            t0_decoding = time.time()
            decoded = [cap[i].read() for i in range(len(cap))]
            for i, c in enumerate(cap):
                cap_next_frame = (next_frame+i*frame_shift)%frame_limit
                c.set(1, cap_next_frame)

            rets = [d[0] for d in decoded]
            frames = [d[1] for i,d in enumerate(decoded) if rets[i]]
            t1_decoding = time.time()
            total_decoding_time = t1_decoding - t0_decoding
    
        if not all(rets):
            print(f'Reached end of at least one stream [{rets}]')
            break

        ts0_bg = time.time()
        
        if replicate_multi:
            regions_proposed, _ = motionDetector[0].detect(frames[0])
            if len(regions_proposed):
                regions_proposed = [regions_proposed for _ in range(num_streams)]
            
            ts1_bg = time.time()
            total_time_bg = ts1_bg-ts0_bg
            total_time_bg *= num_streams
        else:
            regions_proposed = [
                mDetector.detect(frame)
                for mDetector, frame in zip(motionDetector, frames)]
            regions_proposed = [r[0] for r in regions_proposed]

            ts1_bg = time.time()
            total_time_bg = ts1_bg-ts0_bg

        frames_rgb = [cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB) for frame in frames]
        
        num_detections = 0
        total_time_infer = 0
        total_crop_time = 0
        total_decode_infer = 0
        
        num_regions_proposed = sum([len(r) for r in regions_proposed])
        if num_regions_proposed:
            for roi in regions_proposed[0]:
                cv2.rectangle(frames[0], (roi[0], roi[1]), (roi[2], roi[3]), (255, 0, 0), 2)
                cv2.putText(frames[0], f'{roi[2]-roi[0]}x{roi[3]-roi[1]}', (roi[0], roi[1]-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
            merged_frame = None
            object_map = None
            objects = []

            ts0_crop = time.time()
            merged_frame, object_map, objects = crop.combine_border(frames_rgb, regions_proposed, border_size = 5)
            regions_proposed = [[0, 0, merged_frame.shape[1]-1, merged_frame.shape[0]-1]]
            ts1_crop = time.time()
            total_crop_time = ts1_crop - ts0_crop

            ts0_infer = time.time()
            results = edge.run([merged_frame])
            ts1_infer = time.time()
            total_time_infer += (ts1_infer-ts0_infer)
            
            boxes = results[0]['boxes']
            scores = results[0]['scores']
            labels = results[0]['labels']

            ts0_decode_infer =  time.time()
            for i in range(min(len(boxes), max_boxes)):
                label = labels[i]
                score = scores[i]
                if valid_classes is not None and label not in valid_classes:
                    continue 
                
                ymin, xmin, ymax, xmax = tuple(boxes[i])

                # Object/Detection coordinates in merged frame 
                (merged_left, merged_right, merged_top, merged_bottom) = (
                                                int(xmin * merged_frame.shape[1]), int(xmax * merged_frame.shape[1]),
                                                int(ymin * merged_frame.shape[0]), int(ymax * merged_frame.shape[0]))
                
                # Find object id consulting the object map
                obj_id = int(np.median(object_map[merged_top:merged_bottom,merged_left:merged_right]))
                obj = objects[obj_id-1]

                # iou = motion.compute_iou([merged_left, merged_top, merged_right, merged_bottom], obj.inf_box)
                # if iou < 0.1:
                #     color = (255, 255, 255)
                    # display_str = f'{label} (Object {obj_id})'
                if debug and score >= 0.5:
                    color = (0, 255, 0)
                    display_str = f'{label} ({score*100:.2f}%)'
                    cv2.rectangle(merged_frame, (merged_left, merged_top), (merged_right, merged_bottom), color, 2)
                    cv2.putText(merged_frame, display_str, (merged_left, merged_top-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                # if iou < 0.1:
                #     continue

                if debug: # and False:
                    if score >= 0.5:
                        color = (0, 255, 0)
                    else:
                        color = (0, 0, 255)
                    if merged_left < obj.inf_box[0] or merged_right > obj.inf_box[2] \
                        or merged_top < obj.inf_box[1] or merged_bottom > obj.inf_box[3]:
                        color = (255, 255, 255)
                    
                    # unique_values = np.unique(object_map[merged_top:merged_bottom, merged_left:merged_right])
                    # if len(unique_values) > 2:
                        # print(f'obj ids: {unique_values} vs {values}')

                    roi_width = merged_right-merged_left
                    roi_height = merged_bottom-merged_top
                    if roi_width > obj.width() or roi_height > obj.height():
                        color = (255, 0, 0)

                # Translate to coordinates in original frame from the camera
                # roi is in camera frame coordinates  
                roi = obj.box
                # inference box is in merged frame coordinates and includes borders
                box_in_inference = obj.inf_box

                
                # First, we adjust coordinates within merged frame by making sure borders are taken into account and subtracted
                adjusted_coords = [
                    max(merged_left, box_in_inference[0]),
                    max(merged_top, box_in_inference[1]),
                    min(merged_right, box_in_inference[2]),
                    min(merged_bottom, box_in_inference[3]),
                ]

                # Second, we compute the relative object coordinates within RoI by removing box_in_inference coordinates
                relative_coords = [
                    adjusted_coords[0] - box_in_inference[0],
                    adjusted_coords[1] - box_in_inference[1],
                    adjusted_coords[2] - box_in_inference[0],
                    adjusted_coords[3] - box_in_inference[1],
                ]

                # Second, we remove borders such that 0,0 within roi is 0,0
                no_border_coords = [
                    relative_coords[0]-obj.border[0],
                    relative_coords[1]-obj.border[1],
                    relative_coords[2]-obj.border[0],
                    relative_coords[3]-obj.border[1],
                ]

                # Now, we can compute the absolute coordinates within the camera frames by adding roi coordinates
                obj_coords = [
                    no_border_coords[0] + roi[0],
                    no_border_coords[1] + roi[1],
                    no_border_coords[2] + roi[0],
                    no_border_coords[3] + roi[1],
                ]
                                            
                # (left, right, top, bottom) = (roi[0] + obj_coords[0], roi[0] + obj_coords[2],
                #                                 roi[1] + obj_coords[1], roi[1] + obj_coords[3])
                (left, top, right, bottom) = obj_coords

                if left > right or top > bottom:
                    continue
                    import pdb; pdb.set_trace()
                if debug and score >= 0.2:
                    display_str = f'{label} ({score*100:.2f}%)'
                    # display_str = f'{label} (Object {obj_id})'
                    cv2.rectangle(frames[obj.cam_id], (int(left), int(top)), (int(right), int(bottom)), color, 2)
                    cv2.putText(frames[obj.cam_id], display_str, (int(left), int(top)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                num_detections += 1
                # (24.0, 728.0, 19.0, 743.0)
                if left > right:
                    import pdb; pdb.set_trace()
                df.append([
                    obj.cam_id,
                    (next_frame-frame_skip+(obj.cam_id*frame_shift))%frame_limit,
                    label,
                    score,
                    left,
                    top,
                    right,
                    bottom
                ])
            ts1_decode_infer = time.time()
            total_decode_infer = ts1_decode_infer - ts0_decode_infer

        ts1_frame = time.time()
        total_frame_time = ts1_frame - ts0_frame
        df_stats.append([
            next_frame,
            total_frame_time,
            total_decoding_time,
            total_time_bg,    
            total_time_infer,
            total_decode_infer,
            num_regions_proposed,
        ])


        processed_frames += 1
        if processed_frames % update_metrics_interval == 0:
            metric_bgs_lat.set(total_time_bg)
            metric_dec_lat.set(total_decoding_time)
            metric_inf_lat.set(total_time_infer)
            metric_num_dets.set(num_detections)
            metric_num_rois.set(num_regions_proposed)
            metric_fps.set(1000/total_frame_time*update_metrics_interval)
            metric_lat_histo.observe(total_frame_time)
            SLO_metric_fps.set(SLO_target)


        if not no_show:
            for video_id, frame in enumerate(frames):
                frames[video_id] = cv2.resize(frame, (1280, 768))
                frame = frames[video_id]
                cv2.rectangle(frame, (10, frame.shape[0]-50), (len(streams[video_id])*20, frame.shape[0]-5), (255, 255, 255), -1)
                cv2.putText(frame, streams[video_id], (15, frame.shape[0]-25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0))
                # cv2.imshow(f'frame {video_id}', frame)

            img_h, img_w, img_c = frames[0].shape

            m_x = 0
            m_y = 0

            imgmatrix = np.zeros((img_h * cam_grid_h + m_y * (cam_grid_h - 1),
                                img_w * cam_grid_w + m_x * (cam_grid_w - 1),
                                img_c),
                                np.uint8)

            imgmatrix.fill(255)    

            positions = itertools.product(range(cam_grid_w), range(cam_grid_h))
            for (x_i, y_i), img in zip(positions, frames):
                x = x_i * (img_w + m_x)
                y = y_i * (img_h + m_y)
                imgmatrix[y:y+img_h, x:x+img_w, :] = img    

            all_frames = imgmatrix 
            cv2.rectangle(all_frames, (10, 2), (180,130), (255,255,255), -1)
            cv2.putText(all_frames, str(cap[0].get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
            cv2.putText(all_frames, f'Decoding: {total_decoding_time:.3f} sec.', (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
            cv2.putText(all_frames, f'Bg: {total_time_bg:.3f} sec.', (15, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
            cv2.putText(all_frames, f'Infer: {total_time_infer:.3f} sec.', (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
            cv2.putText(all_frames, f'Motion: {1 if not use_motion else len(regions_proposed)} regions.', (15, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
            cv2.putText(all_frames, f'Cropping: {total_crop_time:.3f} sec.', (15, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
            cv2.putText(all_frames, f'Detection: {num_detections} objects.', (15, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

            if debug:
                mdet = motionDetector[0]
                threshold = mdet.current_threshold.copy()
                delta = mdet.current_delta.copy()
                gray = mdet.current_gray.copy()
                bg_color = background[0].background_color.copy()

                threshold = cv2.resize(threshold, window_size)
                delta = cv2.resize(delta, window_size)
                gray = cv2.resize(gray, window_size)
                bg_color = cv2.resize(bg_color, window_size)
                cv2.imshow('Background Color', bg_color)
                cv2.imshow('Threshold', threshold)
                cv2.imshow('Delta', delta)
                cv2.imshow('Gray', gray)

                if merged_frame is not None:
                    merged_frame = cv2.resize(merged_frame, (1000, 1000))
                    merged_frame = merged_frame / 255.0
                    cv2.imshow("merged", merged_frame)

            cv2.imshow('Multi Cam View', all_frames)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord('p'):
                cv2.waitKey(0)

        if save_to:
            recorder.write(frame)

    if not no_show:
        cv2.destroyAllWindows()
    if save_to:
        recorder.release()

    columns = ['cam', 'frame', 'label', 'score', 'xmin', 'ymin', 'xmax', 'ymax']
    df = pd.DataFrame(df, columns=columns)
    df.sort_values(['frame', 'score'])
    df.to_csv(save_detections, sep=',', index=False)

    columns_stats = ['frame', 'total', 'decoding_frame', 'motion', 'inference', 'decoding_inference', 'regions']
    df_stats = pd.DataFrame(df_stats, columns=columns_stats)
    df_stats.to_csv('/tmp/stats.csv', sep=',', index=False)
    

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

