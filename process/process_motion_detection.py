#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.append('../')
from api.server import get_top_torch, get_top_tf
from utils.detector import label_map, init_detector, run_detector
from utils.motion_detection import MotionDetection, Background
from utils.nms import non_max_suppression_fast

detector = None


def process_video_motion(video, no_merge_rois=False, min_area=500, no_average=False):
    background = Background(
        no_average=no_average,
        skip=1,
        take=10,
        use_last=15,
    )

    motionDetector = MotionDetection(
        background=background,
        min_area_contour=min_area,
        merge_rois=(not no_merge_rois),
    )

    cap = cv2.VideoCapture(video)
    ret, frame = cap.read()
    max_boxes = 10
    min_score = 0.5
    frame_id = 0
    frames_with_motion = 0
    infers_per_frame = []
    data = []
    while ret:
        motion_boxes = motionDetector.detect(frame)

        if len(motion_boxes):
            frames_with_motion += 1
            infers_per_frame.append(len(motion_boxes))
            for roi in motion_boxes:
                cropped_roi = np.array(frame[roi[1]:roi[3], roi[0]:roi[2]])
                preds = run_detector(detector, cropped_roi)
                top_preds = get_top_tf(preds)
                data.append([frame_id, top_preds['scores'], top_preds['idxs']])

        ret, frame = cap.read()
        frame_id += 1

    skipped_frames = frame_id-frames_with_motion
    print(f'{video}')
    print(f'\tnum_frames: {frame_id}')
    print(f'\tskipped_frames: {skipped_frames} ({skipped_frames/frame_id:.2f})')
    print(f'\tframes_with_motion: {frames_with_motion}')
    print(f'\tinfers_per_frame:')
    print(f'\t\tmax: {max(infers_per_frame)}')
    print(f'\t\tavg: {np.average(infers_per_frame)}')
    print(f'\t\t99%: {np.percentile(infers_per_frame, 99)}')

    return data


def process_video_parallel(args):
    return process_video_motion(args[0])
    

def split_date(ts):
    ts = pd.to_datetime(ts)
    date = str(ts.date())
    hour = str(ts.hour)
    minute = str(ts.minute)

    return date, hour, minute


def process_dataset(dataset, dataset_name,
                    process_all=True,
                    move_when_done=None,
                    max_workers=1):

    columns = ['cam', 'timestamp', 'date', 'hour', 'minute', 'frame_id',
               'model', 'top_scores', 'top_classes']
    subcolumns = ['frame_id', 'top_scores', 'top_classes']

    detections = pd.DataFrame([], columns=columns)

    processed_ts = defaultdict(bool)

    videos_processed = defaultdict(int)
    models = ['ref', 'edge']
    videos = []
    if process_all:
        videos = [[str(f)] for f in dataset]
    else:
        for f in dataset:
            ts = f.stem.split('.')[0]
            cam = f.stem.split('.')[1]
            date, hour, minute = split_date(ts)

            date_hour = f'[{cam}] {date}:{hour}'
            if processed_ts[date_hour]:
                continue
            else:
                videos.append([str(f)])
                processed_ts[date_hour] = f.stem

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(process_video_parallel, videos)

        for r_idx, r in enumerate(results):
            filename = videos[r_idx][0]
            print(filename)
            model = 'edge'
            ts = Path(filename).stem.split('.')[0]
            cam = Path(filename).stem.split('.')[1]
            date, hour, minute = split_date(ts)

            df = pd.DataFrame(r, columns=subcolumns)
            df['cam'] = cam
            df['model'] = model
            df['timestamp'] = ts
            df['date'] = date
            df['hour'] = hour
            df['minute'] = minute

            detections = detections.append(df, ignore_index=True)
            print(f'{len(detections)} detections.')
            print(f'Writing results to {dataset_name}.tmp.csv')
            detections.to_csv(f'{dataset_name}.tmp.csv',
                              sep=',',
                              float_format='.2f',
                              index=False)

            print(f'Results from {filename} just processed.')

            videos_processed[filename] += 1
            if videos_processed[filename] == len(models):
                shutil.move(filename, move_when_done)
                del videos_processed[filename]

    detections.to_csv(f'{dataset_name}.csv',
                      sep=',',
                      float_format='.2f',
                      index=False)


def main():
    global detector
    args = argparse.ArgumentParser()

    args.add_argument("-i", "--input",
                      default="./",
                      type=str,
                      help="Path to the dataset to process.")

    args.add_argument("-n", "--name",
                      default="dataset",
                      type=str,
                      help="Name of the dataset. Used to name the results file.")

    args.add_argument("--fast",
                      default=False,
                      action="store_true",
                      help="Processes one video per hour "
                      "instead of the whole dataset.")

    args.add_argument("--move",
                      default=None,
                      type=str,
                      help="If specified, videos are moved "
                      "to this path after processing.")

    args.add_argument("-m", "--max-workers",
                      default=1,
                      type=int,
                      help="Max. workers to send parallel requests.")

    # Motion Detection
    args.add_argument("--no-merge-rois", action="store_true", help="Don't merge ROIs on scene")
    args.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
    args.add_argument("--no-average", help="use always first frame as background.", action="store_true")
    
    config = args.parse_args()

    path = Path(config.input)
    if os.path.isfile(config.input):
        dataset = [path]
    else:
        extensions = ['.mkv', '.mp4', '.webm']
        dataset = sorted([f for f in path.rglob('*') if f.suffix in extensions], key=os.path.getmtime)

    detector = init_detector()

    process_dataset(dataset,
                    dataset_name=config.name,
                    process_all=(not config.fast),
                    move_when_done=config.move,
                    max_workers=config.max_workers)


if __name__ == '__main__':
    main()
