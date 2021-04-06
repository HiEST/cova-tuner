#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import pandas as pd

sys.path.append('../')
from utils.datasets import MSCOCO as label_map

COLORS = [
    (0, 255, 0),
    (0, 255, 255),
    (0, 0, 255)
]

def main():
    args = argparse.ArgumentParser()
    args.add_argument("-v", "--video", nargs='+', default=None, help="path to the video file")
    args.add_argument("-d", "--detections", default=None, help="ground truth")
    args.add_argument("-f", "--fps", default=25, type=float, help="play fps")
    args.add_argument("-j", "--jump-to", default=0, type=int, help="Jumpt to frame")

    config = args.parse_args()

    object_labels = ['person', 'car', 'vehicle', 'object', 'bike']
    columns = ['object_id', 'object_duration', 'current_frame', 'xmin', 'ymin', 'width', 'height', 'object_type']

    frame_id = config.jump_to
    for video in config.video:
        cap = cv2.VideoCapture(video)
        num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print(f'{Path(video).stem} ({num_frames})')
        cap.set(1, frame_id)
        ret, frame = cap.read()

        results = pd.read_csv(f'{config.detections}/{Path(video).stem}.viratdata.objects.txt', header=None, sep=' ', index_col=False)
        results.columns = columns
        results['label'] = results['object_type'].apply(lambda x: object_labels[int(x)-1])
        results['xmax'] = results['xmin'] + results['width']
        results['ymax'] = results['ymin'] + results['height']
        print(results.head(1))

        frame_lat = 1.0 / config.fps
        last_frame = time.time()
        # import pdb; pdb.set_trace()
        while ret:
            detections = results[results.current_frame == frame_id]
            for i, det in detections.iterrows():
                (left, right, top, bottom) = det[['xmin', 'xmax', 'ymin', 'ymax']].values 

                # display_str = "{} (id={})".format(det['label'], det['object_id'])
                display_str = "{}".format(det['label'])

                cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 1)
                cv2.putText(frame, display_str, (int(left), int(top)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

            # frame = cv2.resize(frame, (1280, 768))
            cv2.putText(frame, f'frame: {frame_id}', (10, 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            cv2.imshow('Detections', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                sys.exit()
            elif key == ord("c"):
                break
            elif key == ord("s"):
                cv2.imwrite(f'frame_{frame_id}.jpg', frame)

            while time.time() - last_frame < frame_lat:
                time.sleep(time.time() - last_frame)
            last_frame = time.time()

            ret, frame = cap.read()
            frame_id += 1


if __name__ == "__main__":
    main()
