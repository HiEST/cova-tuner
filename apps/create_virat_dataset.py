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
from dnn.tfrecord import generate_tfrecord
from utils.datasets import MSCOCO as label_map

def main():
    args = argparse.ArgumentParser()
    args.add_argument("-v", "--videos", default=None, help="path to the video file")
    args.add_argument("-d", "--detections", default=None, help="ground truth")
    args.add_argument("-f", "--frames-per-scene", default=1000, type=float, help="play fps")
    args.add_argument("-m", "--minutes-per-scene", default=10, type=float, help="play fps")

    config = args.parse_args()

    object_labels = ['person', 'car', 'vehicle', 'object', 'bike']
    columns = ['object_id', 'object_duration', 'current_frame', 'xmin', 'ymin', 'width', 'height', 'object_type']

    scenes = {s: {} for s in ["0000", "0001", "0002", "0100", "0101",
              "0102", "0400", "0401", "0500", "0502", "0503"]}
    video_prefix = 'VIRAT_S_{}'

    # train and eval will be split according to the video's timstamp (in filename) 
    # trying to distance videos from both datasets as much as possible

    # First, compute the number of frames per scene
    for scene in scenes.keys():
        scene_prefix = video_prefix.format(scene)
        videos_in_scene = [str(v) for v in Path(config.videos).glob('{}*.mp4'.format(scene_prefix))]
        num_videos = len(videos_in_scene)
        if num_videos == 0:
            continue
        num_videos = int(num_videos / 2)
        train_videos = videos_in_scene[:num_videos]
        eval_videos = videos_in_scene[num_videos:]
        scenes[scene]['train'] = train_videos
        scenes[scene]['eval'] = eval_videos
        scenes[scene]['train_frames'] = []
        scenes[scene]['eval_frames'] = []

        frames_in_scene = 0
        for video in videos_in_scene:
            cap = cv2.VideoCapture(str(video))
            num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            frames_in_scene += num_frames

            if video in scenes[scene]['train']:
                scenes[scene]['train_frames'].append(num_frames)
            else:
                scenes[scene]['eval_frames'].append(num_frames)

        avg_frames_per_video = int(frames_in_scene / (len(videos_in_scene)))
        
        print(f'[{scene}] {len(videos_in_scene)} videos with {frames_in_scene} total frames (avg. {avg_frames_per_video})')

    secs_per_scene = config.minutes_per_scene * 60
    frames_per_scene = config.frames_per_scene
    pick_one_every_n = int((secs_per_scene * 24) / frames_per_scene)
    for scene, data in scenes.items():
        frames_in_scene = 0
        offset = 0
        for idx, video in enumerate(data['train']):
            frames_in_video = data['train_frames'][idx]
            num_picks = int(frames_in_video / pick_one_every_n)
            if frames_in_video % pick_one_every_n > 0: # We pick one last frame from this video
                num_picks += 1

            cap = cv2.VideoCapture(video)
            for pick in range(num_picks):
                frame_to_pick = pick*pick_one_every_n + offset
                print(f'[{idx}] picking frame {frame_to_pick}')
                cap.set(1, frame_to_pick)
                ret, frame = cap.read()
                assert ret

                
            offset = frames_in_video - num_pick*pick_one_every_n

    sys.exit()
        
    for video in config.video:
        cap = cv2.VideoCapture(video)
        num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print(f'num_frames: {num_frames}')
        frame_id = num_frames - 500
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

                display_str = "{} (id={})".format(det['label'], det['object_id'])

                cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
                cv2.putText(frame, display_str, (int(left), int(top)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # frame = cv2.resize(frame, (1280, 768))
            cv2.putText(frame, f'frame: {frame_id}', (10, 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            cv2.imshow('Detections', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                sys.exit()

            while time.time() - last_frame < frame_lat:
                time.sleep(time.time() - last_frame)
            last_frame = time.time()

            ret, frame = cap.read()
            frame_id += 1


if __name__ == "__main__":
    main()
