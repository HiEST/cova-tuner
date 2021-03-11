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
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange

sys.path.append('../')
from dnn.utils import generate_label_map, save_pbtxt
from dnn.tfrecord import generate_tfrecord
from utils.datasets import MSCOCO as label_map


def process_scenes(path_to_videos):
    scenes = {s: {} for s in ["0000", "0001", "0002", "0100", "0101",
              "0102", "0400", "0401", "0500", "0503"]}
    video_prefix = 'VIRAT_S_{}'

    # train and eval will be split according to the video's timstamp (in filename) 
    # trying to distance videos from both datasets as much as possible

    # First, compute the number of frames per scene
    for scene in scenes.keys():
        scene_prefix = video_prefix.format(scene)
        videos_in_scene = [str(v) for v in Path(path_to_videos).glob('{}*.mp4'.format(scene_prefix))]
        num_videos = len(videos_in_scene)
        if num_videos == 0:
            continue
        num_videos = int(num_videos / 2)
        train_videos = videos_in_scene[:num_videos]
        eval_videos = videos_in_scene[num_videos:]
        scenes[scene]['videos'] = videos_in_scene
        scenes[scene]['train'] = train_videos
        scenes[scene]['eval'] = eval_videos
        scenes[scene]['train_frames'] = []
        scenes[scene]['eval_frames'] = []
        scenes[scene]['num_frames'] = []

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
            scenes[scene]['num_frames'].append(num_frames)

        avg_frames_per_video = int(frames_in_scene / (len(videos_in_scene)))
        frames_to_train = sum(scenes[scene]['num_frames'])
        secs_in_scene = int(frames_to_train / 24)
        mins_in_scene = int(secs_in_scene / 60)
        print(f'[{scene}] {len(videos_in_scene)} videos with a total duration of {mins_in_scene}:{secs_in_scene-(mins_in_scene*60)}min (frames={frames_to_train}, avg={avg_frames_per_video})')

    return scenes


def capture_frames(video, frames_to_pick, imgs_dir, resize_height=None):
    imgs_already_captured = [img for img in Path(f'{imgs_dir}').glob(f'{Path(video).stem}*')]
    left_to_capture = [f for f in frames_to_pick if not os.path.exists(f'{imgs_dir}/{Path(video).stem}_{f}.jpg')]
    # skip_capture = (len(imgs_already_captured) == len(frames_to_pick))
    skip_capture = (len(left_to_capture) == 0)
    if len(imgs_already_captured) > 0 and not skip_capture:
        print(f'Some images captured but some missing: {left_to_capture}')

        frames_to_pick = left_to_capture

    cap = cv2.VideoCapture(video)
    original_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    original_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print(resize_height)
    if resize_height is not None:
        new_height = resize_height
        resize_ratio = new_height / original_height
        new_width =  int(original_width * resize_ratio)
    else:
        new_height = original_height
        new_width = original_width
        resize_ratio = 1

    if not skip_capture:
        for frame_to_pick in tqdm(frames_to_pick, desc='Capturing frames', leave=False):
            cap.set(1, frame_to_pick)
            ret, frame = cap.read()
            assert ret
            
            img = cv2.resize(frame, (new_width, new_height))
            cv2.imwrite(f'{imgs_dir}/{Path(video).stem}_{frame_to_pick}.jpg', img)
    else:
        print(f'Frames for video {video} already captured')
        cap.release()

    return new_width, new_height, resize_ratio


def generate_dataset(scenes, dataset, pick_one_every_n, resize_height, detections_dir, output_dir):
    object_labels = ['person', 'car', 'vehicle', 'object', 'bike']
    df_columns = ['filename', 'width', 'height', 'xmin', 'xmax', 'ymin', 'ymax', 'class']
    columns = ['object_id', 'object_duration', 'current_frame', 'xmin', 'ymin', 'width', 'height', 'object_type']
    annotations = pd.DataFrame([], columns=df_columns)
    
    label_map = generate_label_map([l for l in object_labels if l != 'object'])
    if not os.path.exists(f'{output_dir}/label_map.pbtxt'):
        save_pbtxt(['person', 'car', 'vehicle', 'bike'], output_dir)

    imgs_dir = f'{output_dir}/{dataset}_images'
    os.makedirs(imgs_dir, exist_ok=True)
    if not os.path.exists(f'{output_dir}/{dataset}_annotations.csv'):
        for scene, data in scenes.items():
            print(f'\tGetting frames from scene: {scene}')
            frames_in_scene = 0
            offset = 0

            for idx, video in tqdm(enumerate(data[dataset]), desc=f'Processing scene {scene}', total=len(data[dataset])):
                frames_to_pick = []
                frames_in_video = data['{}_frames'.format(dataset)][idx]
                num_picks = int((frames_in_video-offset) / pick_one_every_n)
                if (frames_in_video-offset) % pick_one_every_n > 0: # We pick one last frame from this video
                    num_picks += 1
                
                # break, if we already captured frames from this scene
                frames_to_pick = [int(pick*pick_one_every_n + offset) for pick in range(num_picks)]
                assert len(frames_to_pick) > 0
                assert frames_in_video > frames_to_pick[-1]

                new_width, new_height, resize_ratio = capture_frames(video, frames_to_pick, imgs_dir, resize_height) 
                
                results = pd.read_csv(f'{detections_dir}/{Path(video).stem}.viratdata.objects.txt', header=None, sep=' ', index_col=False)
                results.columns = columns
                results = results[results['object_type'] != 0]
                results['class'] = results['object_type'].apply(lambda x: object_labels[int(x)-1])
                results = results[results['class'] != 'object']
                results['filename'] = results['current_frame'].apply(lambda x: Path(video).stem + '_' + str(x) + '.jpg')
                results['xmax'] = results['xmin'] + results['width']
                results['ymax'] = results['ymin'] + results['height']
                results['width'] = new_width
                results['height'] = new_height
                for coord in ['xmin', 'xmax', 'ymin', 'ymax']:
                    results[coord] = results[coord].apply(lambda x: int(x * resize_ratio))

                detections = results[results.current_frame.isin(frames_to_pick)][df_columns]
                annotations = annotations.append(detections, ignore_index=True)

                offset = pick_one_every_n - (frames_in_video - frames_to_pick[-1]) # num_picks*pick_one_every_n - frames_in_video
                assert offset >= 0 and offset < pick_one_every_n

        annotations.sort_values('filename', inplace=True)
        annotations.to_csv(f'{output_dir}/{dataset}_annotations.csv', sep=',', index=False)

    else:
        print(f'{output_dir}/{dataset}_annotations.csv already exists.')

    generate_tfrecord(f'{output_dir}/{dataset}.record', imgs_dir, f'{output_dir}/{dataset}_annotations.csv', label_map)


def generate_datasets_per_scene(scenes, frames_per_scene, resize_height, detections_dir, output_dir, train_eval_ratio=0.8):
    object_labels = ['person', 'car', 'vehicle', 'object', 'bike']
    df_columns = ['filename', 'width', 'height', 'xmin', 'xmax', 'ymin', 'ymax', 'class']
    columns = ['object_id', 'object_duration', 'current_frame', 'xmin', 'ymin', 'width', 'height', 'object_type']
    annotations = pd.DataFrame([], columns=df_columns)
    
    label_map = generate_label_map([l for l in object_labels if l != 'object'])
    if not os.path.exists(f'{output_dir}/label_map.pbtxt'):
        save_pbtxt(['person', 'car', 'vehicle', 'bike'], output_dir)

    frames_per_scene = frames_per_scene + (1-train_eval_ratio)*frames_per_scene
    imgs_dir = f'{output_dir}/images'
    os.makedirs(imgs_dir, exist_ok=True)
    if not os.path.exists(f'{output_dir}/train_annotations.csv'):
        for scene, data in scenes.items():
            print(f'\tGetting frames from scene: {scene}')
            pick_one_every_n = int(sum(data['num_frames']) / frames_per_scene)
            offset = 0

            for idx, video in tqdm(enumerate(data['videos']), desc=f'Processing scene {scene}', total=len(data['videos'])):
                frames_to_pick = []
                frames_in_video = data['num_frames'][idx]
                num_picks = int((frames_in_video-offset) / pick_one_every_n)
                if (frames_in_video-offset) % pick_one_every_n > 0: # We pick one last frame from this video
                    num_picks += 1
                
                # break, if we already captured frames from this scene
                frames_to_pick = [int(pick*pick_one_every_n + offset) for pick in range(num_picks)]
                assert len(frames_to_pick) > 0
                assert frames_in_video > frames_to_pick[-1]

                new_width, new_height, resize_ratio = capture_frames(video, frames_to_pick, imgs_dir, resize_height) 
                
                results = pd.read_csv(f'{detections_dir}/{Path(video).stem}.viratdata.objects.txt', header=None, sep=' ', index_col=False)
                results.columns = columns
                results = results[results['object_type'] != 0]
                results['class'] = results['object_type'].apply(lambda x: object_labels[int(x)-1])
                results = results[results['class'] != 'object']
                results['filename'] = results['current_frame'].apply(lambda x: Path(video).stem + '_' + str(x) + '.jpg')
                results['xmax'] = results['xmin'] + results['width']
                results['ymax'] = results['ymin'] + results['height']
                results['width'] = new_width
                results['height'] = new_height
                for coord in ['xmin', 'xmax', 'ymin', 'ymax']:
                    results[coord] = results[coord].apply(lambda x: int(x * resize_ratio))

                detections = results[results.current_frame.isin(frames_to_pick)][df_columns]
                annotations = annotations.append(detections, ignore_index=True)

                offset = pick_one_every_n - (frames_in_video - frames_to_pick[-1]) # num_picks*pick_one_every_n - frames_in_video
                assert offset >= 0 and offset < pick_one_every_n

        annotations.sort_values('filename', inplace=True)
        image_fns = annotations['filename'].unique()
        # num_train_imgs = int(frames_per_scene * train_eval_ratio)
        # num_eval_imgs = int(frames_per_scene * (1-train_eval_ratio))
        # train_imgs = image_fns[:num_train_imgs]
        # eval_imgs = image_fns[num_train_imgs:]
        train_imgs, eval_imgs = train_test_split(image_fns, test_size=1-train_eval_ratio) 

        train_ds = annotations[annotations['filename'].isin(train_imgs)]
        eval_ds = annotations[annotations['filename'].isin(eval_imgs)]
        train_ds.to_csv(f'{output_dir}/train_annotations.csv', sep=',', index=False)
        eval_ds.to_csv(f'{output_dir}/eval_annotations.csv', sep=',', index=False)
        # annotations.to_csv(f'{output_dir}/{dataset}_annotations.csv', sep=',', index=False)

    else:
        print(f'{output_dir}/{dataset}_annotations.csv already exists.')


    generate_tfrecord(f'{output_dir}/train.record', imgs_dir, f'{output_dir}/train_annotations.csv', label_map)
    generate_tfrecord(f'{output_dir}/eval.record', imgs_dir, f'{output_dir}/eval_annotations.csv', label_map)


def generate_scene_dataset(scenes, dataset, output_dir):
    object_labels = ['person', 'car', 'vehicle', 'object', 'bike']
    label_map = generate_label_map([l for l in object_labels if l != 'object'])
    video_prefix = 'VIRAT_S_{}'
    detections = pd.read_csv(f'{output_dir}/{dataset}_annotations.csv')
    columns = detections.columns
    detections['scene'] = detections['filename'].apply(lambda x: Path(x).stem.replace(video_prefix.format(''), '')[:4])
    for scene, data in scenes.items():
        scene_dir = f'{output_dir}/scene_{scene}'
        if os.path.exists(f'{scene_dir}/{dataset}.record'):
            continue

        os.makedirs(scene_dir, exist_ok=True)
        os.makedirs(f'{scene_dir}/{dataset}_images', exist_ok=True)
        
        scene_detections = detections[detections['scene'] == scene]
        scene_images = detections['filename'].unique()

        scene_detections[columns].to_csv(f'{scene_dir}/{dataset}_annotations.csv', sep=',', index=False)
        for img in scene_images:
            shutil.copy(f'{output_dir}/images/{img}', f'{scene_dir}/{dataset}_images') 

        generate_tfrecord(f'{scene_dir}/{dataset}.record', f'{scene_dir}/{dataset}_images', f'{scene_dir}/{dataset}_annotations.csv', label_map)  


def main():
    args = argparse.ArgumentParser()
    args.add_argument("-v", "--videos", default=None, help="path to the video file")
    args.add_argument("-d", "--detections", default=None, help="ground truth")
    args.add_argument("-f", "--frames-per-scene", default=0, type=int, help="play fps")
    args.add_argument("-m", "--minutes-per-scene", default=0, type=float, help="play fps")
    args.add_argument("-t", "--frames_to_train", default=None, type=float, help="play fps")
    args.add_argument("--height", default=None, type=int, help="new height to resize frames to")
    args.add_argument("-o", "--output", default='dataset/', type=str, help="new height to resize frames to")

    config = args.parse_args()

    os.makedirs(config.output, exist_ok=True)
    scenes = process_scenes(config.videos)

    # if config.minutes_per_scene is not None and config.frames_per_scene is not None:
    #     secs_per_scene = config.minutes_per_scene * 60
    #     frames_per_scene = config.frames_per_scene
    #     pick_one_every_n = int((secs_per_scene * 24) / frames_per_scene)
    # else:
    
    for dataset in ['train', 'eval']:
        if os.path.exists(f'{config.output}/{dataset}.record'):
            continue
        # available_examples = sum([sum(s[f'{dataset}_frames']) for _,s in scenes.items()])
        # if config.frames_per_scene > 0:
        #     max_to_pick = config.frames_per_scene
        # max_to_pick = config.frames_to_train
        # if dataset == 'eval':
        #     max_to_pick = max_to_pick * 0.2

        # if available_examples < max_to_pick:
        #     max_to_pick = available_examples
        #     print(f'Not enough available examples {available_examples} vs {max_to_pick}')

        # pick_one_every_n = int(available_examples / max_to_pick)
        # print(f'Total {dataset} frames available: {available_examples}')
        # print(f'Picking one frame every {pick_one_every_n}')

        # generate_dataset(scenes, dataset, pick_one_every_n, config.height, config.detections, config.output)
        generate_datasets_per_scene(
                scenes=scenes,
                frames_per_scene=config.frames_per_scene,
                resize_height=config.height,
                detections_dir=config.detections,
                output_dir=config.output)

    # Now create one train/eval dataset per scene
    for dataset in ['train', 'eval']:
        generate_scene_dataset(scenes, dataset, config.output)
    

   

if __name__ == "__main__":
    main()
