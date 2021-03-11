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
from dnn.tfrecord import generate_tfrecord
from dnn.utils import generate_label_map, save_pbtxt
# from utils.datasets import MSCOCO as label_map


# def generate_dataset(scenes, dataset, pick_one_every_n, resize_height, detections_dir, output_dir):
#     object_labels = ['person', 'car', 'vehicle', 'object', 'bike']
#     df_columns = ['filename', 'width', 'height', 'xmin', 'xmax', 'ymin', 'ymax', 'class']
#     columns = ['object_id', 'object_duration', 'current_frame', 'xmin', 'ymin', 'width', 'height', 'object_type']
#     annotations = pd.DataFrame([], columns=df_columns)
#     
#     label_map = generate_label_map([l for l in object_labels if l != 'object'])
#     if not os.path.exists(f'{output_dir}/label_map.pbtxt'):
#         save_pbtxt(['person', 'car', 'vehicle', 'bike'], output_dir)


def convert_images(imgs_dir, output_dir, pad):
    imgs = [img for img in Path(imgs_dir).glob('*.jpg')]
    
    dest_dir = f'{output_dir}/images'
    os.makedirs(dest_dir, exist_ok=True)

    dims_dict = {s: [] for s in ["0000", "0001", "0002", "0100", "0101",
                                "0102", "0400", "0401", "0500", "0503"]}
    for img in tqdm(imgs):
        scene = img.stem[8:]
        buf = cv2.imread(str(img))
        dest_file = f'{dest_dir}/{img.stem}.jpg'

        if pad:
            stripes_height = int((buf.shape[1] - buf.shape[0])/2)
            padded_img = cv2.copyMakeBorder(buf, stripes_height, stripes_height, 0, 0, cv2.BORDER_CONSTANT)
            cv2.imwrite(dest_file, padded_img)

            img = padded_img

        else: # crop
            h, w, _ = buf.shape
            left = int(w/2 - h/2)
            crop_img = buf[:, left:left+h]
            cv2.imwrite(dest_file, crop_img)

            img = crop_img

        if dims_dict.get(scene, None) is not None:
            if dims_dict[scene] != img.shape:
                print(f'dims for scene {scene} differ: {dims_dict[scene]} vs {crop_img.shape}')
        else:
            dims_dict[scene] = img.shape

    return dims_dict


def main():
    args = argparse.ArgumentParser()
    args.add_argument("-i", "--images", default=None, help="path to the images dir")
    args.add_argument("-d", "--detections", default=None, help="ground truth")
    args.add_argument("--height", default=None, type=int, help="new height to resize frames to")
    args.add_argument("--pad", action="store_true", default=False, help="pad instead of crop. Adds horizontal lines to the image.")
    args.add_argument("-o", "--output", default='dataset/', type=str, help="new height to resize frames to")

    config = args.parse_args()

    os.makedirs(config.output, exist_ok=True)
    
    # Convert all images to squared size/ar
    # dims = convert_images(config.images, config.output, config.pad)

    scenes = ["0000", "0001", "0002", "0100", "0101",
              "0102", "0400", "0401", "0500", "0503"]

    object_labels = ['person', 'car', 'vehicle', 'object', 'bike']
    label_map = generate_label_map([l for l in object_labels if l != 'object'])
    if not os.path.exists(f'{config.output}/label_map.pbtxt'):
        save_pbtxt(['person', 'car', 'vehicle', 'bike'], config.output)

    for dataset in ['train', 'eval']:
        if os.path.exists(f'{config.output}/{dataset}.record'):
            continue
   
        detections = pd.read_csv(f'{config.detections}/{dataset}_annotations.csv')
        w, h = detections.head(1)[['width','height']].values[0]

        if config.pad:
            stripes_height = int((w - h)/2)
            for dim in ['ymin', 'ymax']:
                detections[dim] = detections[dim].apply(lambda x: x+stripes_height)

        else: # image was cropped
            border = int(int(w)/2 - int(h)/2)

            detections['xmin'] = detections['xmin'].apply(lambda x: max(x - border, 0))
            detections['xmax'] = detections['xmax'].apply(lambda x: min(x - border, h))

            detections = detections[detections['xmin'] < detections['xmax']]

        detections['width'] = h
        detections.to_csv(f'{config.output}/{dataset}_annotations.csv', sep=',', index=False)

        imgs_dir = f'{config.output}/images'
        output_dir = f'{config.output}'
        # generate_tfrecord(f'{output_dir}/{dataset}.record', imgs_dir, f'{output_dir}/{dataset}_annotations.csv', label_map) 

        for scene in scenes:
            output_dir = f'{config.output}/scene_{scene}'
            os.makedirs(output_dir, exist_ok=True)
            scene_imgs = [f'{img.stem}.jpg' for img in Path(imgs_dir).glob('*.jpg') if f'VIRAT_S_{scene}' in str(img)]
            scene_detections = detections[detections['filename'].isin(scene_imgs)]
            scene_detections.to_csv(f'{output_dir}/{dataset}_annotations.csv', sep=',', index=False)

            generate_tfrecord(f'{output_dir}/{dataset}.record', imgs_dir, f'{output_dir}/{dataset}_annotations.csv', label_map) 
   

if __name__ == "__main__":
    main()
