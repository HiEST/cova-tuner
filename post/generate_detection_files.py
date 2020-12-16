#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
from pathlib import Path
import sys

import pandas as pd

sys.path.append('../')
from utils.datasets import MSCOCO as label_map


def generate_detection_files(detections, output_dir, prefix, ground_truth=False, threshold=0.0):
    if 'label' not in detections.columns:
        detections['label'] = detections.class_id.apply(lambda x: label_map[str(x)]['name'].replace(' ', '_'))
    else:
        detections['label'] = detections.label.apply(lambda x: x.replace(' ', '_'))


    output_dir = f'{output_dir}/{"groundtruths" if ground_truth else "detections"}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frames = detections.frame.unique()
    for frame in frames:
        frame_detections = detections[(detections.frame == frame) & (detections.score > threshold)]
        columns = ['label', 'score', 'xmin', 'ymin', 'xmax', 'ymax']
        if ground_truth:
            columns = ['label', 'xmin', 'ymin', 'xmax', 'ymax']

        frame_detections = frame_detections[columns]

        output_file = f'{output_dir}/{prefix}_{frame}.txt'
        with open(output_file, 'w') as f:
            frame_detections.to_csv(f, sep=' ', index=False, header=False)


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--ground-truth", default=False, action="store_true", help="detections are ground-truth")
    args.add_argument("-o", "--output", default="detection_files/", type=str, help="path where detection files will be stored")
    args.add_argument("-p", "--prefix", default=None, type=str, help="prefix to name detection files")
    args.add_argument("-t", "--threshold", default=0, type=float, help="score threshold")

    args.add_argument('pickle')

    config = args.parse_args()

    prefix = config.prefix
    if prefix == None:
        prefix = Path(config.pickle).stem.split('.')[0]

    detections = pd.read_pickle(config.pickle, compression='bz2')
    generate_detection_files(detections, config.output, prefix, config.ground_truth, config.threshold)

if __name__ == "__main__":
    main()
