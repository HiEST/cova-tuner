
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

VIRAT = '../data_local/virat'

def read_virat(video_id):
    fn = os.path.join(VIRAT, 'annotations', video_id + '.viratdata.objects.txt')
    annotations = pd.read_csv(fn, header=None, sep=' ', index_col=False)
    annotations.columns = ['object_id', 'object_duration', 'current_frame', 
                            'xmin', 'ymin', 'width', 'height', 'object_type']

    annotations = annotations[annotations.object_type > 0]
    annotations['xmax'] = annotations['xmin'] + annotations['width']
    annotations['ymax'] = annotations['ymin'] + annotations['height']
    object_labels = ['person', 'car', 'vehicle', 'object', 'bike']
    annotations['label'] = annotations['object_type'].apply(lambda obj: object_labels[obj-1])
    annotations = annotations[annotations.label != 'object']
    # annotations = annotations[annotations.label != 'bike']
    annotations = annotations.rename({'current_frame': 'frame_id'}, axis=1)
    return annotations


def main():
    parser = argparse.ArgumentParser(description='This program curates VIRAT dataset by removing static objects from the annotations.')
    parser.add_argument('-v', '--video', type=str, help='Path to a video or a sequence of image.', default=None)
    parser.add_argument('--show', default=False, action='store_true', help='Show window with results.')
    
    args = parser.parse_args()
    
    video_id = Path(args.video).stem
    video = args.video
    assert os.path.isfile(video)
    
    cap = cv2.VideoCapture(video)

def get_virat(video_id):
    annotations = read_virat(video_id)
    return annotations
