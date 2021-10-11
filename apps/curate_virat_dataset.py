
import argparse
import os
from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd

from tqdm import tqdm

DATA = '../data'
if not os.path.isdir(DATA):
    DATA = '../data_local'

VIRAT = f'{DATA}/virat/VIRAT Ground Dataset'

def read_virat(video):
    video_id = Path(video).stem
    video_path = Path(video).parent
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


def mark_static_objects(annotations, offset=25):
    annotations['static_frame'] = False
    annotations['static_object'] = False
    annotations['diff'] = 0

    objects = list(annotations['object_id'].unique())
    for obj in tqdm(objects):
        object_annotations = annotations[annotations['object_id'] == obj].copy()#.reset_index(drop=True)
        frames = np.asarray(object_annotations['frame_id'].unique())
        boxes = list(object_annotations[['xmin', 'ymin', 'xmax', 'ymax']].values)

        # frame_id after subtracting at least offset frames 
        object_annotations['compare_to_frame'] = object_annotations['frame_id'].apply(
            lambda x: 0 if x < frames[0]+offset else frames[frames <= x-offset].max())
        object_annotations['compare_to_index'] = object_annotations['frame_id'].apply(
            lambda x: 0 if x < frames[0]+offset else np.where(frames == frames[frames <= x-offset].max())[0][0])

        for coord_id, coord in enumerate(['xmin', 'ymin', 'xmax', 'ymax']):
            object_annotations[f'diff_{coord}'] = object_annotations.apply(
                lambda x: 0 if x['frame_id'] == 0 else abs(x[coord] - boxes[x['compare_to_index']][coord_id]),
                axis=1
            )

        object_annotations['diff'] = object_annotations.apply(
            lambda x: sum([x['diff_xmin'], x['diff_ymin'], x['diff_xmax'], x['diff_ymax']]),
            axis=1,
        )

        object_annotations['static_frame'] = object_annotations.apply(
            lambda x: False if x['diff'] > 5 else True,
            axis=1
        )
        
        # static_frames = []
        # # for frame_pos, frame_id in enumerate(frames):
        # for frame_pos in tqdm(range(len(frames))):
        #     frame_id = frames[frame_pos]
        #     if frame_pos < offset:
        #         continue

        #     coords = object_annotations[object_annotations['frame_id'] == frame_id][['xmin', 'ymin', 'xmax', 'ymax']].values[0]

        #     prev_frame = frames[frame_pos-offset]
        #     prev = object_annotations[object_annotations['frame_id'] == prev_frame][['xmin', 'ymin', 'xmax', 'ymax']].values[0]

        #     diff = [
        #         abs(coords[0]-prev[0]),
        #         abs(coords[1]-prev[1]),
        #         abs(coords[2]-prev[2]),
        #         abs(coords[3]-prev[3]),
        #     ]

            # if not any([d > 0 for d in diff]):
                # if obj == 56:
                #     print(f'[obj={obj},frame={frame_id}] {diff}')
                # static_frames.append(frame_id)

        # import pdb; pdb.set_trace()
        
        static_frames = object_annotations[object_annotations['static_frame'] == True]
            # if obj == 56:
            #     import pdb; pdb.set_trace()
            # annotations.loc[(annotations['object_id'] == obj) & (annotations['frame_id'].isin(static_frames)), 'static_frame'] = True
        static_object = False
        if len(static_frames) > 0.9*len(frames):
            static_object = True
        
        object_annotations['static_object'] = static_object

        annotations.loc[(annotations['object_id'] == obj), 'static_frame'] = object_annotations['static_frame']
        # import pdb; pdb.set_trace()
        annotations.loc[(annotations['object_id'] == obj), 'static_object'] = object_annotations['static_object']
        annotations.loc[(annotations['object_id'] == obj), 'diff'] = object_annotations['diff']

    return annotations


def main():
    parser = argparse.ArgumentParser(description='This program curates VIRAT dataset by removing static objects from the annotations.')
    parser.add_argument('-v', '--video', type=str, help='Path to a video or a sequence of image.', default=None)
    parser.add_argument('--show', default=False, action='store_true', help='Show window with results.')
    parser.add_argument("-f", "--fps", default=25, type=float, help="play fps")
    parser.add_argument("-j", "--jump-to", default=0, type=int, help="Jumpt to frame")
    
    args = parser.parse_args()
    
    video_id = Path(args.video).stem
    video = args.video
    assert os.path.isfile(video)

    annotations = read_virat(video)
    annotations = mark_static_objects(annotations)

    annotations.to_csv(f'annotations/{video_id}.no-static.csv', sep=',', index=False)
    if not args.show:
        return

    cap = cv2.VideoCapture(video)
    ret, frame = cap.read()

    frame_lat = 1.0 / args.fps
    last_frame = time.time()
    frame_id = args.jump_to

    while ret:
        detections = annotations[annotations.frame_id == frame_id]
        for i, det in detections.iterrows():
            (left, right, top, bottom) = det[['xmin', 'xmax', 'ymin', 'ymax']].values 

            # display_str = "{} (id={})".format(det['label'], det['object_id'])
            display_str = "{} ({}, {})".format(det['label'], det['object_id'], det['diff'])

            if det['static_frame'] or det['static_object']:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), color, 1)
            cv2.putText(frame, display_str, (int(left), int(top)-10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)

        # frame = cv2.resize(frame, (1280, 768))
        cv2.putText(frame, f'frame: {frame_id}', (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

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



if __name__ == '__main__':
    main()

# def get_virat(video):
#     annotations = read_virat(video)
#     return annotations
