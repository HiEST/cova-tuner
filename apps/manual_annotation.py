#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
from pathlib import Path
import sys
import time

import cv2
import imutils
import numpy as np
import pandas as pd

sys.path.append('../')
from utils.datasets import MSCOCO as label_map

COLORS = [
    (0, 255, 0),
    (0, 255, 255),
    (0, 0, 255)
]


refPt = []
selecting = False
detections = []
frame = None
frame_copy = None
annotation_class = ''
draw_frame = True
frame_moving = None
original_shape = None
frame_id = 0
sticky = False
sticky_detections = []

def annotate(event, x, y, flags, param):
    global refPt, selecting, detections, frame, frame_copy, \
        annotation_class, draw_frame, frame_moving, \
        original_shape, frame_id, sticky, sticky_detections

    if event == cv2.EVENT_LBUTTONDOWN:
        # print(f'Down {event}: {x} - {y}')
        refPt = [(x, y)]

        detections_to_remove = []
        for i, det in detections.iterrows():
            x_ = x * (original_shape[1]/frame.shape[1])
            y_ = y * (original_shape[0]/frame.shape[0])
            if x_ >= det['xmin'] and \
                    x_ <= det['xmax'] and \
                    y_ >= det['ymin'] and \
                    y_ <= det['ymax']:
                print(f'Selected detection of class {det["class_id"]}')

                # Remove it from the list of detections
                detections_to_remove.append(i)
                draw_frame = True
                break

        if len(detections_to_remove) > 0:
            detections = detections.drop(detections_to_remove)
            return

        # print(f'No detection selected')
        selecting = True

    elif event == cv2.EVENT_LBUTTONUP:
        if selecting:
            refPt.append((x, y))
            selecting = False
            # print(f'refPt: {refPt}')
            # print(f'reshape x: {original_shape[1]/frame.shape[1]}')
            # print(f'reshape y: {original_shape[0]/frame.shape[0]}')

            x1 = refPt[0][0] * (original_shape[1]/frame.shape[1])
            y1 = refPt[0][1] * (original_shape[0]/frame.shape[0])
            x2 = x * (original_shape[1]/frame.shape[1])
            y2 = y * (original_shape[0]/frame.shape[0])
            # print(f'frame shape: {original_shape}')
            # print(f'x1: {x1}, x2: {x2}')
            # print(f'y1: {y1}, y2: {y2}')
            # frame_id = detections['frame'].values[0]
            xmin = min(x1, x2)
            ymin = min(y1, y2)
            xmax = max(x1, x2)
            ymax = max(y1, y2)
            # print(f'xmin: {xmin}, xmax: {xmax}')
            # print(f'ymin: {ymin}, ymax: {ymax}')
            xmin = max(int(xmin), 0)
            ymin = max(int(ymin), 0)
            xmax = min(int(xmax), original_shape[1])
            ymax = min(int(ymax), original_shape[0])
            # print(f'xmin: {xmin}, xmax: {xmax}')
            # print(f'ymin: {ymin}, ymax: {ymax}')

            row = {
                'frame': frame_id,
                'class_id': annotation_class,
                'score': 1,
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax,
                'sticky': sticky
            }
            print(row)
            df = pd.DataFrame([row], columns=detections.columns)
            detections = detections.append(df, ignore_index=True)
            if sticky:
                detections['sticky'] = True
                if len(sticky_detections) == 0:
                    sticky_detections = df
                else:
                    sticky_detections = sticky_detections.append(df, ignore_index=True)
            cv2.rectangle(frame, refPt[0], refPt[1], (255, 0, 0), 2)
            
            display_str = "{}: 100%".format(label_map[annotation_class]['name'])
            cv2.putText(frame, display_str, (refPt[0][0], refPt[0][1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS[0], 2)
            cv2.imshow('Annotations', frame)

    elif event == cv2.EVENT_MOUSEMOVE:
        if selecting:
            coords = (x, y)
        
            frame_moving = frame.copy()

            cv2.rectangle(frame_moving, refPt[0], coords, (255, 0, 0), 2)
            cv2.imshow('Annotations', frame_moving)


def main():
    global detections, frame, annotation_class, draw_frame, \
        original_shape, frame_id, sticky, sticky_detections

    args = argparse.ArgumentParser()
    args.add_argument("-v", "--videos", required=True, nargs='+', default=None, help="path to the video file")
    args.add_argument("-o", "--output", default=None, help="path save new detections.")
    args.add_argument("-d", "--detections", default=None, help="ground truth")
    args.add_argument("-t", "--threshold", default=0.5, type=float, help="score threshold")
    args.add_argument("-f", "--fps", default=25, type=float, help="play fps")
    args.add_argument("-c", "--classes", nargs='+', default=None, help="valid classes")
    args.add_argument("--dataset", type=str, default=None, help="name of the dataset")
    args.add_argument("--no-stop", default=False, action='store_true', help="Don't stop unless there are detections over the threshold")
    args.add_argument("--loop", default=False, action='store_true', help="Start again after finishing")
    args.add_argument("--append", default=False, action='store_true', help="Append to output")


    config = args.parse_args()

    cv2.namedWindow('Annotations')
    cv2.setMouseCallback('Annotations', annotate)

    valid_class_ids = [c['id'] for c in label_map.values() if c['name'] in config.classes]
    annotation_class = valid_class_ids[0]
    print(f'annotation_class: {annotation_class}')

    max_boxes = 10
    min_score = config.threshold

    sticky = False
    dont_stop = config.no_stop
    max_frames = 20
    output_pkl = None
    skip_until = 0
    for video_id, video in enumerate(config.videos):
        checked_detections = None
        if config.append:
            if os.path.exists('{}/{}.pkl'.format(config.output, Path(video).stem)):
                print(f'Found previous annotations')
                checked_detections = pd.read_csv('{}/{}.csv'.format(config.output, Path(video).stem))
                skip_until = max(checked_detections['frame'].values)
                print(f'skipping until frame {skip_until}')

        previous_frames = []
        oldest_frame_id = 0
        newest_frame_id = 0
        load_prev_frame = False
        frame_id = 0
        keep = False

        cap = cv2.VideoCapture(video)
        ret, frame = cap.read()
        frame_copy = frame.copy()
        original_shape = frame.shape
        previous_frames.append(frame_copy)

        results = pd.read_pickle('{}/{}.pkl'.format(config.detections, Path(video).stem), "bz2")
        results['sticky'] = False
        if checked_detections is not None:
            results = results[results.frame > skip_until]
            results = results.append(checked_detections, ignore_index=True)
        columns = results.columns
        # import pdb; pdb.set_trace()
        columns_to_keep = ['filename', 'class_id', 'xmin', 'ymin', 'xmax', 'ymax', 'score']
        results_to_keep = pd.DataFrame([], columns=columns_to_keep)
        while ret:
            detections = results[
                (results.frame == frame_id) &
                (results.class_id.isin(valid_class_ids)) &
                (results.score >= min_score) &
                (results.sticky == False)
            ]
            num_detections = len(detections)
            if len(sticky_detections) > 0:
                detections = detections.append(sticky_detections, ignore_index=True)

            # import pdb; pdb.set_trace()
            key = 0
            draw_frame = True
            print(f'a total of {len(detections)} detections')
            print(detections)
            while True and skip_until <= frame_id:
                if key == 9 or key == 32:
                    print(key)
                    print('skip')
                    break
                elif key == ord('q'): 
                    print('quit')
                    break
                elif key == ord('z'):
                    print('undo')
                    detections.drop(detections.tail(1).index, inplace=True)
                    draw_frame = True
                elif key == ord('k'):  # keep image and annotation
                    keep = True
                elif key >= ord('1') and key <= ord('9'):
                    n = key - ord('1')
                    print(f'number hit: {n}')
                    if n <= len(valid_class_ids):
                        annotation_class = valid_class_ids[n]
                        draw_frame = True
                elif key == ord('s'): # sticky bounding box
                    if sticky:
                        sticky = False
                        # sticky_detections = None
                    else:
                        sticky = True
                    draw_frame = True
                elif key == ord('d'):
                    sticky_detections = []
                    draw_frame = True
                elif key == ord('c'):
                    if dont_stop:
                        dont_stop = False
                    else:
                        dont_stop = True
                elif key == 44: # left arrow
                    if frame_id > oldest_frame_id:
                        frame_id -= 1
                        load_prev_frame = True
                        print('prev frame')
                        break
                elif key == 46: # right arrow
                    if frame_id < newest_frame_id:
                        frame_id += 1
                        load_prev_frame = True
                        print('next frame')
                        break

                elif key != 255:
                    print(f'key: {key} ({time.time()})')


                if draw_frame:
                    frame = frame_copy.copy()
                    infos = [f'Annotating {label_map[annotation_class]["name"]}']
                    if sticky:
                        infos.append(f'Sticky detection: ON ({len(sticky_detections)} saved)')
                    for i, info in enumerate(infos):
                        cv2.putText(frame, info, (30, 100*(i+1)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 3, COLORS[0], 3)

                    for i, det in detections.iterrows():
                        score = det['score']
                        class_id = int(det['class_id'])
                        class_name = label_map[class_id]['name']
                        if class_name not in config.classes:
                            continue
                        (left, right, top, bottom) = det[['xmin', 'xmax', 'ymin', 'ymax']].values 

                        display_str = "{}: {}%".format(class_name, int(100 * score))

                        cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), COLORS[0], 2)
                        cv2.putText(frame, display_str, (int(left), int(top)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS[0], 2)

                    frame = cv2.resize(frame, (1280, 768))
                    cv2.putText(frame, f'frame: {frame_id}', (10, 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                    cv2.imshow('Annotations', frame)
                    draw_frame = False

                key = cv2.waitKey(1) & 0xFF
                if dont_stop and num_detections == 0:
                    break

            if key == ord("q"):
                break
                sys.exit()
            elif key == ord('c'):
                dont_stop = False

            if checked_detections is None:
                checked_detections = detections
            else:
                checked_detections = checked_detections[checked_detections.frame != frame_id]
                checked_detections = checked_detections.append(detections, ignore_index=True)

            if keep:
                filename = f'{config.dataset}/{Path(video).stem}_{frame_id}.jpg'
                height = 300
                width = 300 * (frame_copy.shape[1]/frame_copy.shape[0])
                img = cv2.resize(frame_copy, (int(width), height))
                # import pdb; pdb.set_trace()
                cv2.imwrite('{}/{}'.format(config.output, filename), img)
                detections['filename'] = filename
                results_to_keep = results_to_keep.append(detections[columns_to_keep], ignore_index=True)
                print(results_to_keep)
                keep = False
            results = results[results.frame != frame_id]
            results = results.append(detections, ignore_index=True)

            print(f'{len(checked_detections)} detections saved ({len(detections)} from current frame).')

            if not load_prev_frame:
                ret, frame = cap.read()
                previous_frames.append(frame.copy())
                newest_frame_id += 1
                frame_id = newest_frame_id
                if len(previous_frames) > max_frames:
                    previous_frames = previous_frames[-max_frames:]
                    oldest_frame_id = newest_frame_id - max_frames
                print('oldest: {oldest_frame_id}')
                print('newest: {newest_frame_id}')
            else:
                print(f'accessing frame {frame_id-oldest_frame_id-1}')
                frame = previous_frames[frame_id-oldest_frame_id-1].copy()
                load_prev_frame = False
                print(f'checked detections: {len(checked_detections)}')

            if ret:
                frame_copy = frame.copy()
            else:
                if config.loop:
                    previous_frames = []
                    oldest_frame_id = 0
                    newest_frame_id = 0
                    load_prev_frame = False
                    frame_id = 0

                    cap = cv2.VideoCapture(video)
                    ret, frame = cap.read()
                    frame_copy = frame.copy()
                    original_shape = frame.shape
                    previous_frames.append(frame_copy)

        output_dir = config.output
        results_to_keep.to_pickle('{}/{}/{}.pkl'.format(output_dir, config.dataset, Path(video).stem), 'bz2')
        results_to_keep.to_csv('{}/{}/{}.csv'.format(output_dir, config.dataset, Path(video).stem), index=False)
        if checked_detections is None:
            checked_detections = pd.DataFrame([], columns=columns)
        print(checked_detections)
        if output_dir is None:
            output_dir = '.'

        checked_detections.to_pickle('{}/{}.pkl'.format(output_dir, Path(video).stem), 'bz2')
        checked_detections.to_csv('{}/{}.csv'.format(output_dir, Path(video).stem))


if __name__ == "__main__":
    main()
