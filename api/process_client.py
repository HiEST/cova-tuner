#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import base64
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import json
import os
from pathlib import Path
import shutil
import sys
import time

import cv2
import pandas as pd
import requests
from tqdm import tqdm

sys.path.append('../')
from utils.detector import label_map

url = 'http://localhost:{}/{}'
with open('imagenet.txt', 'r') as f:
    imagenet = json.load(f)


def draw_bboxes(img, preds, max_boxes=5, min_score=.5, color=(255, 0, 0)):
    boxes_drawn = 0

    img_ = img.copy()
    for idx, bb in enumerate(preds['boxes']):
        if boxes_drawn >= max_boxes:
            break

        if preds['scores'][idx] >= min_score:
            ymin, xmin, ymax, xmax = bb
            (left, right, top, bottom) = (
                xmin * img_.shape[1],  # left
                xmax * img_.shape[1],  # right
                ymin * img_.shape[0],  # top
                ymax * img_.shape[0]   # bottom
            )
            cv2.rectangle(img_, (int(left), int(top)),
                          (int(right), int(bottom)),
                          color, 1)
            obj_class = label_map[str(preds['idxs'][idx])]['name']
            if top - 10 < 0:
                top = top + 20
            cv2.putText(img_, str(obj_class), (int(left), int(top)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return img_


def process_response(response, detection=False):
    top5_str = []
    preds = json.loads(response.text)['data']
    if detection:
        boxes = preds['boxes']
        scores = [conf for conf in preds['scores']]
        idxs = preds['idxs']

        top5 = [idxs, scores, boxes]
        top5_str = []
        for idx, conf in zip(idxs, scores):
            top5_str.append(f'{label_map[str(idx)]["name"]} ({conf*100:.2f}%)')

        return top5, top5_str

    else:
        classes = []
        confs = []

        top5_str = []
        for pred in preds:
            obj_class = int(pred[0])
            conf = float(pred[1])

            top5_str.append(f'{imagenet[obj_class]} ({conf:.2f}%)')

            classes.append(obj_class)
            confs.append(conf)

        top5 = [classes, confs]


def offload_video_frames(filename, url, framework='torch', no_show=False):
    detection = True if framework == 'tf' else False

    ts = filename.stem

    date = '-'.join(str(ts).split('-')[:2])
    hour = str(ts).split('-')[3]
    minute = str(ts).split('-')[4]

    cap = cv2.VideoCapture(str(filename))
    frame_id = 0
    ret, frame = cap.read()
    data = []

    error = False
    while ret:
        frame = cv2.resize(frame, (800, 600))
        _, buf = cv2.imencode('.png', frame)
        png64 = base64.b64encode(buf)

        top5 = {}
        top5_str = {}
        for model in ['edge', 'ref']:
            try:
                r = requests.post(url, data={
                    'model': model,
                    'img': png64,
                    'device': 'cuda',
                    'framework': framework
                })
            except ConnectionResetError:
                error = True
                break

            preds_top5, preds_str = process_response(r, detection)
            top5[model] = preds_top5
            top5_str[model] = preds_str

        if error:
            break

        row = [
            ts,
            date,
            hour,
            minute,
            frame_id,
            top5['ref'][0],
            top5['ref'][1],
            top5['edge'][0],
            top5['edge'][1]
        ]
        data.append(row)

        if not no_show:
            cv2.putText(frame, 'Reference Model:', (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            for j, det in enumerate(top5_str['ref']):
                cv2.putText(frame, det, (10, 20 + 15 * (j+1)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.putText(frame, 'Edge Model:', (10, 500),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            for j, det in enumerate(top5_str['edge']):
                cv2.putText(frame, det, (10, 500 + 15 * (j+1)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if detection:
                predictions = {
                    'ref': {
                        'idxs': top5['ref'][0],
                        'scores': top5['ref'][1],
                        'boxes': top5['ref'][2]
                    },
                    'edge': {
                        'idxs': top5['edge'][0],
                        'scores': top5['edge'][1],
                        'boxes': top5['edge'][2]
                    }
                }

                frame = draw_bboxes(frame, predictions['ref'], max_boxes=5)
                frame = draw_bboxes(frame, predictions['edge'],
                                    max_boxes=5, color=(0, 128, 128))
            cv2.imshow('Detections', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                error = True
                break
                sys.exit()
            elif key == ord("n"):
                break

        ret, frame = cap.read()
        frame_id += 1

    if error:
        return False, data

    return True, data


def offload_video(filename, url, model='edge', framework='torch'):
    print(f'Processing {filename} with model {model}')
    try:
        with open(filename, 'rb') as video:
            r = requests.post(url,
                              files={'video': video},
                              data={
                                  'model': model,
                                  'device': 'cuda',
                                  'framework': framework
                              })
    except ConnectionResetError:
        return False, None
    except Exception:
        raise

    preds = json.loads(r.text)['data']

    return True, preds


def offload_video_threading(query_args):
    filename, url, model, framework = query_args
    ret, data = offload_video(filename, url, model, framework)
    return ret, model, data


def process_video(args):
    filename, url, model = args[:3]
    framework, offload_frames = args[3:5]
    no_show, max_workers = args[5:7]

    ts0 = time.time()
    if offload_frames:
        ret, data = offload_video_frames(
            filename=filename,
            url=url,
            framework=framework,
            model=model,
            no_show=no_show
        )

    else:
        ret, data = offload_video(filename=filename, url=url,
                                  model=model, framework=framework)

    assert ret
    scores = [','.join(f'{s:.3f}'
                       for s in p['scores']) for p in data]
    classes = [','.join(str(c)
                        for c in p['idxs']) for p in data]

    rows = [[i, s, classes[i]] for i, s in enumerate(scores)]

    ts1 = time.time()
    print(f'Time to process video {filename} '
          f'for model {model}: {ts1-ts0:.2f} seconds.')
    return rows


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def process_dataset(dataset, url,
                    framework='torch',
                    offload_frames=False,
                    no_show=False,
                    process_all=True,
                    move_when_done=None,
                    max_workers=1):

    columns = ['timestamp', 'date', 'hour', 'minute', 'frame_id',
               'model', 'top_classes', 'top_scores']
    subcolumns = ['frame_id', 'top_classes', 'top_scores']

    detections = pd.DataFrame([], columns=columns)

    processed_ts = defaultdict(bool)

    models = ['ref', 'edge']
    videos = []
    for f in dataset:
        ts = f.stem

        date = '-'.join(ts.split('-')[:2])
        hour = ts.split('-')[3]
        minute = ts.split('-')[4]

        if not process_all:
            date_hour = f'{date}-{hour}'
            if processed_ts[date_hour]:
                # print(f'Skipping processing {f.stem} because'
                #       f'{processed_ts[date_hour]} has been already processed.')
                continue
            else:
                videos.append(str(f))
                processed_ts[date_hour] = f.stem

    query_args = [
        [
            video, url, model,
            framework, offload_frames,
            no_show, max_workers
        ]
        for video, model in product(videos, models)
    ]

    query_chunks = chunks(query_args, max_workers)

    videos_processed = defaultdict(int)
    for chunk in query_chunks:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(process_video, chunk)

        for r_idx, r in enumerate(results):
            filename = chunk[r_idx][0]
            model = chunk[r_idx][2]
            ts = Path(filename).stem
            df = pd.DataFrame(r, columns=subcolumns)
            df['model'] = model
            df['timestamp'] = ts
            df['date'] = date
            df['hour'] = hour
            df['minute'] = minute

            detections = detections.append(df, ignore_index=True)
            detections.to_csv('detections.tmp.csv',
                              sep=',',
                              float_format='.2f',
                              index=False)

            videos_processed[filename] += 1

        if move_when_done is not None:
            videos_done = [
                k 
                for k, v in videos_processed.items()
                if v == len(models)
            ]

            for v in videos_done:
                shutil.move(v, move_when_done)

    detections.to_csv('detections.csv',
                      sep=',',
                      float_format='.2f',
                      index=False)


def main():
    global url
    args = argparse.ArgumentParser()
    args.add_argument("-i", "--input",
                      default="./",
                      type=str,
                      help="Path to the dataset to process.")

    args.add_argument("-p", "--port",
                      default=5000,
                      type=int,
                      help="Port to connect to.")

    args.add_argument("-f", "--framework",
                      default='torch',
                      choices=['torch', 'tf'],
                      help="Framework to use")

    args.add_argument("--offload-frames",
                      default=False,
                      action="store_true",
                      help="Offload frame by frame instead of the whole video")

    args.add_argument("--no-show",
                      default=False,
                      action="store_true",
                      help="Don't show results in a window.")

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

    config = args.parse_args()
    url = url.format(config.port, 'infer' if config.offload_frames else 'video')

    path = Path(config.input)
    dataset = sorted([f for f in path.glob('*.mkv')], key=os.path.getmtime)
    process_dataset(dataset, url,
                    framework=config.framework,
                    offload_frames=config.offload_frames,
                    no_show=config.no_show,
                    process_all=(not config.fast),
                    move_when_done=config.move,
                    max_workers=config.max_workers)


if __name__ == '__main__':
    main()
