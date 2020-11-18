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
import random
import shutil
import sys
import time

import cv2
import pandas as pd
import requests
from tqdm import tqdm

sys.path.append('../')
from utils.detector import label_map

NUM_SERVERS=2

url = 'http://localhost:{}/{}'
# with open('imagenet.txt', 'r') as f:
#     imagenet = json.load(f)


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


def offload_single_frame(img, url, model, framework):
    _, buf = cv2.imencode('.png', img)
    png64 = base64.b64encode(buf)
    try:
        r = requests.post(url, data={
            'model': model,
            'img': png64,
            'device': 'cuda',
            'framework': framework
        })
    except ConnectionResetError:
        return False, None

    return True, json.loads(r.text)['data']


def offload_video_frames(filename, url, model,
                         framework='torch', no_show=False):

    # detection = True if framework == 'tf' else False

    cap = cv2.VideoCapture(filename)
    frame_id = 0
    ret, frame = cap.read()

    error = False
    detections = []
    while ret:
        frame = cv2.resize(frame, (800, 600))

        ret, data = offload_single_frame(frame, url, model, framework)
        print(f'[{filename}] {ret} --> {data}')
        # detections.append(data)

        assert ret
        yield ret, data

        ret, frame = cap.read()


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
        frame_generator = offload_video_frames(
                                filename=filename,
                                url=url,
                                framework=framework,
                                model=model,
                                no_show=no_show)
        data = [frame for ret, frame in frame_generator if ret]
        print(f'Received frames: {len(data)}')
        print(data)

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


def split_date(ts):
    date = str(ts.date())
    hour = str(ts.hour)
    minute = str(ts.minute)

    return date, hour, minute


def process_dataset(dataset, url, dataset_name,
                    framework='torch',
                    offload_frames=False,
                    no_show=False,
                    process_all=True,
                    move_when_done=None,
                    max_workers=1):

    columns = ['cam', 'timestamp', 'date', 'hour', 'minute', 'frame_id',
               'model', 'top_scores', 'top_classes']
    subcolumns = ['frame_id', 'top_scores', 'top_classes']

    detections = pd.DataFrame([], columns=columns)

    processed_ts = defaultdict(bool)

    models = ['ref', 'edge']
    videos = []
    if process_all:
        videos = [str(f) for f in dataset]
    else:
        for f in dataset:
            ts = f.stem.split('.')[0]
            cam = f.stem.split('.')[1]
            date, hour, minute = split_date(ts)

            date_hour = f'[{cam}] {date}:{hour}'
            if processed_ts[date_hour]:
                continue
            else:
                videos.append(str(f))
                processed_ts[date_hour] = f.stem

    # FIXME: Find another way to load balance requests
    servers = [url, url.replace('5000', '5001')]
    query_args = [
        [
            video, random.choice(servers), model,
            framework, offload_frames,
            no_show, max_workers
        ]
        for video, model in product(videos, models)
    ]

    videos_processed = defaultdict(int)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(process_video, query_args)

        for r_idx, r in enumerate(results):
            filename = query_args[r_idx][0]
            model = query_args[r_idx][2]
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
            detections.to_csv(f'{dataset_name}.tmp.csv',
                              sep=',',
                              float_format='.2f',
                              index=False)

            print(f'Results from {filename} just processed.')

            videos_processed[filename] += 1

        if move_when_done is not None:
            videos_done = [
                k
                for k, v in videos_processed.items()
                if v == len(models)
            ]

            for v in videos_done:
                shutil.move(v, move_when_done)
                del videos_processed[v]

    detections.to_csv(f'{dataset_name}.csv',
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

    args.add_argument("-n", "--name",
                      default="dataset",
                      type=str,
                      help="Name of the dataset. Used to name the results file.")

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
    if os.path.isfile(config.input):
        dataset = [path]
    else:
        extensions = ['.mkv', '.mp4', '.webm']
        dataset = sorted([f for f in path.rglob('*') if f.suffix in extensions], key=os.path.getmtime)

    print(dataset)
    process_dataset(dataset, url,
                    dataset_name=config.name,
                    framework=config.framework,
                    offload_frames=config.offload_frames,
                    no_show=config.no_show,
                    process_all=(not config.fast),
                    move_when_done=config.move,
                    max_workers=config.max_workers)


if __name__ == '__main__':
    main()
