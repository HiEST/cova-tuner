#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import json
import os
from pathlib import Path
import requests
import shutil
import time

import pandas as pd


def offload_video(filename, model='edge', framework='torch'):
    try:
        servers = ['http://localhost:5000/video',
                   'http://localhost:5001/video']
        for server in servers:
            r = requests.get(server,
                             data={
                                'video': filename,
                                'model': model})

            status, data = json.loads(r.text)['data']
            if status == 'ready':
                print(f'status: {status} ({len(data)} results)')
                return True, data

    except ConnectionResetError:
        return False, None
    except Exception:
        raise

    return False, []


def process_video(args):
    filename = args[0]
    model = args[1]
    framework = args[2]

    ts0 = time.time()
    ret, data = offload_video(filename=filename,
                              model=model,
                              framework=framework)

    if not ret:
        return []

    assert ret
    scores = [','.join(f'{s:.3f}'
                       for s in p['scores']) for p in data]
    classes = [','.join(str(c)
                        for c in p['idxs']) for p in data]

    rows = [[i, s, classes[i]] for i, s in enumerate(scores)]

    ts1 = time.time()
    print(f'Obtained {len(rows)} results for {filename} '
          f'and model {model} in {ts1-ts0:.2f} seconds.')
    return rows


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def split_date(ts):
    ts = pd.to_datetime(ts)
    date = str(ts.date())
    hour = str(ts.hour)
    minute = str(ts.minute)

    return date, hour, minute


def process_dataset(dataset, dataset_name,
                    framework='torch',
                    move_when_done=None,
                    max_workers=1):

    columns = ['cam', 'timestamp', 'date', 'hour', 'minute', 'frame_id',
               'model', 'top_scores', 'top_classes']
    subcolumns = ['frame_id', 'top_scores', 'top_classes']

    detections = pd.DataFrame([], columns=columns)

    processed_ts = defaultdict(bool)

    videos_processed = defaultdict(int)
    models = ['ref', 'edge']
    videos = []
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
    query_args = [
        [video, model, framework]
        for video, model in product(videos, models)
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(process_video, query_args)

        for r_idx, r in enumerate(results):
            filename = query_args[r_idx][0]
            model = query_args[r_idx][1]
            if len(r) == 0:
                continue
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
            print(f'{len(detections)} detections.')
            print(f'Writing results to {dataset_name}.tmp.csv')
            detections.to_csv(f'{dataset_name}.tmp.csv',
                              sep=',',
                              float_format='.2f',
                              index=False)

            print(f'Results from {filename} just processed.')

            videos_processed[filename] += 1
            if move_when_done is not None and \
                    videos_processed[filename] == len(models):
                shutil.move(filename, move_when_done)
                del videos_processed[filename]

    detections.to_csv(f'{dataset_name}.csv',
                      sep=',',
                      float_format='.2f',
                      index=False)


def main():
    args = argparse.ArgumentParser()
    args.add_argument("-i", "--input",
                      default="./",
                      type=str,
                      help="Path to the dataset to process.")

    args.add_argument("-n", "--name",
                      default="dataset",
                      type=str,
                      help="Name of the dataset. "
                           "Used to name the results file.")

    args.add_argument("-f", "--framework",
                      default='torch',
                      choices=['torch', 'tf'],
                      help="Framework to use")

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

    path = Path(config.input)
    if os.path.isfile(config.input):
        dataset = [path]
    else:
        extensions = ['.mkv', '.mp4', '.webm']
        dataset = sorted([f for f in path.rglob('*') if f.suffix in extensions],
                         key=os.path.getmtime)

    process_dataset(dataset=dataset,
                    dataset_name=config.name,
                    framework=config.framework,
                    move_when_done=config.move,
                    max_workers=config.max_workers)


if __name__ == '__main__':
    main()
