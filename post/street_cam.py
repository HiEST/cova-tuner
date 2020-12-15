#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

import pandas as pd


def join_results(path, name, output):
    detections = [d for d in Path(path).glob(f'{name}.tmp.csv.*')]
    print(f'Found {len(detections)} results files from {name}.')
    data = pd.read_csv(detections[0])
    for i in range(1, len(detections)):
        data_ = pd.read_csv(detections[i])
        data = data.append(data_, ignore_index=False)

    data = data.rename(columns={
        'top_classes': 'top_scores',
        'top_scores': 'top_classes'
    })
    timestamp = data['timestamp'].apply(lambda x: pd.to_datetime(x))
    data['month'] = timestamp.dt.month
    data['day'] = timestamp.dt.day
    data['hour'] = timestamp.dt.hour
    data.to_csv(f'{output}/{name}.csv', sep=',', index=False)


def create_intermediate_data(data, threshold=0.5):
    columns = ['cam', 'timestamp', 'frame_id', 'model', 'score', 'class'] 
    new_data = []
    for cam in data.cam.unique():
        data_cam = data[data.cam == cam]
        for ts in data_cam.timestamp.unique():
            data_ts = data_cam[data_cam.timestamp == ts]
            for row_id, row in data_ts.iterrows(): 
                top_scores = row['top_scores'].split(',')
                top_classes = row['top_classes'].split(',')
                for s, c in zip(top_scores, top_classes):
                    s = float(s)
                    if s < threshold:
                        continue

                    c = int(c)
                    new_row = [
                        cam,
                        ts,
                        row['frame_id'],
                        row['model'],
                        s,
                        c
                    ]
                    new_data.append(new_row)

    new_data = pd.DataFrame(new_data, columns=columns)
    return new_data

def process_results(data, threshold=0.5):
    with open('../aux/mscoco.json', 'r') as f:
        labels = json.load(f)

    results = {}
    for entry_id in data.id.unique():
        results[entry_id] = {}
        for model in ['ref', 'edge', 'edge-motion']:
            data_ = data[(data.model == model) & (data.id == entry_id)]
            results[entry_id][model] = {
                'frame_results': [],
                'avg_score': 0.0,
                'avg_objects_detected': 0,
                'frames_with_detections': 0,
                'class_histogram': defaultdict(int),
                'class_histogram_norep': defaultdict(int)
            }
            for idx, row in data_.iterrows():
                class_idxs = row['top_classes'].split(',')
                scores = [float(s) for s in row['top_scores'].split(',')]

                results[entry_id][model]['frame_results'].append({
                    'classes': [],
                    'class_idxs': [],
                    'scores': [],
                    'avg_score': 0.0,
                    'class_histogram': defaultdict(int)
                })
                for c, s in zip(class_idxs, scores):
                    s = float(s)
                    if s < threshold:
                        continue

                    c = int(c)
                    obj_class = labels[str(c)]['name']
                    results[entry_id][model]['frame_results'][-1]['classes'].append(obj_class)
                    results[entry_id][model]['frame_results'][-1]['class_idxs'].append(c)
                    results[entry_id][model]['frame_results'][-1]['scores'].append(s)
                    results[entry_id][model]['frame_results'][-1]['class_histogram'][obj_class] += 1

                sum_scores = sum(results[entry_id][model]['frame_results'][-1]['scores'])
                num_scores = len(results[entry_id][model]['frame_results'][-1]['scores'])
                avg_score = 0 if num_scores == 0 else sum_scores / num_scores
                results[entry_id][model]['frame_results'][-1]['avg_score'] = avg_score
                results[entry_id][model]['avg_objects_detected'] += num_scores

                valid_frame = 1 if avg_score > 0 else 0
                results[entry_id][model]['frames_with_detections'] += valid_frame
                results[entry_id][model]['avg_score'] += avg_score
                for c, v in results[entry_id][model]['frame_results'][-1]['class_histogram'].items():
                    results[entry_id][model]['class_histogram'][c] += v
                    results[entry_id][model]['class_histogram_norep'][c] += 1

            valid_frames = results[entry_id][model]['frames_with_detections']
            results[entry_id][model]['avg_score'] = 0 if valid_frames == 0 else results[entry_id][model]['avg_score'] / valid_frames
            results[entry_id][model]['avg_objects_detected'] = 0 if valid_frames == 0 else results[entry_id][model]['avg_objects_detected'] / valid_frames

            del results[entry_id][model]['frame_results']

    return results


def main():
    args = argparse.ArgumentParser()
    args.add_argument("-i", "--input", required=True, default=None, type=str, help="path to the .tmp.csv files with the results")
    args.add_argument("-n", "--name", required=True, default=None, type=str, help="Name of the dataset")
    args.add_argument("-o", "--output", required=True, default=None, type=str, help="path to the output dir where to write the results")
    args.add_argument("-t", "--threshold", default=0.5, type=float, help="Confidence threshold.")

    config = args.parse_args()

    # if config.input is not None:
    if os.path.isdir(config.input):
        join_results(
            path=config.input,
            name=config.name,
            output=config.output)

        data = pd.read_csv(f'{config.output}/{config.name}.csv')
    else:
        data = pd.read_csv(config.input)

    new_data = create_intermediate_data(data, config.threshold)
    new_data.to_csv(f'{config.output}/{config.name}.threshold_{config.threshold}.csv',
                    sep=',', float_format='%.3f', index=False)

    # data['id'] = data.apply(lambda x: f"{x['cam']}/{x['timestamp']}", axis=1)
    # results = process_results(data, threshold=config.threshold)

    # with open(f'{config.output}/{config.name}.json', 'w') as f:
    #     json.dump(results, f)

if __name__ == '__main__':
    main()
