#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import base64
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests

from ..utils.detector import label_map

url = 'http://localhost:{}/infer'
with open('imagenet.txt', 'r') as f:
    imagenet = json.load(f)


def process_response(response, detection=False):
    top5_str = []
    print(response)
    preds = json.loads(response.text)['data']
    if detection:
        boxes = preds['boxes']
        scores = [conf*100 for conf in preds['scores']]
        idxs = preds['idxs']

        top5 = [idxs, scores]
        top5_str = []
        for idx, conf in zip(idxs, scores):
            top5_str.append(f'{label_map[str(idx)]} ({conf:.2f}%)')

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

    config = args.parse_args()
    url = url.format(config.port)

    path = Path(config.input)

    data = []
    error = False
    for f in path.glob('*.mkv'):
        ts = f.stem

        date = '-'.join(str(ts).split('-')[:2])
        hour = str(ts).split('-')[3]
        minute = str(ts).split('-')[4]
        # print(f'day: {date} @ {hour}:{minute}')

        cap = cv2.VideoCapture(str(f))
        frame_id = 0
        ret, frame = cap.read()
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
                        'framework': config.framework
                    })
                except ConnectionResetError:
                    error = True
                    break

                detection = True if config.framework == 'tf' else False
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
    
            cv2.putText(frame, 'Reference Model:', (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            for j, det in enumerate(top5_str['ref']):
                cv2.putText(frame, det, (10, 20 + 15* (j+1)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.putText(frame, 'Edge Model:', (10, 500),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            for j, det in enumerate(top5_str['edge']):
                cv2.putText(frame, det, (10, 500 + 15 * (j+1)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow('Detections', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                sys.exit()
            elif key == ord("n"):
                break

            ret, frame = cap.read()
            frame_id += 1

        if error:
            break

    columns = ['timestamp', 'date', 'hour', 'minute', 'frame_id',
               'top5_ref_classes', 'top5_ref_conf',
               'top5_edge_classes', 'top5_edge_conf']

    detections = pd.DataFrame(data, columns=columns)
    detections.to_csv('detections.csv', sep=',', float_format='.2f', index=False)

if __name__ == '__main__':
    main()
