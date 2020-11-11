#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import base64
import json
import sys
from pathlib import Path

import cv2
import pandas as pd
import requests

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
    print(response)
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


def process_video(filename, url, framework='torch', no_show=False):
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
    print('send video')
    try:
        # headers = {'content-type': 'application/x-www-form-urlencoded'}
        with open(str(filename), 'rb') as video:
            r = requests.post(url,
                              files={'video': video},
                              data={
                                  'model': model,
                                  'device': 'cuda',
                                  'framework': framework
                              })
        print(r.text)
    except ConnectionResetError:
        return False, None
    except Exception:
        raise

    preds = json.loads(r.text)['data']

    return True, preds

def process_dataset(path, url,
                    framework='torch',
                    send_video=False,
                    no_show=False):

    columns = ['timestamp', 'date', 'hour', 'minute', 'frame_id',
               'model', 'top_classes', 'top_scores']
    subcolumns = ['frame_id', 'top_classes', 'top_scores']

    detections = pd.DataFrame([], columns=columns)

    for f in path.glob('*.mkv'):
        ts = f.stem

        date = '-'.join(str(ts).split('-')[:2])
        hour = str(ts).split('-')[3]
        minute = str(ts).split('-')[4]

        for model in ['edge']:
            if not send_video:
                ret, data = process_video(f, url, framework, no_show)

                if not ret:
                    break
            else:
                ret, data = offload_video(f, url, framework=framework)

                # frame_ids = [i for i, _ in enumerate(data)]
                scores = [','.join(f'{s:.3f}'
                                   for s in p['scores']) for p in data]
                classes = [','.join(str(c)
                                    for c in p['idxs']) for p in data]

                rows = [[i, s, classes[i]] for i, s in enumerate(scores)]
                df = pd.DataFrame(rows, columns=subcolumns)
            
            df['model'] = model
            df['timestamp'] = ts
            df['date'] = date
            df['hour'] = hour
            df['minute'] = minute

            detections = detections.append(df, ignore_index=True)
        break

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
    
    args.add_argument("--send-video",
                      default=False,
                      action="store_true",
                      help="Send whole video instead of frame by frame")
    
    args.add_argument("--no-show",
                      default=False,
                      action="store_true",
                      help="Don't show results in a window.")

    config = args.parse_args()
    url = url.format(config.port, 'video' if config.send_video else 'infer')

    path = Path(config.input)
    process_dataset(path, url, config.framework,
                    config.send_video, config.no_show)

if __name__ == '__main__':
    main()
