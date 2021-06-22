import argparse
from pathlib import Path
import os
import sys

import cv2
import numpy as np
import pandas as pd

# import tensorflow as tf

from edge_autotune.motion.motion_detector import non_max_suppression_fast
from edge_autotune.dnn.metrics import get_precision_recall, evaluate_predictions

PATH = '/home/drivas/Workspace/remote/bscdc-ml02/training/edgeautotuner/apps/'
VIRAT = '../data_local/virat/VIRAT Ground Dataset'

colors = {
        'FN': (255, 0, 0),
        'TP': (0, 255, 0),
        'FP': (0, 0, 255)
}


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
    annotations = annotations[annotations.label != 'bike']
    annotations = annotations.rename({'current_frame': 'frame_id'}, axis=1)
    return annotations


def draw_detection(frame, box, label, color=(255,0,0)):
    if frame is None:
        return
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    cv2.putText(frame, label, (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        

def main():
    parser = argparse.ArgumentParser(description='This program evaluates accuracy of a CNN after using different BGS methods.')
    parser.add_argument('-v', '--video', type=str, help='Path to a video or a sequence of image.', default=None)
    # parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='mog')
    # parser.add_argument('--gt', type=str, help='Path to ground-truth.')
    # parser.add_argument('--bgs', type=str, help='Path to BGS results.')
    parser.add_argument('--show', default=False, action='store_true', help='Show window with results.')
    # parser.add_argument('--write', default=False, action='store_true', help='Write results as images.')
    # parser.add_argument('--model', default=None, help='Path to CNN model.')
    parser.add_argument('--methods', default=['gt'], nargs='+', help='Method.')
    parser.add_argument('--classes', default=['person'], nargs='+', help='Valid classes.')
    parser.add_argument('--start', type=int, default=50, help='Start frame.')
    # parser.add_argument('--min-score', type=float, default=0.1, help='Minimum score to accept a detection.')

    args = parser.parse_args()

    if os.path.exists(args.video):
        video_path = args.video
    else:
        video_path = os.path.join(VIRAT, 'videos_original', f'{args.video}.mp4')
    video_id = Path(video_path).stem

    if args.show:
        cap = cv2.VideoCapture(video_path)

    dets = pd.read_csv(os.path.join(os.getcwd(), f'{video_id}_detections.csv'))
    dets = dets[dets.frame_id >= args.start]
    dets = dets[(dets.xmin < dets.xmax) & (dets.ymin < dets.ymax)]
    dets = dets[dets.method.isin(args.methods)].copy().reset_index(drop=True)
    gt = read_virat(video_id)

    frame = None
    frames = sorted(dets.frame_id.unique())

    results = {
        method:{
            c:{'TP': 0, 'FP': 0, 'FN': 0} for c in args.classes
        } for method in args.methods
    }

    last_frame_decoded = -1

    for method in args.methods:
        print(f'{method}:')
        print(results[method])
        df_method = dets[(dets.method == method)].copy().reset_index(drop=True)

        for frame_id in frames:    
            # if args.show and last_frame_decoded < frame_id:
            cap.set(1, frame_id)
            _, frame = cap.read()
            # last_frame_decoded = frame_id

            df_frame = df_method[df_method.frame_id == frame_id].copy().reset_index(drop=True)

            # boxes = df_frame[['xmin', 'ymin', 'xmax', 'ymax']].values
            # scores = df_frame['score'].values
            # labels = df_frame['label'].values
            # selected_indices = tf.image.non_max_suppression(
            #                 boxes=boxes, scores=scores, 
            #                 max_output_size=100,
            #                 iou_threshold=0.5,
            #                 score_threshold=0.05)
                
            # if len(selected_indices) != len(df_frame):
            #     print(f'Dropping {len(df_frame)-len(selected_indices)} boxes with')
            # df_frame = df_frame.iloc[selected_indices].copy().reset_index(drop=True)

            gt_frame = gt[gt.frame_id == frame_id].copy().reset_index(drop=True)
            # for i, box in gt_frame.iterrows():
            #     draw_detection(frame, box[['xmin', 'ymin', 'xmax', 'ymax']], box['label'], color=(255,0,0))

            if not len(df_frame):
                for c in args.classes:
                    gt_class = gt_frame[gt_frame.label == c].copy().reset_index(drop=True)
                    results[method][c]['FP'] += len(gt_class)
                continue

            for c in args.classes:            
                gt_class = gt_frame[gt_frame.label == c].copy().reset_index(drop=True)
                # frame_pr = get_precision_recall(df_frame, gt_class, c)
                # results[method][c]['TP'] += frame_pr[2][0]
                # results[method][c]['FP'] += frame_pr[2][1]
                # results[method][c]['FN'] += frame_pr[2][2]

                frame_pr = evaluate_predictions(df_frame, gt_class, c)
                results[method][c]['TP'] += len(frame_pr['TP'])
                results[method][c]['FP'] += len(frame_pr['FP'])
                results[method][c]['FN'] += len(frame_pr['FN'])

                # print(f'\t[{frame_id}] Precision: {frame_pr[0]:.2f} - Recall: {frame_pr[1]:.2f} '
                #     f'(TP:{frame_pr[2][0]},FP:{frame_pr[2][1]},FN:{frame_pr[2][2]})')

                for m in frame_pr.keys():
                    for box in frame_pr[m]:
                        print(box)
                        draw_detection(frame, box[['xmin', 'ymin', 'xmax', 'ymax']], c, color=colors[m])
                
            if args.show:
                cv2.imshow(method, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    sys.exit(1)
        
        # print(results[method])
        # print(results)

    # print(results)
    df = []
    for method in args.methods:
        # print(method)
        for c in args.classes:
            # print(f'\t{c}:')
            r = results[method][c]
            # print(f'\t\t{r}')
            for metric in ['TP','FP','FN']:
                df.append([
                    method,
                    c,
                    metric,
                    r[metric]
                ])

            precision = 0 if r['TP'] == 0 else r['TP'] / (r['TP'] + r['FP'])
            recall = 0 if r['TP'] == 0 else r['TP'] / (r['TP'] + r['FN'])
            
            # Av. Precision
            df.append([
                method,
                c,
                'precision',
                precision
            ])
            df.append([
                method,
                c,
                'recall',
                recall
            ])
        
    columns = ['method','label', 'metric', 'value']
    df = pd.DataFrame(df, columns=columns)
    print(df)
    df.to_csv(f'{video_id}_map.csv')

if __name__ == '__main__':
    main()