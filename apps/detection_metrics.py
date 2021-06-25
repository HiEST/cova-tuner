import argparse
from pathlib import Path
import os
import sys

import cv2
import numpy as np
import pandas as pd

import src.evaluators.coco_evaluator as coco_evaluator
import src.evaluators.pascal_voc_evaluator as pascal_voc_evaluator
import src.utils.converter as converter
import src.utils.general_utils as general_utils
from src.bounding_box import BoundingBox
from src.utils.enumerators import (BBFormat, BBType, CoordinatesType,
                                   MethodAveragePrecision)


def generate_metric_fns(dets_path, dets):
    os.makedirs(dets_path, exist_ok=True)
    
    dets['width'] = dets.apply(lambda x: x['xmax'] - x['xmin'], axis=1)
    dets['height'] = dets.apply(lambda x: x['ymax'] - x['ymin'], axis=1)
    frames = dets.frame_id.unique()
    for frame_id in frames:
        frame_dets = dets[dets.frame_id == frame_id][['label', 'score', 'xmin', 'ymin', 'width', 'height']]
        frame_dets.to_csv(os.path.join(dets_path, f'{frame_id}.txt'), sep=' ', index=False, header=False)
        
    
def generate_gts(gt_path, gt, frames):
    os.makedirs(gt_path, exist_ok=True)
    
    for frame_id in frames:
        frame_gt = gt[gt.frame_id == frame_id][['label', 'xmin', 'ymin', 'width', 'height']]
        frame_gt.to_csv(os.path.join(gt_path, f'{frame_id}.txt'), sep=' ', index=False, header=False)        


def plot_bb_per_classes(dict_bbs_per_class,
                        horizontally=True,
                        rotation=0,
                        show=False,
                        extra_title=''):
    plt.close()
    if horizontally:
        ypos = np.arange(len(dict_bbs_per_class.keys()))
        plt.barh(ypos, dict_bbs_per_class.values(), align='edge')
        plt.yticks(ypos, dict_bbs_per_class.keys(), rotation=rotation)
        plt.xlabel('amount of bounding boxes')
        plt.ylabel('classes')
    else:
        plt.bar(dict_bbs_per_class.keys(), dict_bbs_per_class.values())
        plt.xlabel('classes')
        plt.ylabel('amount of bounding boxes')
    plt.xticks(rotation=rotation)
    title = f'Distribution of bounding boxes per class {extra_title}'
    plt.title(title)
    if show:
        plt.tick_params(axis='x', labelsize=10) # Set the x-axis label size
        plt.show(block=True)
    return plt  


def evaluate_predictions(dets, gts, methods, valid_classes, output):
    # Generate ground truth files if do not exist yet
    dir_gts = f'{output}/gts'
    frames = dets[dets['method'] == 'full_frame']['frame_id'].unique()
    if not os.path.exists(dir_gts):
        generate_gts(dir_gts, gts, frames)
    
    # Get annotations (ground truth and detections)
    gt_bbs = converter.text2bb(
        dir_gts,
        bb_type=BBType.GROUND_TRUTH,
        bb_format=BBFormat.XYWH,
        type_coordinates=CoordinatesType.ABSOLUTE)

    gt_bbs = [bb for bb in gt_bbs if bb.get_class_id() in valid_classes]
    assert len(gt_bbs)

    coco_metrics = ['class', 'AP', 'total positives', 'TP', 'FP']
    coco_detail = pd.DataFrame([], columns=coco_metrics)
    coco_summary = None
    voc_metrics = ['class', 'AP', 'total positives', 'total TP', 'total FP', 'iou']
    voc_summary = pd.DataFrame([], columns=voc_metrics)
    for method in methods:
        dir_dets = f'{output}/dets/{method}'
        dets_method = dets[dets['method'] == method].copy().reset_index(drop=True)
        generate_metric_fns(dir_dets, dets_method)

        det_bbs = converter.text2bb(dir_dets,
            bb_type=BBType.DETECTED,
            bb_format=BBFormat.XYWH,
            type_coordinates=CoordinatesType.ABSOLUTE)
        det_bbs = [bb for bb in det_bbs if bb.get_class_id() in valid_classes]
        assert len(det_bbs)

        # Uncomment to plot the distribution bounding boxes per classes
        # dict_gt = BoundingBox.get_amount_bounding_box_all_classes(gt_bbs, reverse=False)
        # plot_bb_per_classes(dict_gt, horizontally=True, rotation=0, show=True, extra_title=' (groundtruths)')
        # clases_gt = [b.get_class_id() for b in gt_bbs]
        # dict_det = BoundingBox.get_amount_bounding_box_all_classes(det_bbs, reverse=True)
        # general_utils.plot_bb_per_classes(dict_det, horizontally=False, rotation=80, show=True, extra_title=' (detections)')

        #############################################################
        # EVALUATE WITH COCO METRICS
        #############################################################
        coco_res1 = coco_evaluator.get_coco_summary(gt_bbs, det_bbs)
        coco_res2 = coco_evaluator.get_coco_metrics(gt_bbs, det_bbs)

        if coco_summary is None:
            coco_summary = pd.DataFrame([], columns=coco_res1.keys())
        
        coco_res1['method'] = method
        coco_summary = coco_summary.append(coco_res1, ignore_index=True)
            
        for label in coco_res2.keys():
            coco_values = {k:v for k,v in coco_res2[label].items() if k in coco_metrics}
            coco_values['method'] = method

            recall = coco_res2[label]['recall']
            if recall is not None and len(recall):
                recall = recall.mean()
            precision = coco_res2[label]['precision']
            if precision is not None and len(precision):
                precision = precision.mean()
            
            coco_values['recall'] = recall
            coco_values['precision'] = precision
            coco_detail = coco_detail.append(coco_values, ignore_index=True)

        #############################################################
        # EVALUATE WITH VOC PASCAL METRICS
        #############################################################
        ious = [0.3, 0.5, 0.75]
        voc_res = {}
        for iou in ious:
            dict_res = pascal_voc_evaluator.get_pascalvoc_metrics(gt_bbs, det_bbs, iou, generate_table=True, method=MethodAveragePrecision.ELEVEN_POINT_INTERPOLATION)
            voc_res = dict_res['per_class']
            pascal_voc_evaluator.plot_precision_recall_curves(voc_res, showInterpolatedPrecision=True, showAP=True)

            for label in voc_res.keys():

                voc_class_values = {k:v for k,v in voc_res[label].items() if k in voc_metrics}
                voc_class_values['class'] = label
                voc_class_values['method'] = method
                voc_class_values['recall'] = voc_res[label]['table']['recall'].mean()
                voc_class_values['precision'] = voc_res[label]['table']['precision'].mean()
                voc_summary = voc_summary.append(voc_class_values, ignore_index=True)

            # import pdb; pdb.set_trace()
                
    return coco_summary, coco_detail, voc_summary


def main():
    parser = argparse.ArgumentParser(description='This program evaluates accuracy of a CNN after using different BGS methods.')
    parser.add_argument('-v', '--video', type=str, help='Path to a video or a sequence of image.', default=None)
    
    parser.add_argument('--show', default=False, action='store_true', help='Show window with results.')

    parser.add_argument('--skip', default=100, type=int, help='Number of frames to skip from evaluating.')
    
    parser.add_argument('--model', default=None, help='Path to CNN model.')
    parser.add_argument('--methods', default=['gt', 'full_frame', 'mog', 'mean', 'hybrid'], nargs='+', help='Method.')
    parser.add_argument('--classes', default=['person'], nargs='+', help='Valid classes.')
    parser.add_argument('--min-score', type=float, default=0.1, help='Minimum score to accept a detection.')

    args = parser.parse_args()

    video = args.video
    video_id = video
    if os.path.exists(video):
        video_path = video
        video_id = Path(video_path).stem

    detections_fn = os.path.join(os.getcwd(), f'infer/{video_id}_detections-{args.model}.csv')
    accuracy_fn = os.path.join(os.getcwd(), f'accuracy/{video_id}-{args.model}-coco_summary.csv')
    if os.path.exists(accuracy_fn):
        print(f'[ERROR] {video_id} has already been processed.')
        sys.exit()
    
    dets = pd.read_csv(detections_fn)
    dets = dets[dets.frame_id >= args.skip]
    assert not len(dets[(dets.xmin >= dets.xmax) | (dets.ymin >= dets.ymax)])
    dets = dets[(dets.method.isin(args.methods)) & (dets.score >= args.min_score)].copy().reset_index(drop=True)
    gt = pd.read_csv(f'annotations/{video_id}.no-static.csv')
    gt = gt[gt['static_object'] == False].copy().reset_index(drop=True)

    coco_summary, coco_metrics, voc = evaluate_predictions(dets, gt, args.methods, args.classes, output=f'/tmp/metrics/{video_id}')
    coco_summary.to_csv(f'accuracy/{video_id}-{args.model}-coco_summary.csv', sep=',', index=True)
    coco_metrics.to_csv(f'accuracy/{video_id}-{args.model}-coco_metrics.csv', sep=',', index=True)
    voc.to_csv(f'accuracy/{video_id}-{args.model}-voc.csv', sep=',', index=True)


if __name__ == '__main__':
    main()
