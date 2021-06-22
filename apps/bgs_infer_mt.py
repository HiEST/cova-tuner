#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import os
import sys
import time

from queue import Queue
from threading import Thread

import cv2
import numpy as np
import pandas as pd

from edge_autotune.dnn import infer, metrics
from edge_autotune.motion import object_crop as crop
from edge_autotune.motion.motion_detector import merge_overlapping_boxes, resize_if_smaller


colors = {
    'full_frame': (255, 0, 0),
    'gt': (0, 255, 0),
    'mog': (255, 255, 0),
    'mean': (255, 0, 255),
    'hybrid': (0, 0, 255)
}

def decode_async(q_output, q_visual, cap, frames_with_objects):
    for frame_id in frames_with_objects:
        if frame_id % 10 != 0:
            continue
        cap.set(1, frame_id)
        ret, frame = cap.read()
        frame_bgr = frame
        frame_rgb = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
        
        q_output.put([frame_id, frame, frame_rgb])
        # q_visual.put([frame_id, frame])

    q_output.put([-1, None, None])


def compose_async(q_input, q_output, bgs, input_size):
    frame_id, frame, frame_rgb = q_input.get()
    while frame_id >= 0:
        video_height, video_width, _ = frame.shape
        composed_frames = [cv2.resize(frame.copy(), (300, 300))]
        composed_frames_rgb = [cv2.resize(frame_rgb.copy(), (300, 300))]
        cf2method = ['full_frame']
        object_lists = [None]
        object_maps = [None]

        for method in ['gt', 'mog', 'mean', 'hybrid']:
            regions_proposed = bgs[(bgs.frame_id == frame_id) & (bgs.method == method)][['xmin', 'ymin', 'xmax', 'ymax']].values
            if not len(regions_proposed):
                continue

            rois_proposed = []
            for roi in regions_proposed:
                if any([r>1 or r<0 for r in roi]):
                    import pdb; pdb.set_trace()
                xmin = int(roi[0]*video_width)
                ymin = int(roi[1]*video_height)
                xmax = int(roi[2]*video_width)
                ymax = int(roi[3]*video_height)
                rois_proposed.append([xmin, ymin, xmax, ymax])
            
            # FIXME: Not sure whether this is necessary
            combined_width = sum(roi[2]-roi[0] for roi in rois_proposed)
            combined_height = sum(roi[3]-roi[1] for roi in rois_proposed)
            resize_x, resize_y = (1, 1)
            if combined_width < input_size[0]:
                resize_x = input_size[0] / combined_width
            if combined_height < input_size[1]:
                resize_y = input_size[1] / combined_height
            # increase width to reach model input's width combined
            if resize_x > 1 or resize_y > 1:
                for roi_id, roi in enumerate(rois_proposed):
                    new_size = (int((roi[2]-roi[0])*resize_x), int((roi[3]-roi[1])*resize_y))
                    new_box = resize_if_smaller(roi, max_dims=(video_width, video_height), min_size=new_size)
                    rois_proposed[roi_id] = new_box

            rois_proposed = merge_overlapping_boxes(rois_proposed)

            # Check area covered by RoIs proposed. If > 80% of the frame, just use the whole frame.
            # area_rois = sum([(roi[2]-roi[0])*(roi[3]-roi[1]) for roi in rois_proposed])
            # if area_rois > (video_width*video_height)*0.8:
            #     composed_frames.append(None)
            #     composed_frames_rgb.append(None)
            #     object_maps.append(None)
            #     object_lists.append(None)
            #     cf2method.append(method)
            #     print(f'RoIs take more than 80% of the frame. Skipping')


            #     row = [frame_id, method] + [-1]*10
            #     detection_results.append([row])
                
            #     continue

            composed_frame = None
            object_map = None
            objects = []

            composed_frame, object_map, objects = crop.combine_border([frame], [rois_proposed], 
                                                        border_size = 5, min_combined_size=(300, 300),
                                                        max_dims=(video_width, video_height))

            composed_frame_rgb, _, _ = crop.combine_border([frame_rgb], [rois_proposed],
                                                        border_size = 5, min_combined_size=(300, 300),
                                                        max_dims=(video_width, video_height))
            # import pdb; pdb.set_trace()
            composed_frames.append(cv2.resize(composed_frame, (300, 300)).astype('uint8'))
            composed_frames_rgb.append(cv2.resize(composed_frame_rgb, (300, 300)).astype('uint8'))
            object_maps.append(object_map)
            object_lists.append(objects)
            cf2method.append(method)

        q_output.put([frame_id, frame, [composed_frames, composed_frames_rgb, object_maps, object_lists, cf2method]])
        frame_id, frame, frame_rgb = q_input.get()
    
    q_output.put([-1, None, [None]*5])


def infer_async(q_input, q_output, model, batch_size=1):
    frame_id, frame, composed_data = q_input.get()

    while frame_id >= 0:
        batch_frames = composed_data[1]
        batch_data = [[frame_id, frame, composed_data]]
        for i in range(batch_size-1):    
            frame_id, frame, composed_data = q_input.get()
            if frame_id < 0:
                batch_size = i
                break
                
            batch_frames.extend(composed_data[1])
            batch_data.append([frame_id, frame, composed_data])

        ts0_infer = time.time()
        results = model.run(batch_frames)
        ts1_infer = time.time()
        infer_latency = ts1_infer-ts0_infer
        num_frames = len(batch_frames)
        print(f'[{frame_id}] Took {infer_latency:.2f} seconds to process {num_frames} frames '
                f'({1/infer_latency*num_frames:.2f} fps).')

        start_frame = 0
        for i in range(batch_size):
            end_frame = start_frame + len(batch_data[i][-1][0])
            batch_results = batch_data[i]
            batch_results.extend([results[start_frame:end_frame]])
            
            q_output.put(batch_results)
            start_frame = end_frame
        
        if frame_id >= 0:
            frame_id, frame, composed_data = q_input.get()

    q_output.put([-1, None, [None], [None]])


def recompose_async(q_input, q_output, valid_classes=None, min_score=0.1, max_boxes=100):
    frame_id, frame, composed_data, infer_results = q_input.get()
    
    while frame_id >= 0:
        composed_frames, composed_frames_rgb, object_maps, object_lists, cf2method = composed_data
        detection_results = []
        for method_id, method in enumerate(cf2method):
            object_map = object_maps[method_id]
            objects = object_lists[method_id]
            composed_frame = composed_frames[method_id]
            boxes = infer_results[method_id]['boxes']
            scores = infer_results[method_id]['scores']
            labels = infer_results[method_id]['labels']

            # draw_top5(frame_bgr, labels, scores, method, color=colors[method], pos=method_id)

            for i in range(min(len(boxes), max_boxes)):
                label = labels[i]
                score = scores[i]
                if valid_classes is not None and label not in valid_classes:
                    continue 
                
                if score < min_score:
                    continue
                ymin, xmin, ymax, xmax = tuple(boxes[i])

                # Object/Detection coordinates in merged frame 
                (infer_left, infer_right, infer_top, infer_bottom) = (
                                                int(xmin * composed_frame.shape[1]), int(xmax * composed_frame.shape[1]),
                                                int(ymin * composed_frame.shape[0]), int(ymax * composed_frame.shape[0]))

                
                if method == 'full_frame':
                    (left, right, top, bottom) = (
                                                int(xmin * frame.shape[1]), int(xmax * frame.shape[1]),
                                                int(ymin * frame.shape[0]), int(ymax * frame.shape[0]))

                    detection_results.append([
                        frame_id, method, label, score,
                        left, top, right, bottom,
                        infer_left, infer_top, infer_right, infer_bottom,
                    ])

                    continue
                
                # Find object id consulting the object map
                (composed_left, composed_right, composed_top, composed_bottom) = (
                                                int(xmin * object_map.shape[1]), int(xmax * object_map.shape[1]),
                                                int(ymin * object_map.shape[0]), int(ymax * object_map.shape[0]))
                predicted_box = [composed_left, composed_top, composed_right, composed_bottom]
                obj_id = int(np.median(object_map[composed_top:composed_bottom,composed_left:composed_right]))
                if obj_id == 0:
                    continue
                    # import pdb; pdb.set_trace()
                obj = objects[obj_id-1]

                # Translate to coordinates in original frame from the camera
                # roi is in camera frame coordinates  
                roi = obj.box
                # inference box is in merged frame coordinates and includes borders
                box_in_inference = obj.inf_box

                # Sanity check
                assert predicted_box[0] < predicted_box[2]
                assert predicted_box[1] < predicted_box[3]
                

                # First, we adjust coordinates within merged frame by making sure borders are taken into account and subtracted
                adjusted_coords = [
                    max(predicted_box[0], box_in_inference[0]),
                    max(predicted_box[1], box_in_inference[1]),
                    min(predicted_box[2], box_in_inference[2]),
                    min(predicted_box[3], box_in_inference[3]),
                ]

                # Check if adjusted coordinates still fall within the RoI box.
                if adjusted_coords[0] > box_in_inference[2] or \
                    adjusted_coords[1] > box_in_inference[3] or \
                    adjusted_coords[2] < box_in_inference[0] or \
                    adjusted_coords[3] < box_in_inference[1]:
                    print('coords out of frame')
                    print(adjusted_coords)
                    continue

                # Second, we compute the relative object coordinates within RoI by removing box_in_inference coordinates
                relative_coords = [
                    adjusted_coords[0] - box_in_inference[0],
                    adjusted_coords[1] - box_in_inference[1],
                    adjusted_coords[2] - box_in_inference[0],
                    adjusted_coords[3] - box_in_inference[1],
                ]

                # Second, we remove borders such that 0,0 within roi is 0,0
                no_border_coords = [
                    max(0, relative_coords[0]-obj.border[0]),
                    max(0, relative_coords[1]-obj.border[1]),
                    min(frame.shape[1], relative_coords[2]-obj.border[0]),
                    min(frame.shape[0], relative_coords[3]-obj.border[1]),
                ]

                # Now, we can compute the absolute coordinates within the camera frames by adding roi coordinates
                obj_coords = [
                    no_border_coords[0] + roi[0],
                    no_border_coords[1] + roi[1],
                    min(frame.shape[1], no_border_coords[2] + roi[0]),
                    min(frame.shape[0], no_border_coords[3] + roi[1]),
                ]

                # if new box does not intersect enough with the original detection, skip it
                predicted_coords_origin = [0, 0, predicted_box[2]-predicted_box[0], predicted_box[3]-predicted_box[1]]
                translated_coords_origin = [0, 0, obj_coords[2]-obj_coords[0], obj_coords[3]-obj_coords[1]]
                iou, _ = metrics.get_iou(predicted_coords_origin, translated_coords_origin)
                if iou < 0.5:
                    print('new box is too different. Skipping')
                    continue
                                            
                if any([c < 0 for c in obj_coords]) or \
                    obj_coords[2] > frame.shape[1] or \
                    obj_coords[3] > frame.shape[0]:
                    import pdb; pdb.set_trace()

                (left, top, right, bottom) = obj_coords
                
                detection_results.append([
                    frame_id, method, label, score,
                    left, top, right, bottom,
                    infer_left, infer_top, infer_right, infer_bottom,
                ])

        q_output.put(detection_results)
        frame_id, frame, composed_data, infer_results = q_input.get()

    q_output.put(None)


def read_virat(fn):
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
    return annotations


def draw_detection(frame, box, label, score, color=(255,0,0)):
    x1, y1, x2, y2 = box
    try:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    except Exception as e:
        print(e)
        import pdb; pdb.set_trace()
    cv2.putText(frame, f'{label} ({int(score*100)}%)', (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def draw_top5(frame, labels, scores, method, color=(255,0,0), pos=0):
    if len(labels) < 5:
        print(labels)
        return
    # Draw gray box where detections will be printed
    height, width, _ = frame.shape
    x1, y1 = (width - 200 * (pos+1), 10)
    x2, y2 = (width - 200 * pos, 10+15*7)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (250, 250, 250), -1)
    cv2.putText(frame, method, (x1+10, y1+10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    for i in range(5):
        label = labels[i]
        score = scores[i]

        cv2.putText(frame, f'{label} ({int(score*100)}%)', (x1+10, y1+10+15*(i+1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def main():
    parser = argparse.ArgumentParser(description='This program evaluates accuracy of a CNN after using different BGS methods.')
    parser.add_argument('-v', '--video', type=str, help='Path to a video or a sequence of image.', default=None)
    parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='mog')
    # parser.add_argument('--gt', type=str, help='Path to ground-truth.')
    # parser.add_argument('--bgs', type=str, help='Path to BGS results.')
    parser.add_argument('--show', default=False, action='store_true', help='Show window with results.')
    parser.add_argument('--write', default=False, action='store_true', help='Write results as images.')
    parser.add_argument('--model', default=None, help='Path to CNN model.')
    parser.add_argument('--min-score', type=float, default=0.1, help='Minimum score to accept a detection.')
    parser.add_argument('--input', default=(300,300), nargs='+', help='Models input size.')
    
    args = parser.parse_args()

    valid_classes = ['person', 'car', 'bike']

    cap = cv2.VideoCapture(args.video)
    video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    max_boxes = 100

    detection_results = []
    columns = [
        'frame_id', 'method', 'label', 'score', 
        'xmin', 'ymin', 'xmax', 'ymax',
        'roi_xmin', 'roi_ymin', 'roi_xmax', 'roi_ymax']

    bgs = pd.read_csv(os.path.join(os.getcwd(), f'{Path(args.video).stem}_rois.csv'))
    frames_with_objects = sorted(bgs[bgs.method == 'gt']['frame_id'].unique())

    q_frames = Queue(maxsize=20)
    q_visual = Queue(maxsize=20)
    q_composed = Queue(maxsize=20)
    q_preds = Queue(maxsize=20)
    q_results = Queue(maxsize=20)

    model = infer.Model(
        model_dir=args.model,
        label_map=None,  # Will load MSCOCO
        min_score=0.01,
        iou_threshold=0.3,
    )

    decoder_thread = Thread(target=decode_async, args=(q_frames, q_visual, cap, frames_with_objects), daemon=True)
    composer_thread = Thread(target=compose_async, args=(q_frames, q_composed, bgs, args.input), daemon=True)
    infer_thread = Thread(target=infer_async, args=(q_composed, q_preds, model), daemon=True)
    recomposer_thread = Thread(target=recompose_async, args=(q_preds, q_results, valid_classes, args.min_score), daemon=True)
    decoder_thread.start()
    composer_thread.start()
    infer_thread.start()
    recomposer_thread.start()

    
    results = q_results.get()
    while results is not None:
        detection_results.extend(results)
        t0 = time.time()
        results = q_results.get()
        t1 = time.time()
        print(f'Results in {t1-t0:.2f} sec ({1/(t1-t0):.2f} fps).')

        print('Queues:')
        print(f'\tDecode: {q_frames.qsize()}')
        print(f'\tCompose: {q_composed.qsize()}')
        print(f'\tPreds: {q_preds.qsize()}')
        print(f'\tResults: {q_results.qsize()}')


        # if args.write:
        #     cv2.imwrite(os.path.join(os.getcwd(), 'results', f'{Path(args.video).stem}_{frame_id}_{method}.png'), composed_frame)
        
        # if args.show:
        #     cv2.imshow(method, composed_frame)
        #     if method != 'full_frame':
        #         cv2.setWindowTitle(method, f'{method} ({object_map.shape[1]}x{object_map.shape[0]})')
        #     cv2.imshow('Full Frame', frame_bgr)
        #     key = cv2.waitKey(1) & 0xFF
        #     if key == ord("q"):
        #         sys.exit(1)

            

    detection_results = pd.DataFrame(detection_results, columns=columns)
    detection_results.to_csv(f'{Path(args.video).stem}_detections.csv', index=False, sep=',')

if __name__ == '__main__':
    main()
