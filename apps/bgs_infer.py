#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd

from cova.dnn import infer, metrics
from cova.motion import object_crop as crop
from cova.motion.motion_detector import merge_overlapping_boxes, resize_if_smaller


def read_virat(fn):
    annotations = pd.read_csv(fn, header=None, sep=" ", index_col=False)
    annotations.columns = [
        "object_id",
        "object_duration",
        "current_frame",
        "xmin",
        "ymin",
        "width",
        "height",
        "object_type",
    ]

    annotations = annotations[annotations.object_type > 0]
    annotations["xmax"] = annotations["xmin"] + annotations["width"]
    annotations["ymax"] = annotations["ymin"] + annotations["height"]
    object_labels = ["person", "car", "vehicle", "object", "bike"]
    annotations["label"] = annotations["object_type"].apply(
        lambda obj: object_labels[obj - 1]
    )
    annotations = annotations[annotations.label != "object"]
    annotations = annotations[annotations.label != "bike"]
    return annotations


def draw_detection(frame, box, label, score, color=(255, 0, 0)):
    x1, y1, x2, y2 = box
    try:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    except Exception as e:
        print(e)
        import pdb

        pdb.set_trace()
    cv2.putText(
        frame,
        f"{label} ({int(score*100)}%)",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
    )


def draw_top5(frame, labels, scores, method, color=(255, 0, 0), pos=0):
    if len(labels) < 5:
        print(labels)
        return
    # Draw gray box where detections will be printed
    height, width, _ = frame.shape
    x1, y1 = (width - 200 * (pos + 1), 10)
    x2, y2 = (width - 200 * pos, 10 + 15 * 7)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (250, 250, 250), -1)
    cv2.putText(
        frame, method, (x1 + 10, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
    )

    for i in range(5):
        label = labels[i]
        score = scores[i]

        cv2.putText(
            frame,
            f"{label} ({int(score*100)}%)",
            (x1 + 10, y1 + 10 + 15 * (i + 1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )


def main():
    parser = argparse.ArgumentParser(
        description="This program evaluates accuracy of a CNN after using different BGS methods."
    )
    parser.add_argument(
        "-v",
        "--video",
        type=str,
        help="Path to a video or a sequence of image.",
        default=None,
    )
    parser.add_argument(
        "--algo",
        type=str,
        help="Background subtraction method (KNN, MOG2).",
        default="mog",
    )
    # parser.add_argument('--gt', type=str, help='Path to ground-truth.')
    # parser.add_argument('--bgs', type=str, help='Path to BGS results.')
    parser.add_argument(
        "--show", default=False, action="store_true", help="Show window with results."
    )
    parser.add_argument(
        "--write", default=False, action="store_true", help="Write results as images."
    )
    parser.add_argument("--model", default=None, help="Path to CNN model.")
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.1,
        help="Minimum score to accept a detection.",
    )
    parser.add_argument(
        "--input", default=(300, 300), nargs="+", help="Models input size."
    )

    args = parser.parse_args()

    valid_classes = ["person", "car", "bike"]

    cap = cv2.VideoCapture(args.video)
    video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # video_width = 1280
    # video_height = 720
    max_boxes = 100

    detection_results = []
    columns = [
        "frame_id",
        "method",
        "label",
        "score",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "roi_xmin",
        "roi_ymin",
        "roi_xmax",
        "roi_ymax",
    ]

    # gt_fn = os.path.join(Path(args.input).video, '../annotations', Path(args.video).stem + '.viratdata.objects.txt')
    # gt = read_virat(gt_fn)
    bgs = pd.read_csv(os.path.join(os.getcwd(), f"{Path(args.video).stem}_rois.csv"))

    # We don't consider the first 500 frames
    # gt = gt[gt.current_frame >= 500]
    # bgs = bgs[bgs.frame_id >= 500]

    colors = {
        "full_frame": (255, 0, 0),
        "gt": (0, 255, 0),
        "mog": (255, 255, 0),
        "mean": (255, 0, 255),
        "hybrid": (0, 0, 255),
    }

    model = infer.Model(
        model_dir=args.model,
        label_map=None,  # Will load MSCOCO
        min_score=0.01,
        iou_threshold=0.3,
    )

    frames_with_objects = sorted(bgs[bgs.method == "gt"]["frame_id"].unique())
    for frame_id in frames_with_objects:
        if frame_id % 10 != 0:
            continue
        cap.set(1, frame_id)
        ret, frame = cap.read()
        frame_bgr = frame
        frame_rgb = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
        # import pdb; pdb.set_trace()

        composed_frames = [cv2.resize(frame.copy(), (300, 300))]
        composed_frames_rgb = [cv2.resize(frame_rgb.copy(), (300, 300))]
        cf2method = ["full_frame"]
        object_lists = [None]
        object_maps = [None]

        for method in ["gt", "mog", "mean", "hybrid"]:
            regions_proposed = bgs[(bgs.frame_id == frame_id) & (bgs.method == method)][
                ["xmin", "ymin", "xmax", "ymax"]
            ].values
            if not len(regions_proposed):
                continue

            rois_proposed = []
            for roi in regions_proposed:
                if any([r > 1 or r < 0 for r in roi]):
                    import pdb

                    pdb.set_trace()
                xmin = int(roi[0] * video_width)
                ymin = int(roi[1] * video_height)
                xmax = int(roi[2] * video_width)
                ymax = int(roi[3] * video_height)
                # cropped_rois.append(np.array(frame[ymin:ymax, xmin:xmax]))
                rois_proposed.append([xmin, ymin, xmax, ymax])
                # print(f'roi: {rois_proposed[-1]}')
                # draw_detection(frame=frame_bgr, box=rois_proposed[-1], label=method, score=1, color=(255, 0, 0))

            combined_width = sum(roi[2] - roi[0] for roi in rois_proposed)
            combined_height = sum(roi[3] - roi[1] for roi in rois_proposed)
            resize_x, resize_y = (1, 1)
            if combined_width < args.input[0]:
                resize_x = args.input[0] / combined_width
            if combined_height < args.input[1]:
                resize_y = args.input[1] / combined_height
            # increase width to reach model input's width combined
            if resize_x > 1 or resize_y > 1:
                print((resize_x, resize_y))
                for roi_id, roi in enumerate(rois_proposed):
                    new_size = (
                        int((roi[2] - roi[0]) * resize_x),
                        int((roi[3] - roi[1]) * resize_y),
                    )
                    new_box = resize_if_smaller(
                        roi, max_dims=(video_width, video_height), min_size=new_size
                    )
                    # print(f'new roi: {new_box}')

                    rois_proposed[roi_id] = new_box
                    # draw_detection(frame=frame_bgr, box=new_box, label=method, score=1, color=(0, 255, 0))

            # if method == 'gt':
            rois_proposed = merge_overlapping_boxes(rois_proposed)

            # Check area covered by RoIs proposed. If > 80% of the frame, just use the whole frame.
            area_rois = sum(
                [(roi[2] - roi[0]) * (roi[3] - roi[1]) for roi in rois_proposed]
            )
            if area_rois > (video_width * video_height) * 0.8:
                row = [frame_id, method] + [-1] * 10
                detection_results.append([row])
                print(f"RoIs take more than 80% of the frame. Skipping")
                continue

            composed_frame = None
            object_map = None
            objects = []

            # import pdb; pdb.set_trace()

            # ts0_crop = time.time()
            composed_frame, object_map, objects = crop.combine_border(
                [frame],
                [rois_proposed],
                border_size=5,
                min_combined_size=(300, 300),
                max_dims=(video_width, video_height),
            )

            composed_frame_rgb, _, _ = crop.combine_border(
                [frame_rgb],
                [rois_proposed],
                border_size=5,
                min_combined_size=(300, 300),
                max_dims=(video_width, video_height),
            )
            # import pdb; pdb.set_trace()
            composed_frames.append(
                cv2.resize(composed_frame, (300, 300)).astype("uint8")
            )
            composed_frames_rgb.append(
                cv2.resize(composed_frame_rgb, (300, 300)).astype("uint8")
            )
            object_maps.append(object_map)
            object_lists.append(objects)
            cf2method.append(method)
            # regions_proposed = [[0, 0, composed_frame.shape[1]-1, composed_frame.shape[0]-1]]
            # ts1_crop = time.time()
            # total_crop_time = ts1_crop - ts0_crop

        # if args.show:
        #     for i, method in enumerate(cf2method):
        #         cv2.imshow(method, composed_frames[i])

        #         if method != 'full_frame':
        #             cv2.setWindowTitle(method, f'{method} ({object_maps[i].shape[1]}x{object_map[i].shape[0]})')

        #     cv2.imshow('Full Frame', frame_bgr)
        #     key = cv2.waitKey(1) & 0xFF
        #     if key == ord("q"):
        #         sys.exit(1)

        # time.sleep(2)

        # continue

        ts0_infer = time.time()
        results = model.run(composed_frames_rgb)
        ts1_infer = time.time()
        infer_latency = ts1_infer - ts0_infer
        print(
            f"[{frame_id}] Took {infer_latency:.2f} seconds to process {len(composed_frames)} frames ({1/infer_latency:.2f} fps) -> {cf2method}."
        )
        # total_time_infer += (ts1_infer-ts0_infer)

        for method_id, method in enumerate(cf2method):
            object_map = object_maps[method_id]
            objects = object_lists[method_id]
            composed_frame = composed_frames[method_id]
            boxes = results[method_id]["boxes"]
            scores = results[method_id]["scores"]
            labels = results[method_id]["labels"]

            draw_top5(
                frame_bgr, labels, scores, method, color=colors[method], pos=method_id
            )

            ts0_decode_infer = time.time()
            for i in range(min(len(boxes), max_boxes)):
                label = labels[i]
                score = scores[i]
                if valid_classes is not None and label not in valid_classes:
                    continue

                if score < args.min_score:
                    continue
                ymin, xmin, ymax, xmax = tuple(boxes[i])

                # Object/Detection coordinates in merged frame
                (infer_left, infer_right, infer_top, infer_bottom) = (
                    int(xmin * composed_frame.shape[1]),
                    int(xmax * composed_frame.shape[1]),
                    int(ymin * composed_frame.shape[0]),
                    int(ymax * composed_frame.shape[0]),
                )

                draw_detection(
                    composed_frame,
                    [infer_left, infer_top, infer_right, infer_bottom],
                    label,
                    score,
                )

                if method == "full_frame":
                    (left, right, top, bottom) = (
                        int(xmin * frame_bgr.shape[1]),
                        int(xmax * frame_bgr.shape[1]),
                        int(ymin * frame_bgr.shape[0]),
                        int(ymax * frame_bgr.shape[0]),
                    )

                    detection_results.append(
                        [
                            frame_id,
                            method,
                            label,
                            score,
                            left,
                            top,
                            right,
                            bottom,
                            infer_left,
                            infer_top,
                            infer_right,
                            infer_bottom,
                        ]
                    )

                    draw_detection(frame_bgr, [left, top, right, bottom], label, score)

                    continue

                # Find object id consulting the object map
                (composed_left, composed_right, composed_top, composed_bottom) = (
                    int(xmin * object_map.shape[1]),
                    int(xmax * object_map.shape[1]),
                    int(ymin * object_map.shape[0]),
                    int(ymax * object_map.shape[0]),
                )
                predicted_box = [
                    composed_left,
                    composed_top,
                    composed_right,
                    composed_bottom,
                ]
                obj_id = int(
                    np.median(
                        object_map[
                            composed_top:composed_bottom, composed_left:composed_right
                        ]
                    )
                )
                if obj_id == 0:
                    continue
                    # import pdb; pdb.set_trace()
                obj = objects[obj_id - 1]

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
                if (
                    adjusted_coords[0] > box_in_inference[2]
                    or adjusted_coords[1] > box_in_inference[3]
                    or adjusted_coords[2] < box_in_inference[0]
                    or adjusted_coords[3] < box_in_inference[1]
                ):
                    print("coords out of frame")
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
                    max(0, relative_coords[0] - obj.border[0]),
                    max(0, relative_coords[1] - obj.border[1]),
                    min(frame.shape[1], relative_coords[2] - obj.border[0]),
                    min(frame.shape[0], relative_coords[3] - obj.border[1]),
                ]

                # Now, we can compute the absolute coordinates within the camera frames by adding roi coordinates
                obj_coords = [
                    no_border_coords[0] + roi[0],
                    no_border_coords[1] + roi[1],
                    min(frame_bgr.shape[1], no_border_coords[2] + roi[0]),
                    min(frame_bgr.shape[0], no_border_coords[3] + roi[1]),
                ]

                # if new box does not intersect enough with the original detection, skip it
                predicted_coords_origin = [
                    0,
                    0,
                    predicted_box[2] - predicted_box[0],
                    predicted_box[3] - predicted_box[1],
                ]
                translated_coords_origin = [
                    0,
                    0,
                    obj_coords[2] - obj_coords[0],
                    obj_coords[3] - obj_coords[1],
                ]
                iou, _ = metrics.get_iou(
                    predicted_coords_origin, translated_coords_origin
                )
                if iou < 0.5:
                    print("new box is too different. Skipping")
                    continue

                # (left, right, top, bottom) = (roi[0] + obj_coords[0], roi[0] + obj_coords[2],
                #                                 roi[1] + obj_coords[1], roi[1] + obj_coords[3])
                if (
                    any([c < 0 for c in obj_coords])
                    or obj_coords[2] > frame_bgr.shape[1]
                    or obj_coords[3] > frame_bgr.shape[0]
                ):
                    import pdb

                    pdb.set_trace()

                (left, top, right, bottom) = obj_coords

                detection_results.append(
                    [
                        frame_id,
                        method,
                        label,
                        score,
                        left,
                        top,
                        right,
                        bottom,
                        infer_left,
                        infer_top,
                        infer_right,
                        infer_bottom,
                    ]
                )

                draw_detection(
                    composed_frame,
                    [infer_left, infer_top, infer_right, infer_bottom],
                    label,
                    score,
                    color=colors[method],
                )

                draw_detection(
                    frame_bgr,
                    [left, top, right, bottom],
                    label,
                    score,
                    color=colors[method],
                )

                # ts1_decode_infer = time.time()
                # total_decode_infer = ts1_decode_infer - ts0_decode_infer

            if args.write:
                cv2.imwrite(
                    os.path.join(
                        os.getcwd(),
                        "results",
                        f"{Path(args.video).stem}_{frame_id}_{method}.png",
                    ),
                    composed_frame,
                )
            # import pdb; pdb.set_trace()
            # composed_frame = cv2.cvtColor(np.float32(composed_frame), cv2.COLOR_RGB2BGR)
            if args.show:
                cv2.imshow(method, composed_frame)
                if method != "full_frame":
                    cv2.setWindowTitle(
                        method,
                        f"{method} ({object_map.shape[1]}x{object_map.shape[0]})",
                    )
                cv2.imshow("Full Frame", frame_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    sys.exit(1)

            # if frame_id == 50:
            #     import pdb; pdb.set_trace()

    # import pdb; pdb.set_trace()
    detection_results = pd.DataFrame(detection_results, columns=columns)
    detection_results.to_csv(
        f"{Path(args.video).stem}_detections.csv", index=False, sep=","
    )


if __name__ == "__main__":
    main()
