from typing import Tuple

import numpy as np
import pandas as pd


def get_overlap(
    bb1: Tuple[int, int, int, int], bb2: Tuple[int, int, int, int]
) -> float:
    """
    Computes the overlap between two bounding boxes.
    Returns the ratio of area between bb1 and the intersection of bb1 and bb2.
    """
    intersection = [
        max(bb1[0], bb2[0]),
        max(bb1[1], bb2[1]),
        min(bb1[2], bb2[2]),
        min(bb1[3], bb2[3]),
    ]

    if intersection[0] > intersection[2] or intersection[1] > intersection[3]:
        return 0.0

    area_bb1 = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    area_intersection = (intersection[2] - intersection[0]) * (
        intersection[3] - intersection[1]
    )

    return area_intersection / area_bb1


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert isinstance(bb1, list) or isinstance(bb1, np.ndarray)

    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0, 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou, intersection_area


def is_positive(pred, gt_boxes, iou_level=0.5):
    for gt_id, box in gt_boxes.iterrows():
        iou, _ = get_iou(pred.values, box.values)
        if iou >= iou_level:
            return gt_id
    return -1


def get_precision_recall(preds, gts, label):
    TP = 0
    FP = 0
    FN = 0
    gt_boxes = gts[["xmin", "ymin", "xmax", "ymax"]]

    for i, pred in preds.iterrows():
        box = pred[["xmin", "ymin", "xmax", "ymax"]]

        match_id = is_positive(box, gt_boxes, iou_level=0.5)
        if match_id >= 0:
            if pred["label"] == gts.iloc[match_id]["label"]:
                TP += 1
            else:
                FP += 1
        else:
            FP += 1

    FN = len(gts) - TP

    assert len(preds) == (TP + FP)

    precision = 0 if TP == 0 else TP / (TP + FP)
    recall = 0 if TP == 0 else TP / (TP + FN)
    return [precision, recall, [TP, FP, FN]]


def evaluate_predictions(preds, gts, label):
    results = {
        "TP": [],
        "FP": [],
        "FN": [],
    }

    gt_boxes = gts[["xmin", "ymin", "xmax", "ymax"]]
    gt_positives = []

    for i, pred in preds.iterrows():
        box = pred[["xmin", "ymin", "xmax", "ymax"]]

        match_id = is_positive(box, gt_boxes, iou_level=0.5)
        if match_id >= 0:
            if pred["label"] == gts.iloc[match_id]["label"]:
                results["TP"].append(box)
                gt_positives.append(match_id)
            else:
                results["FP"].append(box)
        else:
            results["FP"].append(box)

    assert len(preds) == (len(results["TP"]) + len(results["FP"]))

    return results


def compute_area_of_union(boxes):
    width = max([box[2] for box in boxes]) + 1
    height = max([box[3] for box in boxes]) + 1
    canvas = np.zeros((width, height))
    for box in boxes:
        canvas[box[1] : box[3], box[0] : box[2]] = 1

    area_of_union = np.sum(canvas > 0)
    return area_of_union


def compute_area_of_intersect(boxes):
    canvases = []
    width = max([box[2] for box in boxes])
    height = max([box[3] for box in boxes])
    for box in boxes:
        canvas = np.zeros(width, height)
        canvas[box[1] : box[3], box[0] : box[2]] = 1
        canvases.append(canvas)

    area_of_intersect = np.sum(np.sum(canvases) > 1)
    return area_of_intersect


def compute_area_match(boxes, gt_boxes, iou_levels=[0.3, 0.5]):
    boxes_area = sum([(box[2] - box[0]) * (box[3] - box[1]) for box in boxes])
    gt_area = 0 if not len(gt_boxes) else compute_area_of_union(gt_boxes)

    results = []
    for iou_threshold in iou_levels:
        intersection_area = 0
        matches = 0
        avg_iou_matches = []
        avg_iou_misses = []

        for gt_id, gt in enumerate(gt_boxes):
            max_iou = 0
            intersection = 0
            for box in boxes:
                iou, intersection = get_iou(gt, box)
                max_iou = max(max_iou, iou)
                if iou >= iou_threshold:
                    break

            if max_iou < iou_threshold:
                avg_iou_misses.append(max_iou)
                continue

            intersection_area += intersection
            matches += 1
            avg_iou_matches.append(max_iou)

        results.append(
            {
                "iou": iou_threshold,
                "intersection_area": intersection_area,
                "matches": matches,
                "misses": len(gt_boxes) - matches,
                "avg_iou_matches": 0
                if not len(avg_iou_matches)
                else sum(avg_iou_matches) / len(avg_iou_matches),
                "avg_iou_misses": 0
                if not len(avg_iou_misses)
                else sum(avg_iou_misses) / len(avg_iou_misses),
            }
        )

    results = {
        "gt_area": gt_area,
        "boxes_area": boxes_area,
        "results": [r for r in results],
    }

    return results
