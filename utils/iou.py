import numpy as np
import itertools

def intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def union(boxA,boxB):
    # x = min(a[0], b[0])
    # y = min(a[1], b[1])
    xA = min(boxA[0], boxB[0])
    yA = min(boxA[1], boxB[1])
    xB = max(boxA[2], boxB[2])
    yB = max(boxA[3], boxB[3])
    return (xA, yA, xB, yB)

def intersection(a,b):
    boxA = a
    boxB = b
#   left, top, right, bottom = tuple(roi)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    w = max(0, xB - xA + 1) 
    h = max(0, yB - yA + 1)
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    return interArea > 0


def merge_recs(rects, max_unions=100):
    unions = 0
    while (1):
        found = 0
        for ra, rb in itertools.combinations(rects, 2):
            intersec = intersection(ra, rb)
            if intersec:
                if ra in rects:
                    rects.remove(ra)
                if rb in rects:
                    rects.remove(rb)
                rects.append((union(ra, rb)))
                unions = unions + 1
                found = 1
                break
        if found == 0 or unions >= max_unions:
            break

    return rects