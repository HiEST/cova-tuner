#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements methods to perform object-level scheduling through cropping and merge of objects"""

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
import logging
import math
import os
import sys
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

from edge_autotune.motion.motion_detector import resize_if_smaller, merge_overlapping_boxes
from edge_autotune.dnn import metrics

logger = logging.getLogger(__name__)


@dataclass
class MovingObject:
    """Class for keeping track of (potential) objects within a multi-camera setup"""
    cam_id: int  # camera identifier
    frame_id: int  # frame number. Relative to the camera
    obj_id: int  # object identifier. Relative to the list of objects within an inference.
    box: list  # bounding box coordinates
    inf_box: list  # coordinates within inference matrix
    border: list  # border size on each coordinate

    def area(self) -> float:
        return self.width()*self.height()

    def width(self) -> int:
        return self.box[2]-self.box[0] + self.border[0]+self.border[2]

    def height(self) -> int:
        return self.box[3]-self.box[1] + self.border[1]+self.border[3]


class MergeHeuristic(Enum):
    # Online
    FIRST_FIT = 1
    BEST_FIT = 2
    # Offline
    FIRST_FIT_DECREASING = 3
    FIRST_FIT_DECREASING_ONE_DIM = 4


def first_fit_decreasing(img, objects: list):
    # sort boxes based on area in decreasing order
    # boxes.sort(key=lambda x: (x[2]-x[0])*(x[3]-x[1]), reverse=True)
    objects.sort(key=lambda x: x.area, reverse=True)

    max_dim = max([
        sum([(x[2+dim]-x[0+dim]) for x in boxes])
        for dim in [0,1]
    ])
    
    # Initialize output matrix with all 0's (black)
    result = np.zeros((max_dim, max_dim, 3))
    x_lim = 0
    y_lim = 0
    x_occupancy = [0]
    y_occupancy = [0]
    grid = np.zeros((max_dim, max_dim))

    for obj in objects:
        if x_lim + obj.width() <= y_lim + obj.height():
            # Place alongside, starting at (x_lim, x_occupancy[y_lim])
            obj.inf_box = [x_lim, x_occupancy[y_lim]]

            # Update limits
            x_lim += obj.width()
            y_lim += obj.height()


def grid_fit_decreasing(objects: list, xlim: int):
    objects.sort(key=lambda x: x.area(), reverse=True)
    # objects[0].inf_box = [
    #     0, 0,
    #     objects[0].width(),
    #     objects[0].height()
    # ]

    row_width = 0
    row_height = 0
    max_row_height = 0
    max_row_width = 0

    for idx, obj in enumerate(objects):
        if row_width + obj.width() > xlim:  
            # Jump to the next row
            row_width = 0
            row_height += max_row_height
            max_row_height = 0
        
        obj.inf_box = [
            row_width,
            row_height,
            row_width+obj.width(),
            row_height+obj.height()]

        # Update row's width
        row_width += obj.width()
        if obj.height() > max_row_height:
            max_row_height = obj.height()
        if row_width > max_row_width:
            max_row_width = row_width

    row_width = max_row_width
    row_height += max_row_height

    return objects, (row_height, row_width)


def bin_packing(objects: list):
    objects.sort(key=lambda x: x.area(), reverse=True)

    packer = newPacker(rotation=False)
    limits = {'x': 1, 'y': 1}
    maxrect = {'x': 0, 'y': 0}

    packer.add_bin(limits['x'], limits['y'])
    for obj_id, obj in enumerate(objects):
        packer.add_rect(obj.width(), obj.height(), rid=obj_id)

        packer.pack()
        objects_packed = len(packer.rect_list())
        if objects_packed < (obj_id+1):  # Increase bin size
            obj_lims = {'x': maxrect['x'] + obj.width(),
                        'y': maxrect['y'] + obj.height()}
            if limits['x'] < limits['y']:  # Start increasing x
                dims = ['x', 'y']
            else:
                dims = ['y', 'x']

            for d in dims:
                limits[d] += limits[d] - (maxrect[d] - obj_lims[d])
                packer._avail_bins[0] = (limits['x'], limits['y'], 1, {})
                packer.pack()
                
                objects_packed = len(packer.rect_list())
                if objects_packed >= obj_id+1:
                    break
            
        # Make sure all objects have been packed
        assert objects_packed == (obj_id+1)


    # import pdb; pdb.set_trace()
    # packer.pack()
    for rect in packer[0]:
        objects[rect.rid].inf_box = [
            rect.x,
            rect.y,
            rect.x+rect.width,
            rect.y+rect.height]

    return objects


def merge(img, boxes: list, heuristic=MergeHeuristic.FIRST_FIT_DECREASING):
    objects = []
    for box_id, box in enumerate(boxes):
        objects.append(MovingObject(0, 0, box_id, box, []))

    # Heuristic for maximum width. Start with sq root of area.
    xlim = math.sqrt(sum([obj.area() for obj in objects]))
    prev_solution = {
        'area': np.inf,
        'shape': None,
        'objects': None,
    }

    # print(f'Trying to find solution with limit {xlim}.')
    while True:
        objects, img_shape = grid_fit_decreasing(objects, xlim)
        area = max(img_shape)*max(img_shape)  # img will be squared, so we compute area wrt larges dim
        if prev_solution['shape'] is not None:
            if area == prev_solution['area']:
                break
            elif area > prev_solution['area']:  # Roll-back before break
                objects = prev_solution['objects']
                img_shape = prev_solution['shape']
                break
            # else:
                # print(f'Found NEW solution with area {area} vs {prev_solution["area"]}')

        prev_solution['shape'] = img_shape
        prev_solution['area'] = area
        prev_solution['objects'] = deepcopy(objects)
        # Try again with wider image
        xlim = max(img_shape)


    if img_shape[0] > img_shape[1]:
        img_shape = (img_shape[0], img_shape[0])
    else:
        img_shape = (img_shape[1], img_shape[1])
    merged_img = np.zeros(img_shape + (3,))
    object_map = np.zeros(img_shape) 

    for obj in objects:
        roi = obj.box
        inf_roi = obj.inf_box

        img_shape = img[roi[1]:roi[3], roi[0]:roi[2]].shape
        merged_shape = merged_img[inf_roi[1]:inf_roi[3], inf_roi[0]:inf_roi[2]].shape
        if img_shape != merged_shape:
            import pdb; pdb.set_trace()

        cropped_object = img[roi[1]:roi[3], roi[0]:roi[2]].copy()
        merged_img[inf_roi[1]:inf_roi[3], inf_roi[0]:inf_roi[2], :3] = cropped_object
        object_map[inf_roi[1]:inf_roi[3], inf_roi[0]:inf_roi[2]] = obj.obj_id+1
    
    objects.sort(key=lambda x: x.obj_id)
    return merged_img, object_map, objects


def combine_streams(frames, box_lists: list, heuristic=MergeHeuristic.FIRST_FIT_DECREASING):
    objects = []
    for stream_id in range(len(frames)):
        boxes = box_lists[stream_id]
        for box_id, box in enumerate(boxes):
            objects.append(MovingObject(stream_id, 0, len(objects), box, []))

    objects, img_shape = grid_fit_decreasing(objects)

    if img_shape[0] > img_shape[1]:
        img_shape = (img_shape[0], img_shape[0])
    else:
        img_shape = (img_shape[1], img_shape[1])
    merged_img = np.zeros(img_shape + (3,))
    object_map = np.zeros(img_shape) 

    for obj in objects:
        frame = frames[obj.cam_id]
        roi = obj.box
        inf_roi = obj.inf_box
        
        img_shape = frame[roi[1]:roi[3], roi[0]:roi[2]].shape
        merged_shape = merged_img[inf_roi[1]:inf_roi[3], inf_roi[0]:inf_roi[2]].shape
        if img_shape != merged_shape:
            import pdb; pdb.set_trace()

        cropped_object = frame[roi[1]:roi[3], roi[0]:roi[2]].copy()
        merged_img[inf_roi[1]:inf_roi[3], inf_roi[0]:inf_roi[2], :3] = cropped_object
        object_map[inf_roi[1]:inf_roi[3], inf_roi[0]:inf_roi[2]] = obj.obj_id+1
    
    objects.sort(key=lambda x: x.obj_id)
    return merged_img, object_map, objects


def combine_boxes(objects: list, heuristic=MergeHeuristic.FIRST_FIT_DECREASING):
    # objects = []
    # for list_id, boxes in enumerate(box_lists):
    #     for box in boxes:
    #         objects.append(MovingObject(list_id, 0, len(objects), box, []))

    xlim = math.sqrt(sum([obj.area() for obj in objects]))
    prev_solution = {
        'area': np.inf,
        'shape': None,
        'objects': None,
    }

    # print(f'Trying to find solution with limit {xlim}.')

    while True:
        objects, img_shape = grid_fit_decreasing(objects, xlim)
        if isinstance(img_shape[0], float):
            import pdb; pdb.set_trace()
        area = max(img_shape)*max(img_shape)  # img will be squared, so we compute area wrt larges dim
        if prev_solution['shape'] is not None:
            if area == prev_solution['area']:
                break
            elif area > prev_solution['area']:  # Roll-back before break
                objects = prev_solution['objects']
                img_shape = prev_solution['shape']
                break
            # else:
            #     print(f'Found NEW solution with area {area} vs {prev_solution["area"]}')

        prev_solution['shape'] = img_shape
        prev_solution['area'] = area
        prev_solution['objects'] = deepcopy(objects)
        # Try again with wider image
        xlim = max(img_shape)
        # print(f'Found solution with area {area}. Trying again with limit {xlim}.')

    objects.sort(key=lambda x: x.obj_id)
    return objects, img_shape


def combine_border(frames: list, box_lists: list, border_size: int = 10, min_combined_size=None, max_dims=None):
    objects = []
    if sum([len(boxes) for boxes in box_lists]) == 1:
            border_size = 0
    for stream_id in range(len(frames)):
        boxes = box_lists[stream_id]
        for box in boxes:
            border = [border_size]*4
            objects.append(MovingObject(stream_id, 0, len(objects), box, [], border))

    objects, img_shape = combine_boxes(objects)
    # img_size = [img_shape[1], img_shape[0]]
    # resize_x = 1 if min_combined_size is None else max(1, min_combined_size[0]/img_size[0])
    # resize_y = 1 if min_combined_size is None else max(1, min_combined_size[1]/img_size[1])

    # new_size = [resize_x*img_size[0], resize_y*img_size[1]]
    # # print(f'first new size: {new_size} ({resize_x} & {resize_y})')
    # if new_size[0] < new_size[1]:
    #     resize_x = new_size[1] / img_size[0]
    # elif new_size[1] < new_size[0]:
    #     resize_y = new_size[0] / img_size[1]

    # if resize_x > 1 or resize_y > 1:
    #     new_size = [int(resize_x*img_size[0]), int(resize_y*img_size[1])]
    #     # TODO: Check if new boxes overlap between them. If so, merge them.
        
    #     for stream_id in range(len(frames)):
    #         boxes = box_lists[stream_id]
    #         new_boxes = []
    #         for box in boxes:
    #             new_w = int((box[2]-box[0])*resize_x)
    #             new_h = int((box[3]-box[1])*resize_y)
    #             box = resize_if_smaller(box, max_dims=max_dims, min_size=(new_w, new_h))
    #             new_boxes.append(box)

    #         new_boxes = merge_overlapping_boxes(new_boxes)

    #         for box in new_boxes:
    #             border = [border_size]*4
    #             objects.append(MovingObject(stream_id, 0, len(objects), box, [], border))
                
    #     # print(f'Min combined size: {min_combined_size}')
    #     # print(f'Resizing objects by {resize_x:.2f}, {resize_y:.2f}. From ({img_size}) to ({new_size})')
    #     objects, img_shape = combine_boxes(objects)
    #     # print(f'\tGot: {img_shape[1]}, {img_shape[0]}')
    #         # obj.inf_box = [
    #         #     int(obj.inf_box[0]*resize_x),
    #         #     int(obj.inf_box[1]*resize_y),
    #         #     int(obj.inf_box[2]*resize_x),
    #         #     int(obj.inf_box[3]*resize_y),
    #         # ]
    #     # img_shape = (int(resize_x*img_shape[0]), int(resize_y*img_shape[1]))

    if img_shape[0] > img_shape[1]:
        img_shape = (img_shape[0], img_shape[0])
    else:
        img_shape = (img_shape[1], img_shape[1])
    merged_img = np.zeros(img_shape + (3,))
    object_map = np.zeros(img_shape) 

    for obj in objects:
        frame = frames[obj.cam_id]
        roi = obj.box
        inf_roi = obj.inf_box
        
        roi_shape = frame[roi[1]:roi[3], roi[0]:roi[2]].shape
        merged_shape = merged_img[inf_roi[1]:inf_roi[3], inf_roi[0]:inf_roi[2]].shape
        cropped_object = frame[roi[1]:roi[3], roi[0]:roi[2]].copy()
        crop_with_border = cv2.copyMakeBorder(cropped_object,
                                            obj.border[1], obj.border[3],
                                            obj.border[0], obj.border[2],
                                            cv2.BORDER_CONSTANT, (0, 0, 0))
        if crop_with_border.shape != merged_shape:
            import pdb; pdb.set_trace()
        merged_img[inf_roi[1]:inf_roi[3], inf_roi[0]:inf_roi[2], :3] = crop_with_border
        object_map[
            (inf_roi[1]+obj.border[1]):(inf_roi[3]-obj.border[3]),
            (inf_roi[0]+obj.border[0]):(inf_roi[2]-obj.border[2])] = obj.obj_id+1
    
    objects.sort(key=lambda x: x.obj_id)
    return merged_img, object_map, objects


def combine_resize(frames: list, box_lists: list, roi_size: list = (100, 100), border_size: int = 10):
    objects = []
    all_boxes = []
    for stream_id in range(len(frames)):
        boxes = box_lists[stream_id]
        for box in boxes:
            # border = [border_size]*4
            border = [0]*4
            new_box = (0, 0) + roi_size
            objects.append(MovingObject(stream_id, 0, len(objects), new_box, [], border))
            all_boxes.append(box)

    objects, img_shape = combine_boxes(objects)

    if img_shape[0] > img_shape[1]:
        img_shape = (img_shape[0], img_shape[0])
    else:
        img_shape = (img_shape[1], img_shape[1])
    merged_img = np.zeros(img_shape + (3,))
    object_map = np.zeros(img_shape) 

    for obj in objects:
        frame = frames[obj.cam_id]
        roi = all_boxes[obj.obj_id]
        inf_roi = obj.inf_box
        
        img_shape = frame[roi[1]:roi[3], roi[0]:roi[2]].shape
        merged_shape = merged_img[inf_roi[1]:inf_roi[3], inf_roi[0]:inf_roi[2]].shape
        cropped_object = cv2.resize(frame[roi[1]:roi[3], roi[0]:roi[2]].copy(), roi_size)
        crop_with_border = cv2.copyMakeBorder(cropped_object,
                                            obj.border[1], obj.border[3],
                                            obj.border[0], obj.border[2],
                                            cv2.BORDER_CONSTANT, (0, 0, 0))
        if crop_with_border.shape != merged_shape:
            import pdb; pdb.set_trace()
        merged_img[inf_roi[1]:inf_roi[3], inf_roi[0]:inf_roi[2], :3] = crop_with_border
        object_map[inf_roi[1]:inf_roi[3], inf_roi[0]:inf_roi[2]] = obj.obj_id+1
    
    objects.sort(key=lambda x: x.obj_id)
    return merged_img, object_map, objects
    

def prediction_to_object(
    predicted: Tuple[int, int, int, int],
    objects: List[MovingObject],
    object_map: Optional[List[List[np.uint8]]] = None) -> Tuple[MovingObject]:


    if object_map is None:
        max_iou = [0, None]
        for obj in objects:
            iou, _ = metrics.get_iou(predicted, obj.inf_box)
        
            if max_iou[0] < iou:
                max_iou = [iou, obj]
        
        obj = max_iou[1]
    else:
        xmin, ymin, xmax, ymax = predicted
        
        if not (xmin < xmax and ymin < ymax):
            return None
        if xmin < 0 or ymin < 0: # TODO: Check why this might happen
            return None

        try:
            obj_id = int(np.median(object_map[ymin:ymax,xmin:xmax]))
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.error(f'{exc_type} in {fname}:{exc_tb.tb_lineno}')
            logger.debug(f'coords {ymin}:{ymax},{xmin}:{xmax}')
            logger.debug(f'object_map.shape: {object_map.shape}')
            raise e

        #FIXME: Instead of checking only for 0, check if it is a float number (i.e. check % of one object) 
        if obj_id == 0:
            return None
        obj = objects[obj_id-1]

    return obj


def adjust_predicted_to_object_placement(
    predicted: Tuple[int, int, int, int],
    object: MovingObject) -> Tuple[int, int, int, int]:
    """Adjustes the predicted bounding box and moves its coordinates within the object's box within the composed frame,
    i.e., returns a bounding box contained completely within the object (no borders).
    Returns None if predicted and object's box do not overlap."""

    if predicted[0] > object.inf_box[2] or \
        predicted[1] > object.inf_box[3] or \
        predicted[2] < object.inf_box[0] or \
        predicted[3] < object.inf_box[1]:
        return None

    return [
        max(predicted[0], object.inf_box[0]+object.border[0]),
        max(predicted[1], object.inf_box[1]+object.border[1]),
        min(predicted[2], object.inf_box[2]-object.border[2]),
        min(predicted[3], object.inf_box[3]-object.border[3]),
    ]


def translate_to_object_coordinates(
    predicted: Tuple[int, int, int, int],
    object: MovingObject) -> Tuple[int, int, int, int]:
    """Returns predicted coordinates translated using the object's origin as their origin.
        Pre-requisit: predicted coordinates are within object.inf_box.
    """

    adjusted = adjust_predicted_to_object_placement(predicted, object)
    if adjusted is None:
        return None

    origin_object = [
        object.inf_box[0] + object.border[0],
        object.inf_box[1] + object.border[1],
    ]

    return [
        adjusted[0] - origin_object[0],
        adjusted[1] - origin_object[1],
        adjusted[2] - origin_object[0],
        adjusted[3] - origin_object[1],
    ]


def translate_to_frame_coordinates(
    predicted: Tuple[int, int, int, int],
    object_map: List[List[np.uint8]],
    objects: List[MovingObject],
    min_overlap: float = 0) -> Tuple[int, int, int, int]:

    # 1. Obtain the object that matches the predicted bounding box.
    obj = prediction_to_object(predicted, objects, object_map=object_map)
    if obj is None:
        return None

    # 2. Adjust the predicted bounding box to the region the object occupies within the composed frame
    adjusted = adjust_predicted_to_object_placement(predicted, obj)
    print(f'adjusted: {adjusted}')
    print(f'predicted: {predicted}')

    # 3. Check that the overlap is greater than the minimum specified.
    overlap = metrics.get_overlap(predicted, adjusted)
    print(f'overlap: {overlap}')
    if overlap < min_overlap:
        return None

    # 4. Translate to object coordinates
    object_coordinates = translate_to_object_coordinates(predicted, obj)
    print(f'object_coordinates: {object_coordinates}')

    # 5. Translate to frame coordinates
    return [
        object_coordinates[0] + obj.box[0],
        object_coordinates[1] + obj.box[1],
        object_coordinates[2] + obj.box[0],
        object_coordinates[3] + obj.box[1],
    ]
