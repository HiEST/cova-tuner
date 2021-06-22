#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements methods to perform object-level scheduling through cropping and merge of objects"""

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
import math

import cv2
import numpy as np

# from rectpack import newPacker


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
        # return (self.box[2]-self.box[0])*(self.box[3]-self.box[1])

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

    print(f'Trying to find solution with limit {xlim}.')
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
            else:
                print(f'Found NEW solution with area {area} vs {prev_solution["area"]}')

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

    print(f'Trying to find solution with limit {xlim}.')

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
            else:
                print(f'Found NEW solution with area {area} vs {prev_solution["area"]}')

        prev_solution['shape'] = img_shape
        prev_solution['area'] = area
        prev_solution['objects'] = deepcopy(objects)
        # Try again with wider image
        xlim = max(img_shape)
        # print(f'Found solution with area {area}. Trying again with limit {xlim}.')

    objects.sort(key=lambda x: x.obj_id)
    return objects, img_shape


def combine_border(frames: list, box_lists: list, border_size: int = 10):
    objects = []
    for stream_id in range(len(frames)):
        boxes = box_lists[stream_id]
        for box in boxes:
            # border = [
            #     min(box[0], border_size),
            #     min(box[1], border_size),
            #     min(border_size, frames[stream_id].shape[1]-box[2]),
            #     min(border_size, frames[stream_id].shape[0]-box[3]),
            # ]
            border = [border_size]*4
            objects.append(MovingObject(stream_id, 0, len(objects), box, [], border))

    objects, img_shape = combine_boxes(objects)

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
    