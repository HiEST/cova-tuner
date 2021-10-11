import unittest

import numpy as np
import pytest

import edge_autotune.motion.object_crop as crop


@pytest.fixture
def boxes():
    return [
            [0, 0, 100, 100],
            [0, 0, 100, 100],
            [0, 0, 10, 10],
        ]


@pytest.fixture
def border():
    return[10]*4


@pytest.fixture
def widths(boxes, border):
    return [
        box[2]-box[0]+border[2]+border[0]
        for box in boxes
    ]
        
        
@pytest.fixture
def heights(boxes, border):
    return [
        box[3]-box[1]+border[3]+border[1]
        for box in boxes
    ]
    
    
@pytest.fixture
def areas(widths, heights):
    return [
        (widths[i])*(heights[i])
        for i in range(len(widths))
    ]


@pytest.fixture
def objects(boxes, border):
    return [
        crop.MovingObject(0, 0, i+1, box, [], border)
        for i,box in enumerate(boxes)
    ]


@pytest.fixture
def objects_placement():
    return [
        [0, 0, 120, 120],
        [120, 0, 240, 120],
        [0, 120, 30, 150],
    ]


@pytest.fixture
def objects_with_placement(boxes, objects_placement, border):
    return [
        crop.MovingObject(0, 0, i+1, boxes[i], placement, border)
        for i,placement in enumerate(objects_placement)
    ]
    

@pytest.fixture
def object_map(objects_placement):
    object_map = np.zeros((150, 240))
    object_map[objects_placement[0][1]:objects_placement[0][3], objects_placement[0][0]:objects_placement[0][2]] = 1
    object_map[objects_placement[1][1]:objects_placement[1][3], objects_placement[1][0]:objects_placement[1][2]] = 2
    object_map[objects_placement[2][1]:objects_placement[2][3], objects_placement[2][0]:objects_placement[2][2]] = 3
    return object_map


def test_objects_properties(widths, heights, areas, objects):
    for i,obj in enumerate(objects):
        assert obj.width() == widths[i]
        assert obj.height() == heights[i]
        assert obj.area() == areas[i]


def test_combine_boxes(objects, objects_placement, object_map):
    output_objects, img_shape = crop.combine_boxes(objects, heuristic=crop.MergeHeuristic.FIRST_FIT_DECREASING)
    output_objects_placement = []
    for i, obj in enumerate(output_objects):
        output_objects_placement.append(obj.inf_box)
        assert obj.obj_id == i+1
        
    assert output_objects_placement == objects_placement
    assert img_shape[:2] == object_map.shape


@pytest.mark.parametrize("example", [
    [1, [0, 0, 120, 120]],
    [1, [100, 100, 120, 120]],
    [2, [140, 10, 200, 100]],
    [3, [10, 130, 20, 140]],
])
def test_prediction_to_object_obj_full(example, objects, object_map):
    obj_id, predicted = example
    assert crop.prediction_to_object(predicted, objects, object_map).obj_id == obj_id


# FIXME: Is this the expected behaviour? only borders but count as object
@pytest.mark.parametrize("example", [
    [1, [0, 0, 10, 10]],
    [1, [110, 110, 120, 120]],
])
def test_prediction_to_object_border(example, objects, object_map):
    obj_id, predicted = example
    assert crop.prediction_to_object(predicted, objects, object_map).obj_id == obj_id


@pytest.mark.parametrize("example", [
    [1, [0, 0, 240, 120]],
    [2, [90, 70, 200, 120]],
])
def test_prediction_to_object_overlap(example, objects, object_map):
    obj_id, predicted = example
    assert crop.prediction_to_object(predicted, objects, object_map).obj_id == obj_id


@pytest.mark.parametrize("predicted", [
    [120, 0, 100, 120],
    [-1, 10, 10, 10],
    [0, -3, 10, -5],
    [0, 300, 0, 300],
])
def test_prediction_to_object_none(predicted, objects, object_map):
    assert crop.prediction_to_object(predicted, objects, object_map) is None


def test_prediction_to_object_no_map():
    assert crop.prediction_to_object([], [], None) is None


@pytest.mark.parametrize(("obj_id", "predicted", "expected"), [
    [1, [60, 60, 110, 110], [60, 60, 110, 110]],
    [1, [60, 60, 140, 140], [60, 60, 110, 110]],
    [2, [100, 50, 150, 100], [130, 50, 150, 100]],
    [3, [0, 110, 30, 150], [10, 130, 20, 140]],
    [3, [0, 0, 300, 300], [10, 130, 20, 140]],
    [1, [1000, 1000, 2000, 2000], None],
])
def test_adjust_predicted_to_object_placement(obj_id, predicted, expected, objects_with_placement):
    assert crop.adjust_predicted_to_object_placement(predicted=predicted, object=objects_with_placement[obj_id-1]) == expected


@pytest.mark.parametrize(("obj_id", "predicted", "expected"), [
    [1, [60, 60, 110, 110], [50, 50, 100, 100]],
    [1, [60, 60, 140, 140], [50, 50, 100, 100]],
    [2, [100, 50, 150, 120], [0, 40, 20, 100]],
    [2, [130, 0, 240, 20], [0, 0, 100, 10]],
    [2, [120, 50, 300, 100], [0, 40, 100, 90]],
    [2, [120, 50, 150, 100], [0, 40, 20, 90]],
    [2, [90, 120, 40, 60], None],
    [3, [0, 110, 30, 200], [0, 0, 10, 10]],
    [3, [0, 0, 300, 300], [0, 0, 10, 10]],
])
def test_translate_to_object_coordinates(obj_id, predicted, expected, objects_with_placement):
    assert crop.translate_to_object_coordinates(predicted=predicted, object=objects_with_placement[obj_id-1]) == expected


@pytest.mark.parametrize(("predicted", "expected", "min_overlap"), [
    [[60, 60, 110, 110], [50, 50, 100, 100], 1],
    [[60, 60, 110, 110], [50, 50, 100, 100], 0.5],
    [[60, 60, 110, 120], None, 1],
    [[120, 50, 150, 100], [0, 40, 20, 90], 0.5],
    [[120, 50, 150, 100], [0, 40, 20, 90], 0.4],
    [[120, 80, 140, 100], [0, 70, 10, 90], 0.5],
    [[120, 80, 140, 100], None, 0.6],
    [[120, 80, 140, 140], [0, 70, 10, 100], 0.25],
    [[100, 80, 160, 140], None, 0.4],
])
def test_translate_to_frame_coordinates(predicted, expected, min_overlap, objects_with_placement, object_map):
    coords = crop.translate_to_frame_coordinates(
        predicted=predicted, object_map=object_map, 
        objects=objects_with_placement, min_overlap=min_overlap)
    
    assert coords == expected    


# def test_translate_to_frame_coordinates_invalid():
#     # Object map is None
#     assert crop.translate_to_frame_coordinates(predicted=[], object_map=None, objects=[], frame_size=None) == (None, 0, None)

#     # Invalid predicted
#     assert crop.translate_to_frame_coordinates(predicted=[-1, 0, 10, 10], object_map=None, objects=[], frame_size=None) == (None, 0, None)

#     # predicted coordinates are outside frame coordinates
#     assert crop.translate_to_frame_coordinates(predicted=[0, 120, 0, 120], object_map=None, objects=[], frame_size=(100, 100)) == (None, 0, None)

#     # predicted coordinates are outside frame coordinates
#     assert crop.translate_to_frame_coordinates(predicted=[0, 99, 0, 99], object_map=None, objects=[], frame_size=(100, 100)) == (None, 0, None)



if __name__ == '__main__':
    unittest.main()