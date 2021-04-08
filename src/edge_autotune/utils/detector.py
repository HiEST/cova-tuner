import datetime
import time
import os

import numpy as np
import cv2
import imutils

import tensorflow as tf
import tensorflow_hub as hub
import json

from utils.datasets import MSCOCO as label_map

COLOR_GOOD_CLASSIF = (255, 0, 255)
COLOR_POOR_CLASSIF = (255, 255, 255)
COLOR_DETECTION = (0, 0, 255)
COLOR_MOTION = (0, 255, 0)
COLOR_ROI = (255, 0, 0)

ALL_MODELS = {
'CenterNet HourGlass104 512x512' : 'https://tfhub.dev/tensorflow/centernet/hourglass_512x512/1',
'CenterNet HourGlass104 Keypoints 512x512' : 'https://tfhub.dev/tensorflow/centernet/hourglass_512x512_kpts/1',
'CenterNet HourGlass104 1024x1024' : 'https://tfhub.dev/tensorflow/centernet/hourglass_1024x1024/1',
'CenterNet HourGlass104 Keypoints 1024x1024' : 'https://tfhub.dev/tensorflow/centernet/hourglass_1024x1024_kpts/1',
'CenterNet Resnet50 V1 FPN 512x512' : 'https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512/1',
'CenterNet Resnet50 V1 FPN Keypoints 512x512' : 'https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512_kpts/1',
'CenterNet Resnet101 V1 FPN 512x512' : 'https://tfhub.dev/tensorflow/centernet/resnet101v1_fpn_512x512/1',
'CenterNet Resnet50 V2 512x512' : 'https://tfhub.dev/tensorflow/centernet/resnet50v2_512x512/1',
'CenterNet Resnet50 V2 Keypoints 512x512' : 'https://tfhub.dev/tensorflow/centernet/resnet50v2_512x512_kpts/1',
'EfficientDet D0 512x512' : 'https://tfhub.dev/tensorflow/efficientdet/d0/1',
'EfficientDet D1 640x640' : 'https://tfhub.dev/tensorflow/efficientdet/d1/1',
'EfficientDet D2 768x768' : 'https://tfhub.dev/tensorflow/efficientdet/d2/1',
'EfficientDet D3 896x896' : 'https://tfhub.dev/tensorflow/efficientdet/d3/1',
'EfficientDet D4 1024x1024' : 'https://tfhub.dev/tensorflow/efficientdet/d4/1',
'EfficientDet D5 1280x1280' : 'https://tfhub.dev/tensorflow/efficientdet/d5/1',
'EfficientDet D6 1280x1280' : 'https://tfhub.dev/tensorflow/efficientdet/d6/1',
'EfficientDet D7 1536x1536' : 'https://tfhub.dev/tensorflow/efficientdet/d7/1',
'SSD MobileNet v2 320x320' : 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2',
'SSD MobileNet V1 FPN 640x640' : 'https://tfhub.dev/tensorflow/ssd_mobilenet_v1/fpn_640x640/1',
'SSD MobileNet V2 FPNLite 320x320' : 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1',
'SSD MobileNet V2 FPNLite 640x640' : 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_640x640/1',
'SSD ResNet50 V1 FPN 640x640 (RetinaNet50)' : 'https://tfhub.dev/tensorflow/retinanet/resnet50_v1_fpn_640x640/1',
'SSD ResNet50 V1 FPN 1024x1024 (RetinaNet50)' : 'https://tfhub.dev/tensorflow/retinanet/resnet50_v1_fpn_1024x1024/1',
'SSD ResNet101 V1 FPN 640x640 (RetinaNet101)' : 'https://tfhub.dev/tensorflow/retinanet/resnet101_v1_fpn_640x640/1',
'SSD ResNet101 V1 FPN 1024x1024 (RetinaNet101)' : 'https://tfhub.dev/tensorflow/retinanet/resnet101_v1_fpn_1024x1024/1',
'SSD ResNet152 V1 FPN 640x640 (RetinaNet152)' : 'https://tfhub.dev/tensorflow/retinanet/resnet152_v1_fpn_640x640/1',
'SSD ResNet152 V1 FPN 1024x1024 (RetinaNet152)' : 'https://tfhub.dev/tensorflow/retinanet/resnet152_v1_fpn_1024x1024/1',
'Faster R-CNN ResNet50 V1 640x640' : 'https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1',
'Faster R-CNN ResNet50 V1 1024x1024' : 'https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_1024x1024/1',
'Faster R-CNN ResNet50 V1 800x1333' : 'https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_800x1333/1',
'Faster R-CNN ResNet101 V1 640x640' : 'https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_640x640/1',
'Faster R-CNN ResNet101 V1 1024x1024' : 'https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_1024x1024/1',
'Faster R-CNN ResNet101 V1 800x1333' : 'https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_800x1333/1',
'Faster R-CNN ResNet152 V1 640x640' : 'https://tfhub.dev/tensorflow/faster_rcnn/resnet152_v1_640x640/1',
'Faster R-CNN ResNet152 V1 1024x1024' : 'https://tfhub.dev/tensorflow/faster_rcnn/resnet152_v1_1024x1024/1',
'Faster R-CNN ResNet152 V1 800x1333' : 'https://tfhub.dev/tensorflow/faster_rcnn/resnet152_v1_800x1333/1',
'Faster R-CNN Inception ResNet V2 640x640' : 'https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_640x640/1',
'Faster R-CNN Inception ResNet V2 1024x1024' : 'https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_1024x1024/1',
'Mask R-CNN Inception ResNet V2 1024x1024' : 'https://tfhub.dev/tensorflow/mask_rcnn/inception_resnet_v2_1024x1024/1'
}

COCO17_HUMAN_POSE_KEYPOINTS = [(0, 1),
 (0, 2),
 (1, 3),
 (2, 4),
 (0, 5),
 (0, 6),
 (5, 7),
 (7, 9),
 (6, 8),
 (8, 10),
 (5, 6),
 (5, 11),
 (6, 12),
 (11, 12),
 (11, 13),
 (13, 15),
 (12, 14),
 (14, 16)]


def print_tf():
    # Print Tensorflow version
    print(tf.__version__)

    # Check available GPU devices.
    print("The following GPU devices are available: %s" % tf.test.gpu_device_name())

def init_detector(model="SSD MobileNet v2 320x320"):
    model_handle = ALL_MODELS[model]
    print(f'Loading model {model}...')
    detector = hub.load(model_handle)
    print(f'Model loaded!')

    
    return detector

def draw_boxes(frame, boxes, class_names, scores, 
                max_boxes=10, min_score=0.5, 
                inference_time=None,
                save_annotations=False):

    detections_str = ''
    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            (left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1],
                                ymin * frame.shape[0], ymax * frame.shape[0])
            
            class_id = int(class_names[i])
            display_str = "{}: {}%".format(label_map[class_id]['name'],
                                            int(100 * scores[i]))

            detections_str = f'{detections_str}\n{display_str}'
            
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), COLOR_DETECTION, 2)
            cv2.putText(frame, display_str, (int(left), int(top)-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
    if inference_time:
        cv2.putText(frame, "Inference time: {:.2f}".format(inference_time), (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    return frame, detections_str

def run_detector(detector, roi, input_size=(320, 320)):

    img = imutils.resize(roi, width=input_size[0], height=input_size[1])
    converted_img  = tf.image.convert_image_dtype(img, tf.uint8)[tf.newaxis, ...]
    
    result = detector(converted_img)
    result = {key:value.numpy() for key,value in result.items()}

    return result

def draw_predictions(img, result, max_boxes=30, min_score=.5):
    # result = {key:value.numpy() for key,value in results.items()}
    label_id_offset = 0
    image_np_with_detections = img.copy()

    # Use keypoints if available in detections
    keypoints, keypoint_scores = None, None
    if 'detection_keypoints' in result:
      keypoints = result['detection_keypoints'][0]
      keypoint_scores = result['detection_keypoint_scores'][0]

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections[0],
          result['detection_boxes'][0],
          (result['detection_classes'][0] + label_id_offset).astype(int),
          result['detection_scores'][0],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=max_boxes,
          min_score_thresh=min_score,
          agnostic_mode=False,
          keypoints=keypoints,
          keypoint_scores=keypoint_scores,
          keypoint_edges=COCO17_HUMAN_POSE_KEYPOINTS)

    return image_np_with_detections[0], result


def detect_and_draw(detector, roi, 
        frame=None, 
        min_score=0.1, 
        results_file=None, 
        frame_id=0):

    start_time = time.time()
    result = run_detector(detector, roi)
    end_time = time.time()

    inference_time = None #(end_time-start_time)

    if frame is None:
        frame = roi

    frame_with_bb, detections_str = draw_boxes(
        frame, result["detection_boxes"][0],
        result["detection_classes"][0], result["detection_scores"][0],
        min_score=min_score, inference_time=inference_time)

    if results_file:
        for i, box in enumerate(result["detection_boxes"][0]):
            score = result["detection_scores"][0][i]
            class_name = result["detection_classes"][0][i].decode("ascii")
            results_file.write(f'{frame_id},{class_name},{score},{box}\n')

    return frame_with_bb, detections_str, (end_time-start_time)
