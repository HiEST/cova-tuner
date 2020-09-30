import datetime
import time

import numpy as np
import cv2
import imutils

import tensorflow as tf
import tensorflow_hub as hub

COLOR_GOOD_CLASSIF = (255, 0, 255)
COLOR_POOR_CLASSIF = (255, 255, 255)
COLOR_DETECTION = (0, 0, 255)
COLOR_MOTION = (0, 255, 0)
COLOR_ROI = (255, 0, 0)

def print_tf():
    # Print Tensorflow version
    print(tf.__version__)

    # Check available GPU devices.
    print("The following GPU devices are available: %s" % tf.test.gpu_device_name())

def init_detector(model="MobileNetV2"):
    if model == "RCNN":
        module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
    elif model == "MobileNetV2":
        module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
    else:
        raise ValueError("Unknown model")

    detector = hub.load(module_handle).signatures['default']
    return detector

def draw_boxes(frame, boxes, class_names, scores, 
                max_boxes=10, min_score=0.1, 
                inference_time=None,
                save_annotations=False):

    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            (left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1],
                                ymin * frame.shape[0], ymax * frame.shape[0])
            
            display_str = "{}: {}%".format(class_names[i].decode("ascii"),
                                            int(100 * scores[i]))
            
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), COLOR_DETECTION, 2)
            cv2.putText(frame, display_str, (int(left), int(top)-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
    if inference_time:
        cv2.putText(frame, "Inference time: {:.2f}".format(inference_time), (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    return frame

def run_detector(detector, roi):

    img = imutils.resize(roi, width=256, height=256)
    converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    
    result = detector(converted_img)
    result = {key:value.numpy() for key,value in result.items()}

    return result


def detect_and_draw(detector, roi, 
        frame=None, 
        min_score=0.1, 
        results_file=None, 
        frame_id=0):

    start_time = time.time()
    result = run_detector(detector, roi)
    end_time = time.time()

    if frame is None:
        frame = roi

    frame_with_bb = draw_boxes(
        frame, result["detection_boxes"],
        result["detection_class_entities"], result["detection_scores"],
        min_score=min_score, inference_time=(end_time-start_time))

    if results_file:
        for i, box in enumerate(result["detection_boxes"]):
            score = result["detection_scores"][i]
            class_name = result["detection_class_entities"][i].decode("ascii")
            results_file.write(f'{frame_id},{class_name},{score},{box}\n')

    return frame_with_bb, (end_time-start_time)