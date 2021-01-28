# import the necessary packages
import sys
import time
import random
import argparse
import datetime
from os.path import isfile

import cProfile
import pstats

import numpy as np
import pandas as pd

from imutils.video import VideoStream
import imutils
import cv2

# Tensorflow
import tensorflow as tf
import tensorflow_hub as hub

# Auxiliary functions
from utils.motion_detection import MotionDetection, Background
from utils.classification import Classifier
from utils.detector import init_detector, run_detector, detect_and_draw, label_map
# from utils.capture_screen import CaptureScreen
from utils.constants import *
from utils.datasets import MSCOCO


COLORS = [
    (0, 255, 0),
    (0, 255, 255),
    (0, 0, 255)
]


def load_pbtxt(filename):
    label_map = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    label = ''
    for l in lines:
        if 'name' in l:
            label = l.split('"')[1]
        elif 'id' in l:
            class_id = int(l.split(':')[1])
            label_map[class_id] = {
                'name': label,
                'id': class_id
            }

    return label_map


def open_video(video):
    # if the video argument is None, then we are reading from webcam
    if video is None:
        vs = VideoStream(src=0).start()
        time.sleep(2.0)
    # or monitor
    elif video == "screen":
        vs = CaptureScreen()
    # otherwise, we are reading from a video file
    else:
        vs = cv2.VideoCapture(video)

    return vs


def pause():
    cv2.waitKey(0)


def load_saved_model(model):
    return tf.saved_model.load(model)


def main():
    # construct the argument parser and parse the arguments
    args = argparse.ArgumentParser()

    # Required
    args.add_argument("-v", "--video", default=None, help="path to the video file")

    args.add_argument("--compare", default=None, nargs='+', help="ground truth to compare to")

    # Application control
    args.add_argument("--max-frames", type=int, default=0, help="maximum frames to process")
    args.add_argument("--skip-frames", type=int, default=0, help="number of frames to skip for each frame processed")
    args.add_argument("--loop", action="store_true", default=False, help="loop video")
    args.add_argument("--resize-width", type=int, default=None, help="New height to resize images keeping aspect ratio")

    # Save results
    args.add_argument("-o", "--output", default=None, help="path to the output video file")
    args.add_argument("-r", "--results", default=None, help="detection results")
    args.add_argument("-s", "--save", default=None, help="Path to pickle where detections will be saved.")
    args.add_argument("--save-ratio", type=float, default=0.1, help="Ratio of ROIs that are saved")
    
    # Motion Detection
    # args.add_argument("--no-motion", action="store_true", help="Disable motion detection. All frames are fully analysed")
    # args.add_argument("--no-merge-rois", action="store_true", help="Don't merge ROIs on scene")
    # args.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
    # args.add_argument("--no-average", help="use always first frame as background.", action="store_true")
    
    # Detection/Classification
    args.add_argument("-m", "--model", default="resnet", help="Model for image classification")
    args.add_argument("-l", "--label-map", default=None, help="Label map for the model")
    args.add_argument("--min-score", type=float, default=0.6, help="minimum score for detections")
    args.add_argument("--max-boxes", type=float, default=10, help="maximumim number of bounding boxes per frame")
    args.add_argument("--detect", help="Detect objects using DNN.", action="store_true")
    args.add_argument("--wait-for-detection", help="Number of frames to wait before first detection.", default=None, type=int)
    args.add_argument("--classify", help="Run image classification using DNN.", action="store_true")
    
    # Control what's shown
    args.add_argument("--debug", help="Show debug information and display all steps.", action="store_true")
    args.add_argument("--no-show", help="Don't show any windows.", action="store_true")
    
    config = args.parse_args()

    play_fps = 30
    frame_frequency = 1.0 / play_fps
    pause = False
    img_id = 1

    max_boxes = config.max_boxes
    min_score = config.min_score

    # Stats
    frames_with_detection = 0
    total_detections = 0
    num_frames = 0
    frames_skipped = 0
    # Time counters
    start_frame = 0
    end_frame = 0
    
    vs = open_video(config.video)

    # Load pre-computed results to compare to
    results_to_compare = []
    if config.compare is not None:
        for r in config.compare:
            results_to_compare.append(pd.read_pickle(r, "bz2"))

    classifier = None
    detector = None
    if config.detect or config.classify:
        if config.detect:
            saved_model = False
            if 'saved_model' in config.model:
                if config.label_map is None:
                    label_map_model = MSCOCO
                else:
                    label_map_model = load_pbtxt(config.label_map)
                saved_model = True
                detector = load_saved_model(config.model)
            else:
                label_map_model = MSCOCO
                detector = init_detector('Faster R-CNN Inception ResNet V2 1024x1024')
            print(f'Detector initialized.')

        if config.classify:
            classifier = Classifier(config.model)

        if config.save is not None:
            columns = ['frame', 'class_id', 'score',
                       'xmin', 'ymin', 'xmax', 'ymax', 'label']
            save_results = []

    prev_scores = None
    prev_classes = None

    while True:
        ret, frame = vs.read()
        # if the frame could not be grabbed, then we have reached the end
        # of the video
        if not ret:
            if config.loop:
                vs.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = vs.read()
            
            if not ret:
                break

        if config.skip_frames > 0:
            if config.skip_frames > frames_skipped:
                frames_skipped += 1
                continue
            else:
                frames_skipped = 0
        
        time_since_last_frame = time.time() - start_frame
        if time_since_last_frame < frame_frequency:
            time.sleep(frame_frequency-time_since_last_frame)

        start_frame = time.time()
        # read next frame
        if pause:
            while True: 
                k = cv2.waitKey(0)
                if key == ord("p"):
                    break
            pause = False
            
        original_width, original_height, _ = frame.shape
        if config.resize_width is not None:
            frame = imutils.resize(frame, height=config.resize_width) 
            (width, height, _) = frame.shape
            resize_width = width / original_width
            resize_height = height / original_height
        else:
            resize_width = 1
            resize_height = 1

        start_time = time.time()
        if config.detect:
            if config.wait_for_detection is None or num_frames >= config.wait_for_detection:

                results = run_detector(detector, frame) 
                boxes = results['detection_boxes'][0]
                scores = results['detection_scores'][0]
                class_ids = results['detection_classes'][0]

                if prev_scores is not None:

                    same_scores = all([s == prev_scores[i] for i,s in enumerate(scores)])
                    same_classes = all([c == prev_classes[i] for i,c in enumerate(class_ids)])

                prev_scores = scores
                prev_classes = class_ids

                # selected_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold=0.3)
                # selected_boxes = np.array(tf.gather(boxes, selected_indices))

                # boxes = selected_boxes
                # scores = np.array(tf.gather(scores, selected_indices))
                # class_ids = np.array(tf.gather(class_ids, selected_indices))
               
                detections_full_str = []
                for i in range(min(boxes.shape[0], max_boxes)):
                    
                    ymin, xmin, ymax, xmax = tuple(boxes[i])
                    (left, right, top, bottom) = (
                        xmin * frame.shape[1], 
                        xmax * frame.shape[1],
                        ymin * frame.shape[0], 
                        ymax * frame.shape[0]
                    )

                    if config.save is not None:
                        save_results.append([
                            num_frames,
                            int(class_ids[i]),
                            scores[i],
                            left, top,
                            right, bottom,
                            label_map_model[int(class_ids[i])]['name']
                        ])

                    if scores[i] >= min_score:
                        # ymin, xmin, ymax, xmax = tuple(boxes[i])
                        # 
                        # (left, right, top, bottom) = (
                        #     xmin * frame.shape[1], 
                        #     xmax * frame.shape[1],
                        #     ymin * frame.shape[0], 
                        #     ymax * frame.shape[0]
                        # )

                        class_id = int(class_ids[i])
                        display_str = "{}: {}%".format(label_map_model[class_id]['name'],
                                                        int(100 * scores[i]))

                        detections_full_str.append(display_str)
                        
                        cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), 1)
                        cv2.putText(frame, display_str, (int(left), int(top)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)

                for j, det in enumerate(detections_full_str):
                    coords = (10, 20 + 15*j)
                    if coords[1] >= frame.shape[0]-20:
                        break
                    cv2.putText(frame, detections_full_str[j], coords,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                

        end_time = time.time() 

        # Draw pre-computed detections
        for r_id, r in enumerate(results_to_compare):
            detections = r[r.frame == num_frames]
            if len(detections) > 0:
                boxes_drawn = 0
                for i, det in detections.iterrows():
                    score = det['score']
                    if score > min_score and boxes_drawn < max_boxes:
                        boxes_drawn += 1
                        # if boxes[i] not in boxes_nms:
                        #     continue
                        class_id = int(det['class_id'])
                        class_name = label_map[class_id]['name']
                        (left, right, top, bottom) = det[['xmin', 'xmax', 'ymin', 'ymax']].values 
                        left = int(left * resize_width)
                        right = int(right * resize_width)
                        top = int(top * resize_height)
                        bottom = int(bottom * resize_height)

                        display_str = "{}: {}%".format(class_name, int(100 * score))

                        cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), COLORS[r_id], 2)
                        cv2.putText(frame, display_str, (int(left), int(top)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS[r_id], 2)
       
        if not config.no_show: 
            # frame = imutils.resize(frame, width=1024)  
            cv2.putText(frame, f'Infer: {end_time-start_time:.3f} sec.', (frame.shape[1]-120, frame.shape[0]-20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
            cv2.putText(frame, f'Frame: {num_frames}', (10, 10),
               cv2.FONT_HERSHEY_SIMPLEX, 1 , (0,0,0))
            if config.resize_width is None:
                frame = imutils.resize(frame, height=1024) 

            cv2.imshow("Detections", frame)
        

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            pause = not pause
        elif key == ord("n"):
            mergeROIs = not mergeROIs
        elif key == ord("1"):
            play_fps = play_fps - 1
            print(f'fps: {play_fps}')
            frame_frequency = 1.0 / play_fps
        elif key == ord("2"):
            play_fps = play_fps + 1
            print(f'fps: {play_fps}')
            frame_frequency = 1.0 / play_fps

        if not pause:
            num_frames = num_frames + 1

        if config.max_frames > 0 and num_frames >= config.max_frames:
            break


    # cleanup the camera and close any open windows
    vs.stop() if config.video is None else vs.release()
    if not config.no_show:
        cv2.destroyAllWindows()

    if config.save is not None:
        save_results = pd.DataFrame(save_results, columns=columns)
        if '.pkl' in config.save:
            save_results.to_pickle(config.save, 'bz2')
        else:
            save_results.to_csv(config.save, index=False)

    print(f'Total frames: {num_frames}')
    print(f'Frames with motion detected: {frames_with_detection}')
    print(f'Number of ROIs proposed from motion: {total_detections}')


if __name__ == "__main__":
    # pr = cProfile.Profile()
    # pr.enable()
    main()
    # pr.disable()
    # ps = pstats.Stats(pr).print_stats()
    # pr.print_stats(sort='time')
    # pr.dump_stats('main.profile')
