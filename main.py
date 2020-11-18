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
from utils.capture_screen import CaptureScreen
from utils.constants import *


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


def main():
    # construct the argument parser and parse the arguments
    args = argparse.ArgumentParser()

    # Required
    args.add_argument("-v", "--video", default=None, help="path to the video file")

    # Application control
    args.add_argument("--max-frames", type=int, default=0, help="maximum frames to process")
    args.add_argument("--skip-frames", type=int, default=0, help="number of frames to skip for each frame processed")
    args.add_argument("--loop", action="store_true", default=False, help="loop video")

    # Save results
    args.add_argument("-o", "--output", default=None, help="path to the output video file")
    args.add_argument("-r", "--results", default=None, help="detection results")
    args.add_argument("-s", "--save", action="store_true", help="Save ROIs as images")
    args.add_argument("--save-ratio", type=float, default=0.1, help="Ratio of ROIs that are saved")
    
    # Motion Detection
    args.add_argument("--no-merge-rois", action="store_true", help="Don't merge ROIs on scene")
    args.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
    args.add_argument("--no-average", help="use always first frame as background.", action="store_true")
    
    # Detection/Classification
    args.add_argument("-m", "--model", default="resnet", help="Model for image classification")
    args.add_argument("--min-score", type=float, default=0.6, help="minimum score for detections")
    args.add_argument("--max-boxes", type=float, default=10, help="maximumim number of bounding boxes per frame")
    args.add_argument("--detect", help="Detect objects using DNN.", action="store_true")
    args.add_argument("--wait-for-detection", help="Number of frames to wait before first detection.", default=None, type=int)
    args.add_argument("--classify", help="Run image classification using DNN.", action="store_true")
    
    # Control what's shown
    args.add_argument("--debug", help="Show debug information and display all steps.", action="store_true")
    args.add_argument("--no-show", help="Don't show any windows.", action="store_true")
    
    config = args.parse_args()

    frame_width = 1920
    frame_height = 1080
    play_fps = 1
    frame_frequency = 1.0 / play_fps
    min_score = config.min_score
    mergeROIs = not config.no_merge_rois
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

    recorder = None
    if config.output:
        if isfile(config.output):
            print("ERROR: output file already exists. Move it or use a different name and try again.")
            sys.exit()
        recorder = cv2.VideoWriter(config.output, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))

    results_file = None
    classifier = None
    detector = None
    if config.detect or config.classify:

        if config.detect:
            detector = init_detector()
            print(f'Detector initialized.')

        if config.classify:
            classifier = Classifier(config.model)

        if config.results:
            results_file = open(config.results, 'w')
            results_file.write('frame,class,score,BBox\n')


    background = Background(
        no_average=config.no_average,
        skip=10,
        take=10,
        use_last=15,
    )

    motionDetector = MotionDetection(
        background=background,
        min_area_contour=config.min_area,
        merge_rois=(not config.no_merge_rois),
    )    

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
            
        
        if recorder:
            recorder.write(frame)

        if config.detect:
            if config.wait_for_detection is None or num_frames >= config.wait_for_detection:

                full_frame_detections = frame.copy()
                results = run_detector(detector, full_frame_detections) 
                boxes = results['detection_boxes'][0]
                scores = results['detection_scores'][0]
                class_ids = results['detection_classes'][0]

                
                detections_full_str = []
                for i in range(min(boxes.shape[0], max_boxes)):
                    if scores[i] >= min_score:
                        ymin, xmin, ymax, xmax = tuple(boxes[i])
                        
                        (left, right, top, bottom) = (
                            xmin * full_frame_detections.shape[1], 
                            xmax * full_frame_detections.shape[1],
                            ymin * full_frame_detections.shape[0], 
                            ymax * full_frame_detections.shape[0]
                        )


                        class_id = int(class_ids[i])
                        display_str = "{}: {}%".format(label_map[str(class_id)]['name'],
                                                        int(100 * scores[i]))

                        detections_full_str.append(display_str)
                        
                        cv2.rectangle(full_frame_detections, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), 2)
                        cv2.putText(full_frame_detections, display_str, (int(left), int(top)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                joint_detections = full_frame_detections.copy()
                
        start_time = time.time()

        boxes = motionDetector.detect(frame)            

        
        if len(boxes):
            
            frames_with_detection += 1
            total_detections += len(boxes)

            if config.classify or config.detect:
                detection_frame = frame.copy()
                detections_str = []
                for roi in boxes:
                    cropped_roi = np.array(frame[roi[1]:roi[3], roi[0]:roi[2]])

                    if config.classify:
                        pred = classifier.classify(cropped_roi)
                        left, top, right, bottom = tuple(roi)
                        
                        (imagenetID, label, prob) = pred[0]   
                        if prob > min_score:
                            pred_str = "{} ({:.2f}%)".format(label, prob*100)
                            print(pred_str)                             
                            
                            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), COLOR_GOOD_CLASSIF, 1)
                            cv2.putText(frame, pred_str, (int(left), int(top)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)


                            cv2.imshow(f"Detection ROI", cropped_roi)
                            cv2.putText(frame, pred_str, (int(left), int(top)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                        else:
                            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), COLOR_POOR_CLASSIF, 1)

                    if config.detect:
                        if config.wait_for_detection is None or num_frames >= config.wait_for_detection:

                            results = run_detector(detector, cropped_roi) 
                            boxes = results['detection_boxes'][0]
                            scores = results['detection_scores'][0]
                            class_ids = results['detection_classes'][0]
                            
                            cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), COLOR_MOTION, 2)
                            cv2.putText(frame, 'ROI', (roi[0], roi[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_MOTION, 2)

                            for i in range(min(boxes.shape[0], max_boxes)):
                                if scores[i] >= min_score:
                                    ymin, xmin, ymax, xmax = tuple(boxes[i])
                                    
                                    (left, right, top, bottom) = (roi[0] + xmin * cropped_roi.shape[1], roi[0] + xmax * cropped_roi.shape[1],
                                                                roi[1] + ymin * cropped_roi.shape[0], roi[1] + ymax * cropped_roi.shape[0])


                                    class_id = int(class_ids[i])
                                    display_str = "{}: {}%".format(label_map[str(class_id)]['name'],
                                                                    int(100 * scores[i]))

                                    detections_str.append(display_str)
                                    
                                    cv2.rectangle(joint_detections, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
                                    cv2.putText(joint_detections, display_str, (int(left), int(top)-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                                        
                            
            if config.save:
                print("Saving...")
                frame_saved = False
                for roi in boxes:
                    save_roi = (random.random() < config.save_ratio)
                    if not save_roi:
                        print('not this one')
                        continue
                    print(f'Saving in {DATA_PATH}/saved_rois/{num_frames}.{img_id}.jpg')
                    cropped_roi = np.array(frame[roi[1]:roi[3], roi[0]:roi[2]])
                    cv2.imwrite(f'{DATA_PATH}/saved_rois/{num_frames}.{img_id}.jpg', cropped_roi)
                    img_id = img_id + 1

                # save whole frame for comparison
                cv2.imwrite(f'{DATA_PATH}/frames/{num_frames}.jpg', frame)
                        

        end_time = time.time() 
       
        if not config.no_show: 
            # Draw ROIs
            for box in boxes:
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), COLOR_MOTION, 1)            
            

            if config.detect:
                joint_detections = imutils.resize(joint_detections, width=800)
                for j, det in enumerate(detections_full_str):
                    coords = (10, 20 + 15*j)
                    if coords[1] >= joint_detections.shape[0]-20:
                        break
                    cv2.putText(joint_detections, detections_full_str[j], coords,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                if len(boxes):
                    for j, det in enumerate(detections_str):
                        coords = (joint_detections.shape[1]-100, 20 + 15*j)
                        if coords[1] >= joint_detections.shape[0]-20:
                            break
                        cv2.putText(joint_detections, detections_str[j], coords,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                if config.wait_for_detection is None or num_frames >= config.wait_for_detection:
                    cv2.imshow("Detections", joint_detections)

            if config.debug:
                gray = motionDetector.current_gray
                thresh = motionDetector.current_threshold
                frameDelta = motionDetector.current_delta
                background_color = background.background_color

                cv2.imshow("Gray", imutils.resize(gray, width=500))
                cv2.imshow("Thresh", imutils.resize(thresh, width=500))
                cv2.imshow("Frame Delta", imutils.resize(frameDelta, width=500))
                cv2.imshow("Background", imutils.resize(background_color, width=500))
                    
            frame = imutils.resize(frame, width=800)  
            cv2.imshow("Motion Detection", frame)
        

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
