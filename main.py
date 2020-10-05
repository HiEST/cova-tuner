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
from utils.detector import init_detector, run_detector, detect_and_draw
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


def main():
    # construct the argument parser and parse the arguments
    args = argparse.ArgumentParser()

    # Required
    args.add_argument("-v", "--video", default=None, help="path to the video file")

    # Application control
    args.add_argument("--max-frames", type=int, default=0, help="maximum frames to process")

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
    args.add_argument("--min-score", type=float, default=0.3, help="minimum score for detections")
    args.add_argument("--detect", help="Detect objects using DNN.", action="store_true")
    args.add_argument("--classify", help="Run image classification using DNN.", action="store_true")
    
    # Control what's shown
    args.add_argument("--debug", help="Show debug information and display all steps.", action="store_true")
    args.add_argument("--no-show", help="Don't show any windows.", action="store_true")
    
    config = args.parse_args()

    frame_width = 1920
    frame_height = 1080
    play_fps = 1000
    frame_frequency = 1.0 / play_fps
    min_score = config.min_score
    mergeROIs = not config.no_merge_rois
    pause = False
    img_id = 1

    # Stats
    frames_with_detection = 0
    total_detections = 0
    num_frames = 0
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
    )

    motionDetector = MotionDetection(
        background=background,
        min_area_contour=config.min_area,
        merge_rois=(not config.no_merge_rois),
    )    

    while True:

        time_since_last_frame = time.time() - start_frame
        if time_since_last_frame < frame_frequency:
            time.sleep(frame_frequency-time_since_last_frame)

        start_frame = time.time()
        # read next frame
        if pause:
            frame = prev_frame.copy()
        else:
            ret, frame = vs.read()
            # if the frame could not be grabbed, then we have reached the end
            # of the video
            if not ret:
                break
            prev_frame = frame.copy()

        if recorder:
            recorder.write(frame)

        if config.detect:
            detection, inf_time = detect_and_draw(
                detector=detector, 
                roi=imutils.resize(frame, width=800), 
                frame=None, 
                min_score=min_score, 
                results_file=results_file, 
                frame_id=num_frames)


        start_time = time.time()

        boxes = motionDetector.detect(frame)            

        if len(boxes):
            frames_with_detection += 1
            total_detections += len(boxes)

            if config.classify or config.detect:
                # roi_proposals = region_proposal(boxes.tolist())  
                # frame_roi = frame.copy()      
                # for i, roi in enumerate(boxes):
                detection_frame = frame.copy()
                for roi in boxes:
                    cropped_roi = np.array(frame[roi[1]:roi[3], roi[0]:roi[2]])

                    if config.classify:
                        pred = classifier.classify(cropped_roi)
                        left, top, right, bottom = tuple(roi)
                        # (left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1],
                        #                     ymin * frame.shape[0], ymax * frame.shape[0])
                        
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

                            # cv2.rectangle(frame_roi, (roi[0], roi[1]), (roi[2], roi[3]), (255, 0, 255), 1)
                        else:
                            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), COLOR_POOR_CLASSIF, 1)

                    if config.detect:
                        # cropped_roi = np.array(frame[roi[1]:roi[3], roi[0]:roi[2]])
                        detection_roi, inf_time = detect_and_draw(
                                                            detector=detector, 
                                                            roi=cropped_roi,
                                                            frame=None, 
                                                            min_score=min_score, 
                                                            results_file=results_file, 
                                                            frame_id=num_frames
                                                        )

                        # replace roi in original detection frame
                        detection_frame[roi[1]:roi[3], roi[0]:roi[2]] = detection_roi
                    cv2.imshow(f"Detection ROI", imutils.resize(detection_frame, width=800))
        
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
            # frame = imutils.resize(frame, width=800)  
            # Draw ROIs
            for box in boxes:
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), COLOR_MOTION, 1)            
            frame = imutils.resize(frame, width=800)  
            
            cv2.putText(frame, "Background substraction time: {:.3f}".format(end_time - start_time), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            
            cv2.imshow("Motion detection", frame)

            if config.debug:
                gray = motionDetector.current_gray
                thresh = motionDetector.current_threshold
                frameDelta = motionDetector.current_delta
                background_color = background.background_color

                cv2.imshow("Gray", imutils.resize(gray, width=500))
                cv2.imshow("Thresh", imutils.resize(thresh, width=500))
                cv2.imshow("Frame Delta", imutils.resize(frameDelta, width=500))
                cv2.imshow("Background", imutils.resize(background_color, width=500))
                
            if config.detect:
                cv2.imshow("Full frame Detection", imutils.resize(detection, width=800))
        

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            pause = not pause
        elif key == ord("n"):
            mergeROIs = not mergeROIs

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
    pr = cProfile.Profile()
    pr.enable()
    main()
    pr.disable()
    # ps = pstats.Stats(pr).print_stats()
    pr.print_stats(sort='time')
    pr.dump_stats('main.profile')
