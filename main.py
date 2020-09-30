# import the necessary packages
import sys
import numpy as np
from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2
import random

from os.path import isfile

# Tensorflow
import tensorflow as tf
import tensorflow_hub as hub
# from tensorflow.keras.models import save_model

# Auxiliary functions
from utils.classification import Classifier
from utils.detector import init_detector, run_detector, detect_and_draw
from utils.nms import non_max_suppression_fast
from utils.iou import merge_recs, intersection
from utils.capture_screen import CaptureScreen


# define colors
COLOR_GOOD_CLASSIF = (255, 0, 255)
COLOR_POOR_CLASSIF = (255, 255, 255)
COLOR_DETECTION = (0, 0, 255)
COLOR_MOTION = (0, 255, 0)
COLOR_ROI = (255, 0, 0)

# path to save images and annotations
DATA_PATH = 'data/'

# Sort bounding rects by x coordinate
def getXFromRect(item):
    return item[0]

def getYFromRect(item):
    return item[1]

def merge_boxes(rects, mergeXThreshold=5, min_width=256, min_height=256):
    rects.sort(key = getXFromRect)
    rectsUsed = [False for _ in rects]

    # Array of accepted rects
    acceptedRects = []

    # Merge threshold for x coordinate distance
    xThr = mergeXThreshold

    # Iterate all initial bounding rects
    for supIdx, supVal in enumerate(rects):
        if (rectsUsed[supIdx] == False):

            # Initialize current rect
            currxMin = supVal[0]
            currxMax = supVal[0] + supVal[2]
            curryMin = supVal[1]
            curryMax = supVal[1] + supVal[3]

            # This bounding rect is used
            rectsUsed[supIdx] = True

            # Iterate all initial bounding rects
            # starting from the next
            for subIdx, subVal in enumerate(rects[(supIdx+1):], start = (supIdx+1)):

                # Initialize merge candidate
                candxMin = subVal[0]
                candxMax = subVal[0] + subVal[2]
                candyMin = subVal[1]
                candyMax = subVal[1] + subVal[3]

                # Check if x distance between current rect
                # and merge candidate is small enough
                if (candxMin <= currxMax + xThr):

                    # Reset coordinates of current rect
                    currxMax = candxMax
                    curryMin = min(curryMin, candyMin)
                    curryMax = max(curryMax, candyMax)

                    # Merge candidate (bounding rect) is used
                    rectsUsed[subIdx] = True
                else:
                    break

            # No more merge candidates possible, accept current rect
            # but first, check if it has the minimum height and width.
            # otherwise, take center of rectangle and expand it on both ways.
            currWidth = (currxMax - currxMin)
            currHeight = (curryMax - curryMin)
            if currWidth < min_width:
                currxMax = currxMin + min_width
                # print(f'Width was {currWidth}. Now is {min_width}')
            if currHeight < min_height:
                curryMax = curryMin + min_height
                # print(f'Height was {currHeight}. Now is {min_height}')
            acceptedRects.append([currxMin, curryMin, currxMax - currxMin, curryMax - curryMin])
    
    return acceptedRects

def merge_all_boxes(boxes):
    minX = np.min(boxes[:,0])
    minY = np.min(boxes[:,1])
    maxX = np.max(boxes[:,2])
    maxY = np.max(boxes[:,3])

    return (minX, minY, maxX, maxY)

def single_box(boxes, frame):
    roi = merge_all_boxes(np.array(boxes))
    # preserve aspect ratio
    h_roi = roi[3] - roi[1]
    w_roi = roi[2]-roi[0]
    ar_roi = w_roi / h_roi
    ar_frame = (frame.shape[1]/frame.shape[0])
    
    if ar_roi < ar_frame: # height too large, expand width
        new_width = h_roi * ar_frame
        minX = roi[0] + w_roi/2 - new_width/2
        maxX = roi[0] + w_roi/2 + new_width/2
        assert int(new_width) <= frame.shape[1]
        if minX < 0:
            maxX = maxX - minX
            minX = 0
        elif maxX > frame.shape[1]:
            minX = minX - (maxX - frame.shape[1])
            maxX = frame.shape[1]
        roi = [int(minX), roi[1], int(maxX), roi[3]]
    else:
        new_height = w_roi / ar_frame
        minY = roi[1] + h_roi/2 - new_height/2
        maxY = roi[1] + h_roi/2 + new_height/2
        assert int(new_height) <= frame.shape[0]
        if minY < 0:
            maxY = maxY - minY
            minY = 0
        elif maxY > frame.shape[0]:
            minY = minY - (maxY - frame.shape[0])
            maxY = frame.shape[0]
        roi = [roi[0], int(minY), roi[2], int(maxY)]

    return roi

def region_proposal(boxes, roi_width=256, roi_height=256, max_width=1920, max_height=1080, random_factor=1):
    roi_proposals = []
    # if not isinstance(boxes, np.ndarray):
    boxes = np.array(boxes)

    minBox = False # MinBox means a minimum box (squared) containing the object
    if roi_width == 0 or roi_height == 0:
        minBox = True

    while len(boxes) > 0:
        minX = np.min(boxes[:,0])
        minY = np.min(boxes[:,1])

        roi = [minX, minY, minX+roi_width, minY+roi_height]
        if roi[2] > max_width:
            roi[2] = max_width
        if roi[3] > max_height:
            roi[3] = max_height

        num_boxes = len(boxes)
        boxes = np.array([box for box in boxes if not intersection(roi, box)])
        if len(boxes) < num_boxes:
            roi_proposals.append(roi)
        else:
            # roi_proposals.append(boxes.tolist())
            for box in boxes:
                roi_proposals.append(box)
            break

    return roi_proposals


    for i in range(len(boxes)):
        box = boxes[i]
        roi = [box[0], box[1], box[0]+roi_width, box[1]+roi_height]
        
        np_boxes = np.append(np.array(boxes), np.array([roi]), axis=0)

        num_boxes_prev = len(np_boxes)
        np_boxes = non_max_suppression_fast(np_boxes,0.01)
        num_boxes_after = len(np_boxes)
        if num_boxes_prev == num_boxes_after:
            break
        roi_proposals.append(roi)

    roi_proposals = non_max_suppression_fast(np.array(roi_proposals),0.2).tolist()
    return roi_proposals



def main():
    # construct the argument parser and parse the arguments
    args = argparse.ArgumentParser()
    args.add_argument("-m", "--model", default="resnet", help="Model for image classification")
    args.add_argument("-v", "--video", default=None, help="path to the video file")
    args.add_argument("-o", "--output", default=None, help="path to the output video file")
    args.add_argument("-r", "--results", default=None, help="detection results")
    args.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
    args.add_argument("--min-score", type=float, default=0.3, help="minimum score for detections")
    args.add_argument("--merge-rois", action="store_true", help="Merge ROIs on scene")
    args.add_argument("-s", "--save", action="store_true", help="Save ROIs as images")
    args.add_argument("--save-ratio", type=float, default=0.1, help="Ratio of ROIs that are saved")
    args.add_argument("--max-frames", type=int, default=0, help="maximum frames to process")
    args.add_argument("--no-average", help="use always first frame as background.", action="store_true")
    args.add_argument("--detect", help="Detect objects using DNN.", action="store_true")
    args.add_argument("--classify", help="Run image classification using DNN.", action="store_true")
    
    config = args.parse_args()
    # if the video argument is None, then we are reading from webcam
    if config.video is None:
        vs = VideoStream(src=0).start()
        time.sleep(2.0)
    # or monitor
    elif config.video == "screen":
        vs = CaptureScreen()
    # otherwise, we are reading from a video file
    else:
        vs = cv2.VideoCapture(config.video)

    frame_width = 1920
    frame_height = 1080
    fps = 20

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

    min_area_contour = config.min_area
    overlapThresh = 0.1
    min_score = config.min_score
    mergeXThreshold = 5
    mergeROIs = config.merge_rois
    expansionBBFactor = 0.1
    deltaDiff = 25
    pause = False
    max_unions = 100
    img_id = 1

    # initialize the first frame in the video stream
    firstFrame = None

    avg_every_frame = 10
    avg_skip_frames = 20

    avg_frames = []
    avg_backgrounds = []
    max_backgrounds = 15

    # loop over the frames of the video
    num_frames = 0
    background = None

    start_frame = 0
    end_frame = 0
    change = False
    while True:
        # print(f'num frames: {num_frames}')
        end_frame = time.time()
        if (end_frame - start_frame) < (1.0/fps):
            continue

        start_frame = time.time()
        # read next frame
        if pause:
            frame = prev_frame.copy()
        else:
            ret, frame = vs.read()
            prev_frame = frame.copy()

        # if the frame could not be grabbed, then we have reached the end
        # of the video
        if not ret:
            break

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

        # resize the frame, convert it to grayscale, and blur it
        start_time = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
       
        # if the first frame is None, initialize it
        if background is None:
            background = gray
            avg_background = frame
            continue

        if not config.no_average:
            if num_frames % avg_skip_frames == 0:
                avg_frames.append(frame.copy())
            
                if len(avg_frames) == avg_every_frame:
                    # t0 = time.time()
                    avg_img = np.mean(avg_frames, axis=0)
                    avg_img = avg_img.astype(np.uint8)
                    avg_frames = [avg_img.copy()]
                    
                    avg_backgrounds.append(avg_img)
                    if len(avg_backgrounds) > max_backgrounds:
                        avg_backgrounds = avg_backgrounds[1:]
                    avg_background = np.mean(avg_backgrounds, axis=0)
                    avg_background = avg_background.astype(np.uint8)
                    
                    gray = cv2.cvtColor(avg_background, cv2.COLOR_BGR2GRAY)
                    gray = cv2.GaussianBlur(gray, (21, 21), 0)
                    background = gray

                    # t1 = time.time()
                    # print(f'new avg. frame: {t1-t0:.2f}')
                    

        # compute the absolute difference between the current frame and
        frameDelta = cv2.absdiff(background, gray)
        thresh = cv2.threshold(frameDelta, deltaDiff, 255, cv2.THRESH_BINARY)[1]
        
        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        boxes = [] #np.empty((0,4))
        # loop over the contours
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < min_area_contour:
                continue

            (x, y, w, h) = cv2.boundingRect(c)
            boxes.append([x, y, x+w, y+h])

        if mergeROIs and len(boxes):
            # boxes = merge_recs(boxes, max_unions)
            # boxes = np.array(boxes)
            # boxes = non_max_suppression_fast(boxes,overlapThresh)
            # boxes = boxes.tolist()

            # merged_boxes = [] # merge_boxes(boxes, bbUsed, mergeXThreshold)
        
            boxes = region_proposal(boxes)
        # else:
        #     boxes = random_center(boxes)

        if len(boxes):
            # roi = single_box(boxes)
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
                    print(f'Saving in {DATA_PATH}/saved_rois/{img_id}.jpg')
                    cropped_roi = np.array(frame[roi[1]:roi[3], roi[0]:roi[2]])
                    cv2.imwrite(f'{DATA_PATH}/saved_rois/{img_id}.jpg', cropped_roi)

                    # save whole frame for comparison
                    if not frame_saved:
                        cv2.imwrite(f'{DATA_PATH}/saved_rois/frames/{img_id}.jpg', frame)
                        frame_saved = True
                        
                    img_id = img_id + 1


            for box in boxes:
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), COLOR_MOTION, 1)

            # cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), COLOR_ROI, 1)
            
        frame = imutils.resize(frame, width=800)
        end_time = time.time() 
       
        # draw the text and timestamp on the frame
        # frame = imutils.resize(frame, width=500)
        cv2.putText(frame, "Background substraction time: {:.3f}".format(end_time - start_time), (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
            (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        
        # show the frame and record if the user presses a key
        cv2.imshow("Motion detection", frame)
        cv2.imshow("Thresh", imutils.resize(thresh, width=500))
        cv2.imshow("Frame Delta", imutils.resize(frameDelta, width=500))
        cv2.imshow("Background", imutils.resize(avg_background, width=500))
        
        if config.detect:
            cv2.imshow("Full frame Detection", imutils.resize(detection, width=800))
        

        change = False
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key is pressed, break from the loop
        if key == ord("q"):
            break
        elif key == ord("p"):
            pause = not pause
        elif key == ord("1"):
            min_area_contour = min_area_contour - 10
            print(f"min area contour: {min_area_contour}")
        elif key == ord("2"):
            min_area_contour = min_area_contour + 10
            print(f"min area contour: {min_area_contour}")
        elif key == ord("3"):
            max_unions = max_unions - 1
            print(f"max_unions: {max_unions}")
            change = True
        elif key == ord("4"):
            max_unions = max_unions + 1
            print(f"max_unions: {max_unions}")
            change = True
        elif key == ord("5"):
            min_score = min_score - 0.05
            print(f"min_score: {min_score}")
        elif key == ord("6"):
            min_score = min_score + 0.05
            print(f"min_score: {min_score}")
        elif key == ord("7"):
            deltaDiff = deltaDiff - 1
            print(f"deltaDiff: {deltaDiff}")
        elif key == ord("8"):
            deltaDiff = deltaDiff + 1
            print(f"deltaDiff: {deltaDiff}")
        elif key == ord("n"):
            mergeROIs = not mergeROIs
            print(f"using NMS: {useNMS}")
            change = True

        if not pause:
            num_frames = num_frames + 1

        if config.max_frames > 0 and num_frames >= config.max_frames:
            break

    # cleanup the camera and close any open windows
    vs.stop() if args.get("video", None) is None else vs.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()