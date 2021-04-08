import datetime
import time
import argparse

import numpy as np
import cv2

from imutils.video import VideoStream
import imutils


from nms import non_max_suppression_fast

def gray_and_blur(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    return gray

def get_motion_bb(background, gray, min_area=500, overlapThr=0.1):
    # compute the absolute difference between the current frame and
    # first frame
    frameDelta = cv2.absdiff(background, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), 
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    boxes = np.empty((0,4))
    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < min_area:
            continue
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        boxes = np.append(boxes, np.array([[x, y, x+w, y+h]]), axis=0)

    boxes = non_max_suppression_fast(boxes, overlapThr)
    return boxes, frameDelta, thresh

def draw_bb(frame, boxes):
    text = "Unoccupied"
    for box in boxes:
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        text = "Occupied"

    # draw the text and timestamp on the frame
    cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)


def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to the video file")
    ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
    args = vars(ap.parse_args())
    # if the video argument is None, then we are reading from webcam
    if args.get("video", None) is None:
        vs = VideoStream(src=0).start()
        time.sleep(2.0)
    # otherwise, we are reading from a video file
    else:
        vs = cv2.VideoCapture(args["video"])
    # initialize the first frame in the video stream
    firstFrame = None

    while True:
        # grab the current frame and initialize the occupied/unoccupied
        # text
        frame = vs.read()
        frame = frame if args.get("video", None) is None else frame[1]
        
        # if the frame could not be grabbed, then we have reached the end
        # of the video
        if frame is None:
            break

        # resize the frame, convert it to grayscale, and blur it
        frame = imutils.resize(frame, width=500)
        gray = gray_and_blur(frame)
        # if the first frame is None, initialize it
        if firstFrame is None:
            firstFrame = gray
            continue

        boxes, thresh, frameDelta = get_motion_bb(firstFrame, gray)
        frame = draw_bb(frame, boxes)
    
        # show the frame and record if the user presses a key
        cv2.imshow("Security Feed", frame)
        cv2.imshow("Thresh", thresh)
        cv2.imshow("Frame Delta", frameDelta)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
            break

if __name__ == "__main__":
    main()