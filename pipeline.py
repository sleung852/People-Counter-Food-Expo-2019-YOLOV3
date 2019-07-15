# import the necessary packages
import numpy as np
import imutils
import time
import cv2
import os
from imutils.video import FPS

# yolov3 pytorch
from sys import platform
from models import *
from utils.datasets import *
from utils.utils import *
import torch

# own packages
from headdetection import HeadDetection
import dlib
from trackingalgo.centroidtracker import CentroidTracker
from trackingalgo.trackableobject import TrackableObject

# setup
display = True
print_result = True

# obtain image data from webcam
#vs = cv2.VideoCapture(0)
vs = cv2.VideoCapture("input/fwss1.MOV")

frameIndex = 0
count = 0
time_per_frame = []
counter = 0
skip_frames = 30
W, H = None, None

# initialise object detection model
headdet = HeadDetection()

# initilise object tracking model
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}
rects = []

# object counting
totalFrames = 0
totalLeft = 0
totalRight = 0

# initialise all the timers to evaluate the pipeline speed
obj_detection_timer = []
obj_tracking_timer = []
displaying_timer = []

# start the frames per second throughput estimator
fps = FPS().start()

# loop over frames from the video file stream
while True:
    # read the next frame from the file
    (grabbed, im0) = vs.read()
    count += 1

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = im0.shape[:2]
        line = [(int(W/2),0),(int(W/2),H)]
        #line = [(0, int(W/2)), (H, int(W/2))]
        print(line)

    # feed the frame into the model
    dets = []
    dets = headdet.detect_one(im0)

    # print detection results
    if print_result:
        pass

    # draw dots and bbox on the frame
    if dets is not None:
        for *xyxy, conf, cls_conf, cls in dets:
            centre_x = int((xyxy[0] + xyxy[2])/2)
            centre_y = int((xyxy[1] + xyxy[3])/2)

            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(xyxy[0], xyxy[1], xyxy[2], xyxy[3])
            tracker.start_track(im0, rect)
            trackers.append(tracker)

            if display & False:
                cv2.circle(im0,(centre_x,centre_y), 2, (0,0,255), -1)
                cv2.rectangle(im0, (xyxy[0],xyxy[1]), (xyxy[2],xyxy[3]), (0,0,255), 2)
    
    ## Object Tracking
    objects = ct.update(rects)
    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # check to see if a trackable object exists for the current
        # object ID
        to = trackableObjects.get(objectID, None)
 
        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)
 
        # otherwise, there is a trackable object so we can utilize it
        # to determine direction
        else:
            # the difference between the y-coordinate of the *current*
            # centroid and the mean of *previous* centroids will tell
            # us in which direction the object is moving (negative for
            # 'up' and positive for 'down')
            x = [c[0] for c in to.centroids]
            direction = centroid[0] - np.mean(x)
            to.centroids.append(centroid)
 
            # check to see if the object has been counted or not
            if not to.counted:
                # if the direction is negative (indicating the object
                # is moving up) AND the centroid is above the center
                # line, count the object
                if direction < 0 and centroid[0] < W // 2 + 100:
                    totalLeft += 1
                    to.counted = True
 
                # if the direction is positive (indicating the object
                # is moving down) AND the centroid is below the
                # center line, count the object
                elif direction > 0 and centroid[0] > W // 2 + 100:
                    totalRight += 1
                    to.counted = True
 
        # store the trackable object in our dictionary
        trackableObjects[objectID] = to

        # draw
        text = "ID {}".format(objectID)
        cv2.putText(im0, text, (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(im0, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    info = [
    ("Left", totalLeft),
    ("Right", totalRight),
    ]

    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(im0, text, (10, H - ((i * 20) + 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    print('Total Left: {}'.format(totalLeft))
    print('Total Right: {}'.format(totalRight))

    # display line
    if display:
        cv2.line(im0, line[0], line[1], (0,255,0), 3)
        # display the drawn frame
        cv2.imshow('frame', im0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # increase frame index
    fps.update()
    frameIndex += 1

# release the file pointers
print("[INFO] cleaning up...")
vs.release()

#dets = np.asarray(dets)

time_took = time.time() - timer_start

print('Time took: {}'.format(time_took))
print('Pixel {} x {}'.format(H, W))
print('FPS: {:.2f}'.format(frameIndex/time_took))

cv2.destroyAllWindows()
