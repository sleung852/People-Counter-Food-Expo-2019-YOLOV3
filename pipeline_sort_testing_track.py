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
from sort import *
from objtracksort import SortAlgorithm

# setup
display = True
print_result = True

# obtain image data from webcam
#vs = cv2.VideoCapture(0)
vs = cv2.VideoCapture("input/fwss1.MOV")
#vs = cv2.VideoCapture("input/fwss2.MOV")

frameIndex = 0
time_per_frame = []
counter = 0
skip_frames = 30
W, H = None, None

# initialise object detection model
headdet = HeadDetection()

# initilise object tracking model
# Return true if line segments AB and CD intersect
objtrack = SortAlgorithm()
COLORS = np.random.randint(0, 255, size=(200, 3),
    dtype="uint8")

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

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = im0.shape[:2]
        line = [(int(W/2)+100,0),(int(W/2)+100,H)] #Vertical Line
        #line = [(0, int(H/2)), (W, int(H/2))] #Horizontal Line
        objtrack.set_line(line[0], line[1])
        print(line)

    # feed the frame into the model
    dets = []
    dets = headdet.detect_one(im0)

    # print detection results
    if print_result:
        pass
        
    ## Object Tracking

    counter = objtrack.output_counter(dets)

    if len(objtrack.boxes) > 0:
        i = int(0)
        for box in objtrack.boxes:
            # extract the bounding box coordinates
            (x11, y11) = (int(box[0]), int(box[1]))
            (x12, y12) = (int(box[2]), int(box[3]))

            # display each bounding box/dot of the detected
            if display:
                color = [int(c) for c in COLORS[objtrack.indexIDs[i] % len(COLORS)]]        
                text = "{}".format(objtrack.indexIDs[i])
                #cv2.rectangle(im0, (x, y), (w, h), color, 2)
                centre_x = int((x11 + x12)/2)
                centre_y = int((y11 + y12)/2)
                cv2.circle(im0,(centre_x,centre_y), 2, color, 2)
                cv2.putText(im0, text, (x11, y11 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            i += 1

    obj_tracker_end = time.time()
    #tracker_timer.append(obj_tracker_end - obj_tracker_start)

    # display line
    if display:
        # draw the line
        cv2.line(im0, line[0], line[1], (0,255,0), 3)
        # draw counter
        cv2.putText(im0, str(counter), (100,200), cv2.FONT_HERSHEY_DUPLEX, 5.0, (0, 255, 255), 10)
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

#time_took = time.time() - timer_start

print('Time took: {}'.format(time_took))
print('Pixel {} x {}'.format(H, W))
print('FPS: {:.2f}'.format(frameIndex/time_took))

cv2.destroyAllWindows()
