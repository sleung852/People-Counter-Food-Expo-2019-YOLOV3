#!/usr/bin/env python -W ignore::DeprecationWarning
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
stack_frames = 30
W, H = None, None

# initialise object detection model
headdet = HeadDetection()

# initilise object tracking model
# Return true if line segments AB and CD intersect
objtrack = SortAlgorithm()

# object counting
totalFrames = 0

# initialise all the timers to evaluate the pipeline speed
class PipelineTimer:
    def __init__(self):
        self.total_time = 0
        self.timer_start = 0
        self.timer_end = 0
        self.record = []
    def start(self):
        self.timer_start = time.time()
    def end(self):
        self.timer_end = time.time() - self.timer_start
        self.record.append(self.timer_end)
    def report(self):
        return sum(self.record)/len(self.record)

frames_storing_timer = PipelineTimer()
obj_detection_timer = PipelineTimer()
obj_tracking_timer = PipelineTimer()
#displaying_timer = PipelineTimer()

# start the frames per second throughput estimator
fps = FPS().start()

# stack related
stack_frames_count = 0
im0s=[]

# make a txt file
#txtfile = open('objdetectresult.txt', 'w+')
dims = []

# loop over frames from the video file stream
while True:
    frames_storing_timer.start()
    # read the next frame from the file
    (grabbed, im0) = vs.read()
    stack_frames_count += 1

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = im0.shape[:2]
        #line = [(int(W/2),0),(int(W/2),H)]
        #line = [(0, int(H/2)), (W, int(H/2))]
        line = [(int(W/2)+100,0),(int(W/2)+100,H)] # for fwss1.MOV
        objtrack.set_line(line[0], line[1])

    im0s.append(im0)

    if stack_frames_count == 30:

        frames_storing_timer.end()
        obj_detection_timer.start()

        dets = headdet.detect_mult(im0s)

        obj_detection_timer.end()
        obj_tracking_timer.start()

        for i in range(len(dets)):
            print(type(dets[i]))
            dims.append(dets[i].shape)
            counter = objtrack.output_counter(dets[i])
            print(counter)

        obj_tracking_timer.end()

        stack_frames_count = 0

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

print(dims)

print('Pixel {} x {}'.format(H, W))
#print('FPS: {:.2f}'.format(frameIndex/time_took))

#report
print('Time took for Reading 30 frames: {}'.format(frames_storing_timer.report()))
print('Time took for ObjDetect 30 frames: {}'.format(obj_detection_timer.report()))
print('Time took for ObjTrack 30 frames: {}'.format(obj_tracking_timer.report()))

cv2.destroyAllWindows()

