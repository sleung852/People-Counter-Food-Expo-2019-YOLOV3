#!/usr/bin/env python -W ignore::DeprecationWarning
# import the necessary packages
import numpy as np
import cv2
import os, shutil
import argparse
import time

# yolov3 using pytorch
from models import *
from utils.datasets import *
from utils.utils import *

# own packages
from headdetection import HeadDetection
from objtracksort import SortAlgorithm
from tools.pipelinetimer import PipelineTimer

def main(display, stacknum, test_mode, conf_thres, hor, savebool):
    # setup
    display = opt.display
    stack_num = opt.stacknum
    if stacknum > 1:
        stack_mode = True
    else:
    	stack_mode = False
    test_mode = opt.testmode
    conf_thres = opt.conf

    # setup recording result
    if savebool:
	    if os.path.exists('record'):
	        shutil.rmtree('record')  # delete output folder
	    os.makedirs('record')
	    record_perNframe = 1800 #30frames*60seconds -> record once per minute
	    result_file = open("result.txt", "w+")

    # obtain image data from webcam
    if test_mode:
        vs = cv2.VideoCapture("input/fwss1.MOV")
        #vs = cv2.VideoCapture("input/fwss2.MOV")
    else:
       vs = cv2.VideoCapture(0) 

    frameIndex = 0
    time_per_frame = []
    left_counter = 0
    right_counter = 0
    stack_frames = 30
    W, H = None, None

    # initialise object detection model
    headdet = HeadDetection(conf_thres=conf_thres)

    # initilise object tracking model
    objtrack = SortAlgorithm(hor=True)

    # FPS counting
    totalFrames = 0

    # initialise all the timers to evaluate the pipeline speed
    frames_storing_timer = PipelineTimer() # suggest disable during production
    obj_detection_timer = PipelineTimer()
    obj_tracking_timer = PipelineTimer()
    displaying_timer = PipelineTimer()
    #overall_timer = PipelineTimer()
    fps_timer = PipelineTimer()

    # stack related variables initiation
    stack_frames_count = 0
    im0s=[]

    # make a txt file
    #txtfile = open('objdetectresult.txt', 'w+')
    dims = []

    # loop over frames from the video file stream
    while True:
        frames_storing_timer.start()
        fps_timer.start()
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
            if hor:
                line = [(int(W/2)+100,0),(int(W/2)+100,H)] # for fwss1.MOV
                #line = [(int(W/2),0),(int(W/2),H)]
            else:
                line = [(0, (int(H/2)), (W, int(H/2)))] # for fwss2.MOV or fwss3.MOV
            objtrack.set_line(line[0], line[1])

        if stack_mode:
            im0s.append(im0)

        if stack_frames_count == stack_num:

            frames_storing_timer.end()
            obj_detection_timer.start()

            if stack_mode:
                dets = headdet.detect_mult(im0s)
                obj_detection_timer.end()
                for i in range(len(dets)):
                    #dims.append(dets[i].shape)
                    left_counter, right_counter = objtrack.output_counter(dets[i])
                im0s = []
            else:
                dets = headdet.detect_one(im0)
                obj_detection_timer.end()
                obj_tracking_timer.start()

                left_counter, right_counter = objtrack.output_counter(dets)

            obj_tracking_timer.end()

            stack_frames_count = 0

        if hor:
            info_str = 'Left: {} | Right: {}'.format(left_counter, right_counter)
        else:
            info_str = 'Up: {} | Down: {}'.format(left_counter, right_counter)
        print(info_str)

        # display option
        if display:
            # add a title
            cv2.putText(im0, 'YOLOV3TINY+SORT HUMAN HEAD TRAFFIC COUNTER - DEVELOPED BY SH LEUNG', (50,50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
            # draw the line
            cv2.line(im0, line[0], line[1], (0,255,0), 3)
            # draw counters
            cv2.putText(im0, info_str, (100,100), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 3)
            # display the drawn frame
            cv2.imshow('frame', im0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        

        # increase frame index
        frameIndex += 1
        
        if test_mode:
            if frameIndex > 450: # around 15s testing
                break

        if savebool:
	        if frameIndex % record_perNframe == 0:
	            result_file.write("{}, {}, {}\n".format(time.time(), left_counter, right_counter))

    # release the file pointers
    print("[INFO] cleaning up...")
    vs.release()

    print('Pixel {} x {}'.format(H, W))
    #print('FPS: {:.2f}'.format(fps.fps()))

    #report
    print('Time took for Reading {} frame(s): {}'.format(stacknum, frames_storing_timer.report()))
    print('Time took for ObjDetect {} frame(s): {}'.format(stacknum, obj_detection_timer.report()))
    print('Time took for ObjTrack {} frame(s): {}'.format(stacknum, obj_tracking_timer.report()))

    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testmode', type=bool, default=False)
    parser.add_argument('--display', type=bool, default=False)
    parser.add_argument('--stacknum', type=int, default=1)
    parser.add_argument('--hormode', type=bool, default=True)
    parser.add_argument('--conf', type=float, default=0.4)
    parser.add_argument('--savetxt', type=bool, default=True)
    opt = parser.parse_args()

    with torch.no_grad():
        main(display = opt.display,
            stacknum = opt.stacknum,
            test_mode = opt.testmode,
            conf_thres = opt.conf,
            hor = opt.hormode,
            savebool = opt.savetxt)

