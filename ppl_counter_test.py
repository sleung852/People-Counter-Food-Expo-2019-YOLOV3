# import the necessary packages
from trackingalgo.centroidtracker import CentroidTracker
from trackingalgo.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import dlib
import cv2
import os

class TrackableObject:
	def __init__(self, objectID, centroid):
		# store the object ID, then initialize a list of centroids
		# using the current centroid
		self.objectID = objectID
		self.centroids = [centroid]
 
		# initialize a boolean used to indicate if the object has
		# already been counted or not
		self.counted = False

#settings
save_bool = False
webcam_bool = False
confidence_level = 0.4
skip_frames = 30

writer = None

# derive the paths to the YOLO weights and model configuration
model_dir = 'prod_model'
weightsPath = os.path.sep.join([model_dir, "yolov3-tiny_final-TL.weights"])
configPath = os.path.sep.join([model_dir, "yolov3-tiny.cfg"])
 
# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# if a video path was not supplied, grab a reference to the webcam
if webcam_bool:
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)

# otherwise, grab a reference to the video file
else:
	print("[INFO] opening video file...")
	vs = cv2.VideoCapture('input/fwss3.MOV')

# initialize the video writer (we'll instantiate later if need be)
if save_bool:
	writer = None
 
# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None
 
# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}
 
# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalDown = 0
totalUp = 0

# initialise all the timers to evaluate the pipeline speed
obj_detection_timer = []
obj_tracking_timer = []
displaying_timer = []
 
# start the frames per second throughput estimator
fps = FPS().start()

# loop over frames from the video stream
while True:
	# grab the next frame and handle if we are reading from either
	# VideoCapture or VideoStream
	frame = vs.read()
	if not webcam_bool:
		frame = frame[1]
 
	# if we are viewing a video and we did not grab a frame then we
	# have reached the end of the video
	if frame is None:
		break

	# resize the frame to have a maximum width of 500 pixels (the
	# less data we have, the faster we can process it), then convert
	# the frame from BGR to RGB for dlib
	frame = imutils.resize(frame, width=500)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 
	# if the frame dimensions are empty, set them
	if W is None or H is None:
		(H, W) = frame.shape[:2]
 
	# if we are supposed to be writing a video to disk, initialize
	# the writer
	if save_bool:
		if writer is None:
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter('somename.mp4', fourcc, 30,
				(W, H), True)

	# initialize the current status along with our list of bounding
	# box rectangles returned by either (1) our object detector or
	# (2) the correlation trackers
	status = "Waiting"
	rects = []
 
	obj_detection_timer_start = time.time()

	# check to see if we should run a more computationally expensive
	# object detection method to aid our tracker
	if totalFrames % skip_frames == 0:
		# set the status and initialize our new set of object trackers
		status = "Detecting"
		trackers = []
 
		# convert the frame to a blob and pass the blob through the
		# network and obtain the detections
		blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
		net.setInput(blob)
		detections = net.forward()

		# loop over each of the layer outputs
		for detection in detections:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			confidence = detection[4]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > confidence_level:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# Convert Yolo output to match
				startX = int(centerX-width/2)
				startY = int(centerY-height/2)
				endX = int(centerX+width/2)
				endY = int(centerY+height/2)			

				tracker = dlib.correlation_tracker()
				rect = dlib.rectangle(startX, startY, endX, endY)
				tracker.start_track(rgb, rect)
 
				# add the tracker to our list of trackers so we can
				# utilize it during skip frames
				trackers.append(tracker)


	# object *detectors* to obtain a higher frame processing throughput
	else:
		# loop over the trackers
		for tracker in trackers:
			# set the status of our system to be 'tracking' rather
			# than 'waiting' or 'detecting'
			status = "Tracking"
 
			# update the tracker and grab the updated position
			tracker.update(rgb)
			pos = tracker.get_position()
 
			# unpack the position object
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())
 
			# add the bounding box coordinates to the rectangles list
			rects.append((startX, startY, endX, endY))

	obj_detection_timer.append(time.time() - obj_detection_timer_start)

	obj_tracking_timer_start = time.time()

	# draw a horizontal line in the center of the frame -- once an
	# object crosses this line we will determine whether they were
	# moving 'up' or 'down'
	cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)
 
	# use the centroid tracker to associate the (1) old object
	# centroids with (2) the newly computed object centroids
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
			y = [c[1] for c in to.centroids]
			direction = centroid[1] - np.mean(y)
			to.centroids.append(centroid)
 
			# check to see if the object has been counted or not
			if not to.counted:
				# if the direction is negative (indicating the object
				# is moving up) AND the centroid is above the center
				# line, count the object
				if direction < 0 and centroid[1] < H // 2:
					totalUp += 1
					to.counted = True
 
				# if the direction is positive (indicating the object
				# is moving down) AND the centroid is below the
				# center line, count the object
				elif direction > 0 and centroid[1] > H // 2:
					totalDown += 1
					to.counted = True
 
		# store the trackable object in our dictionary
		trackableObjects[objectID] = to

		# draw both the ID of the object and the centroid of the
		# object on the output frame
		text = "ID {}".format(objectID)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
	
	obj_tracking_timer.append(time.time() - obj_tracking_timer_start)

	displaying_timer_start = time.time()

	# construct a tuple of information we will be displaying on the
	# frame
	info = [
		("Up", totalUp),
		("Down", totalDown),
		("Status", status),
	]
 
	# loop over the info tuples and draw them on our frame
	for (i, (k, v)) in enumerate(info):
		text = "{}: {}".format(k, v)
		cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

	# check to see if we should write the frame to disk
	if writer is not None:
		writer.write(frame)
 
	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
 
	# increment the total number of frames processed thus far and
	# then update the FPS counter
	totalFrames += 1
	fps.update()

	displaying_timer.append(time.time() - displaying_timer_start)

	if totalFrames > 500:
		break


# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

print("[Pipeline Breakdown]")
print("Object Detection Average Time per frame: {}".format(np.mean(obj_detection_timer)))
print("Object Tracking Average Time per frame: {}".format(np.mean(obj_tracking_timer)))
print("OpenCV IMSHOW Average Time per frame: {}".format(np.mean(displaying_timer)))
 
# check to see if we need to release the video writer pointer
if writer is not None:
	writer.release()
 
# if we are not using a video file, stop the camera video stream
if webcam_bool:
	vs.stop()
 
# otherwise, release the video file pointer
else:
	vs.release()
 
# close any open windows
cv2.destroyAllWindows()