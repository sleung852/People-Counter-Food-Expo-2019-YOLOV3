# import the necessary packages
import numpy as np
import imutils
import time
import cv2
import os
import glob

#yolov3 pytorch
from sys import platform
from models import *
from utils.datasets import *
from utils.utils import *

timer_start = time.time()

vs = cv2.VideoCapture("input/medical2.avi")
#vs = cv2.VideoCapture(0)

# derive the paths to the YOLO weights and model configuration
#model_dir = 'prod_model'
#weightsPath = os.path.sep.join([model_dir, "yolov3-tiny_final-TL.weights"])
#configPath = os.path.sep.join([model_dir, "yolov3-tiny.cfg"])
W, H = None, None

### DETECTION FUNCTION - START ###

def detect(im0,
			cfg = "prod_model/yolov3-tiny.cfg",
			weights = "prod_model/yolov3-tiny_final-TL.weights",
			conf_thres = 0.4,
			nms_thres = 0.4):

	img = preprocess_numpy_img(im0)

	device = torch_utils.select_device()
	torch.backends.cudnn.benchmark = False  # set False for reproducible results

	model = Darknet(cfg, 416)

	_ = load_darknet_weights(model, weights)

	# Fuse Conv2d + BatchNorm2d layers
	model.fuse()

	# Eval mode
	model.to(device).eval()

	# Get detections
	img = torch.from_numpy(img).unsqueeze(0).to(device)
	pred, _ = model(img)
	det = non_max_suppression(pred, conf_thres, nms_thres)[0]

	#H_ratio, W_ratio = H_org / H_tran, W_org / W_tran

	if det is not None and len(det) > 0:
		# Rescale boxes from 416 to true image size
		det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
		#det[:, 0] = det[:, 0] * H_ratio
		#det[:, 1] = det[:, 1] * W_ratio
		#det[:, 2] = det[:, 2] * H_ratio
		#det[:, 3] = det[:, 3] * W_ratio

	if det is not None:
		return det.detach().numpy()

	return None

### DETECTION FUNCTION - END ###

frameIndex = 0
count = 0
time_per_frame = []

# loop over frames from the video file stream
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()
	count += 1

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	dets = []
	dets = detect(frame)
	im0 = frame
	if dets is not None:
		for *xyxy, conf, cls_conf, cls in dets:
			centre_x = int((xyxy[0] + xyxy[2])/2)
			centre_y = int((xyxy[1] + xyxy[3])/2)
			cv2.circle(im0,(centre_x,centre_y), 2, (0,0,255), -1)
			cv2.rectangle(im0, (xyxy[0],xyxy[1]), (xyxy[2],xyxy[3]), (0,0,255), 2)

		if False:
			for det in dets:
				x1, y1, x2, y2 = det[0:4]
				cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
		
	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	
	# counter += 1

	# increase frame index
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