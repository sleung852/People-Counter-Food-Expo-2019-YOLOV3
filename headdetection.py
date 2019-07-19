# import the necessary packages
import numpy as np
import imutils
import cv2

#yolov3 pytorch
from sys import platform
from models import *
from utils.datasets import *
from utils.utils import *
import torch

class HeadDetection:

	def __init__(self, cfg = "prod_model/yolov3-tiny.cfg", weights = "prod_model/yolov3-tiny_final-TL.weights", conf_thres = 0.4):
		## setup device
		self.device = torch_utils.select_device()

		torch.backends.cudnn.benchmark = True  # set False for reproducible results
		self.conf_thres = conf_thres
		self.nms_thres = 0.4

		## setup model
		self.model = Darknet(cfg, 416)
		_ = load_darknet_weights(self.model, weights)
		# Fuse Conv2d + BatchNorm2d layers
		self.model.fuse()
		# Eval mode
		self.model.to(self.device).eval()

	def detect_one(self, im0):

		img = preprocess_numpy_img(im0)
		print(img.shape)
		# Get detections
		img = torch.from_numpy(img).unsqueeze(0).to(self.device)
		pred, _ = self.model(img)
		det = non_max_suppression(pred, self.conf_thres, self.nms_thres)[0]

		#H_ratio, W_ratio = H_org / H_tran, W_org / W_tran

		if det is not None and len(det) > 0:
			# Rescale boxes from 416 to true image size
			det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

		if det is not None:
			result = det.detach()
			if torch.cuda.is_available():
				result = result.cpu()
				return result.numpy()
			return result.numpy()

		return []

	def detect_mult(self, im0_list):
		im0_list_output = []
		for im0 in im0_list:
			im0_list_output.append(preprocess_numpy_img(im0))

		imgs_input = np.stack(im0_list_output)

		# Get detections
		imgs = torch.from_numpy(imgs_input).to(self.device)
		pred, _ = self.model(imgs)
		dets = non_max_suppression(pred, self.conf_thres, self.nms_thres) #see whether can solve this via 

		#H_ratio, W_ratio = H_org / H_tran, W_org / W_tran

		if dets is not None and len(dets) > 0:
			# Rescale boxes from 416 to true image size
			#print((imgs_input[0].shape[1:])) correct!
			#print((im0_list[0].shape[:2])) correct!

			for i in range(len(dets)):
				dets[i][:, :4] = scale_coords(imgs_input[0].shape[1:], dets[i][:, :4], im0_list[0].shape[:2]).round()
				dets[i] = dets[i].detach().cpu().numpy()
			#print(dets.shape)
			#dets[:, :, :4] = dets[:, :, :4] * np.array([])
		return dets