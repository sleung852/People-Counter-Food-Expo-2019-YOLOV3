from torch2trt import torch2trt
from models import *
from tools.pipelinetimer import PipelineTimer
import torch

torch_timer = PipelineTimer()
trt_timer = PipelineTimer()

model = Darknet('prod_model/yolov3-tiny.cfg', 416)

model = model(pretrained=True).eval().cuda()

x = torch.ones((1, 3, 256, 416)).cuda()

model_trt = torch2trt(model, [x])

torch_timer.start()
y = model(x)
torch_timer.end()

trt_timer.start()
y_trt = model_trt(x)
trt_timer.end()

print('Result Test - Difference between the Two:')

print(torch.max(torch.abs(y - y_trt)))

print('Speed Test:')

print('Torch: {}s'.format(torch_timer.report()))
print('TRT: {}s'.format(trt_timer.report()))