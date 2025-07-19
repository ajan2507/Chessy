import shutil
import webbrowser
import cv2
from PIL import Image
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from torch.nn.modules.container import Sequential, ModuleList
from ultralytics.nn.modules.conv import Conv, Concat
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.activation import SiLU
from ultralytics.nn.modules.block import C2f, Bottleneck, SPPF, DFL
from torch.nn.modules.pooling import MaxPool2d
from torch.nn.modules.upsampling import Upsample
from ultralytics.nn.modules.head import Detect
from ultralytics.utils import IterableSimpleNamespace
from ultralytics.utils.loss import v8DetectionLoss, BboxLoss  # <-- Add BboxLoss
from torch.nn.modules.loss import BCEWithLogitsLoss
from ultralytics.utils.tal import TaskAlignedAssigner

torch.serialization.add_safe_globals([
    DetectionModel,
    Sequential,
    ModuleList,
    Conv,
    Concat,
    Conv2d,
    BatchNorm2d,
    SiLU,
    C2f,
    Bottleneck,
    SPPF,
    MaxPool2d,
    Upsample,
    Detect,
    DFL,
    IterableSimpleNamespace,
    v8DetectionLoss,
    BCEWithLogitsLoss,
    TaskAlignedAssigner,
    BboxLoss  # <-- Add here
])       


 

 
########################
image_path='/Users/karthicks7/Downloads/Chess_v1_v2/LICH/ultralytics/test2.png'
#######################




model = YOLO('/Users/karthicks7/Downloads/Chess_v1_v2/LICH/ultralytics/chess2.pt')
results = model.predict(image_path,save=True)
box = results[0].boxes[0]
coord=box.xyxy[0].tolist()


height_diff=abs(coord[3]-coord[1])/8
inter_points_height=[]
inter_points_width=[]
inter_points_height.append(coord[1])
moving=coord[1]


for i in range(8):
   moving+=height_diff
   inter_points_height.append(moving)

width_diff=abs(coord[2]-coord[0])/8
inter_points_width.append(coord[0])
moving=coord[0]

for i in range(8):
   moving+=width_diff
   inter_points_width.append(moving)  



names= {0: 'b', 1: 'b', 2: 'k', 3: 'k', 4: 'n', 5: 'n', 6: 'board', 7: 'p', 8: 'p', 9: 'q', 10: 'q', 11: 'r', 12: 'r', 13: 'eb', 14: 'ew', 15: 'B', 16: 'B', 17: 'K', 18: 'K', 19: 'N', 20: 'N', 21: 'P', 22: 'P', 23: 'Q', 24: 'Q', 25: 'R', 26: 'R'}
model = YOLO('/Users/karthicks7/Downloads/Chess_v1_v2/LICH/ultralytics/best.pt')
# Iterate through cropped images
counter=0
req_url=''
cropped_imager=Image.open(image_path)
for h_start, h_end in zip(inter_points_height[:-1], inter_points_height[1:]):
  
    temp=''
    space=0
    for w_start, w_end in zip(inter_points_width[:-1], inter_points_width[1:]):
        counter+=1
        cropped_image = cropped_imager.crop((w_start, h_start, w_end, h_end))
        results = model.predict(cropped_image, save=True)
        classid=names[int(results[0].boxes[0].cls[0].item())]
        if classid=='board':
           continue
        elif classid=='eb' or classid=='ew':
           space+=1
      
        else:
           if space!=0:
             temp+=str(space)
             space=0 
           temp+=classid               
           
    if space!=0:
     
       temp+=str(space)
    req_url+=temp+"/" 
  
req_url=req_url[:-1]
req_url+="%5B%5D_w_-_-_0_1?color=white"
req_url="https://lichess.org/editor/"+req_url

main_directory = '/Users/karthicks7/Downloads/Chess_v1_v2/LICH/ultralytics/runs/detect'


for subdir in os.listdir(main_directory):
 
    subdir_path = os.path.join(main_directory, subdir)

    if os.path.isdir(subdir_path):
      
        shutil.rmtree(subdir_path)


webbrowser.open(req_url)
