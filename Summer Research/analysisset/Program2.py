'''
Program 2: goes through all the images and write the possible object types to the csv file
'''

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms.functional import convert_image_dtype

import torchvision.transforms.functional as F
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn
from  PIL import Image, ImageFont, ImageDraw
plt.rcParams["savefig.bbox"] = 'tight'

import os

def main():
    path = '/Volumes/Personal/ykang23/Summer/analysisset/CU'

    files = os.listdir(path)
    '''
    for f in files:
        print(f)

    '''
    objtype1 = []
    objtype2 = []

    #mask rcnn for classification
    model = maskrcnn_resnet50_fpn(pretrained=True, progress=False)
    model = model.eval()

    # instant classes 
    inst_classes = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    # looping through all the images in the directory  
    for f in files:
        img = read_image(path+'/'+str(f))
        batch_int = torch.stack([img])
        batch = convert_image_dtype(batch_int, dtype = torch.float)
        output = model(batch)

        img_output = output[0]
        img_masks = img_output['masks']
        print(f"shape = {img_masks.shape}, dtype = {img_masks.dtype}, "
            f"min = {img_masks.min()}, max = {img_masks.max()}")

        inst_class_to_idx = {cls: idx for (idx, cls) in enumerate(inst_classes)}

        #print("The object in the picture is: ")
        # get possible object type 
        possible_categories = []
        for label in img_output['labels']:
            possible_categories.append(inst_classes[label])
        print(possible_categories[:2])
        objtype1.append(possible_categories[0])
        objtype2.append(possible_categories[1])

    # save it as csv
    information = np.array([files,objtype1,objtype2])
    information = information.reshape((1,information.shape[0]))
    np.savetxt("object_categories.csv", information, delimiter = ",", header ="Image Title, Object Type I, Object Type II",fmt=['%s','%s','%s'],comments = '')

if __name__ == '__main__':
    main()

