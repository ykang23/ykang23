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

from PIL import Image, ImageFont, ImageDraw

plt.rcParams["savefig.bbox"] = 'tight'

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def main(argv):
    '''
    # takes image as a command line argument
    if len(argv) < 2:
        print("Usage: python3 categorization.py %s <image>" % (argv[0]) )
        return
    '''

    #image = argv[0]
    fig, ax = plt.subplots()
    img = read_image("analysisset/CU/1330_002062.jpg")
    imgg = T.ToPILImage()(img.to('cpu'))
    ax.imshow(np.asarray(imgg))
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    #img = read_image(image)
    batch_int = torch.stack([img])
    batch = convert_image_dtype(batch_int, dtype = torch.float)

    #mask rcnn for classification
    model = maskrcnn_resnet50_fpn(pretrained=True, progress=False)
    model = model.eval()
    output = model(batch)
    #print(output)

    img_output = output[0]
    img_masks = img_output['masks']
    print(f"shape = {img_masks.shape}, dtype = {img_masks.dtype}, "
        f"min = {img_masks.min()}, max = {img_masks.max()}")

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

    inst_class_to_idx = {cls: idx for (idx, cls) in enumerate(inst_classes)}

    #print("The object in the picture is: ")
    possible_categories = []
    for label in img_output['labels']:
        possible_categories.append(inst_classes[label])
    #print(possible_categories[:2])
    #print([inst_classes[label] for label in img_output['labels']])
    my_image = Image.open("analysisset/CU/1330_002062.jpg")
    image_editable = ImageDraw.Draw(my_image)
    text_font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 25, encoding="unic")
    text = "Most likely categories are: "
    text1 = "1: " + possible_categories[0]
    text2 = "2: " + possible_categories[1]
    image_editable.text((0,0), text, (0,0,255), font = text_font)
    image_editable.text((0,30), text1, (0,0,255), font = text_font)
    image_editable.text((0,60), text2, (0,0,255), font = text_font)

    my_image.save("categorized.jpg")
    fig, ax = plt.subplots()
    img = read_image("categorized.jpg")
    imgg = T.ToPILImage()(img.to('cpu'))
    ax.imshow(np.asarray(imgg))
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    batch_int = torch.stack([img])
    batch = convert_image_dtype(batch_int, dtype = torch.float)
    
    # faster rcnn for bouding boxes 
    model = fasterrcnn_resnet50_fpn(pretrained=True, progress=False)
    model = model.eval()
    outputs = model(batch)
    #print(outputs)

    # draw bounding boxes 
    score_threshold = .8
    img_with_boxes = [
        draw_bounding_boxes(img, boxes=output['boxes'][output['scores'] > score_threshold], width=4)
        for img, output in zip(batch_int, outputs)
    ]
    #show(img_with_boxes)
    for img in img_with_boxes:
        tensor = 
    print("the inst is ", type(img_with_boxes))

if __name__ == "__main__":
    main(sys.argv)