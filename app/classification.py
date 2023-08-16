from torchvision.io import read_image
from torchvision.models.detection import fcos_resnet50_fpn
from torchvision.models.detection import FCOS_ResNet50_FPN_Weights
import torch
from glob import glob
import os
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from collections import Counter
import numpy as np
import re
import requests
from PIL import Image
import torchvision.transforms as transforms
from io import BytesIO



if torch.backends.mps.is_available():
    device = torch.device("mps")


# initiate model and weights
weights = FCOS_ResNet50_FPN_Weights.DEFAULT
model = fcos_resnet50_fpn(weights=weights)
model.eval()




# pass a specific image url (confidence -- 1 to 100)
def inference(confidence ,url):

    item_list = []

    # read and preprocess image

    response = requests.get(url)
    image = Image.open(BytesIO(response.content)) 
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image)
    preprocess = weights.transforms()

    processed_sample_img = preprocess(image_tensor).unsqueeze(0)

    # model inference
    prediction = model(processed_sample_img)[0]

    # prediction: labels, bounding boxes, scores
    labels = [weights.meta["categories"][i] for i in prediction["labels"]] ## model labels
    scores = prediction['scores']
    b_boxes = prediction['boxes']

    # confidence threshold / get list indices for items above threshold
    confidence = confidence / 100
    i_threshold =  [i for i in range(len(scores)) if scores[i] >= confidence] 
    labels_threshold = [labels[i] for i in i_threshold]
    b_boxes_threshold = torch.stack([b_boxes[i] for i in i_threshold])

    # add detected objects to initial item count
    item_list.extend(labels_threshold)
    item_count = Counter(item_list)

    return(item_count)
        


