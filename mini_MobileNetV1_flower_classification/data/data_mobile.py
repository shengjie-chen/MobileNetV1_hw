import cv2
import numpy as np
import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")
# img = cv2.imread("tulip.jpg")
# img = cv2.resize(img, (320, 160))

data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.PILToTensor()])

# img_path = "./rose.jpeg"
img_path = "./tulip.jpg"
assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
img = Image.open(img_path)
img = data_transform(img)
img = img.numpy()
img = img.transpose((1,2,0))
img = img[:,:,::-1]

cv2.imshow('image', img)
cv2.waitKey(0)

print(img.shape)
# img.tofile("rose.bin")
img.tofile("tulip.bin")


