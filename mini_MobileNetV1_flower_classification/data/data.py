import cv2
import numpy as np
import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

img1 = cv2.imread("boat6_0.jpg")
# img = cv2.resize(img, (320, 160))

print(img1.shape)
cv2.imshow('image1', img1)
img1.tofile("boat6_0.bin")
cv2.waitKey(0)

data_transform = transforms.Compose(
        [transforms.PILToTensor()])

# data_transform = transforms.Compose(
#         [transforms.Resize(256),
#          transforms.CenterCrop(224),
#          transforms.PILToTensor()])

img_path = "./boat6_0.jpg"
assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
img2 = Image.open(img_path)
img2 = data_transform(img2)
img2 = img2.numpy()
img2 = img2.transpose((1,2,0))
img2 = img2[:,:,::-1]
cv2.imshow('image2', img2)
cv2.waitKey(0)

# if img1 == img2:
#         print("success")
# print(img2)