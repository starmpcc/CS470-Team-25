import torch
import torch.nn
from Classifier import CatFaceDataset, val_transform
import os
import torchvision.transforms as transforms


root = os.getcwd()


dataset = CatFaceDataset(root, val_transform)

length = len(dataset)
mean = 0
std = 0

for i in range(length):
    image = dataset.__getitem__(i)["image"]
    mean += image.mean((1,2), True)
    std += image.std((1,2), True)

print(mean/length)
print(std/length)