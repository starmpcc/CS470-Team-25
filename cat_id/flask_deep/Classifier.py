import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet101
from PIL import Image
from sklearn.model_selection import train_test_split
from skimage.feature import hog
import time


root = os.getcwd()
device = torch.device("cuda")


#define hyperparameters
val_set_ratio = 0.25
learning_rate = 0.1
num_epoches = 1000
num_classes = 88
batch_size = 32
aug_mul = 1

#Utility Functions
def rec_freeze(model):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False
        rec_freeze(child)


# Transforms and Transform-Support Functions for data
class SquarePad:
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return transforms.functional.pad(image, padding, 0, 'constant')

    
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

val_transform = transforms.Compose([
                                    SquarePad(),
                                    transforms.Resize([224, 224]),
                                    transforms.ToTensor(),
                                    normalize
                                    ])

aug_transform = transforms.Compose([
                                    transforms.RandomRotation(180),
                                    transforms.RandomHorizontalFlip(),
                                    SquarePad(),
                                    transforms.Resize([224, 224]),
                                    transforms.ToTensor(),
                                    normalize
])


class CatFaceDataset(torch.utils.data.Dataset):
    #Dict {image:Tensor(B*224*224), label:int, index:int}

    def __init__(self, root, transform):
        self.root = root
        self.imgs = []
        self.cats = list(sorted(os.listdir(os.path.join(root, "data_collect","cat"))))
        for cat in self.cats:
            imagelist = list(sorted(os.listdir(os.path.join(root,"data_collect","cat", cat))))
            self.imgs += [os.path.join(root,"data_collect","cat", cat, i) for i in imagelist]
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert("RGB")
        img = self.transform(img)
        label = self.imgs[idx].split('/')[-2].split('_')[-1]
        target = {}        
        target["image"] = img

        #To prevent Cuda error
        target["label"] = int(label)
        return target
    
    def __len__(self):
        return len(self.imgs)


class CatFaceIdentifier(nn.Module):
    def __init__(self):
        super(CatFaceIdentifier, self).__init__()

        #Get layers from pretrained resnet101
        resnet = resnet101(pretrained = True)
        l = []
        for child in resnet.children():
            l.append(child)
        
        #Original layers from resnet101
        self.conv1 = l[0]
        self.bn1 = l[1]
        self.relu = l[2]
        self.maxpool = l[3]
        self.layer1 = l[4]
        self.layer2 = l[5]
        self.layer3 = l[6]
        self.layer4 = l[7]
        self.avgpool = l[8]
#        self.fc = l[9]

        #Re-Define final fc layer to adapt our model
        self.fc = nn.Linear(512*4, num_classes)

        rec_freeze(self.conv1)
        rec_freeze(self.layer1)
        rec_freeze(self.layer2)

    def forward(self, x):
        #B*3*224*224
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #B*64*32*32
        x = self.layer1(x)
        #B*64*32*32
        x = self.layer2(x)
        #B*128*16*16
        x = self.layer3(x)
        #B*256*8*8
        x = self.layer4(x)
        #B*512*4*4
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        #B*512
        x = self.fc(x)

        return x
