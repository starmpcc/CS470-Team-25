import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet34
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import csv

root = os.getcwd()
device = torch.device("cuda")

#define hyperparameters
val_set_ratio = 0.25
learning_rate = 0.01
num_epoches = 50
num_classes = 91


def rec_freeze(model):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False
        rec_freeze(child)


#Read Pre-processed data
face_data = torch.zeros(100, 10000, 10).to(device)

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#temporary loader for raw image
temp_transform = transforms.Compose([
                                    transforms.Resize(224),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize,
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
        index = int(os.path.basename(self.imgs[idx]).split('.')[0])
        target = {}        
        target["image"] = img

        #To prevent Cuda error
        target["label"] = int(label)-1
        target["index"] = int(index)
        return target
    
    def __len__(self):
        return len(self.imgs)



class ACNN(nn.Module):
    def __init__(self):
        super(ACNN, self).__init__()

        #Get layers from pretrained resnet34
        resnet = resnet34(pretrained = True)
        l = []
        for child in resnet.children():
            l.append(child)
        
        #Original layers from resnet34
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
        self.fc = nn.Linear(512, num_classes)

        self.fmn1_1 = nn.Linear(18, 60)
        self.fmn1_2 = nn.Linear(60, 200)
        self.fmn1_3 = nn.Linear(200,32768)

        self.fmn2_1 = nn.Linear(18, 60)
        self.fmn2_2 = nn.Linear(60, 200)
        self.fmn2_3 = nn.Linear(200, 16384)

        rec_freeze(self.conv1)
        rec_freeze(self.layer1)
        rec_freeze(self.layer2)
        #define new layers for Adaptive Convolution
        self.param_ln1 = nn.Linear(1, 1)

    def forward(self, x, cat_face_data):
        #B*3*224*224
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #B*64*32*32
        x = self.layer1(x)
        #B*64*32*32
        x = self.layer2(x)

        fmn = self.fmn1_1(cat_face_data)
        fmn = self.fmn1_2(fmn)
        fmn = self.fmn1_3(fmn)
        fmn = fmn.view(128, 16, 16)
        x = x * fmn
        
        #B*128*16*16
        x = self.layer3(x)
        
        fmn = self.fmn2_1(cat_face_data)
        fmn = self.fmn2_2(fmn)
        fmn = self.fmn2_3(fmn)
        fmn = fmn.view(256, 8, 8)
        x = x*fmn

        #B*256*8*8
        x = self.layer4(x)
        #B*512*4*4
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        #B*512
        x = self.fc(x)

        return x

def run_epoches(model, train_dataloader, val_dataloader, optimizer, fitness):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    for epoch in range(num_epoches):
        model.train()
        cnt = 0
        correct_cnt = 0
        train_loss = 0.0
        for target in train_dataloader:
            x = target["image"].to(device)
            label = target["label"].to(device)
            cat_face_data = face_data[label, target["index"], :]

            pred = model(x, cat_face_data)
            optimizer.zero_grad()
            train_loss = fitness(pred, label)
            train_loss.backward()
            optimizer.step()
            
            _, correct = torch.max(pred, 1)
            correct_cnt += (correct == label).sum().item()
            cnt += x.data.size(0)

        train_losses.append(train_loss)
        train_accs.append(correct_cnt/cnt)

        model.eval()
        cnt = 0
        correct_cnt = 0
        val_loss = 0.0
        for target in val_dataloader:
            with torch.no_grad():
                x = target["image"].to(device)
                label = target["label"].to(device)
                cat_face_data = face_data[label, target["index"], :]

                pred = model(x, cat_face_data)
                val_loss = fitness(pred, label)
                _, correct = torch.max(pred, 1)
                correct_cnt += (correct == label).sum().item()
                cnt += x.data.size(0)

        val_losses.append(val_loss)
        val_accs.append(correct_cnt/cnt)

        print(f"{epoch}th epoch,    train_loss: {train_loss}, val_loss: {val_loss}, train_acc: {train_accs[-1]}, val_acc: {val_accs[-1]}")

    return train_losses, val_losses, train_accs, val_accs



if __name__=="__main__":
    dataset = CatFaceDataset(root, temp_transform)
    #Use ConcatDataset to use refined data
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size = val_set_ratio)
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, 10, True, num_workers = 8)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, 10, True, num_workers = 8)

    # Define Model
    model = ACNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), learning_rate)
    fitness = nn.CrossEntropyLoss()

    train_losses, val_losses, train_accs, val_accs = run_epoches(model, train_dataloader, val_dataloader, optimizer, fitness)

    #Save Model
    torch.save({'epoch':num_epoches, 'model_state_dict':model.state_dict(), 'optimizer_state_dict':optimizer.state_dict(),
                 'train_losses':train_losses, 'val_losses':val_losses, 'train_accs':train_accs, 'val_accs':val_accs, 
                 "train_dataloader":train_dataloader, "val_dataloader":val_dataloader}, os.path.join(root, "ckpt.pt"))