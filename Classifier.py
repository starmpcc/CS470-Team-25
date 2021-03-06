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
import time


root = os.getcwd()
device = torch.device("cuda")


#define hyperparameters
val_set_ratio = 0.25
learning_rate = 0.1
num_epoches = 200
num_classes = 88
batch_size = 64

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


def run_epoches(model, train_dataloader, val_dataloader, optimizer, fitness, best_accs):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    train_top5 = []
    val_top5 = []
    best_model = None
    best_optimizer = None
    for epoch in range(num_epoches):
        model.train()
        cnt = 0
        correct_cnt = 0
        train_loss = 0.0
        top5_cnt = 0
        for target in train_dataloader:
            x = target["image"].to(device)
            label = target["label"].to(device)
            pred = model(x)
            optimizer.zero_grad()
            train_loss = fitness(pred, label)
            train_loss.backward()
            optimizer.step()
            _, correct = torch.max(pred, 1)
            correct_cnt += (correct == label).sum().item()
            _, top5 = torch.topk(pred, 5, 1)
            for i in range(len(label)):
                if (label[i] in top5[i]):
                    top5_cnt+=1
            cnt += x.data.size(0)

        train_losses.append(train_loss)
        train_accs.append(correct_cnt/cnt)
        train_top5.append(top5_cnt/cnt)
        model.eval()
        cnt = 0
        correct_cnt = 0
        val_loss = 0.0
        top5_cnt = 0
        for target in val_dataloader:
            with torch.no_grad():
                x = target["image"].to(device)
                label = target["label"].to(device)

                pred = model(x)
                val_loss = fitness(pred, label)
                _, correct = torch.max(pred, 1)
                correct_cnt += (correct == label).sum().item()
                _, top5 = torch.topk(pred, 5, 1)
                for i in range(len(label)):
                    if (label[i] in top5[i]):
                        top5_cnt+=1
                cnt += x.data.size(0)

        val_losses.append(val_loss)
        val_accs.append(correct_cnt/cnt)
        val_top5.append(top5_cnt/cnt)
        if ((epoch >=2) and (val_accs[-1]<val_accs[-2])):
            scheduler.step()
        print(f"{epoch}th epoch,    train_loss: {train_loss}, val_loss: {val_loss}, train_acc: {train_accs[-1]}, val_acc: {val_accs[-1]}, train_top5: {train_top5[-1]}, val_top5: {val_top5[-1]}")
        if ((epoch >=2) and (val_accs[-1] > best_accs)):
            best_accs = val_accs[-1]
            best_model = model.state_dict()
            best_optimizer = optimizer.state_dict() 

    torch.save({'epoch':num_epoches, 'model_state_dict':best_model, 'optimizer_state_dict':best_optimizer,
            'train_losses':train_losses, 'val_losses':val_losses, 'train_accs':train_accs, 'val_accs':val_accs, 
            "train_dataloader":train_dataloader, "val_dataloader":val_dataloader}, os.path.join(root, "101_200.pt"))
if __name__=="__main__":
    t = time.time()

    dataset_train = CatFaceDataset(root, aug_transform)
    dataset_val = CatFaceDataset(root, val_transform)
    #Use ConcatDataset to use refined data
    train_idx, val_idx = train_test_split(list(range(len(dataset_train))), test_size = val_set_ratio)
    train_dataset = torch.utils.data.Subset(dataset_train, train_idx)
    val_dataset = torch.utils.data.Subset(dataset_val, val_idx)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, True, num_workers = 8)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size, True, num_workers = 8)


    # Define Model
    model = CatFaceIdentifier().to(device)
    optimizer = torch.optim.SGD(model.parameters(), learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.8)
    fitness = nn.CrossEntropyLoss()

    run_epoches(model, train_dataloader, val_dataloader, optimizer, fitness, 0.7)
    print(time.gmtime(time.time()-t))

