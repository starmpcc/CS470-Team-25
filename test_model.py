from Classifier import CatFaceIdentifier, CatFaceDataset, SquarePad, val_transform
import torch
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

root = os.getcwd()
device = torch.device('cuda')

#load Model
model = CatFaceIdentifier().to(device)
checkpoint = torch.load(os.path.join(root, "101_200.pt"))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

val_dataloader = checkpoint['val_dataloader']


def show_plot():
    plt.plot(range(1, checkpoint["epoch"]+1), checkpoint['train_losses'], label='train loss')
    plt.plot(range(1, checkpoint["epoch"]+1), checkpoint['val_losses'], label = 'val loss')
    plt.legend(loc="upper left")
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.show()

    plt.plot(range(1, checkpoint["epoch"]+1), checkpoint['train_accs'], label = 'train acc')
    plt.plot(range(1, checkpoint["epoch"]+1), checkpoint['val_accs'], label='val acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel("accuracy")
    plt.show()


def show_result():
    data = None
    for i, j in enumerate(val_dataloader):
        if i==1:
            data = j
            break
    fig, axes = plt.subplots(2, 5)
    imgs = data['image']
    pred = model(imgs.to(device))
    _, correct = torch.max(pred, 1)

    for i in range(10):
        ax = axes[i%2, i%5]

        img = imgs[i].cpu()
        img = transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225))(img)
        img = transforms.ToPILImage()(img)
        ax.imshow(img)
        ax.set_title(f'Ground Truth : {data["label"][i]} \n output : {correct[i]}')
    plt.tight_layout()
    plt.show()

show_plot()
show_result()

def model_apply(img):
    img = val_transform(img)
    return model.eval(img)