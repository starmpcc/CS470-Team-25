from Classifier import CatFaceIdentifier, CatFaceDataset, num_epoches, SquarePad, val_transform
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
checkpoint = torch.load(os.path.join(root, "ckpt_101_1000epoch.pt"))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

val_dataloader = checkpoint['val_dataloader']


def show_plot():
    l1 = plt.plot(range(1, num_epoches+1), checkpoint['train_losses'], range(1, num_epoches+1), checkpoint['val_losses'])
    plt.show()

    l2 = plt.plot(range(1, num_epoches+1), checkpoint['train_accs'], range(1, num_epoches+1), checkpoint['val_accs'])
    plt.show()


def show_result():
    data = None
    for i, j in enumerate(val_dataloader):
        if i==24:
            data = j
            break
    fig, axes = plt.subplots(2, 5)
    imgs = data['image']
    pred = model(imgs.to(device), None)
    _, correct = torch.max(pred, 1)

    for i in range(10):
        ax = axes[i%2, i%5]
        img = imgs[i].cpu()
        img = transforms.ToPILImage()(img)
        ax.imshow(img)
        ax.set_title(f'Ground Truth : {data["label"][i]} \n output : {correct[i]}')
    plt.tight_layout()
    plt.show()

show_plot()
#show_result()

def model_apply(img):
    img = val_transform(img)
    return model.eval(img)