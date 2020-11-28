import torch
from Classifier import CatFaceIdentifier, CatFaceDataset, SquarePad, val_transform
import os

root= os.getcwd()

checkpoint = torch.load(os.path.join(root, "data_added.pt"))
torch.save({'model_state_dict':checkpoint['model_state_dict']}, os.path.join(root, "resave.pt"))
