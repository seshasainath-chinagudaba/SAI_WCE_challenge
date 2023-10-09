import torch.nn as nn
import torch.optim
import torchvision
import torch.nn.functional as F

import numpy as np
import pandas as pd
from torch.utils.data import Subset

from dataset import WCEClassDataset, WCEClassSubsetDataset
from data_augmentation import WCEImageTransforms
from model import EnsembleModel,CnnModel,SEBlock,GeM,CBAM_Module,AdaptiveConcatPool2d,Flatten
from train import train

def ensemble_loss(outputs, targets, device):
    # BCELoss for each model in the ensemble
    bce_loss = nn.BCEWithLogitsLoss().to(device)

    # Compute individual losses and sum them up
    losses = [bce_loss(F.sigmoid(output), targets.unsqueeze(1)) for output in outputs]
    total_loss = sum(losses)

    return total_loss

model = EnsembleModel(num_models=1)
root_dir = '../datasets/WCEBleedGen'
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
dataset = WCEClassDataset(root_dir=root_dir,num_models=model.num_models)
num_epochs = 18

batch_size = 64

validation_split = 0.2
shuffle_dataset = True
random_seed= 36
save_dir = './'
model_name = 'WCE_class'

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices,val_indices= indices[split:],indices[:split]

rotation_degrees = [30,60,90, 120,150, 180,210, 240,270, 300,330]
blur_parameters = [(11, 9), (9, 7), (7, 5), (5, 3), (3, 1)]
train_transform = WCEImageTransforms(rotation_degrees, blur_parameters)
valid_transform = torchvision.transforms.ToTensor()
train_dataset= WCEClassSubsetDataset(dataset, train_indices, train_transform)
valid_dataset = WCEClassSubsetDataset(dataset, val_indices, valid_transform)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True,num_workers=2,pin_memory=True)
validation_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-3)
criterion = ensemble_loss
lr_scheduler=""
train(model, train_loader, validation_loader, optimizer, lr_scheduler, criterion, device, num_epochs, save_dir, model_name)
