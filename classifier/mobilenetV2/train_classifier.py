from __future__ import print_function, division

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import models
import time
import os

from dataset import get_data_loaders, get_img_files, imshow
from train import train_model, visualize_model
from MobileNetV2 import MobileNetV2
from sklearn.model_selection import KFold

np.random.seed(1)
torch.backends.cudnn.deterministic = True
torch.manual_seed(1)

# %%
N_CV = 5
#LR = 1e-4

N_EPOCHS = 100
#N_EPOCHS = 35
IMG_SIZE = 224
RANDOM_STATE = 1

EXPERIMENT = 'train_unet'
OUT_DIR = 'outputs/{}'.format(EXPERIMENT)

data_dir = '../Database/'
img_size = 224
pre_trained_mobnet2='./mobilenet_v2.pth.tar'
n_class=9

plt.ion()   # interactive mode

#def run_cv(img_size, pre_trained, target):
if __name__ == '__main__':
    image_files = get_img_files(data_dir)
    kf = KFold(n_splits=N_CV, random_state=RANDOM_STATE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for n, (train_idx, val_idx) in enumerate(kf.split(image_files)):
#----Prepare Data-------------------------------------------------------------------------------
        train_files = image_files[train_idx]
        val_files = image_files[val_idx]
        data_loaders = get_data_loaders(train_files, val_files, img_size)
        dataset_sizes = [len(train_files), len(val_files)]
        print('dataset_sizes:', dataset_sizes)
        inputs, classes = next(iter(data_loaders[0]))
        #out = torchvision.utils.make_grid(inputs)
        #imshow(out, title=[x for x in classes]) 
#----Prepare Model-------------------------------------------------------------------------------
        model_mobnet2 = MobileNetV2()
        model_mobnet2.load_state_dict(torch.load(pre_trained_mobnet2))
        model_mobnet2.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model_mobnet2.last_channel, n_class),
        )
        model_mobnet2 = model_mobnet2.to(device)
        criterion = nn.CrossEntropyLoss()
        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(model_mobnet2.parameters(), lr=0.001, momentum=0.9)
        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
        model_mobnet2 = train_model(model_mobnet2, criterion, optimizer_ft, exp_lr_scheduler,N_EPOCHS, data_loaders, dataset_sizes, device)
        visualize_model(model_mobnet2, 20, data_loaders, device)

        torch.save(model_mobnet2.state_dict(), 'HairClassifier_mobnet2.pth')
        break;

