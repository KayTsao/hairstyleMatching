import os
import sys
from glob import glob
import random
import cv2
import re
from PIL import Image

import torch
import torchvision
import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import Compose, RandomResizedCrop, RandomRotation, RandomHorizontalFlip, ToTensor, \
    Resize, RandomAffine, ColorJitter, Normalize
from sklearn.model_selection import KFold

data_dir = './Database/'
N_CV = 5
BATCH_SIZE = 8
RANDOM_STATE = 1
img_size = 224

def getClass(x):
    return {
        'lc0': 0,
        'lcb': 1,
        'ls0': 2,
        'lsb': 3,
        'mc0': 4,
        'mcb': 5,
        'ms0': 6,
        'msb': 7,
        'pt': 8,
    }[x]

def _check_img(img_file):
    if os.path.splitext(img_file)[-1] == ".jpg":
        return img_file

def get_img_files(i_dir):
    img_files = sorted(glob('{}/*.jpg'.format(i_dir)))
    #print(img_files)
    return np.array([_check_img(f) for f in img_files])

class HairStyleDataset(Dataset):
    """Hair Style Dataset."""
    def __init__(self, img_files, transform=None):
        """
        Args:
            img_files (string): Directory with all the images
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.img_files = img_files
        self.transform = transform
        
    def __getitem__(self, idx):
        img = cv2.imread(self.img_files[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_name = self.img_files[idx]
        img_name = img_name.split("/")[-1]
        classname = img_name.split('-')[0]
        type_Id = getClass(classname)
        seed = random.randint(0, 2 ** 32)
        # Apply transform to img
        random.seed(seed)
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, type_Id

    def __len__(self):
        return len(self.img_files)

def get_data_loaders(train_files, val_files, img_size=224):
    train_transform = Compose([
        #ColorJitter(0.3, 0.3, 0.3, 0.3),
        RandomResizedCrop(img_size, scale=(0.8, 1.2)),
        RandomAffine(10.),
        RandomRotation(13.),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transform = Compose([
        Resize((img_size, img_size)),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_loader = DataLoader(HairStyleDataset(train_files, train_transform),
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=4)
    val_loader = DataLoader(HairStyleDataset(val_files, val_transform),
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=4)

    return train_loader, val_loader

#def imshow1(img):
    #plt.imshow(img.numpy().transpose((1, 2, 0)) * 255)
    #plt.show()

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(3)  # pause a bit so that plots are updated
    plt.show()  # pause a bit so that plots are updated

if __name__ == '__main__':
    pass
'''
    for phase in ['train', 'val']:
        print(phase)

    image_files = get_img_files(data_dir)
    kf = KFold(n_splits=N_CV, random_state=RANDOM_STATE, shuffle=True)
    for n, (train_idx, val_idx) in enumerate(kf.split(image_files)):
        train_files = image_files[train_idx]
        val_files = image_files[val_idx]

        data_loaders = get_data_loaders(train_files, val_files, img_size)
        print('Train:', len(train_files), 'Val:', len(val_files))
# Get a batch of training data
        inputs, classes = next(iter(data_loaders[0]))
# Make a grid from batch
        out = torchvision.utils.make_grid(inputs)
        imshow(out, title=[x for x in classes]) 
        plt.show()
'''
