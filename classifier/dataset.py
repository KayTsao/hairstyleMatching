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
    Resize, RandomAffine, ColorJitter
from sklearn.model_selection import KFold

data_dir = './Database/'
N_CV = 5
BATCH_SIZE = 8
RANDOM_STATE = 1
img_size = 224

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

        seed = random.randint(0, 2 ** 32)
        # Apply transform to img
        random.seed(seed)
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, classname

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
    ])

    val_transform = Compose([
        Resize((img_size, img_size)),
        ToTensor(),
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

def imshow(img):
    plt.imshow(img.numpy().transpose((1, 2, 0)) * 255)
    plt.show()

if __name__ == '__main__':
    image_files = get_img_files(data_dir)
    kf = KFold(n_splits=N_CV, random_state=RANDOM_STATE, shuffle=True)
    for n, (train_idx, val_idx) in enumerate(kf.split(image_files)):
        train_files = image_files[train_idx]
        val_files = image_files[val_idx]

        data_loaders = get_data_loaders(train_files, val_files, img_size)
        print('Train:', len(train_files), 'Val:', len(val_files))
        dataiter = iter(data_loaders[1])  
        images, labels = dataiter.next()
        print(labels)
        imshow(torchvision.utils.make_grid(images,nrow=8))  
'''        

        for i_batch, sample_batches in enumerate(data_loaders[0]):
            print(i_batch, sample_batches[0].size(), len(sample_batches[1]))

        for inputs, labels in data_loaders[0]:
            print(inputs[0].size(), len(inputs))
'''
