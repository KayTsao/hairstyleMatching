import logging
import sys
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse

from glob import glob
from PIL import Image
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from nets.MobileNetV2_unet import MobileNetV2_unet

from dataset import MaskTestSet, get_test_files

np.random.seed(1)
torch.backends.cudnn.deterministic = True
torch.manual_seed(1)

IMG_SIZE = 224
EXPERIMENT = 'train_unet'
OUT_DIR = 'outputs/{}'.format(EXPERIMENT)


def get_data_loaders(val_files):
    val_transform = Compose([
        Resize((IMG_SIZE, IMG_SIZE)),
        ToTensor(),
    ])
    val_loader = DataLoader(MaskTestSet(val_files, val_transform),
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=4)
    return val_loader


def evaluate(model, i_dir, o_dir):
    start = time.clock() #timer start
    image_files = get_test_files(i_dir)
    data_loader = get_data_loaders(image_files)
    elapsed = time.clock() - start #timer end
    print('load data cost:', elapsed, 's')
    with torch.no_grad():
        for inputs in data_loader:
            start = time.clock() #timer start
            inputs = inputs.to(device)
            outputs = model(inputs)
            i = inputs[0].cpu().numpy().transpose((1, 2, 0)) * 255
            o = outputs[0].cpu().numpy().reshape(int(IMG_SIZE / 2), int(IMG_SIZE / 2)) * 255  
            i = cv2.resize(i.astype(np.uint8), (IMG_SIZE, IMG_SIZE))
            o = cv2.resize(o.astype(np.uint8), (IMG_SIZE, IMG_SIZE))
            elapsed = time.clock() - start #timer end
            print('process img cost:', elapsed, 's')
            
            plt.subplot(121)
            plt.imshow(i)
            ax = plt.gca()
            ax.set_xticks(np.arange(0, 224, 10))
            ax.set_yticks(np.arange(0, 224, 10))
            plt.grid(c='g')
            plt.subplot(122)
            plt.imshow(o)
            plt.grid(c='g')
            plt.show()


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("i", help = "Input folder directory")
    parser.add_argument("o", help = "Output folder directory")    
    args = parser.parse_args()
    return args.i, args.o


if __name__ == '__main__':
    i_dir, o_dir = parse_command_line()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MobileNetV2_unet()
    model.load_state_dict(torch.load('{}/{}-best.pth'.format(OUT_DIR, 0)))
    model.to(device)
    model.eval()
    
    evaluate(model, i_dir, o_dir)
    
