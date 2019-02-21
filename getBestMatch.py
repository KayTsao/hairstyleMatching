import logging
import sys
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
import os
from glob import glob
from PIL import Image
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

from SemSeg.nets.MobileNetV2_unet import MobileNetV2_unet
from SemSeg.dataset import MaskTestSet, get_test_files

from StyleMatching.utils import Point, GenLine
from StyleMatching.LandmarkDetector import KPDtc

np.random.seed(1)
torch.backends.cudnn.deterministic = True
torch.manual_seed(1)

IMG_SIZE = 224
EXPERIMENT = 'train_unet'
OUT_DIR = 'SemSeg/outputs/{}'.format(EXPERIMENT)


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


def ImgSegmentation(model, i_dir, o_dir):
    image_files = get_test_files(i_dir)
    data_loader = get_data_loaders(image_files)
    idx = 0
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
            fname = image_files[idx]
            filename = fname.split("/")[-1]
            filename = o_dir + filename
            #plt.imshow(o)
            #plt.show()
            cv2.imwrite(filename, o)#单通道输出
            print('o:', o.shape)
            idx+=1

def ImgRegionAlignment( i_dir, o_dir):
    for fname in os.listdir(i_dir):
        i_img = os.path.join(i_dir, fname) 
        i_msk = os.path.join(o_dir, fname) 
        
        p_eye1 = Point(1,0)
        p_eye2 = Point(2,0)
        p_nose = Point(3,0)
        p_chin = Point(4,0)
        p_eye1, p_eye2, p_nose, p_chin = KPDtc(i_img)
        #print('eye[', p_eye1.x, p_eye1.y,'], [', p_eye2.x, p_eye2.y, ']; Nose[', p_nose.x, p_nose.y,']; Chin[', p_chin.x, p_chin.y, ']') 
        #基准区域划分
        l1=GenLine(0,0)
        l2=GenLine(0,0)
        l3=GenLine(0,0)
        l1.frm2P(p_eye1, p_eye2)
        l2.frmKP(l1.k, p_nose)
        l3.frmKP(l1.k, p_chin)
        
        img = Image.open(i_img)
        msk = Image.open(i_msk)
        msk = msk.convert("RGBA")
        msk = msk.resize((img.size), Image.ANTIALIAS)
        #i = 1
        #j = 1
        region1 = 0
        region2 = 0
        region3 = 0
        region4 = 0
        for i in range (1, msk.size[0]):
            for j in range(1, msk.size[1]):
                msk_data = (msk.getpixel((i,j)))
                img_data = (img.getpixel((i,j)))
                if(j < (l1.k * i + l1.b)):
                    img.putpixel((i,j),(img_data[0]+255, img_data[1]+0, img_data[2]+0, img_data[3]+25))
                    if(msk_data[0]>125):
                        region1 += 1   
                elif(j < (l2.k * i + l2.b)):
                    img.putpixel((i,j),(img_data[0]+0, img_data[1]+255, img_data[2]+0, img_data[3]+25))
                    if(msk_data[0]>125):
                        region2 += 1   
                elif(j < (l3.k * i + l3.b)):
                    img.putpixel((i,j),(img_data[0]+0, img_data[1]+0, img_data[2]+255, img_data[3]+25))
                    if(msk_data[0]>125):
                        region3 += 1   
                else:
                    img.putpixel((i,j),(img_data[0]+255, img_data[1]+255, img_data[2]+0, img_data[3]+25))
                    if(msk_data[0]>125):
                        region4 += 1   
        plt.imshow(img)
        plt.show()
        
        Hair_Area = region1 + region2 + region3 + region4
        print('RATIO:', region1/Hair_Area, region2/Hair_Area, region3/Hair_Area, region4/Hair_Area)



def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("i", help = "Input folder directory")
    parser.add_argument("o", help = "Output folder directory") 
    parser.add_argument("t", help = "segmentation target [hair/face]")    
    args = parser.parse_args()
    return args.i, args.o, args.t


if __name__ == '__main__':
    i_dir, o_dir, model_type = parse_command_line()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MobileNetV2_unet()
    model.load_state_dict(torch.load('{}/{}-best-{}.pth'.format(OUT_DIR, 0, model_type)))
    model.to(device)
    model.eval()
    
    ImgSegmentation(model, i_dir, o_dir)
    ImgRegionAlignment(i_dir, o_dir)
    
