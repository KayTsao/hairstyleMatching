import random
import re
from glob import glob
import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from config import IMG_DIR

import matplotlib.pyplot as plt
#-----------------------------Test-----------------------------
def _check_img(img_file):
    if os.path.splitext(img_file)[-1] == ".png":
        return img_file

def get_test_files(i_dir):
    files = sorted(glob('{}/*.png'.format(i_dir)))
    return np.array([_check_img(f) for f in files])


class MaskTestSet(Dataset):
    def __init__(self, img_files, transform, mask_transform=None):
        self.img_files = img_files
        self.transform = transform
        
    def __getitem__(self, idx):
        img = cv2.imread(self.img_files[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Apply transform to img
        img = Image.fromarray(img)
        img = self.transform(img)
        return img
    def __len__(self):
        return len(self.img_files)

#-----------------------------Train&Eval------------------------
def _mask_to_img(mask_file):
    img_file = re.sub('^{}/masks'.format(IMG_DIR), '{}/images'.format(IMG_DIR), mask_file)
    img_file = re.sub('\.jpg$', '.jpg', img_file)#img_file = re.sub('\.png$', '.jpg', img_file)
    return img_file


def _img_to_mask(img_file):
    mask_file = re.sub('^{}/images'.format(IMG_DIR), '{}/masks'.format(IMG_DIR), img_file)
    mask_file = re.sub('\.jpg$', '.jpg', mask_file)#mask_file = re.sub('\.jpg$', '.png', mask_file)
    return mask_file


def get_img_files():
    mask_files = sorted(glob('{}/masks/*.jpg'.format(IMG_DIR)))#mask_files = sorted(glob('{}/masks/*.png'.format(IMG_DIR)))
    return np.array([_mask_to_img(f) for f in mask_files])

#mask_axis=0[hair] mask_axis=1[face]
class MaskDataset(Dataset):
    def __init__(self, img_files, transform, mask_transform=None, mask_axis=1):
        self.img_files = img_files
        self.mask_files = [_img_to_mask(f) for f in img_files]
        self.transform = transform
        if mask_transform is None:
            self.mask_transform = transform
        else:
            self.mask_transform = mask_transform
        self.mask_axis = mask_axis

    def __getitem__(self, idx):
        img = cv2.imread(self.img_files[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_files[idx])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = mask[:, :, self.mask_axis]

        seed = random.randint(0, 2 ** 32)

        # Apply transform to img
        random.seed(seed)
        img = Image.fromarray(img)
        img = self.transform(img)

        # Apply same transform to mask
        random.seed(seed)
        mask = Image.fromarray(mask)
        mask = self.mask_transform(mask)

        return img, mask

    def __len__(self):
        return len(self.img_files)


if __name__ == '__main__':
    pass
'''
    mask = cv2.imread('{}/masks/Aaron_Peirsol_0001.jpg'.format(IMG_DIR))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    for i in range(mask.shape[0]) :
        for j in range(mask.shape[1]):
            if mask[i][j][1]<200:# or mask[i][j][1]!=255 or mask[i][j][2]!=255 :
                mask[i][j][0] = mask[i][j][1] =mask[i][j][2]=255

    #mask = mask[:, :, 1]
    print(mask.shape)
    plt.imshow(mask)
    plt.show()

'''
