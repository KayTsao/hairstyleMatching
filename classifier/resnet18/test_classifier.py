from __future__ import print_function, division

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import models
import time
import os
from PIL import Image
from dataset import get_testdata_loader, get_img_files, imshow
plt.ion()   # interactive mode
#np.random.seed(1)
#torch.backends.cudnn.deterministic = True
#torch.manual_seed(1)
#class_names=['长－卷－无刘海', '长－卷－有刘海', '长－直－无刘海', '长－直－有刘海', '中－卷－无刘海', '中－卷－有刘海', '中－直－无刘海', '中－直－有刘海', '马尾束发']

#HairStyle=['Long-Curly-NoBangs', 'Long-Curly-Bangs', 'Long-Straight-NoBangs', 'Long-Straight-Bangs', 'Short-Curly-NoBangs', 'Short-Curly-Bangs', 'Short-Straight-NoBangs', 'Short-Straight-Bangs', 'Ponytail']
HairStyle=['l1','l2', 'l3','l4', 'l5', 'l6', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'p1', 'p2', 'p3']

num_classes = 21

def HairstyleParsing(model, i_dir, o_dir):
    test_files = get_img_files(i_dir)
    data_loader = get_testdata_loader(test_files)
    dataset_size = len(test_files)

    inputs = next(iter(data_loader))
    
    idx = 0
    fig = plt.figure()
    with torch.no_grad():
        for inputs in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for j in range(inputs.size()[0]):
                print(HairStyle[preds[j]])                
                """Imshow for Tensor."""
                inp = inputs[j].cpu().numpy().transpose((1, 2, 0))
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                inp = std * inp + mean
                inp = np.clip(inp, 0, 1)
                title = HairStyle[preds[j]]
                model_name = '../hairModel/' + title + '.png'
                model = Image.open(model_name)
                plt.title(title)
                plt.subplot(121)
                plt.imshow(inp)
                plt.subplot(122)
                plt.imshow(model)
                plt.pause(3)  # pause a bit so that plots are updated
                plt.show()  # pause a bit so that plots are updated
                fname = o_dir+ str(j+ (idx*inputs.size()[0])) + '.png'
                fig.savefig(fname)
            idx+=1
    
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load('HairClassifier_resnet18_21classes_RGB.pth'))
    model.to(device)
    model.eval()
    i_dir = '../T/'
    o_dir = '../result/'
    HairstyleParsing(model, i_dir, o_dir)
