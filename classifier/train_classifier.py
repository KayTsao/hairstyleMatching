
from __future__ import print_function, division

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy

from dataset import get_data_loaders, get_img_files, imshow
from train import train_model, visualize_model
#from MobileNetV2 import MobileNetV2
from sklearn.model_selection import KFold

np.random.seed(1)
torch.backends.cudnn.deterministic = True
torch.manual_seed(1)

# %%
N_CV = 5
BATCH_SIZE = 8
LR = 1e-4

#N_EPOCHS = 100
N_EPOCHS = 35
IMG_SIZE = 224
RANDOM_STATE = 1

EXPERIMENT = 'train_unet'
OUT_DIR = 'outputs/{}'.format(EXPERIMENT)


data_dir = './Database/'
img_size = 224


plt.ion()   # interactive mode

#def run_cv(img_size, pre_trained, target):
if __name__ == '__main__':
    image_files = get_img_files(data_dir)
    kf = KFold(n_splits=N_CV, random_state=RANDOM_STATE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for n, (train_idx, val_idx) in enumerate(kf.split(image_files)):
        train_files = image_files[train_idx]
        val_files = image_files[val_idx]
        data_loaders = get_data_loaders(train_files, val_files, img_size)
        dataset_sizes = [len(train_files), len(val_files)]
        print('dataset_sizes:', dataset_sizes)
        inputs, classes = next(iter(data_loaders[0]))
        out = torchvision.utils.make_grid(inputs)
        imshow(out, title=[x for x in classes]) 

#-------------------------------------------------------------------------------
        model_ft = models.resnet18(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 9)
        model_ft = model_ft.to(device)
        criterion = nn.CrossEntropyLoss()
        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
        model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,N_EPOCHS, data_loaders, dataset_sizes, device)
        visualize_model(model_ft, 20, data_loaders, device)

        torch.save(model_ft.state_dict(), 'HairClassifier_resnet18.pth')
        break;
'''
        writer = SummaryWriter()

        def on_after_epoch(m, df_hist):
            save_best_model(n, m, df_hist, target)
            write_on_board(writer, df_hist)
            log_hist(df_hist)

        criterion = nn.CrossEntropyLoss()
        data_loaders = get_data_loaders(train_files, val_files, img_size)
        trainer = Trainer(data_loaders, criterion, device, on_after_epoch)

        model = MobileNetV2(pre_trained=True)
        num_ftrs = model_ft.fc.in_features
        model.to(device)
        optimizer = Adam(model.parameters(), lr=LR)

        hist = trainer.train(model, optimizer, num_epochs=N_EPOCHS)
        hist.to_csv('{}/{}-hist.csv'.format(OUT_DIR, n), index=False)

        writer.close()

        break


if __name__ == '__main__':
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)
    if not logger.hasHandlers():
        logger.addHandler(logging.FileHandler(filename="outputs/{}.log".format(EXPERIMENT)))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--img_size',
        type=int,
        default=224,
        help='image size',
    )
    parser.add_argument(
        '--pre_trained',
        type=str,
        help='path of pre trained weight',
    )
    parser.add_argument(
        '--target',
        type=str,
        help='training target [hair/face]',
    )
    args, _ = parser.parse_known_args()
    print(args)
    run_cv(**vars(args))
'''
