import os.path as path
import sys

sys.path.append(path.join(path.dirname(__file__), '..'))

import torchvision
import torch
import numpy as np
import pandas as pd
import numba, cv2, gc
import pathlib, sys, os, random, time
import matplotlib.pyplot as plt
import albumentations as A
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D
from dataloader import HubDataset
from model import *
from loss import *
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

DATA_PATH = '/data/home/zhaohj/dev/dataset/kaggle-hubmap-kidney-segmentation'
pth_save_path = 'path/model_best.pth'
EPOCHES = 5
BATCH_SIZE = 16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
WINDOW = 1024
MIN_OVERLAP = 32
NEW_SIZE = 256

trfm = A.Compose([
    A.Resize(NEW_SIZE, NEW_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),

    A.OneOf([
        A.RandomContrast(),
        A.RandomGamma(),
        A.RandomBrightness(),
        A.ColorJitter(brightness=0.07, contrast=0.07,
                      saturation=0.1, hue=0.1, always_apply=False, p=0.3),
    ], p=0.3),
    A.OneOf([
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        A.GridDistortion(),
        A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
    ], p=0.0),
    A.ShiftScaleRotate(),
])

ds = HubDataset(DATA_PATH, window=WINDOW, overlap=MIN_OVERLAP, transform=trfm)

image, mask = ds[2]
# plt.figure(figsize=(16,8))
# plt.subplot(121)
# plt.imshow(mask[0], cmap='gray')
# plt.subplot(122)
# plt.imshow(image[0])

# _ = rle_numba_encode(mask[0])  # compile function with numba

valid_idx, train_idx = [], []
for i in range(len(ds)):
    if ds.slices[i][0] == 7:
        valid_idx.append(i)
    else:
        train_idx.append(i)

train_ds = D.Subset(ds, train_idx)
valid_ds = D.Subset(ds, valid_idx)

# define training and validation data loaders
loader = D.DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

vloader = D.DataLoader(
    valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


def validation(model: torch.nn.Module, loader, loss_fn):
    losses = []
    model.eval()
    for image, target in loader:
        image, target = image.to(DEVICE), target.float().to(DEVICE)
        output = model(image)
        loss = loss_fn(output, target)
        losses.append(loss.item())

    return np.array(losses).mean()


model = get_unet_model()
model.to(DEVICE)
model = nn.DataParallel(model)
optimizer = torch.optim.AdamW(model.parameters(),
                              lr=1e-4, weight_decay=1e-3)
header = r'''
        Train | Valid
Epoch |  Loss |  Loss | Time, m
'''
#          Epoch         metrics            time
raw_line = '{:6d}' + '\u2502{:7.3f}' * 2 + '\u2502{:6.2f}'

# train

best_loss = 10
EPOCHES = 20
for epoch in range(1, EPOCHES + 1):
    losses = []
    start_time = time.time()
    model.train()
    for image, target in loader:
        image, target = image.to(DEVICE), target.float().to(DEVICE)
        optimizer.zero_grad()
        output = model(image)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    vloss = validation(model, vloader, loss_fn)
    writer.add_scalar('Loss/train', np.array(losses).mean(), epoch)
    writer.add_scalar('Loss/test', vloss, epoch)
    print(raw_line.format(epoch, np.array(losses).mean(), vloss,
                          (time.time() - start_time) / 60 ** 1))
    if vloss < best_loss:
        best_loss = vloss
        torch.save(model.state_dict(), pth_save_path)
del loader, vloader, train_ds, valid_ds, ds
gc.collect()
