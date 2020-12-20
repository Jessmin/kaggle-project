import glob
import os.path as path
import sys

sys.path.append(path.join(path.dirname(__file__), '..'))

import numpy as np
import numba, cv2, gc
import pathlib, sys, os, random, time
import matplotlib.pyplot as plt
import albumentations as A
import torch.utils.data as D
from dataloader import HubDataset
from model import *
from loss import *
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
import logging

writer = SummaryWriter()

DATA_PATH = 'F:/Data/kaggle/kaggle-hubmap-kidney-segmentation/'
pth_save_path = 'path/model_best.pth'
EPOCHES = 5
BATCH_SIZE = 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
WINDOW = 1024
MIN_OVERLAP = 40
NEW_SIZE = 256


# plt.figure(figsize=(16,8))
# plt.subplot(121)
# plt.imshow(mask[0], cmap='gray')
# plt.subplot(122)
# plt.imshow(image[0])

# _ = rle_numba_encode(mask[0])  # compile function with numba

def np_dice_score(probability, mask):
    p = probability.reshape(-1)
    t = mask.reshape(-1)

    p = p > 0.5
    t = t > 0.5
    uion = p.sum() + t.sum()

    overlap = (p * t).sum()
    dice = 2 * overlap / (uion + 0.001)
    return dice


def validation(model, val_loader, criterion):
    val_probability, val_mask = [], []
    model.eval()
    with torch.no_grad():
        for image, target in val_loader:
            image, target = image.to(DEVICE), target.float().to(DEVICE)
            output = model(image)

            output_ny = output.sigmoid().data.cpu().numpy()
            target_np = target.data.cpu().numpy()

            val_probability.append(output_ny)
            val_mask.append(target_np)

    val_probability = np.concatenate(val_probability)
    val_mask = np.concatenate(val_mask)

    return np_dice_score(val_probability, val_mask)


train_trfm = A.Compose([
    # A.RandomCrop(NEW_SIZE*3, NEW_SIZE*3),
    A.Resize(NEW_SIZE, NEW_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(),
    A.OneOf([
        A.RandomContrast(),
        A.RandomGamma(),
        A.RandomBrightness(),
        A.ColorJitter(brightness=0.07, contrast=0.07,
                      saturation=0.1, hue=0.1, always_apply=False, p=0.3),
    ], p=0.3),
    #     A.OneOf([
    #         A.OpticalDistortion(p=0.5),
    #         A.GridDistortion(p=0.5),
    #         A.IAAPiecewiseAffine(p=0.5),
    #     ], p=0.3),
    #     A.ShiftScaleRotate(),
])

val_trfm = A.Compose([
    # A.CenterCrop(NEW_SIZE, NEW_SIZE),
    A.Resize(NEW_SIZE, NEW_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(),
    #     A.OneOf([
    #         A.RandomContrast(),
    #         A.RandomGamma(),
    #         A.RandomBrightness(),
    #         A.ColorJitter(brightness=0.07, contrast=0.07,
    #                    saturation=0.1, hue=0.1, always_apply=False, p=0.3),
    #         ], p=0.3),
    #     A.OneOf([
    #         A.OpticalDistortion(p=0.5),
    #         A.GridDistortion(p=0.5),
    #         A.IAAPiecewiseAffine(p=0.5),
    #     ], p=0.3),
    #     A.ShiftScaleRotate(),
])


def train(model, train_loader, criterion, optimizer):
    losses = []
    for i, (image, target) in enumerate(train_loader):
        image, target = image.to(DEVICE), target.float().to(DEVICE)
        optimizer.zero_grad()

        output = model(image)
        loss = criterion(output, target, 1, False)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        # print('train, ', loss.item())
    return np.array(losses).mean()


tiff_ids = np.array([x.split('\\')[-1][:-5] for x in glob.glob(f'{DATA_PATH}train/*.tiff')])
# tiff_ids = np.array([x.split('/')[-1][:-5] for x in glob.glob(f'{DATA_PATH}train/*.tiff')])
skf = KFold(n_splits=8)
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(tiff_ids, tiff_ids)):
    print(tiff_ids[val_idx])
    # break
    train_ds = HubDataset(DATA_PATH, tiff_ids[train_idx], window=WINDOW, overlap=MIN_OVERLAP,
                          threshold=100, transform=train_trfm)
    valid_ds = HubDataset(DATA_PATH, tiff_ids[val_idx], window=WINDOW, overlap=MIN_OVERLAP,
                          threshold=100, transform=val_trfm, isvalid=False)
    print(len(train_ds), len(valid_ds))
    # define training and validation data loaders
    train_loader = D.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=12)

    val_loader = D.DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=12)

    model = get_unet_model()
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    lr_step = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2)
    # lr_step = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    header = r'''
            Train | Valid
    Epoch |  Loss |  Loss | Time, m
    '''
    #          Epoch         metrics            time
    raw_line = '{:6d}' + '\u2502{:7.3f}' * 2 + '\u2502{:6.2f}'

    best_dice = 0
    for epoch in range(1, EPOCHES + 1):
        start_time = time.time()
        model.train()
        train_loss = train(model, train_loader, loss_fn, optimizer)
        val_dice = validation(model, val_loader, loss_fn)
        lr_step.step(val_dice)

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), 'fold_{0}.pth'.format(fold_idx))
        logging.info(raw_line.format(epoch, train_loss, val_dice, best_dice, (time.time() - start_time) / 60 ** 1))

    del train_loader, val_loader, train_ds, valid_ds
    gc.collect()
    break
