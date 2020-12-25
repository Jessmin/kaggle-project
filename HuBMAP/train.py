import glob
import os.path as path
import sys

sys.path.append(path.join(path.dirname(__file__), '..'))

import numpy as np
import gc
import os, random, time
import albumentations as A
import torch.utils.data as D
from dataloader import HubDataset
from model import *
from loss import *
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold


def set_seeds(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


set_seeds()

writer = SummaryWriter(log_dir='logs', flush_secs=60)

DATA_PATH = '/home/zhaohoj/development_sshfs/dataset/kaggle-hubmap-kidney-segmentation/'
# DATA_PATH = 'F:/Data/kaggle/kaggle-hubmap-kidney-segmentation/'
pth_save_path = './pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHES = 5
BATCH_SIZE = 4
WINDOW = 1024
MIN_OVERLAP = 40
NEW_SIZE = 256


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
])

val_trfm = A.Compose([
    A.Resize(NEW_SIZE, NEW_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(),
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


if __name__ == '__main__':
    start_time_program = time.time()
    import platform

    if platform.system() == 'Linux':
        tiff_ids = np.array([x.split('/')[-1][:-5] for x in glob.glob(f'{DATA_PATH}train/*.tiff')])
    else:
        tiff_ids = np.array([x.split('\\')[-1][:-5] for x in glob.glob(f'{DATA_PATH}train/*.tiff')])

    skf = KFold(n_splits=4)
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

        # lr_step = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2)
        lr_step = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
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
            writer.add_scalar(f'Train_loss_{fold_idx}', train_loss, epoch)
            writer.add_scalar(f'Train_val_dice_{fold_idx}', val_dice, epoch)
            lr_step.step(val_dice)

            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(model.state_dict(), os.path.join(pth_save_path, 'fold_{0}.pth'.format(fold_idx)))
            print(raw_line.format(epoch, train_loss, val_dice, best_dice, (time.time() - start_time) / 60 ** 1))

        del train_loader, val_loader, train_ds, valid_ds
        gc.collect()
        break
    end_time_program = time.time()
    print(f'use-time:{end_time_program - start_time_program}')
