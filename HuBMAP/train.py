import rasterio
import torchvision
import torch
import numpy as np
import pandas as pd
import numba, cv2, gc
import pathlib, sys, os, random, time
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A
from rasterio.windows import Window
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D

import torchvision
from torchvision import transforms as T

DATA_PATH = '/data/home/zhaohj/dev/dataset/kaggle-hubmap-kidney-segmentation'
fcn_pth_path = "input/pretrain-coco-weights-pytorch/fcn_resnet50_coco-1167a1af.pth"

EPOCHES = 5
BATCH_SIZE = 16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# used for converting the decoded image to rle mask
def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(256, 256)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')


@numba.njit()
def rle_numba(pixels):
    size = len(pixels)
    points = []
    if pixels[0] == 1: points.append(0)
    flag = True
    for i in range(1, size):
        if pixels[i] != pixels[i - 1]:
            if flag:
                points.append(i + 1)
                flag = False
            else:
                points.append(i + 1 - points[-1])
                flag = True
    if pixels[-1] == 1: points.append(size - points[-1] + 1)
    return points


def rle_numba_encode(image):
    pixels = image.flatten(order='F')
    points = rle_numba(pixels)
    return ' '.join(str(x) for x in points)


def make_grid(shape, window=256, min_overlap=32):
    """
        Return Array of size (N,4), where N - number of tiles,
        2nd axis represente slices: x1,x2,y1,y2
    """
    x, y = shape
    nx = x // (window - min_overlap) + 1
    x1 = np.linspace(0, x, num=nx, endpoint=False, dtype=np.int64)
    x1[-1] = x - window
    x2 = (x1 + window).clip(0, x)
    ny = y // (window - min_overlap) + 1
    y1 = np.linspace(0, y, num=ny, endpoint=False, dtype=np.int64)
    y1[-1] = y - window
    y2 = (y1 + window).clip(0, y)
    slices = np.zeros((nx, ny, 4), dtype=np.int64)

    for i in range(nx):
        for j in range(ny):
            slices[i, j] = x1[i], x2[i], y1[j], y2[j]
    return slices.reshape(nx * ny, 4)


identity = rasterio.Affine(1, 0, 0, 0, 1, 0)


class HubDataset(D.Dataset):

    def __init__(self, root_dir, transform,
                 window=256, overlap=32, threshold=100):
        self.path = pathlib.Path(root_dir)
        self.overlap = overlap
        self.window = window
        self.transform = transform
        self.csv = pd.read_csv((self.path / 'train.csv').as_posix(),
                               index_col=[0])
        self.threshold = threshold

        self.x, self.y = [], []
        self.build_slices()
        self.len = len(self.x)
        self.as_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize([0.625, 0.448, 0.688],
                        [0.131, 0.177, 0.101]),
        ])

    def build_slices(self):
        self.masks = []
        self.files = []
        self.slices = []
        for i, filename in enumerate(self.csv.index.values):
            filepath = (self.path / 'train' / (filename + '.tiff')).as_posix()
            self.files.append(filepath)

            print('Transform', filename)
            with rasterio.open(filepath, transform=identity) as dataset:
                self.masks.append(rle_decode(self.csv.loc[filename, 'encoding'], dataset.shape))
                slices = make_grid(dataset.shape, window=self.window, min_overlap=self.overlap)

                for slc in tqdm(slices):
                    x1, x2, y1, y2 = slc
                    if self.masks[-1][x1:x2, y1:y2].sum() > self.threshold or np.random.randint(100) > 120:
                        self.slices.append([i, x1, x2, y1, y2])

                        image = dataset.read([1, 2, 3],
                                             window=Window.from_slices((x1, x2), (y1, y2)))

                        #                         if image.std().mean() < 10:
                        #                             continue

                        # print(image.std().mean(), self.masks[-1][x1:x2,y1:y2].sum())
                        image = np.moveaxis(image, 0, -1)
                        self.x.append(image)
                        self.y.append(self.masks[-1][x1:x2, y1:y2])

    # get data operation
    def __getitem__(self, index):
        image, mask = self.x[index], self.y[index]
        augments = self.transform(image=image, mask=mask)
        return self.as_tensor(augments['image']), augments['mask'][None]

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len


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

_ = rle_numba_encode(mask[0])  # compile function with numba

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


def get_model():
    model = torchvision.models.segmentation.fcn_resnet50(False)

    pth = torch.load(fcn_pth_path)
    for key in ["aux_classifier.0.weight", "aux_classifier.1.weight", "aux_classifier.1.bias",
                "aux_classifier.1.running_mean", "aux_classifier.1.running_var", "aux_classifier.1.num_batches_tracked",
                "aux_classifier.4.weight", "aux_classifier.4.bias"]:
        del pth[key]

    model.classifier[4] = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
    return model


def get_unet_model():
    model = smp.Unet(
        encoder_name='efficientnet-b7',
        encoder_weights='imagenet',
        in_channels=3,
        classes=1
    )
    return model

def validation(model, loader, loss_fn):
    losses = []
    model.eval()
    for image, target in loader:
        image, target = image.to(DEVICE), target.float().to(DEVICE)
        output = model(image)
        loss = loss_fn(output, target)
        losses.append(loss.item())

    return np.array(losses).mean()


# !mkdir -p /root/.cache/torch/hub/checkpoints/
# !cp ../input/pytorch-pretrained-models/resnet50-19c8e357.pth /root/.cache/torch/hub/checkpoints/
# !cp ../input/pretrain-coco-weights-pytorch/fcn_resnet50_coco-1167a1af.pth /root/.cache/torch/hub/checkpoints/


model = get_unet_model()
# model = get_model()
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


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1., dims=(-2, -1)):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims

    def forward(self, x, y):
        tp = (x * y).sum(self.dims)
        fp = (x * (1 - y)).sum(self.dims)
        fn = ((1 - x) * y).sum(self.dims)

        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()

        return 1 - dc


# train

bce_fn = nn.BCEWithLogitsLoss()
dice_fn = SoftDiceLoss()


def loss_fn(y_pred, y_true):
    bce = bce_fn(y_pred, y_true)
    dice = dice_fn(y_pred.sigmoid(), y_true)
    return 0.8 * bce + 0.2 * dice


print(header)

best_loss = 10
EPOCHES = 20
for epoch in range(1, EPOCHES + 1):
    losses = []
    start_time = time.time()
    model.train()
    for image, target in loader:
        image, target = image.to(DEVICE), target.float().to(DEVICE)
        optimizer.zero_grad()
        # output = model(image)['out']
        output = model(image)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    vloss = validation(model, vloader, loss_fn)
    print(raw_line.format(epoch, np.array(losses).mean(), vloss,
                          (time.time() - start_time) / 60 ** 1))
    losses = []

    if vloss < best_loss:
        best_loss = vloss
        torch.save(model.state_dict(), 'model_best.pth')
del loader, vloader, train_ds, valid_ds, ds
gc.collect()
