import torch
import pathlib
import rasterio
from rasterio.windows import Window
import cv2
import gc
import pandas as pd
import numpy as np
import torchvision
from torch import nn
import albumentations as A
import numba
from torchvision import transforms as T
import segmentation_models_pytorch as smp

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_PATH = '../input/hubmap-kidney-segmentation'
# DATA_PATH = '/data/home/zhaohj/dev/dataset/kaggle-hubmap-kidney-segmentation'
identity = rasterio.Affine(1, 0, 0, 0, 1, 0)
WINDOW = 1024
MIN_OVERLAP = 32
NEW_SIZE = 256


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


trfm = T.Compose([
    T.ToPILImage(),
    T.Resize(NEW_SIZE),
    T.ToTensor(),
    T.Normalize([0.625, 0.448, 0.688],
                [0.131, 0.177, 0.101]),
])


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


def get_model():
    model = torchvision.models.segmentation.fcn_resnet50(False)

    pth = torch.load("../input/pretrain-coco-weights-pytorch/fcn_resnet50_coco-1167a1af.pth")
    for key in ["aux_classifier.0.weight", "aux_classifier.1.weight", "aux_classifier.1.bias",
                "aux_classifier.1.running_mean", "aux_classifier.1.running_var", "aux_classifier.1.num_batches_tracked",
                "aux_classifier.4.weight", "aux_classifier.4.bias"]:
        del pth[key]
    model.classifier[4] = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
    return model


def get_unet_model():
    model = smp.Unet(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        in_channels=3,
        classes=1
    )
    return model


model = get_unet_model()
model.to(DEVICE)
model = nn.DataParallel(model)
model.load_state_dict(torch.load('../input/unetbest/model_best_unet.pth'))
model.eval()

p = pathlib.Path(DATA_PATH)
subm = {}
print('start')
for i, filename in enumerate(p.glob('test/*.tiff')):
    dataset = rasterio.open(filename.as_posix(), transform=identity)
    slices = make_grid(dataset.shape, window=WINDOW, min_overlap=MIN_OVERLAP)
    preds = np.zeros(dataset.shape, dtype=np.uint8)

    for (x1, x2, y1, y2) in slices:
        image = dataset.read([1, 2, 3],
                             window=Window.from_slices((x1, x2), (y1, y2)))
        image = np.moveaxis(image, 0, -1)
        image = trfm(image)
        with torch.no_grad():
            image = image.to(DEVICE)[None]
            score = model(image)[0][0]

            score2 = model(torch.flip(image, [0, 3]))
            score2 = torch.flip(score2, [3, 0])[0][0]

            score3 = model(torch.flip(image, [1, 2]))
            score3 = torch.flip(score3, [2, 1])[0][0]

            score_mean = (score + score2 + score3) / 3.0
            score_sigmoid = score_mean.sigmoid().cpu().numpy()
            score_sigmoid = cv2.resize(score_sigmoid, (WINDOW, WINDOW))

            preds[x1:x2, y1:y2] = (score_sigmoid > 0.5).astype(np.uint8)
    print(f'Done : {i}')
    subm[i] = {'id': filename.stem, 'predicted': rle_numba_encode(preds)}
    del preds
    gc.collect()

submission = pd.DataFrame.from_dict(subm, orient='index')
submission.to_csv('submission.csv', index=False)

