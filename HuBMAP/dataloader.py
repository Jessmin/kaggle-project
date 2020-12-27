import torch.utils.data as D
import rasterio
import pathlib
from torchvision import transforms as T
import pandas as pd
from tqdm import tqdm
import numpy as np
import numba
from rasterio.windows import Window
import cv2
from PIL import Image
import os

identity = rasterio.Affine(1, 0, 0, 0, 1, 0)


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


def rle_encode_less_memory(img):
    pixels = img.T.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
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


idx = 0


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


class HubDataset(D.Dataset):

    def __init__(self, path, tiff_ids, transform,
                 window=256, overlap=32, threshold=100, isvalid=False):
        self.path = pathlib.Path(path)
        self.tiff_ids = tiff_ids
        self.overlap = overlap
        self.window = window
        self.transform = transform
        self.csv = pd.read_csv((self.path / 'train.csv').as_posix(),
                               index_col=[0])
        self.threshold = threshold
        self.isvalid = isvalid

        self.x, self.y, self.id = [], [], []
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
            if not filename in self.tiff_ids:
                continue

            filepath = (self.path / 'train' / (filename + '.tiff')).as_posix()
            self.files.append(filepath)

            # print('Transform', filename)
            with rasterio.open(filepath, transform=identity) as dataset:
                self.masks.append(rle_decode(self.csv.loc[filename, 'encoding'], dataset.shape))
                slices = make_grid(dataset.shape, window=self.window, min_overlap=self.overlap)
                idx = 0

                for slc in slices:
                    x1, x2, y1, y2 = slc
                    # print(slc)
                    image = dataset.read([1, 2, 3],
                                         window=Window.from_slices((x1, x2), (y1, y2)))
                    image = np.moveaxis(image, 0, -1)

                    image = cv2.resize(image, (256, 256))
                    masks = cv2.resize(self.masks[-1][x1:x2, y1:y2], (256, 256))
                    if self.isvalid:
                        self.slices.append([i, x1, x2, y1, y2])
                        self.x.append(image)
                        self.y.append(masks)
                        self.id.append(filename)
                    else:
                        image_dir = f'F:/Data/kaggle/kaggle-hubmap-kidney-segmentation/use/{filename}/image/{idx:05d}.png'
                        mask_dir = f'F:/Data/kaggle/kaggle-hubmap-kidney-segmentation/use/{filename}/mask/{idx:05d}.png'
                        if not os.path.exists(os.path.dirname(image_dir)):
                            os.makedirs(os.path.dirname(image_dir))
                        if not os.path.exists(os.path.dirname(mask_dir)):
                            os.makedirs(os.path.dirname(mask_dir))
                        cv2.imwrite(image_dir, image)
                        if masks.max() > 0:
                            masks = (255 * (masks - masks.min()) / (masks.max() - masks.min())).astype(np.uint8)
                        cv2.imwrite(mask_dir, masks)
                        image = image_dir
                        masks = mask_dir

                        if self.masks[-1][x1:x2, y1:y2].sum() >= self.threshold or (image > 32).mean() > 0.99:
                            self.slices.append([i, x1, x2, y1, y2])
                            self.x.append(image)
                            self.y.append(masks)
                            self.id.append(filename)
                    idx += 1

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
