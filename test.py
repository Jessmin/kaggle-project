import torch
import cv2
import rasterio
import numpy as np
from dataloader import make_grid, identity, Window, rle_numba_encode
import gc
import pathlib
import pandas as pd
import glob
import os
from tqdm import tqdm
from torchvision import transforms as T
import segmentation_models_pytorch as smp

DATA_PATH = '/home/zhaohoj/development_sshfs/dataset/kaggle-hubmap-kidney-segmentation/'
# DATA_PATH = '/data/home/zhaohj/dev/dataset/kaggle-hubmap-kidney-segmentation/'
pth_path = '/home/zhaohoj/development_sshfs/dataset/pth/pth'
# pth_path = '/data/home/zhaohj/dev/dataset/pth/pth'
WINDOW = 1024
MIN_OVERLAP = 40
def get_unet_model():
    model = smp.Unet(
        encoder_name='efficientnet-b7',
        encoder_weights='imagenet',
        in_channels=3,
        classes=1
    )
    return model

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"'Total': {total_num}, 'Trainable': {trainable_num}")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model_filepaths = [os.path.join(pth_path,filename) for filename in os.listdir(pth_path)]

p = pathlib.Path(DATA_PATH)
NEW_SIZE = 256
fold_models = []


test_fns = glob.glob(os.path.join(DATA_PATH, 'test/*.tiff'))
subm = {}
THRESHOLD = 0.45
trfm = T.Compose([
    T.ToPILImage(),
    T.Resize(NEW_SIZE),
    T.ToTensor(),
    T.Normalize([0.625, 0.448, 0.688],
                [0.131, 0.177, 0.101]),
])
for i, filename in tqdm(enumerate(test_fns), total = len(test_fns)):
    print(f"{i + 1} Predicting {os.path.basename(filename).split('.')[0]}")
    dataset = rasterio.open(filename, transform=identity)
    slices = make_grid(dataset.shape, window=WINDOW, min_overlap=MIN_OVERLAP)
    preds = np.zeros(dataset.shape, dtype=np.uint8)

    for (x1, x2, y1, y2) in slices:
        image = dataset.read([1, 2, 3],
                             window=Window.from_slices((x1, x2), (y1, y2)))
        image = np.moveaxis(image, 0, -1)
        image = cv2.resize(image, (NEW_SIZE, NEW_SIZE))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = trfm(image)
        image = image.unsqueeze(0)

        pred = None
        for fold_model_path in model_filepaths:
            model = get_unet_model()
            model.to(DEVICE)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(torch.load(fold_model_path))
            model.eval()
            image = image.to(DEVICE)
            with torch.no_grad():
                score = model(image).squeeze()
            if pred is None:
                pred = score
            else:
                pred += score
        pred = pred / len(fold_models)
        pred = pred.sigmoid().cpu().numpy()
        pred = cv2.resize(pred, (WINDOW, WINDOW))
        preds[x1:x2, y1:y2] = (pred > THRESHOLD).astype(np.uint8)
    subm[i] = {'id': os.path.basename(filename).split('.')[0], 'predicted': rle_numba_encode(preds)}
    del preds
    gc.collect()

submission = pd.DataFrame.from_dict(subm, orient='index')
submission.to_csv('submission.csv', index=False)