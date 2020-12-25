from torch import nn
from ASPP import ASPP
import segmentation_models_pytorch as smp


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.down = True
        self.startconv = nn.Conv2d(1, 3, kernel_size=1)
        self.basemodel = smp.Unet(
            encoder_name='efficientnet-b0',
            encoder_weights='imagenet',
            in_channels=3,
            classes=1
        )
        self.planes = [32, 48, 136, 384]
        self.down = False
        self.center = ASPP(self.planes[3], self.planes[2])

    def forward(self, x):
        pass
