import segmentation_models_pytorch as smp
import torchvision
import torch
from torch import nn


def get_unet_model():
    model = smp.Unet(
        encoder_name='efficientnet-b7',
        encoder_weights='imagenet',
        in_channels=3,
        classes=1
    )
    return model


def get_fcn_model():
    fcn_pth_path = "input/pretrain-coco-weights-pytorch/fcn_resnet50_coco-1167a1af.pth"
    model = torchvision.models.segmentation.fcn_resnet50(False)
    pth = torch.load(fcn_pth_path)
    for key in ["aux_classifier.0.weight", "aux_classifier.1.weight", "aux_classifier.1.bias",
                "aux_classifier.1.running_mean", "aux_classifier.1.running_var", "aux_classifier.1.num_batches_tracked",
                "aux_classifier.4.weight", "aux_classifier.4.bias"]:
        del pth[key]

    model.classifier[4] = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
    return model
