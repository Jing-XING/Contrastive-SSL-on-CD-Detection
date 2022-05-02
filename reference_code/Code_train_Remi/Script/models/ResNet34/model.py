from torchvision.models import resnet34, resnet101, resnet152, vgg16_bn, vgg19_bn
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def Model(nb_classes):
    # Pretrained Resnet 34
    base_resnet101 = resnet34(pretrained=True)
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=1000)
    model.load_state_dict(base_resnet101.state_dict())
    model.fc = nn.Linear(512, nb_classes)

    return model
