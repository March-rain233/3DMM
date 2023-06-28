import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

class FitModel(nn.Module):
    def __init__(self, shapeNum=40, expNum=10):
        super(FitModel, self).__init__()
        self.module = models.resnet50()
        self.module.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.module.fc = nn.Linear(self.module.fc.in_features, shapeNum + expNum + 12)  # [f:1 t:2 pi:9 sp:shapeNUm ep:expNum]

    def forward(self, x):
        x = self.module(x)
        return x