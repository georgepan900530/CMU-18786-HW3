import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mytorch import MyConv2D, MyMaxPool2D
from torchvision import models


class FCNN(nn.Module):
    """
    Fully connected neural network for CIFAR100 classification

    Parameters
    -----
    in_dim: int
        The dimension of the input data. In the case of CIFAR100, this is 3*32*32=3072 (unless we resize the images)
    out_dim: int
        The dimension of the output data. In the case of CIFAR100, this is 100 (number of classes)

    Returns
    -----
    torch.Tensor
        The output of the network. This is a tensor of size (batch_size, out_dim)
    """

    def __init__(self, in_dim=3072, out_dim=100):
        super(FCNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, out_dim),
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


class CNN(nn.Module):
    """
    Convolutional neural network for CIFAR100 classification

    Parameters
    -----
    in_channels: int
        The number of channels in the input data. In the case of CIFAR100, this is 3 in default
        Note that the input size is b x 3 x 32 x 32 in default

    out_dim: int
        The dimension of the output data. In the case of CIFAR100, this is 100 in default (number of classes)

    Returns
    -----
    torch.Tensor
        The output of the network. This is a tensor of size (batch_size, out_dim)
    """

    def __init__(self, in_channels=3, out_dim=100, in_size=32):
        super(CNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),  # 3 x 32 x 32 -> 64 x 32 x 32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=None),  # 64 x 32 x 32 -> 64 x 16 x 16
            nn.Conv2d(64, 128, 3, 1, 1),  # 64 x 16 x 16 -> 128 x 16 x 16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=None),  # 128 x 16 x 16 -> 128 x 8 x 8
            nn.Conv2d(128, 64, 3, 1, 1),  # 128 x 8 x 8 -> 64 x 8 x 8
        )
        self.fc = nn.Linear(64 * (in_size // 4) ** 2, out_dim)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


class ResNet18(nn.Module):
    def __init__(self, out_dim=100):
        super(ResNet18, self).__init__()
        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, out_dim)
        # Initialize the model with Xavier initialization
        for m in self.resnet18.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.resnet18(x)
        return x


class ResNet50(nn.Module):
    def __init__(self, out_dim=100):
        super(ResNet50, self).__init__()
        self.resnet50 = models.resnet50(pretrained=False)
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, out_dim)
        # Initialize the model with Xavier initialization
        for m in self.resnet50.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.resnet50(x)
        return x


class VGG16(nn.Module):
    def __init__(self, out_dim=100):
        super(VGG16, self).__init__()
        self.vgg16 = models.vgg16_bn(pretrained=False)
        self.vgg16.classifier[6] = nn.Linear(
            self.vgg16.classifier[6].in_features, out_dim
        )
        # Initialize the model with Xavier initialization
        for m in self.vgg16.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.vgg16(x)
        return x


class EfficientNet(nn.Module):
    def __init__(self, out_dim=100):
        super(EfficientNet, self).__init__()
        self.efficientnet = models.efficientnet_v2_m(pretrained=False)
        self.efficientnet.classifier[1] = nn.Linear(
            self.efficientnet.classifier[1].in_features, out_dim
        )
        # Initialize the model with Xavier initialization
        for m in self.efficientnet.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.efficientnet(x)
        return x
