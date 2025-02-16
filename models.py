import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mytorch import MyConv2D, MyMaxPool2D


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

    def __init__(self, in_channels=3, out_dim=100):
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
        self.fc = nn.Linear(64 * 8 * 8, out_dim)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
