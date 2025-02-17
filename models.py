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


# This Resnet implementation is based on my homework 3 in the course Machine Learning Spring 2022 at National Taiwan University (https://speech.ee.ntu.edu.tw/~hylee/ml/2022-spring.php)
class Residual_Block(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, down_sample=None):
        super(Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels=out_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.down_sample = down_sample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.down_sample is not None:
            residual = self.down_sample(x)
        out += residual
        out = self.relu(out)

        return out


class MyResNet(nn.Module):
    def __init__(self, block=Residual_Block, layers=[3, 4, 6, 3], num_classes=100):
        super(MyResNet, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.in_channels = 64
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def make_layer(self, block, out_channels, blocks, stride=1):
        down_sample = None
        if (stride != 1) or (self.in_channels != out_channels):
            down_sample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, down_sample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = self.dropout(out)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, out_dim=100, pretrained=False):
        super(ResNet18, self).__init__()
        self.resnet18 = models.resnet18(pretrained=pretrained)
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, out_dim)

    def forward(self, x):
        x = self.resnet18(x)
        return x


class ResNet50(nn.Module):
    def __init__(self, out_dim=100, pretrained=False):
        super(ResNet50, self).__init__()
        self.resnet50 = models.resnet50(pretrained=pretrained)
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, out_dim)

    def forward(self, x):
        x = self.resnet50(x)
        return x


class VGG16(nn.Module):
    def __init__(self, out_dim=100, pretrained=False):
        super(VGG16, self).__init__()
        self.vgg16 = models.vgg16_bn(pretrained=pretrained)
        self.vgg16.classifier[6] = nn.Linear(
            self.vgg16.classifier[6].in_features, out_dim
        )

    def forward(self, x):
        x = self.vgg16(x)
        return x


class EfficientNet(nn.Module):
    def __init__(self, out_dim=100, pretrained=False):
        super(EfficientNet, self).__init__()
        self.efficientnet = models.efficientnet_v2_m(pretrained=pretrained)
        self.efficientnet.classifier[1] = nn.Linear(
            self.efficientnet.classifier[1].in_features, out_dim
        )

    def forward(self, x):
        x = self.efficientnet(x)
        return x
