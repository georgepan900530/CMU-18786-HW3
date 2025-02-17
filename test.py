import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from models import MyResNet, Residual_Block
import random
import numpy as np

model = MyResNet(Residual_Block, [3, 3, 3, 3])
dummy_input = torch.randn(1, 3, 224, 224)

rand_classes = list(np.random.randint(0, 100, 10))
print(rand_classes)
