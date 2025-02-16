import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


model = models.resnet18(pretrained=False)


print(model)
