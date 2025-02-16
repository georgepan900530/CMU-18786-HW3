import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


model = models.efficientnet_v2_m(pretrained=False)
dummy_input = torch.randn(1, 3, 224, 224)


print(model)
