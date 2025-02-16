import torch
import torch.nn as nn
import torch.nn.functional as F


a = torch.randn(3, 10, 10)
b = F.pad(a, (1, 1, 1, 1))
print(b.shape)
