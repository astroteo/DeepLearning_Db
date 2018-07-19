from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from first_torch_net import Net

net = Net()
#recap net architectures
print(net)
#recap net trainable params
params = list(net.parameters())
print(len(params))
for param in params:
    print(param.size())
