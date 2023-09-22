"""
This file contains functions for calculating gradient statistics.
IMPORTANT:
The implememtation is in the lines of "https://github.com/mosaic-group/inverse-dirichlet-pinn" 
"""
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad


class Layer(nn.Module):
    def __init__(self, in_features, out_features, seed, activation):
        super(Layer, self).__init__()
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        gain = 5/3 if isinstance(activation, nn.Tanh) else 1
        nn.init.xavier_normal_(self.linear.weight, gain=gain)
        nn.init.zeros_(self.linear.bias)
        self.linear = self.linear
    def forward(self, x):
        return self.linear(x)

class PINN(nn.Module):
    def __init__(self, sizes, mean=0, std=1, seed=0, activation=nn.Tanh()):
        super(PINN, self).__init__()
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.mean=mean
        self.std= std
        layer = []
        for i in range(len(sizes)-2):
            linear = Layer(sizes[i], sizes[i+1], seed, activation)
            layer += [linear, activation]
        layer += [Layer(sizes[-2], sizes[-1], seed, activation)]
        self.net = nn.Sequential(*layer)
    def forward(self, x):
        X = (x-self.mean)/self.std
        return self.net(X)


