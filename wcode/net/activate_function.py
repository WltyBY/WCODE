import torch

from torch import nn

def Swish(x):
    # swish
    return x * torch.sigmoid(x)

ACTIVATE_LAYER = {
    "leakyrelu": nn.LeakyReLU,
    "relu": nn.ReLU,
}