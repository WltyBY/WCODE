from torch import nn

ACTIVATE_LAYER = {
    "leakyrelu": nn.LeakyReLU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
}