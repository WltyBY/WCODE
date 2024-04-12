from torch import nn

ACTIVATE_LAYER = {
    "leakyrelu": nn.LeakyReLU,
    "relu": nn.ReLU,
    "prelu": nn.PReLU    # When using this, weight decay should not be used 
}