from torch import nn

from wcode.net.baseblock import module_generate
from wcode.net.activate_function import ACTIVATE_LAYER


# downpooling blocks
class ConvDownPool(nn.Module):
    def __init__(
        self,
        in_channels,
        dim=3,
        pool_kernel_size=(2, 2, 2),
        normalization="batchnorm",
        activate="leakyrelu",
    ):
        super(ConvDownPool, self).__init__()
        Conv_layer, Norm_layer = module_generate(dim, normalization)
        Activate_layer = ACTIVATE_LAYER[activate.lower()]

        self.pool = nn.Sequential(
            Conv_layer(
                in_channels,
                in_channels * 2,
                kernel_size=pool_kernel_size,
                stride=pool_kernel_size,
                bias=False
            ),
            Norm_layer(in_channels * 2, affine=True),
            Activate_layer(),
        )

    def forward(self, inputs):
        return self.pool(inputs)


# uppooling blocks
class ConvUpPool(nn.Module):
    def __init__(
        self,
        in_channels,
        dim=3,
        pool_kernel_size=(2, 2, 2),
        normalization="batchnorm",
        activate="leakyrelu",
    ):
        super(ConvUpPool, self).__init__()
        _, Norm_layer = module_generate(dim, normalization)
        if dim == 2:
            ConvTrans_layer = nn.ConvTranspose2d
        elif dim == 3:
            ConvTrans_layer = nn.ConvTranspose3d
        Activate_layer = ACTIVATE_LAYER[activate.lower()]

        self.pool = nn.Sequential(
            ConvTrans_layer(
                in_channels,
                in_channels // 2,
                kernel_size=pool_kernel_size,
                stride=pool_kernel_size,
                bias=False
            ),
            Norm_layer(in_channels // 2, affine=True),
            Activate_layer(),
        )

    def forward(self, inputs):
        return self.pool(inputs)
