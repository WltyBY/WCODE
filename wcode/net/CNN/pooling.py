import torch.nn.functional as F

from torch import nn

from wcode.net.CNN.baseblock_CNN import module_generate
from wcode.net.activate_function import ACTIVATE_LAYER


# downpooling blocks
class ConvDownPool(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        dim=3,
        pool_kernel_size=(2, 2, 2),
        normalization="batchnorm",
        activate="leakyrelu",
        need_bias=True,
    ):
        super(ConvDownPool, self).__init__()
        assert len(pool_kernel_size) == dim
        Conv_layer, Norm_layer = module_generate(dim, normalization)
        if out_channels is None:
            out_channels = in_channels * 2
        Activate_layer = ACTIVATE_LAYER[activate.lower()]

        self.pool = nn.Sequential(
            Conv_layer(
                in_channels,
                out_channels,
                kernel_size=pool_kernel_size,
                stride=pool_kernel_size,
                bias=need_bias,
            ),
            Norm_layer(out_channels, affine=True),
            Activate_layer(),
        )

    def forward(self, inputs):
        return self.pool(inputs)


# uppooling blocks
class ConvUpPool(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        dim=3,
        pool_kernel_size=(2, 2, 2),
        normalization="batchnorm",
        activate="leakyrelu",
        need_bias=True,
    ):
        super(ConvUpPool, self).__init__()
        assert len(pool_kernel_size) == dim
        _, Norm_layer = module_generate(dim, normalization)
        if out_channels is None:
            out_channels = in_channels // 2

        if dim == 2:
            ConvTrans_layer = nn.ConvTranspose2d
        elif dim == 3:
            ConvTrans_layer = nn.ConvTranspose3d
        Activate_layer = ACTIVATE_LAYER[activate.lower()]

        self.pool = nn.Sequential(
            ConvTrans_layer(
                in_channels,
                out_channels,
                kernel_size=pool_kernel_size,
                stride=pool_kernel_size,
                bias=need_bias,
            ),
            Norm_layer(out_channels, affine=True),
            Activate_layer(),
        )

    def forward(self, inputs):
        return self.pool(inputs)


"""
Below are personalized customization.
"""
############################## Used in VQGAN implementation ####################################
class Downsample_with_asymmetric_padding(nn.Module):
    """
    Only support /2
    """
    def __init__(self, in_channels, dim):
        super().__init__()
        if dim == 2:
            Conv_layer = nn.Conv2d
        elif dim == 3:
            Conv_layer = nn.Conv3d

        # no asymmetric padding in torch conv, must do it ourselves
        self.conv = Conv_layer(
            in_channels, in_channels, kernel_size=3, stride=2, padding=0
        )

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x

class Upsample_with_interpolation_and_Conv(nn.Module):
    """
    Only support *2
    """
    def __init__(self, in_channels, dim):
        super(Upsample_with_interpolation_and_Conv, self).__init__()
        if dim == 2:
            Conv_layer = nn.Conv2d
        elif dim == 3:
            Conv_layer = nn.Conv3d

        self.conv = Conv_layer(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x
############################## Used in VQGAN implementation ####################################
