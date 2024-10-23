from torch import nn

from wcode.net.activate_function import ACTIVATE_LAYER


def module_generate(dim, normalization):
    if dim == 2:
        if normalization.lower() == "batchnorm":
            return nn.Conv2d, nn.BatchNorm2d
        elif normalization.lower() == "instancenorm":
            return nn.Conv2d, nn.InstanceNorm2d
        else:
            raise Exception()
    elif dim == 3:
        if normalization.lower() == "batchnorm":
            return nn.Conv3d, nn.BatchNorm3d
        elif normalization.lower() == "instancenorm":
            return nn.Conv3d, nn.InstanceNorm3d
        else:
            raise Exception()


######################### blocks for CNN #########################
class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dropout_p,
        dim=3,
        num_conv=2,
        kernel_size=(3, 3, 3),
        padding_size=(1, 1, 1),
        normalization="batchnorm",
        activate="leakyrelu",
        need_bias=True,
    ):
        super(ConvBlock, self).__init__()
        Conv_layer, Norm_layer = module_generate(dim, normalization)
        Activate_layer = ACTIVATE_LAYER[activate.lower()]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = nn.Dropout(dropout_p)

        layer = []
        for i in range(num_conv):
            layer.append(
                nn.Sequential(
                    Conv_layer(
                        self.in_channels if i == 0 else self.out_channels,
                        self.out_channels,
                        kernel_size=kernel_size,
                        padding=padding_size,
                        bias=need_bias,
                    ),
                    Norm_layer(self.out_channels, affine=True),
                    Activate_layer(),
                    nn.Dropout(dropout_p) if i != num_conv - 1 else nn.Identity(),
                )
            )
        self.layers = nn.Sequential(*layer)

    def forward(self, inputs):
        return self.layers(inputs)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dropout_p,
        dim=3,
        num_conv=2,
        kernel_size=(3, 3, 3),
        padding_size=(1, 1, 1),
        normalization="batchnorm",
        activate="leakyrelu",
        need_bias=True,
    ):
        super(ResidualBlock, self).__init__()
        Conv_layer, Norm_layer = module_generate(dim, normalization)
        Activate_layer = ACTIVATE_LAYER[activate.lower()]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = nn.Dropout(dropout_p)

        if self.in_channels != self.out_channels:
            self.conv_trans = Conv_layer(
                self.in_channels, self.out_channels, kernel_size=1, bias=need_bias
            )

        layer = []
        for i in range(num_conv):
            layer.append(
                nn.Sequential(
                    Conv_layer(
                        self.in_channels if i == 0 else self.out_channels,
                        self.out_channels,
                        kernel_size=kernel_size,
                        padding=padding_size,
                        bias=need_bias,
                    ),
                    Norm_layer(self.out_channels, affine=True),
                    Activate_layer() if i != num_conv - 1 else nn.Identity(),
                    nn.Dropout(dropout_p) if i != num_conv - 1 else nn.Identity(),
                )
            )
        self.layers = nn.Sequential(*layer)
        self.activate = Activate_layer()

    def forward(self, inputs):
        outputs = self.layers(inputs)
        if self.in_channels != self.out_channels:
            inputs = self.conv_trans(inputs)

        return self.activate(inputs + outputs)
