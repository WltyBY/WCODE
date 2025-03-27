import torch
import torch.nn.functional as F

from torch import nn

from wcode.net.activate_function import ACTIVATE_LAYER, Swish


def GroupNormalize(in_channels, num_groups=32):
    return nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )


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
                    Activate_layer(inplace=True),
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
        conv_shortcut=False,
    ):
        super(ResidualBlock, self).__init__()
        Conv_layer, Norm_layer = module_generate(dim, normalization)
        Activate_layer = ACTIVATE_LAYER[activate.lower()]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = nn.Dropout(dropout_p)
        self.use_conv_shortcut = conv_shortcut

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_trans = Conv_layer(
                    self.in_channels,
                    self.out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            else:
                self.conv_trans = Conv_layer(
                    self.in_channels,
                    self.out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
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
                    (
                        Activate_layer(inplace=True)
                        if i != num_conv - 1
                        else nn.Identity()
                    ),
                    nn.Dropout(dropout_p) if i != num_conv - 1 else nn.Identity(),
                )
            )
        self.layers = nn.Sequential(*layer)
        self.activate = Activate_layer(inplace=True)

    def forward(self, inputs):
        outputs = self.layers(inputs)
        if self.in_channels != self.out_channels:
            inputs = self.conv_trans(inputs)

        return self.activate(inputs + outputs)


"""
Below are personalized customization.
"""
############################## Used in VQGAN implementation ####################################
class ResidualBlock_with_GroupNorm_Swish(nn.Module):
    """
    Although the written ResidualBlock above, I have no idea to combine all nn.Module(s) in the same way.
    So just write some personalized customization.
    """

    def __init__(
        self,
        *,
        in_channels,
        out_channels,
        dropout_p,
        dim=3,
        kernel_size=(3, 3, 3),
        padding_size=(1, 1, 1),
        conv_shortcut=False,
        temb_channels=512
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.dim = dim
        if self.dim == 2:
            Conv_layer = nn.Conv2d
        elif self.dim == 3:
            Conv_layer = nn.Conv3d

        self.norm1 = GroupNormalize(in_channels)
        self.conv1 = Conv_layer(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding_size
        )

        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)

        self.norm2 = GroupNormalize(out_channels)
        self.dropout = nn.Dropout(dropout_p)
        self.conv2 = Conv_layer(
            out_channels, out_channels, kernel_size=kernel_size, padding=padding_size
        )

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_trans = Conv_layer(
                    self.in_channels,
                    self.out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            else:
                self.conv_trans = Conv_layer(
                    self.in_channels,
                    self.out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = Swish(h)
        h = self.conv1(h)

        if temb is not None:
            if self.dim == 2:
                h = h + self.temb_proj(Swish(temb))[:, :, None, None]
            elif self.dim == 3:
                h = h + self.temb_proj(Swish(temb))[:, :, None, None, None]

        h = self.norm2(h)
        h = Swish(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int, dim: int):
        super().__init__()
        self.in_channels = in_channels

        if dim == 2:
            Conv_layer = nn.Conv2d
        elif dim == 3:
            Conv_layer = nn.Conv3d

        self.norm = GroupNormalize(in_channels)

        self.q = Conv_layer(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = Conv_layer(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = Conv_layer(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = Conv_layer(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        shp_q = q.shape
        q = q.reshape(*shp_q[:2], -1)  # b, c, (z)yx
        q = q.permute(0, 2, 1)  # b, (z)yx, c
        k = k.reshape(*shp_q[:2], -1)  # b, c, (z)yx
        w_ = torch.bmm(q, k)  # b, (z)yx, (z)yx    w[b,i,j]=sum_c q[b,i,c] k[b,c,j]
        w_ = w_ * (int(shp_q[1]) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(*shp_q[:2], -1)  # b, c, (z)yx
        w_ = w_.permute(0, 2, 1)  # b, (z)yx, (z)yx    (first hw of k, second of q)
        h_ = torch.bmm(
            v, w_
        )  # b, c, (z)yx (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(*shp_q)

        h_ = self.proj_out(h_)

        return x + h_

############################## Used in VQGAN implementation ####################################