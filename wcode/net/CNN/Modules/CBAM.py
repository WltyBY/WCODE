import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16, dim=3):
        super(ChannelAttention, self).__init__()
        if dim == 3:
            ConvLayer = nn.Conv3d
            self.max_pool = nn.AdaptiveMaxPool3d(1)
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
        elif dim == 2:
            ConvLayer = nn.Conv2d
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        else:
            raise ValueError

        self.shared_MLP = nn.Sequential(
            ConvLayer(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(),
            ConvLayer(in_channels // ratio, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, dim=3):
        super(SpatialAttention, self).__init__()
        if dim == 3:
            ConvLayer = nn.Conv3d
        elif dim == 2:
            ConvLayer = nn.Conv2d
        else:
            raise ValueError

        self.conv = ConvLayer(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return out


class CBAM(nn.Module):
    def __init__(self, in_channels, ratio=8, kernel_size=5, dim=3):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, ratio, dim)
        self.spatial_attention = SpatialAttention(kernel_size, dim)

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out
