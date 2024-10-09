import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, List


class LoRAModule(nn.Module):
    def __init__(self, layer, rank):
        super(LoRAModule, self).__init__()
        self.layer = layer
        self.rank = rank

        if self.rank > 0:
            # layer can be nn.Linear nn.Conv1d, nn.Conv2d and nn.Conv3d
            self.layer: Union[nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d]
            if isinstance(layer, nn.Linear):
                self.lora_A = nn.Parameter(torch.zeros(layer.in_features, rank))
                self.lora_B = nn.Parameter(torch.zeros(layer.out_features, rank))
            elif isinstance(layer, nn.Conv1d, nn.Conv2d, nn.Conv3d):
                self.lora_A = nn.Parameter(
                    self.layer.weight.new_zeros(
                        (
                            self.rank * self.layer.kernel_size,
                            self.layer.in_channels * self.layer.kernel_size,
                        )
                    )
                )
                self.lora_B = nn.Parameter(
                    self.layer.weight.new_zeros(
                        self.layer.out_channels
                        // self.layer.groups
                        * self.layer.kernel_size,
                        self.rank * self.layer.kernel_size,
                    )
                )
            self.layer.weight.requires_grad = False

        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        self.layer.reset_parameters()
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # 应用低秩矩阵变换
        x = torch.matmul(x, self.U).view(x.shape[0], -1)
        x = self.layer(x)
        x = torch.matmul(x, self.V.view(self.V.size(1), -1))
        return x


class LoRAlinear(nn.Module):
    def __init__(
        self, in_features, out_features, merge, rank=16, lora_alpha=16, dropout=0.5
    ):
        super(LoRAlinear, self).__init__()
        self.in_feature = in_features
        self.out_feature = out_features
        self.merge = merge
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.dropout_rate = dropout

        self.linear = nn.Linear(in_features, out_features)
        if rank > 0:
            self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
            self.scale = self.lora_alpha / self.rank
            self.linear.weight.requires_grad = False

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)
        else:
            self.dropout = nn.Identity()

        self.weight_init()

    def forward(self, x):
        if self.rank and self.merge:
            output = F.linear(
                x,
                self.linear.weight + self.lora_B @ self.lora_A * self.scale,
                self.linear.bias,
            )
            output = self.dropout(output)
            return output
        else:
            return self.dropout(self.linear(x))

    def weight_init(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
