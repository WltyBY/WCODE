import torch
import torch.nn as nn

from typing import Union, List, Tuple


class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        weight: Union[List, Tuple, torch.tensor] = [1, 1, 1],
        ignore_index: Union[None, List] = None,
        reduction: str = "mean",
        apply_nonlin: bool = True,
    ):
        super(CrossEntropyLoss, self).__init__()
        # ignore_index None or list
        self.ignore_index = ignore_index

        if not isinstance(weight, torch.Tensor):
            self.weight = torch.tensor(weight[:, None]).float()
        else:
            self.weight = weight[:, None].float()

        self.reduction = reduction
        self.apply_nonlin = apply_nonlin

    def forward(self, x, y, loss_mask=None):
        # loss_mask in b, 1, z, x, y
        shp_x, shp_y = x.shape, y.shape

        if self.apply_nonlin:
            x = torch.softmax(x, dim=1)

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(shp_x, shp_y)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                gt = y.long()
                y_onehot = torch.zeros(shp_x, device=x.device)
                y_onehot.scatter_(1, gt, 1)

        x = x.view(*shp_x[:2], -1)
        y_onehot = y_onehot.view(*shp_x[:2], -1)

        ce_loss = -self.weight * y_onehot * torch.log(x.clamp(min=1e-8))
        if self.ignore_index is not None:
            mask = y_onehot[:, self.ignore_index] != 1
            ce_loss = ce_loss * mask.float()
        else:
            mask = torch.ones(x.shape[0], 1, *x.shape[2:])

        # loss mask is the pixel weight
        ce_loss = (
            ce_loss if loss_mask is None else ce_loss * loss_mask.view(shp_x[0], 1, -1)
        )

        if self.reduction == "mean":
            return torch.sum(ce_loss) / torch.sum(mask)
        elif self.reduction == "sum":
            return torch.sum(ce_loss)
        elif self.reduction == "none":
            return torch.sum(ce_loss, dim=1).view(shp_x[0], *shp_x[2:])
        else:
            raise ValueError("Invalid reduction mode")

# can not use
# class FocalLoss(nn.Module):
#     def __init__(self, alpha=1, gamma=2, reduction="mean", apply_nonlin=True):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction
#         self.apply_nonlin = apply_nonlin

#     def forward(self, x, y, loss_mask=None):
#         shp_x, shp_y = x.shape, y.shape

#         with torch.no_grad():
#             if len(shp_x) != len(shp_y):
#                 y = y.view((shp_y[0], 1, *shp_y[1:]))

#             if all([i == j for i, j in zip(shp_x, shp_y)]):
#                 # if this is the case then gt is probably already a one hot encoding
#                 y_onehot = y
#             else:
#                 gt = y.long()
#                 y_onehot = torch.zeros(shp_x, device=x.device)
#                 y_onehot.scatter_(1, gt, 1)

#         if self.apply_nonlin:
#             prob = torch.softmax(x, dim=1)

#         focal_loss = -self.alpha * (1 - x) ** self.gamma * torch.log(x.clamp(min=1e-8))
#         focal_loss = focal_loss if loss_mask is None else focal_loss * loss_mask

#         if self.reduction == "mean":
#             return torch.mean(torch.sum(focal_loss, dim=1))
#         elif self.reduction == "sum":
#             return torch.sum(focal_loss)
#         elif self.reduction == "none":
#             return torch.sum(focal_loss, dim=1)
#         else:
#             raise ValueError("Invalid reduction mode")


if __name__ == "__main__":
    weight = torch.tensor([1, 1, 1]).float()
    x = torch.rand(2, 3, 128, 128, 128)
    y = torch.randint(0, 3, (2, 128, 128, 128))
    ce = nn.CrossEntropyLoss(weight, reduction="mean")
    our_ce = CrossEntropyLoss(weight, reduction="mean")
    print(ce(x, y), our_ce(x, y))
