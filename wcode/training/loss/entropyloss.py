import torch
import numpy as np
from torch import nn, Tensor


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension

    input must be logits, not probabilities!
    """

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]

        return super().forward(input, target.long())


class EntropyMinimizeLoss(nn.Module):
    def __init__(
        self,
        reduction: str = "mean",
        apply_nonlin: bool = True,
    ):
        super(EntropyMinimizeLoss, self).__init__()

        self.reduction = reduction
        self.apply_nonlin = apply_nonlin

    def forward(self, x):
        # loss_mask in b, 1, z, x, y
        # x in b, c, z, y, x
        shp_x = x.shape

        if self.apply_nonlin:
            x = torch.softmax(x, dim=1)

        # b, c, (z * y * x)
        x = x.view(*shp_x[:2], -1)

        em_loss = -x * torch.log(x.clamp(min=1e-8))

        if self.reduction == "mean":
            return torch.mean(em_loss)
        elif self.reduction == "sum":
            return torch.sum(em_loss)
        elif self.reduction == "none":
            return torch.sum(em_loss, dim=1).view(shp_x[0], *shp_x[2:])
        else:
            raise ValueError("Invalid reduction mode")


class SymmetricCrossEntropyLoss(nn.Module):
    """
    @inproceedings{wang2019symmetric,
        title={Symmetric cross entropy for robust learning with noisy labels},
        author={Wang, Yisen and Ma, Xingjun and Chen, Zaiyi and Luo, Yuan and Yi, Jinfeng and Bailey, James},
        booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
        pages={322--330},
        year={2019}
    }
    """

    def __init__(self, alpha=1, beta=0.5, eps=1e-7):
        super(SymmetricCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.cross_entropy = RobustCrossEntropyLoss()

    def forward(self, pred, labels):
        shp_x, shp_y = pred.shape, labels.shape

        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = torch.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=self.eps, max=1.0)

        ## trans the labels to one-hot
        if len(shp_x) != len(shp_y):
            labels = labels.view((shp_y[0], 1, *shp_y[1:]))
        gt = labels.long()
        label_one_hot = torch.zeros(shp_x, device=pred.device)
        label_one_hot.scatter_(1, gt, 1)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)

        rce = -1 * torch.sum(pred * torch.log(label_one_hot), dim=1)

        loss = self.alpha * ce + self.beta * rce.mean()
        return loss


class KL_CE_loss(nn.Module):
    """
    @article{zheng2021rectifying,
        title={Rectifying pseudo label learning via uncertainty estimation for domain adaptive semantic segmentation},
        author={Zheng, Zhedong and Yang, Yi},
        journal={International Journal of Computer Vision},
        volume={129},
        number={4},
        pages={1106--1120},
        year={2021},
        publisher={Springer}
    }
    """

    def __init__(self, weight=None, apply_nonlin=True):
        super(KL_CE_loss, self).__init__()
        self.KL_loss = nn.KLDivLoss(reduction="none")
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none")

        self.apply_nonlin = apply_nonlin

    def forward(self, pred_main, pred_aux, label):
        """
        the three input are the net output without sigmoid or softmax
         pred_main, pred_aux: [batchsize, channel, (z,) x, y]
         target must be b, c, x, y(, z) with c=1
        """
        ce_loss = self.ce(pred_main, label)

        if self.apply_nonlin:
            variance = torch.sum(
                self.KL_loss(
                    torch.log_softmax(pred_main, dim=1), torch.softmax(pred_aux, dim=1)
                ),
                dim=1,
            )
        else:
            variance = torch.sum(self.KL_loss(torch.log(pred_main), pred_aux), dim=1)
        exp_variance = torch.exp(-variance)

        loss = torch.sum(ce_loss * exp_variance + variance, dim=1)

        return loss.mean()


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

    em = EntropyMinimizeLoss(apply_nonlin=False)
    x1 = torch.stack(
        [
            torch.ones((2, 128, 128, 128)),
            torch.zeros((2, 128, 128, 128)),
            torch.zeros((2, 128, 128, 128)),
        ],
        dim=1,
    )
    print(em(x1))
