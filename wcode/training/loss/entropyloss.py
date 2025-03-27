import torch
import torch.nn.functional as F

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

        em_loss = -x * torch.log2(x.clamp(min=1e-7))

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

    def forward(self, pred, label):
        shp_x, shp_y = pred.shape, label.shape

        ce = self.cross_entropy(pred, label)

        # RCE
        pred = torch.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=self.eps, max=1.0)

        ## trans the label to one-hot
        if len(shp_x) != len(shp_y):
            label = label.view((shp_y[0], 1, *shp_y[1:]))
        gt = label.long()
        label_one_hot = torch.zeros(shp_x, device=pred.device)
        label_one_hot.scatter_(1, gt, 1)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)

        rce = -1 * torch.sum(pred * torch.log(label_one_hot), dim=1)

        loss = self.alpha * ce + self.beta * rce.mean()
        return loss


class GeneralizedCrossEntropyLoss(nn.Module):
    """
    @article{zhang2018generalized,
        title={Generalized cross entropy loss for training deep neural networks with noisy labels},
        author={Zhang, Zhilu and Sabuncu, Mert},
        journal={Advances in neural information processing systems},
        volume={31},
        year={2018}
    }
    """

    def __init__(
        self,
        q: float = 0.8,
        apply_nonlin: bool = True,
    ):
        super(GeneralizedCrossEntropyLoss, self).__init__()
        self.q = q
        self.apply_nonlin = apply_nonlin

    def forward(self, pred: torch.Tensor, label: torch.Tensor):
        b, c, *_ = pred.shape

        if len(pred.shape) != len(label.shape):
            label = label.unsqueeze(1)

        if self.apply_nonlin:
            pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        pred = torch.gather(pred, 1, label.long()).transpose(0, 1).reshape(1, -1)

        gce = ((1.0 - torch.pow(pred, self.q)) / self.q).transpose(0, 1)

        return gce.mean()


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

        loss = (ce_loss * exp_variance).sum() / exp_variance.sum() + variance.mean()

        return loss


class reliability_based_co_teaching_loss(nn.Module):
    def __init__(self, weight=None):
        super(reliability_based_co_teaching_loss, self).__init__()
        self.ce = RobustCrossEntropyLoss(weight=weight, reduction="none")

    def forward(self, pred_main, pred_aux, feat_main, feat_aux):
        b, c, *_ = pred_main.shape

        with torch.no_grad():
            # generate pseudo label
            hard_main = torch.argmax(pred_main, dim=1, keepdim=True)
            hard_aux = torch.argmax(pred_aux, dim=1, keepdim=True)
            soft_main = torch.softmax(pred_main, dim=1)
            soft_aux = torch.softmax(pred_aux, dim=1)
            confidence_main, _ = torch.max(soft_main, dim=1, keepdim=True)
            confidence_aux, _ = torch.max(soft_aux, dim=1, keepdim=True)

            main_mask = confidence_main >= confidence_aux
            pseudo_label = (
                main_mask * hard_main + (torch.logical_not(main_mask)) * hard_aux
            )

            # get reliability map
            reliability_map_main = torch.zeros_like(
                hard_main, device=hard_main.device
            ).float()
            reliability_map_aux = torch.zeros_like(
                hard_main, device=hard_main.device
            ).float()
            for m in range(b):
                for i in range(c):
                    mask_main = torch.where(hard_main[m] == i)
                    # c_feat, *
                    class_feat_main = feat_main[m][:, *mask_main[1:]]
                    # 1, *
                    class_confidence_main = confidence_main[m][:, *mask_main[1:]]
                    # c_feat
                    class_center_main = (class_feat_main * class_confidence_main).mean(
                        dim=1
                    )
                    reliability_map_main[m][mask_main] = F.cosine_similarity(
                        class_feat_main, class_center_main[:, None], dim=0
                    )

                    mask_aux = torch.where(hard_aux[m] == i)
                    # c_feat, *
                    class_feat_aux = feat_aux[m][:, *mask_aux[1:]]
                    # 1, *
                    class_confidence_aux = confidence_aux[m][:, *mask_aux[1:]]
                    class_center_aux = (class_feat_aux * class_confidence_aux).mean(
                        dim=1
                    )
                    reliability_map_aux[m][mask_aux] = F.cosine_similarity(
                        class_feat_aux, class_center_aux[:, None], dim=0
                    )

        loss_main = (
            self.ce(pred_main, pseudo_label) * reliability_map_aux
        ).sum() / reliability_map_aux.sum()
        loss_aux = (
            self.ce(pred_aux, pseudo_label) * reliability_map_main
        ).sum() / reliability_map_main.sum()
        return loss_main + loss_aux


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
    y = torch.randint(0, 3, (2, 1, 128, 128, 128))

    em = GeneralizedCrossEntropyLoss(apply_nonlin=True)
    print(em(x, y))
