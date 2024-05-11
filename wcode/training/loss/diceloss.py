import torch

from torch import nn

from wcode.utils.Tensor_operations import AllGatherGrad


class TverskyLoss(nn.Module):
    def __init__(
        self,
        alpha=0.5,
        beta=0.5,
        smooth=1e-5,
        batch_dice=True,
        do_bg=True,
        ddp=False,
        apply_nonlin=True,
    ):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        shp_x, shp_y = x.shape, y.shape

        # comment out if your model contains a sigmoid or equivalent activation layer
        if self.apply_nonlin:
            x = torch.softmax(x, dim=1)

        if not self.do_bg:
            x = x[:, 1:]

        # make everything shape (b, c)
        axes = list(range(2, len(shp_x)))

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

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

        tp = (
            (x * y_onehot).sum(axes)
            if loss_mask is None
            else (x * y_onehot * loss_mask).sum(axes)
        )
        fp = (
            (x * (1 - y_onehot)).sum(axes)
            if loss_mask is None
            else (x * (1 - y_onehot) * loss_mask).sum(axes)
        )
        fn = (
            ((1 - x) * y_onehot).sum(axes)
            if loss_mask is None
            else ((1 - x) * y_onehot * loss_mask).sum(axes)
        )

        if self.ddp and self.batch_dice:
            tp = AllGatherGrad.apply(tp).sum(0)
            fp = AllGatherGrad.apply(fp).sum(0)
            fn = AllGatherGrad.apply(fn).sum(0)

        if self.batch_dice:
            tp = tp.sum(0)
            fp = fp.sum(0)
            fn = fn.sum(0)

        Tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )

        Tversky = Tversky.mean()

        return 1 - Tversky
