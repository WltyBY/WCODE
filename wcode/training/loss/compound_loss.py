import torch
from torch import nn

from wcode.training.loss.diceloss import TverskyLoss


class Tversky_and_CE_loss(nn.Module):
    def __init__(self, tversky_kwargs, ce_kwargs, weight_ce=1, weight_tversky=1, ignore_label=None):
        super(Tversky_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_tversky = weight_tversky
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = nn.CrossEntropyLoss(**ce_kwargs)
        self.tverskyloss = TverskyLoss(**tversky_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, z, y, x with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(Tversky_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        tversky_loss = self.tverskyloss(net_output, target_dice, loss_mask=mask) \
            if self.weight_tversky != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0].long()) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        # print("Tversky Loss:", tversky_loss.item())
        # print("CE Loss:", ce_loss.item())
        result = self.weight_ce * ce_loss + self.weight_tversky * tversky_loss
        return result