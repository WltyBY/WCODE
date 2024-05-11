import torch
import warnings

from torch import nn

from wcode.training.loss.diceloss import TverskyLoss


class Tversky_and_CE_loss(nn.Module):
    def __init__(
        self,
        tversky_kwargs,
        ce_kwargs,
        weight_ce=1,
        weight_tversky=1,
        ignore_label=None,
    ):
        super(Tversky_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs["ignore_index"] = ignore_label

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
            assert target.shape[1] == 1, (
                "ignore label is not implemented for one hot encoded target variables "
                "(Tversky_and_CE_loss)"
            )
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        tversky_loss = (
            self.tverskyloss(net_output, target_dice, loss_mask=mask)
            if self.weight_tversky != 0
            else 0
        )
        ce_loss = (
            self.ce(net_output, target[:, 0].long())
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0)
            else 0
        )
        # print("Tversky Loss:", tversky_loss.item())
        # print("CE Loss:", ce_loss.item())
        result = self.weight_ce * ce_loss + self.weight_tversky * tversky_loss
        return result


class Hinton_distillaton_loss(nn.Module):
    """
    Paper: Distilling the knowledge in a neural network (https://doi.org/10.48550/arXiv.1503.02531)
    """

    def __init__(
        self, ce_kwargs, temperature_index, weight_of_distill, ignore_label=None
    ):
        super(Hinton_distillaton_loss, self).__init__()
        self.weight_of_distill = weight_of_distill
        if self.weight_of_distill <= 0.5:
            warnings.warn(
                "According to the paper, it is best for the weight of distillation loss to be"
                "greater than that of normal classification loss."
            )

        if ignore_label is not None:
            ce_kwargs["ignore_index"] = ignore_label

        self.temperature_index = temperature_index
        self.gradient_compensation = temperature_index * temperature_index
        self.weight_of_distill = weight_of_distill
        self.weight_of_classify = 1 - weight_of_distill
        self.ignore_label = ignore_label

        self.ce = nn.CrossEntropyLoss(**ce_kwargs)

    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        target: torch.Tensor,
    ):
        """
        target must be b, c, z, y, x with c=1
        :param student_output: output logits of the student model
        :param teacher_output: output logits of the teacher model
        :param target: ground truth
        :return: total loss
        """
        student_loss = self.ce(student_output, target[:, 0].long())

        # generate soft target through the output of the teacher model
        soft_target = torch.softmax(teacher_output / self.temperature_index, dim=1)
        student_soft_logits = student_output / self.temperature_index

        distillation_loss = self.ce(student_soft_logits, soft_target)

        loss = (
            self.weight_of_classify * student_loss
            + self.weight_of_distill * self.gradient_compensation * distillation_loss
        )

        return loss
