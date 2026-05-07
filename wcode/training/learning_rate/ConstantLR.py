from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional


class ConstantLRScheduler(_LRScheduler):
    """
    Constant LR scheduler.
    lr = initial_lr (never changes)
    """

    def __init__(self, optimizer, warmup_steps: int = 0, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        # backup initial_lr for each group
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Return constant initial_lr for each group."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup: increase from 0 to initial_lr
            warmup_percent = self.last_epoch / self.warmup_steps
            return [
                max(group["initial_lr"] * warmup_percent, 1e-7)
                for group in self.optimizer.param_groups
            ]

        return [group["initial_lr"] for group in self.optimizer.param_groups]

    def step(self, current_step: Optional[int] = None):
        """
        Update last_epoch and lr manually.
        If current_step is None, last_epoch += 1 (for non-DDP fallback).
        """
        if current_step is None:
            self.last_epoch += 1
        else:
            self.last_epoch = current_step

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr
