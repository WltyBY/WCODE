from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional


class PolyLRScheduler(_LRScheduler):
    """
    Polynomial decay scheduler.
    lr = initial_lr * (1 - current_step / max_steps) ** exponent
    Inherits _LRScheduler only for state_dict/load_state_dict compatibility.
    """

    def __init__(
        self, optimizer, max_steps: int, exponent: float = 0.9, last_epoch: int = -1
    ):
        self.max_steps = max_steps
        self.exponent = exponent
        # backup initial_lr for each group
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute lr for each group."""
        decay = (1 - self.last_epoch / self.max_steps) ** self.exponent
        # print(f"{self.last_epoch}/{self.max_steps} ** {self.exponent}, decay={decay}")
        return [group["initial_lr"] * decay for group in self.optimizer.param_groups]

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
