import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional


class CosineAnnealingLRScheduler(_LRScheduler):
    """
    Cosine Annealing decay scheduler for multiple parameter groups.
    lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(current_step / max_steps * pi))
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_steps: int,
        warmup_steps: int = 0,
        min_lr: float = 1e-7,
        last_epoch: int = -1,
    ):
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = max_steps - warmup_steps

        # backup initial_lr for each group
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute lr for each group."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup: increase from 0 to initial_lr
            warmup_percent = self.last_epoch / self.warmup_steps
            return [
                max(group["initial_lr"] * warmup_percent, 1e-7)
                for group in self.optimizer.param_groups
            ]
        
        curr_decay_step = min(self.last_epoch - self.warmup_steps, self.decay_steps)
        decay = 0.5 * (1 + math.cos(math.pi * curr_decay_step / self.decay_steps))

        return [
            self.min_lr + (group["initial_lr"] - self.min_lr) * decay
            for group in self.optimizer.param_groups
        ]

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
