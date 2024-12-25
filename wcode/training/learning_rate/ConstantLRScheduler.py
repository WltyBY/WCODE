from torch.optim.lr_scheduler import _LRScheduler


class ConstantLRScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer,
        initial_lr: float,
        max_steps: int,
        current_step: int = None,
    ):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.ctr = 0
        super().__init__(
            optimizer, current_step if current_step is not None else -1
        )

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr
