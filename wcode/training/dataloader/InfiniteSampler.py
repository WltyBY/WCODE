import torch
from torch.utils.data import Sampler
from typing import Iterator


class InfiniteSampler(Sampler):
    """
    Wrap any sampler to make it infinite.
    Works for both DistributedSampler and normal Sampler.
    """

    def __init__(self, sampler: Sampler):
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        while True:
            yield from iter(self.sampler)

    def __len__(self):
        # Length is meaningless for infinite sampler
        return 2**31

    def set_epoch(self, epoch):
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)
