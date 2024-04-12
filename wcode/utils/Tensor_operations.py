import torch
import numpy as np

from typing import Union, List, Tuple


def sum_tensor(
    inp: torch.Tensor, axes: Union[np.ndarray, Tuple, List], keepdim: bool = False
) -> torch.Tensor:
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp
