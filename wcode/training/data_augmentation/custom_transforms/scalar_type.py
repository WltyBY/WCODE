# Copyright [2019] [Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany]
from typing import Union, Tuple, Callable
import numpy as np


RandomScalar = Union[int, float, Tuple[float, float], Callable[..., Union[int, float]]]


def sample_scalar(scalar_type: RandomScalar, *args, **kwargs):
    if isinstance(scalar_type, (int, float)):
        return scalar_type
    elif isinstance(scalar_type, (list, tuple)):
        assert len(scalar_type) == 2, 'if list is provided, its length must be 2'
        assert scalar_type[0] <= scalar_type[1], 'if list is provided, first entry must be smaller or equal than second entry, ' \
                                                'otherwise we cannot sample using np.random.uniform'
        if scalar_type[0] == scalar_type[1]:
            return scalar_type[0]
        return np.random.uniform(*scalar_type)
    elif callable(scalar_type):
        return scalar_type(*args, **kwargs)
    else:
        raise RuntimeError('Unknown type: %s. Expected: int, float, list, tuple, callable', type(scalar_type))