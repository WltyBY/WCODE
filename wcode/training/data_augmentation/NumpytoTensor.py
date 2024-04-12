import numpy as np

from wcode.training.data_augmentation.AbstractTransform import AbstractTransform


class NumpyToTensor(AbstractTransform):

    def __init__(self, keys=None, cast_to=None):
        """Utility function for pytorch. Converts data (and seg) numpy ndarrays to pytorch tensors
        :param keys: specify keys to be converted to tensors. If None then all keys will be converted
        (if value id np.ndarray). Can be a key (typically string) or a list/tuple of keys
        :param cast_to: if not None then the values will be cast to what is specified here. Currently only half, float
        and long supported (use string)
        """
        if keys is not None and not isinstance(keys, (list, tuple)):
            keys = [keys]
        self.keys = keys
        self.cast_to = cast_to

    def cast(self, tensor):
        if self.cast_to is not None:
            if self.cast_to == 'half':
                tensor = tensor.half()
            elif self.cast_to == 'float':
                tensor = tensor.float()
            elif self.cast_to == 'long':
                tensor = tensor.long()
            elif self.cast_to == 'bool':
                tensor = tensor.bool()
            else:
                raise ValueError('Unknown value for cast_to: %s' %
                                 self.cast_to)
        return tensor

    def __call__(self, **data_dict):
        import torch

        if self.keys is None:
            for key, val in data_dict.items():
                if isinstance(val, np.ndarray):
                    data_dict[key] = self.cast(
                        torch.from_numpy(val)).contiguous()
                elif isinstance(val, (list, tuple)) and all(
                    [isinstance(i, np.ndarray) for i in val]):
                    data_dict[key] = [
                        self.cast(torch.from_numpy(i)).contiguous()
                        for i in val
                    ]
        else:
            for key in self.keys:
                if isinstance(data_dict[key], np.ndarray):
                    data_dict[key] = self.cast(torch.from_numpy(
                        data_dict[key])).contiguous()
                elif isinstance(data_dict[key], (list, tuple)) and all(
                    [isinstance(i, np.ndarray) for i in data_dict[key]]):
                    data_dict[key] = [
                        self.cast(torch.from_numpy(i)).contiguous()
                        for i in data_dict[key]
                    ]

        return data_dict
