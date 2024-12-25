# Copyright [2019] [Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany]
import torch

from .BasicTransform import BasicTransform


class RandomTransform(BasicTransform):
    def __init__(self, transform: BasicTransform, apply_probability: float = 1):
        super().__init__()
        self.transform = transform
        self.apply_probability = apply_probability

    def get_parameters(self, **data_dict) -> dict:
        return {"apply_transform": torch.rand(1).item() < self.apply_probability}

    def apply(self, data_dict: dict, **params) -> dict:
        if params["apply_transform"]:
            return self.transform(**data_dict)
        else:
            return data_dict

    def __repr__(self):
        ret_str = f"{type(self).__name__}(p={self.apply_probability}, transform={self.transform})"
        return ret_str
