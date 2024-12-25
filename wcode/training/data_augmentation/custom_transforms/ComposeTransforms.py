# Copyright [2019] [Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany]
from typing import List

from .BasicTransform import BasicTransform


class ComposeTransforms(BasicTransform):
    def __init__(self, transforms: List[BasicTransform]):
        super().__init__()
        self.transforms = transforms

    def apply(self, data_dict, **params):
        for t in self.transforms:
            data_dict = t(**data_dict)
        return data_dict
