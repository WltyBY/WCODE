import random
from typing import List
from .AbstractTransform import AbstractTransform


class RandomFlip(AbstractTransform):
    def __init__(
        self,
        keys: List,
        prob: float,
    ) -> None:
        """
        keys: this transform needs to preprocess image, label or both
        prob: the probability to do flip
        """
        super().__init__(self, keys)
        self.prob = prob

    def _get_crop_params(**data_dict):
        shape = data_dict["image"].shape[]

    def __call__(self, **data_dict):
        # all datas, no matter the image or label, should be a NDarray in [b, c, (z,) y, x]
        if random.random() < self.prob:
            if "image" in self.keys:
                image = data_dict["image"]
        else:
            return data_dict
