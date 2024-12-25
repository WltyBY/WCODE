# Copyright [2019] [Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany]
import torch
import numpy as np

from .scalar_type import RandomScalar, sample_scalar
from .BasicTransform import ImageOnlyTransform


class BGContrast:
    def __init__(self, contrast_range):
        self.contrast_range = contrast_range

    def sample_contrast(self, *args, **kwargs):
        if callable(self.contrast_range):
            factor = self.contrast_range()
        else:
            if np.random.random() < 0.5 and self.contrast_range[0] < 1:
                factor = np.random.uniform(self.contrast_range[0], 1)
            else:
                factor = np.random.uniform(
                    max(self.contrast_range[0], 1), self.contrast_range[1]
                )
        return factor

    def __call__(self, *args, **kwargs):
        return self.sample_contrast(*args, **kwargs)

    def __repr__(self):
        return self.__class__.__name__ + f"(contrast_range={self.contrast_range})"


class ContrastTransform(ImageOnlyTransform):
    def __init__(
        self,
        contrast_range: RandomScalar,
        preserve_range: bool,
        synchronize_channels: bool,
        p_per_channel: float = 1,
    ):
        super().__init__()
        self.contrast_range = contrast_range
        self.preserve_range = preserve_range
        self.synchronize_channels = synchronize_channels
        self.p_per_channel = p_per_channel

    def get_parameters(self, **data_dict) -> dict:
        shape = data_dict["image"].shape
        apply_to_channel = torch.where(torch.rand(shape[0]) < self.p_per_channel)[0]
        if self.synchronize_channels:
            multipliers = torch.Tensor(
                [
                    sample_scalar(
                        self.contrast_range, image=data_dict["image"], channel=None
                    )
                ]
                * len(apply_to_channel)
            )
        else:
            multipliers = torch.Tensor(
                [
                    sample_scalar(
                        self.contrast_range, image=data_dict["image"], channel=c
                    )
                    for c in apply_to_channel
                ]
            )
        return {"apply_to_channel": apply_to_channel, "multipliers": multipliers}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        if len(params["apply_to_channel"]) == 0:
            return img
        # array notation is not faster, let's leave it like this
        for i in range(len(params["apply_to_channel"])):
            c = params["apply_to_channel"][i]
            mean = img[c].mean()
            if self.preserve_range:
                minm = img[c].min()
                maxm = img[c].max()

            # this is faster than having it in one line because this circumvents reallocating memory
            img[c] -= mean
            img[c] *= params["multipliers"][i]
            img[c] += mean

            if self.preserve_range:
                img[c].clamp_(minm, maxm)

        return img
