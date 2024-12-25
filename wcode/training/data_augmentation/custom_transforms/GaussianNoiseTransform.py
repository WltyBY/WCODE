# Copyright [2019] [Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany]
import torch

from typing import Tuple

from .scalar_type import RandomScalar, sample_scalar
from .BasicTransform import ImageOnlyTransform

class GaussianNoiseTransform(ImageOnlyTransform):
    def __init__(self,
                 noise_variance: RandomScalar = (0, 0.1),
                 p_per_channel: float = 1.,
                 synchronize_channels: bool = False):
        self.noise_variance = noise_variance
        self.p_per_channel = p_per_channel
        self.synchronize_channels = synchronize_channels
        super().__init__()

    def get_parameters(self, **data_dict) -> dict:
        shape = data_dict['image'].shape
        dct = {}
        dct['apply_to_channel'] = torch.rand(shape[0]) < self.p_per_channel
        dct['sigmas'] = \
            [sample_scalar(self.noise_variance, data_dict['image'])
             for i in range(sum(dct['apply_to_channel']))] if not self.synchronize_channels \
                else sample_scalar(self.noise_variance, data_dict['image'])
        return dct

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        if sum(params['apply_to_channel']) == 0:
            return img
        gaussian_noise = self._sample_gaussian_noise(img.shape, **params)
        img[params['apply_to_channel']] += gaussian_noise
        return img

    def _sample_gaussian_noise(self, img_shape: Tuple[int, ...], **params):
        if not isinstance(params['sigmas'], list):
            num_channels = sum(params['apply_to_channel'])
            # gaussian = torch.tile(torch.normal(0, params['sigmas'], size=(1, *img_shape[1:])),
            #                       (num_channels, *[1]*(len(img_shape) - 1)))
            gaussian = torch.normal(0, params['sigmas'], size=(1, *img_shape[1:]))
            gaussian.expand((num_channels, *[-1]*(len(img_shape) - 1)))
        else:
            gaussian = [
                torch.normal(0, i, size=(1, *img_shape[1:])) for i in params['sigmas']
            ]
            gaussian = torch.cat(gaussian, dim=0)
        return gaussian