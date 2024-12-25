# Copyright [2019] [Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany]
import torch

from .scalar_type import RandomScalar, sample_scalar
from .BasicTransform import ImageOnlyTransform


class GammaTransform(ImageOnlyTransform):
    def __init__(self, gamma: RandomScalar, p_invert_image: float, synchronize_channels: bool, p_per_channel: float,
                 p_retain_stats: float):
        super().__init__()
        self.gamma = gamma
        self.p_invert_image = p_invert_image
        self.synchronize_channels = synchronize_channels
        self.p_per_channel = p_per_channel
        self.p_retain_stats = p_retain_stats

    def get_parameters(self, **data_dict) -> dict:
        shape = data_dict['image'].shape
        apply_to_channel = torch.where(torch.rand(shape[0]) < self.p_per_channel)[0]
        retain_stats = torch.rand(len(apply_to_channel)) < self.p_retain_stats
        invert_image = torch.rand(len(apply_to_channel)) < self.p_invert_image

        if self.synchronize_channels:
            gamma = torch.Tensor([sample_scalar(self.gamma, image=data_dict['image'], channel=None)] * len(apply_to_channel))
        else:
            gamma = torch.Tensor([sample_scalar(self.gamma, image=data_dict['image'], channel=c) for c in apply_to_channel])
        return {
            'apply_to_channel': apply_to_channel,
            'retain_stats': retain_stats,
            'invert_image': invert_image,
            'gamma': gamma
        }

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        for c, r, i, g in zip(params['apply_to_channel'], params['retain_stats'], params['invert_image'], params['gamma']):
            if i:
                img[c] *= -1
            if r:
                # std_mean is for whatever reason slower than doing the computations separately!?
                # std, mean = torch.std_mean(img[c])
                mean = torch.mean(img[c])
                std = torch.std(img[c])
            minm = torch.min(img[c])
            rnge = torch.max(img[c]) - minm
            img[c] = torch.pow(((img[c] - minm) / torch.clamp(rnge, min=1e-7)), g) * rnge + minm
            if r:
                # std_here, mn_here = torch.std_mean(img[c])
                mn_here = torch.mean(img[c])
                std_here = torch.std(img[c])
                img[c] -= mn_here
                img[c] *= (std / torch.clamp(std_here, min=1e-7))
                img[c] += mean

            if i:
                img[c] *= -1
        return img