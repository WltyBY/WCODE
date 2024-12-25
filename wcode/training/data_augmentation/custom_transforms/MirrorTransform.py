# Copyright [2019] [Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany]
import torch

from typing import Tuple

from .BasicTransform import BasicTransform


class MirrorTransform(BasicTransform):
    def __init__(self, allowed_axes: Tuple[int, ...]):
        super().__init__()
        self.allowed_axes = allowed_axes

    def get_parameters(self, **data_dict) -> dict:
        axes = [i for i in self.allowed_axes if torch.rand(1) < 0.5]
        return {"axes": axes}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        if len(params["axes"]) == 0:
            return img
        axes = [i + 1 for i in params["axes"]]
        return torch.flip(img, axes)

    def _apply_to_segmentation(
        self, segmentation: torch.Tensor, **params
    ) -> torch.Tensor:
        if len(params["axes"]) == 0:
            return segmentation
        axes = [i + 1 for i in params["axes"]]
        return torch.flip(segmentation, axes)

    def _apply_to_regr_target(self, regression_target, **params) -> torch.Tensor:
        if len(params["axes"]) == 0:
            return regression_target
        axes = [i + 1 for i in params["axes"]]
        return torch.flip(regression_target, axes)

    def _apply_to_bbox(self, bbox, **params):
        raise NotImplementedError

    def _apply_to_keypoints(self, keypoints, **params):
        raise NotImplementedError
