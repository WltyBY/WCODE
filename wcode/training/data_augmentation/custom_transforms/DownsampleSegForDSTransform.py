# Copyright [2019] [Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany]
import torch

from typing import Tuple, List, Union
from torch.nn.functional import interpolate

from .BasicTransform import SegOnlyTransform


class DownsampleSegForDSTransform(SegOnlyTransform):
    def __init__(self, ds_scales: Union[List, Tuple]):
        super().__init__()
        self.ds_scales = ds_scales

    def _apply_to_segmentation(
        self, segmentation: torch.Tensor, **params
    ) -> List[torch.Tensor]:
        results = []
        for s in self.ds_scales:
            if not isinstance(s, (tuple, list)):
                s = [s] * (segmentation.ndim - 1)
            else:
                assert (
                    len(s) == segmentation.ndim - 1
                ), "len(s): {}, segmentation.ndim - 1: {}".format(
                    len(s), segmentation.ndim - 1
                )

            if all([i == 1 for i in s]):
                results.append(segmentation)
            else:
                new_shape = [round(i * j) for i, j in zip(segmentation.shape[1:], s)]
                dtype = segmentation.dtype
                # interpolate is not defined for short etc
                results.append(
                    interpolate(
                        segmentation[None].float(), new_shape, mode="nearest-exact"
                    )[0].to(dtype)
                )
        return results
