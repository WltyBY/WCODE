import torch

from typing import Union, Tuple, List

from .BasicTransform import SegOnlyTransform


class Label2OnehotTransform(SegOnlyTransform):
    def __init__(self, num_of_classes: int, dimension: int):
        self.num_of_classes = num_of_classes
        self.dim = dimension
        super().__init__()

    def _apply_to_segmentation(
        self, segmentation: torch.Tensor, **params
    ) -> torch.Tensor:
        target_shape = (self.num_of_classes, *segmentation.shape[-self.dim:])
        seg_shape = segmentation.shape

        with torch.no_grad():
            if len(target_shape) != len(seg_shape):
                # we think it (z,) y, x, so add c dimension as c, (z,) y, x.
                assert len(target_shape) == len(seg_shape) + 1
                segmentation = segmentation.view((1, *seg_shape[1:]))

            if all([i == j for i, j in zip(target_shape, seg_shape)]):
                # if this is the case then gt is probably already a one hot encoding
                segmentation_onehot = segmentation
            else:
                gt = segmentation.long()
                segmentation_onehot = torch.zeros(target_shape, device=segmentation.device)
                segmentation_onehot.scatter_(0, gt, 1)
        
        return segmentation_onehot
