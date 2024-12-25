import numpy as np
from scipy import ndimage

from wcode.utils.NDarray_operations import (
    get_largest_k_components,
    get_ND_bounding_box,
    crop_ND_volume_with_bounding_box,
)


def get_human_region_mask_one_channel(img, threshold=-600):
    """
    Get the mask of human region in CT volumes
    """
    dim = len(img.shape)
    if dim == 4:
        img = img[0]
    mask = np.asarray(img > threshold)
    se = np.ones([3, 3, 3])
    mask = ndimage.binary_opening(mask, se, iterations=2)
    mask = get_largest_k_components(mask, 1)
    mask_close = ndimage.binary_closing(mask, se, iterations=2)

    D, H, W = mask.shape
    for d in [1, 2, D - 3, D - 2]:
        mask_close[d] = mask[d]
    for d in range(0, D, 2):
        mask_close[d, 2:-2, 2:-2] = np.ones((H - 4, W - 4))

    # get background component
    bg = np.zeros_like(mask)
    bgs = get_largest_k_components(1 - mask_close, 10)
    for bgi in bgs:
        indices = np.where(bgi)
        if bgi.sum() < 1000:
            break
        if (
            indices[0].min() == 0
            or indices[1].min() == 0
            or indices[2].min() == 0
            or indices[0].max() == D - 1
            or indices[1].max() == H - 1
            or indices[2].max() == W - 1
        ):
            bg = bg + bgi
    fg = 1 - bg

    fg = ndimage.binary_opening(fg, se, iterations=1)
    fg = get_largest_k_components(fg, 1)
    if dim == 4:
        fg = np.expand_dims(fg, 0)
    fg = np.asarray(fg, np.uint8)
    return fg


def create_mask_base_on_human_body(data, threshold=-600):
    """
    data in (c, z, y, x)
    """
    assert (
        len(data.shape) == 4 or len(data.shape) == 3
    ), "data must have shape (C, X, Y, Z) or shape (C, X, Y)"

    mask = np.zeros(data.shape[1:], dtype=bool)
    for c in range(data.shape[0]):
        # each channel
        this_mask = get_human_region_mask_one_channel(data[c], threshold)
        mask = mask | this_mask
    mask = ndimage.binary_fill_holes(mask)
    return mask


def create_mask_base_on_threshold(data, threshold=None):
    assert (
        len(data.shape) == 4 or len(data.shape) == 3
    ), "data must have shape (C, X, Y, Z) or shape (C, X, Y)"

    if threshold is None:
        threshold = np.min(data)

    mask = np.zeros(data.shape[1:], dtype=bool)
    for c in range(data.shape[0]):
        # each channel
        this_mask = data[c] != threshold
        mask = mask | this_mask
    mask = ndimage.binary_fill_holes(mask)
    return mask


def crop_to_mask(data, seg=None, crop_fun_args={}, create_mask=create_mask_base_on_threshold):
    data_nonzero_mask = create_mask(data, **crop_fun_args)
    data_bbmin, data_bbmax = get_ND_bounding_box(data_nonzero_mask)

    if seg is not None:
        seg_nonzero_mask = create_mask_base_on_threshold(seg, threshold=0)
        if not(np.any(seg_nonzero_mask)):
            # all zero in seg_nonzero_mask
            seg_bbmin, seg_bbmax = data_bbmin, data_bbmax
        else:
            seg_bbmin, seg_bbmax = get_ND_bounding_box(seg_nonzero_mask)

        bbmin, bbmax = [], []
        for i in range(len(data_bbmin)):
            bbmin.append(min(data_bbmin[i], seg_bbmin[i]))
            bbmax.append(max(data_bbmax[i], seg_bbmax[i]))

        data_cropped_lst = []
        for i in range(data.shape[0]):
            data_cropped_lst.append(
                crop_ND_volume_with_bounding_box(data[i], bbmin, bbmax)[None]
            )
        data_cropped = np.vstack(data_cropped_lst)
        
        seg_cropped_lst = []
        for i in range(seg.shape[0]):
            seg_cropped_lst.append(
                crop_ND_volume_with_bounding_box(seg[i], bbmin, bbmax)[None]
            )
        seg_cropped = np.vstack(seg_cropped_lst)

        return data_cropped, seg_cropped, [bbmin, bbmax]
    else:
        assert len(data_bbmin) == len(
            data_bbmax
        ), "Length of bbox from data and seg should be the same."

        data_cropped_lst = []
        for i in range(data.shape[0]):
            data_cropped_lst.append(
                crop_ND_volume_with_bounding_box(data[i], data_bbmin, data_bbmax)[None]
            )
        data_cropped = np.vstack(data_cropped_lst)

        return data_cropped, np.zeros_like(data_cropped[0])[None], [data_bbmin, data_bbmax]
