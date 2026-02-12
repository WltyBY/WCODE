import numpy as np

from scipy import ndimage
from typing import Union, Tuple, List


ANISO_THRESHOLD = 3.0


def compute_new_shape(
    old_shape: Union[Tuple[int, ...], List[int], np.ndarray],
    old_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
    new_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
) -> np.ndarray:
    assert len(old_spacing) == len(old_shape)
    assert len(old_shape) == len(new_spacing)
    new_shape = np.array(
        [int(round(i / j * k)) for i, j, k in zip(old_spacing, new_spacing, old_shape)]
    )
    return new_shape


def resample_ND_data_to_given_spacing(
    data: np.ndarray,
    original_spacing: List[float],
    target_spacing: List[float],
    is_seg=False,
):
    """
    Resample an image/seg array (z, y, x) to a given spacing.

    :param data: The input image/seg array.
    :param origin_spacing: Original spacing along x, y, z direction.
    :param target_spacing: (list/tuple) Target spacing along x, y, z direction.
    :param order: (int) Order for interpolation.

    :return: A resampled image/seg array in z, y, x order.
    """
    assert data.ndim == 3

    original_spacing = np.array(original_spacing)[::-1]
    target_spacing = np.array(target_spacing)[::-1]
    zoom_factors = original_spacing / target_spacing

    if is_seg:
        return ndimage.zoom(data, zoom_factors, order=0)

    is_aniso = original_spacing.max() / original_spacing.min() > ANISO_THRESHOLD

    if is_aniso:
        lowres_axes = [
            i
            for i in range(len(original_spacing))
            if original_spacing[i] / original_spacing.min() > ANISO_THRESHOLD
        ]

        # high-res axes: cubic
        zoom_hr = [
            zoom_factors[i] if i not in lowres_axes else 1.0
            for i in range(len(zoom_factors))
        ]
        data = ndimage.zoom(data, zoom_hr, order=3)

        # low-res axes: linear
        zoom_lr = [
            zoom_factors[i] if i in lowres_axes else 1.0
            for i in range(len(zoom_factors))
        ]
        data = ndimage.zoom(data, zoom_lr, order=1)
    else:
        data = ndimage.zoom(data, zoom_factors, order=3)

    return data


def resample_ND_data_to_given_shape(
    data: np.ndarray, target_shape: List, current_spacing: List = None, is_seg=False
):
    """
    data in (z, y, x)
    target_shape in (z, y, x)
    current_spacing in (x, y, z), can be None
    """
    assert data.ndim == 3

    original_shape = np.array(data.shape)
    target_shape = np.array(target_shape)
    assert len(original_shape) == len(target_shape)

    zoom_factors = target_shape / original_shape

    if is_seg:
        return ndimage.zoom(data, zoom_factors, order=0)

    if current_spacing is None:
        return ndimage.zoom(data, zoom_factors, order=3)
    else:
        # x, y, z to z, y, x
        current_spacing = np.array(current_spacing)[::-1]
        is_aniso = current_spacing.max() / current_spacing.min() > ANISO_THRESHOLD
        if is_aniso:
            lowres_axes = [
                i
                for i in range(len(current_spacing))
                if current_spacing[i] / current_spacing.min() > ANISO_THRESHOLD
            ]

            # high-res axes cubic
            zoom_hr = [
                zoom_factors[i] if i not in lowres_axes else 1.0
                for i in range(len(zoom_factors))
            ]
            data = ndimage.zoom(data, zoom_hr, order=3)

            # low-res axes linear
            zoom_lr = [
                zoom_factors[i] if i in lowres_axes else 1.0
                for i in range(len(zoom_factors))
            ]
            data = ndimage.zoom(data, zoom_lr, order=1)
        else:
            data = ndimage.zoom(data, zoom_factors, order=3)

    return data


def resample_npy_with_channels_on_spacing(
    data: np.ndarray,
    original_spacing: List[float],
    target_spacing: List[float],
    channel_names: List[str],
    shapes: List[List[int]],
):
    """
    resample_npy_with_channels_on_spacing is only used during preprocessing

    data: 4D np array with shape (C, Z, Y, X)
    original_spacing/target_spacing: in x, y, z order but data in z, y, x order
    channel_names: Dict, names of each modality
    shapes: shapes of each modality

    ##### Do not use it in other places #####
    """
    assert data.ndim == 4
    # x, y, z to z, y, x
    original_spacing = np.array(original_spacing)[::-1]
    target_spacing = np.array(target_spacing)[::-1]
    
    is_aniso = original_spacing.max() / original_spacing.min() > ANISO_THRESHOLD
    # Determine whether it is anisotropic in each dimension
    lowres_axes = [
        i
        for i in range(len(original_spacing))
        if original_spacing[i] / original_spacing.min() > ANISO_THRESHOLD
    ]
    zoom_factors = original_spacing / target_spacing

    channel_lst = []
    for shape in shapes:
        channel_lst.append(shape[0])
    cumsum_id = np.cumsum(channel_lst)
    data_resampled = []
    for c in range(data.shape[0]):
        modality_id = (cumsum_id > c).tolist().index(True)
        channel_name = channel_names[str(modality_id)]
        if any(
            True if s in channel_name.lower() else False
            for s in ["mask", "label", "seg"]
        ):
            is_seg = True
        else:
            is_seg = False

        data_this = data[c]
        if is_seg:
            data_this = ndimage.zoom(data_this, zoom_factors, order=0)
        else:
            if is_aniso:
                # step 1: cubic (order=3) on all high-res axes
                zoom_hr = [
                    zoom_factors[i] if i not in lowres_axes else 1.0
                    for i in range(len(zoom_factors))
                ]
                data_this = ndimage.zoom(data_this, zoom_hr, order=3)

                # step 2: linear on all low-res axes
                zoom_lr = [
                    zoom_factors[i] if i in lowres_axes else 1.0
                    for i in range(len(zoom_factors))
                ]
                data_this = ndimage.zoom(data_this, zoom_lr, order=1)              
            else:
                data_this = ndimage.zoom(data_this, zoom_factors, order=3)

        data_resampled.append(data_this[None])

    return np.vstack(data_resampled)
