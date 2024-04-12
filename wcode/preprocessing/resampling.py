import numpy as np

from scipy import ndimage
from typing import Union, Tuple, List


ANISO_THRESHOLD = 3

def whether_anisotropy(
    spacing: Union[Tuple[float, ...], List[float], np.ndarray],
    anisotropy_threshold=ANISO_THRESHOLD,
):
    anisotropy_flag = (np.max(spacing) / np.min(spacing)) > anisotropy_threshold
    return anisotropy_flag


def get_lowres_axis(new_spacing: Union[Tuple[float, ...], List[float], np.ndarray]):
    # find which axis is anisotropic
    axis = np.argmax(new_spacing)
    return axis


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


def resample_ND_data_to_given_spacing(data, origin_spacing, target_spacing, is_seg):
    """
    Resample an image/seg array to a given spacing. z, y, x

    :param data: The input image/seg array.
    :param origin_spacing: Original spacing along x, y, z direction.
    :param target_spacing: (list/tuple) Target spacing along x, y, z direction.
    :param order: (int) Order for interpolation.

    :return: A resampled image/seg array.
    """
    assert data.ndim == 3
    zoom = [origin_spacing[i] / target_spacing[i] for i in range(3)]
    zoom = [zoom[2], zoom[1], zoom[0]]

    if is_seg:
        return ndimage.interpolation.zoom(data, zoom, order=0)
    else:
        return ndimage.interpolation.zoom(data, zoom, order=3)


def resample_ND_data_to_given_shape(data, out_shape, is_seg):
    assert data.ndim == 3
    shape0 = data.shape
    assert len(shape0) == len(out_shape)
    scale = [(out_shape[i] + 0.0) / shape0[i] for i in range(len(shape0))]

    if is_seg:
        return ndimage.interpolation.zoom(data, scale, order=0)
    else:
        return ndimage.interpolation.zoom(data, scale, order=3)


def resample_npy_with_channels_on_spacing(data, origin_spacing, target_spacing, is_seg=False):
    assert data.ndim == 4

    zoom = [origin_spacing[i] / target_spacing[i] for i in range(3)]
    zoom = [zoom[2], zoom[1], zoom[0]]
    data_resampled = []
    for i in range(data.shape[0]):
        if is_seg:
            data_resampled.append(
                ndimage.interpolation.zoom(data[i], zoom, order=0)[None]
            )
        else:
            data_resampled.append(
                ndimage.interpolation.zoom(data[i], zoom, order=3)[None]
            )
    data_resampled = np.vstack(data_resampled)

    return data_resampled


def resample_npy_with_channels_on_shape(data, target_shape, is_seg=False):
    assert data.ndim == 4
    shape0 = data.shape[1:]
    assert len(shape0) == len(target_shape)

    scale = [(target_shape[i] + 0.0) / shape0[i] for i in range(len(shape0))]

    data_resampled = []
    for i in range(data.shape[0]):
        if is_seg:
            data_resampled.append(
                ndimage.interpolation.zoom(data[i], scale, order=0)[None]
            )
        else:
            data_resampled.append(
                ndimage.interpolation.zoom(data[i], scale, order=3)[None]
            )
    data_resampled = np.vstack(data_resampled)

    return data_resampled


if __name__ == "__main__":
    image_path = "./Dataset/RADCURE/images/RADCURE-3953_0000.nii.gz"
    img_output_path = "./Dataset_preprocessed/RADCURE/RADCURE-3953_0000.nii.gz"
    seg_path = "./Dataset/RADCURE/labels/RADCURE-3953.nii.gz"
    seg_output_path = "./Dataset_preprocessed/RADCURE/RADCURE-3953.nii.gz"
    import SimpleITK as sitk

    itk_obj = sitk.ReadImage(image_path)
    print(itk_obj.GetSpacing())
    data = sitk.GetArrayFromImage(itk_obj)
    data_resampled = resample_ND_data_to_given_spacing(
        data, itk_obj.GetSpacing(), [0.9760000109672546, 0.9760000109672546, 2.0], False
    )
    output_obj = sitk.GetImageFromArray(data_resampled)
    output_obj.SetDirection(itk_obj.GetDirection())
    output_obj.SetOrigin(itk_obj.GetOrigin())
    output_obj.SetSpacing([0.9760000109672546, 0.9760000109672546, 2.0])
    sitk.WriteImage(output_obj, img_output_path)

    itk_obj = sitk.ReadImage(seg_path)
    data = sitk.GetArrayFromImage(itk_obj)
    data_resampled = resample_ND_data_to_given_spacing(
        data, itk_obj.GetSpacing(), [0.9760000109672546, 0.9760000109672546, 2.0], True
    )
    output_obj = sitk.GetImageFromArray(data_resampled)
    output_obj.SetDirection(itk_obj.GetDirection())
    output_obj.SetOrigin(itk_obj.GetOrigin())
    output_obj.SetSpacing([0.9760000109672546, 0.9760000109672546, 2.0])
    sitk.WriteImage(output_obj, seg_output_path)
