import numpy as np
from scipy import ndimage
from typing import Dict, List


def get_ND_bounding_box(volume, margin=None):
    """
    Get the bounding box of nonzero region in an ND volume.
    :param volume: An ND numpy array.
    :param margin: (list)
        The margin of bounding box along each axis.
    :return bb_min: (list) A list for the minimal value of each axis
            of the bounding box.
    :return bb_max: (list) A list for the maximal value of each axis
            of the bounding box.
    """
    input_shape = volume.shape
    if margin is None:
        margin = [0] * len(input_shape)
    assert len(input_shape) == len(margin)
    indxes = np.nonzero(volume)
    bb_min = []
    bb_max = []
    for i in range(len(input_shape)):
        bb_min.append(int(indxes[i].min()))
        bb_max.append(int(indxes[i].max()) + 1)

    for i in range(len(input_shape)):
        bb_min[i] = max(bb_min[i] - margin[i], 0)
        bb_max[i] = min(bb_max[i] + margin[i], input_shape[i])

    return bb_min, bb_max


def crop_ND_volume_with_bounding_box(volume, bb_min, bb_max):
    """
    Extract a subregion form an ND image.
    :param volume: The input ND array (z, y, x).
    :param bb_min: (list) The lower bound of the bounding box for each axis. z, y, x
    :param bb_max: (list) The upper bound of the bounding box for each axis. z, y, x
    :return: A croped ND image (z, y, x).
    """
    dim = len(volume.shape)
    assert dim >= 2 and dim <= 5
    assert bb_max[0] - bb_min[0] <= volume.shape[0]
    if dim == 2:
        output = volume[bb_min[0] : bb_max[0], bb_min[1] : bb_max[1]]
    elif dim == 3:
        output = volume[
            bb_min[0] : bb_max[0], bb_min[1] : bb_max[1], bb_min[2] : bb_max[2]
        ]
    elif dim == 4:
        output = volume[
            bb_min[0] : bb_max[0],
            bb_min[1] : bb_max[1],
            bb_min[2] : bb_max[2],
            bb_min[3] : bb_max[3],
        ]
    elif dim == 5:
        output = volume[
            bb_min[0] : bb_max[0],
            bb_min[1] : bb_max[1],
            bb_min[2] : bb_max[2],
            bb_min[3] : bb_max[3],
            bb_min[4] : bb_max[4],
        ]
    else:
        raise ValueError("the dimension number shoud be 2 to 5")
    return output


def get_largest_k_components(
    image: np.ndarray, k: int = 1, threshold: int = None, connectivity: int = 1
):
    """
    Get the largest K components from 2D or 3D binary image.
    inputs:
        image: The input ND array for binary segmentation.
        k: number of selected largest components.
        threshold: a size threshold to filter the components.
        connectivity: connectivity to get the component
    outputs:
        return: An output array (k == 1) or a list of ND array (k>1)
                with only the largest K components of the input.
    """
    dim = len(image.shape)
    if image.sum() == 0:
        print("the largest component is null")
        return image
    if dim < 2 or dim > 3:
        raise ValueError("the dimension number should be 2 or 3")
    s = ndimage.generate_binary_structure(dim, connectivity)
    labeled_array, numpatches = ndimage.label(image, s)
    sizes = ndimage.sum(image, labeled_array, range(1, numpatches + 1))
    sizes_sort = sorted(sizes, reverse=True)

    kmin = min(k, numpatches)
    output = []
    if threshold:
        max_label = min(np.where(sizes == sizes_sort[0])[0]) + 1
        output.append(np.asarray(labeled_array == max_label, np.uint8))
        for i in range(1, kmin):
            if sizes_sort[i] > threshold:
                labeli = min(np.where(sizes == sizes_sort[i])[0]) + 1
                output_i = np.asarray(labeled_array == labeli, np.uint8)
                output.append(output_i)
    else:
        for i in range(kmin):
            labeli = min(np.where(sizes == sizes_sort[i])[0]) + 1
            output_i = np.asarray(labeled_array == labeli, np.uint8)
            output.append(output_i)
            
    return output[0] if k == 1 else output


def create_h_component_image(image):
    """
    Get the H-component image for nuclei image.
    :param image: The input ND array for nuclei image, channel in RGB.
    :return: An output array of H-component image.
    """
    # define stain_matrix
    H = np.array([0.650, 0.704, 0.286])
    E = np.array([0.072, 0.990, 0.105])
    R = np.array([0.268, 0.570, 0.776])
    HDABtoRGB = [
        (H / np.linalg.norm(H)).tolist(),
        (E / np.linalg.norm(E)).tolist(),
        (R / np.linalg.norm(R)).tolist(),
    ]
    stain_matrix = HDABtoRGB
    im_inv = np.linalg.inv(stain_matrix)

    # transform
    im_temp = (-255) * np.log((np.float64(image) + 1) / 255) / np.log(255)
    image_out = np.reshape(
        np.dot(np.reshape(im_temp, [-1, 3]), im_inv), np.shape(image)
    )
    image_out = np.exp((255 - image_out) * np.log(255) / 255)
    image_out[image_out > 255] = 255
    image_h = image_out[:, :, 0].astype(np.uint8)

    return image_h


def rgb_seg_to_index(
    seg: np.ndarray,
    label_dict: Dict[str, List[int]],
    ignore_label: int = 0,
):
    """
    Convert an RGB-like segmentation map to class indices.

    Parameters
    ----------
    seg : np.ndarray
        Segmentation array of shape (3, H, W) or (H, W, 3), dtype uint8 or int.
    label_dict : dict
        Mapping from label name to RGB value, e.g.
        {
            "background": [0, 0, 0], -> 0
            "car": [255, 255, 0], -> 1
            "people": [0, 255, 0]  -> 2
        }
    ignore_label : int
        Class index to use for unknown / unmatched colors.

    Returns
    -------
    seg_idx : np.ndarray
        Segmentation map of shape (1, H, W), dtype int32
    """

    assert seg.ndim in (3,), "seg must in c, h, w"

    # Ensure HWC
    if seg.shape[0] == 3:
        seg_hwc = seg.transpose(1, 2, 0)
    else:
        seg_hwc = seg

    seg_hwc = seg_hwc.astype(np.int32)
    h, w, _ = seg_hwc.shape

    # Pack RGB -> int: 0xRRGGBB
    seg_int = (seg_hwc[..., 0] << 16) | (seg_hwc[..., 1] << 8) | seg_hwc[..., 2]

    # Build LUT: packed RGB -> class index
    color_to_index = {}
    for idx, (_, rgb) in enumerate(label_dict.items()):
        r, g, b = rgb
        packed = (r << 16) | (g << 8) | b
        color_to_index[packed] = idx

    # Vectorized mapping
    seg_flat = seg_int.reshape(-1)
    seg_idx_flat = np.full(seg_flat.shape, ignore_label, dtype=np.int32)

    for packed_color, class_idx in color_to_index.items():
        seg_idx_flat[seg_flat == packed_color] = class_idx

    seg_idx = seg_idx_flat.reshape(h, w)[None, ...]

    return seg_idx