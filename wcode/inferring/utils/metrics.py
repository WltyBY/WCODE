import numpy as np

from scipy import ndimage
from typing import Union, Tuple

def compute_tp_fp_fn_tn(pred, target, ignore_mask=None):
    if ignore_mask is None:
        use_mask = np.ones_like(target, dtype=bool)
    else:
        use_mask = ~ignore_mask
    tp = np.sum((target & pred) & use_mask)
    fp = np.sum(((~target) & pred) & use_mask)
    fn = np.sum((target & (~pred)) & use_mask)
    tn = np.sum(((~target) & (~pred)) & use_mask)
    return tp, fp, fn, tn


def region_or_label_to_mask(segmentation: np.ndarray, region_or_label: Union[int, Tuple[int, ...]]) -> np.ndarray:
    if np.isscalar(region_or_label):
        return segmentation == region_or_label
    else:
        mask = np.zeros_like(segmentation, dtype=bool)
        for r in region_or_label:
            mask[segmentation == r] = True
    return mask


def DSC(pred, target, axes=None, smooth=1e-5):
    tp, fp, fn, _ = compute_tp_fp_fn_tn(pred, target, axes)
    return (2 * tp + smooth) / (2 * tp + fp + fn + smooth)


def IoU(pred, target, axes=None, mask=None, square=False, smooth=1e-5):
    tp, fp, fn, _ = compute_tp_fp_fn_tn(pred, target, axes, mask, square)
    return (tp + smooth) / (tp + fp + fn + smooth)


def Sensitivity(pred, target, axes=None, mask=None, square=False, smooth=1e-5):
    tp, _, fn, _ = compute_tp_fp_fn_tn(pred, target, axes, mask, square)
    return (tp + smooth) / (tp + fn + smooth)


def get_edge_points(img):
    """
    Get edge points of a binary segmentation result.

    :param img: (numpy.array) a 2D or 3D array of binary segmentation.
    :return: an edge map.
    """
    dim = len(img.shape)
    if dim == 2:
        strt = ndimage.generate_binary_structure(2, 1)
    else:
        strt = ndimage.generate_binary_structure(3, 1)
    ero = ndimage.binary_erosion(img, strt)
    edge = np.asarray(img, np.uint8) - np.asarray(ero, np.uint8)
    return edge


def HD95(s, g, spacing=None):
    """
    Get the 95 percentile of hausdorff distance between a binary segmentation
    and the ground truth.

    :param s: (numpy.array) a 2D or 3D binary image for segmentation.
    :param g: (numpy.array) a 2D or 2D binary image for ground truth.
    :param spacing: (list) A list for image spacing, length should be 2 or 3.

    :return: The HD95 value.
    """
    s_edge = get_edge_points(s)
    g_edge = get_edge_points(g)
    ns = s_edge.sum()
    ng = g_edge.sum()
    if ns + ng == 0:
        hd95 = 0.0
    elif ns * ng == 0:
        hd95 = 100.0
    else:
        image_dim = len(s.shape)
        assert image_dim == len(g.shape)
        if spacing == None:
            spacing = [1.0] * image_dim
        else:
            assert image_dim == len(spacing)
        s_dis = ndimage.distance_transform_edt(1 - s_edge, sampling=spacing)
        g_dis = ndimage.distance_transform_edt(1 - g_edge, sampling=spacing)

        dist_list1 = s_dis[g_edge > 0]
        dist_list1 = sorted(dist_list1)
        dist1 = dist_list1[int(len(dist_list1) * 0.95)]
        dist_list2 = g_dis[s_edge > 0]
        dist_list2 = sorted(dist_list2)
        dist2 = dist_list2[int(len(dist_list2) * 0.95)]
        hd95 = max(dist1, dist2)
    return hd95


def ASSD(s, g, spacing=None):
    """
    Get the Average Symetric Surface Distance (ASSD) between a binary segmentation
    and the ground truth.

    :param s: (numpy.array) a 2D or 3D binary image for segmentation.
    :param g: (numpy.array) a 2D or 2D binary image for ground truth.
    :param spacing: (list) A list for image spacing, length should be 2 or 3.

    :return: The ASSD value.
    """
    s_edge = get_edge_points(s)
    g_edge = get_edge_points(g)
    image_dim = len(s.shape)
    assert image_dim == len(g.shape)
    if spacing == None:
        spacing = [1.0] * image_dim
    else:
        assert image_dim == len(spacing)
    s_dis = ndimage.distance_transform_edt(1 - s_edge, sampling=spacing)
    g_dis = ndimage.distance_transform_edt(1 - g_edge, sampling=spacing)

    ns = s_edge.sum()
    ng = g_edge.sum()
    if ns + ng == 0:
        assd = 0.0
    elif ns * ng == 0:
        assd = 20.0
    else:
        s_dis_g_edge = s_dis * g_edge
        g_dis_s_edge = g_dis * s_edge
        assd = (s_dis_g_edge.sum() + g_dis_s_edge.sum()) / (ns + ng)
    return assd
