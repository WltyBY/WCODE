import json
import numpy as np

from scipy import ndimage
from typing import Union, Tuple
from itertools import product

from sklearn import metrics
from sympy import per


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


def region_or_label_to_mask(
    segmentation: np.ndarray, region_or_label: Union[int, Tuple[int, ...]]
) -> np.ndarray:
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


def get_instances(mask: np.ndarray, connectivity: int):
    mask = mask.astype(bool)
    struct = ndimage.generate_binary_structure(mask.ndim, connectivity)
    inst, num = ndimage.label(mask, structure=struct)
    return inst, num


def compute_iou_matrix(target_inst, pred_inst, num_target, num_pred):
    """Compute IoU matrix (num_target × num_pred)."""
    t = target_inst.ravel()
    p = pred_inst.ravel()

    # (gt_id, pred_id) → count
    pairs = np.stack([t, p], axis=1)
    uniq, cnt = np.unique(pairs, axis=0, return_counts=True)

    area_target = np.bincount(t, minlength=num_target + 1)
    area_pred = np.bincount(p, minlength=num_pred + 1)

    inter = np.zeros((num_target, num_pred), dtype=float)

    # fill intersection matrix - vectorized
    valid = (uniq[:, 0] > 0) & (uniq[:, 1] > 0)
    tids = uniq[valid, 0] - 1
    pids = uniq[valid, 1] - 1
    inter[tids, pids] = cnt[valid]

    union = area_target[1:, None] + area_pred[1:][None, :] - inter

    return inter / np.maximum(union, 1e-9)


def region_level_metrics_no_conf(
    pred: np.ndarray,
    target: np.ndarray,
    connectivity: int = 2,
    IoU_thresholds: list = list(np.arange(0.05, 0.51, 0.05)),
    method: str = "interp",
):
    """
    Get the Region-level Average Precision and F1 Score between a binary segmentation and the ground truth with the IoU as the confidence.

    :param pred: (numpy.array) a 2D or 3D binary image for segmentation.
    :param target: (numpy.array) a 2D or 2D binary image for ground truth.
    :param connectivity: (int) determines which elements of the output array belong
        to the structure, i.e., are considered as neighbors of the central element.
        Elements up to a squared distance of `connectivity` from the center are considered
        neighbors. `connectivity` may range from 1 (no diagonal elements are neighbors) to
        `ndim` (all elements are neighbors).
    :param IoU_threshold: (float) The threshold to decide whether a region is detected successfully.
    :param method: (str) The way to get PR curve and get AP.

    :return: The Region-level Average Precision, F1 Score
    """
    ndim = pred.ndim
    assert ndim in [1, 2, 3]
    assert np.all(0 <= np.array(IoU_thresholds)) and np.all(
        np.array(IoU_thresholds) <= 1
    )
    assert 1 <= connectivity <= ndim

    # label regions in the two binary masks
    pred_inst, num_pred = get_instances(pred, connectivity)
    target_inst, num_target = get_instances(target, connectivity)

    # no GT: AP deined as 1 if no pred, else 0
    all_0 = [0.0 for _ in range(len(IoU_thresholds))]
    all_1 = [1.0 for _ in range(len(IoU_thresholds))]
    if num_target == 0:
        return all_1, all_1 if num_pred == 0 else all_0, all_0
    if num_pred == 0:
        return all_0, all_0

    # IoU matrix
    iou = compute_iou_matrix(target_inst, pred_inst, num_target, num_pred)

    # Compute soft score for eac prediction
    ## score = max IoU with any GT
    scores = np.max(iou, axis=0) if num_pred > 0 else np.array([])

    # Sort prediction ids by score descending
    order = np.argsort(-scores)

    # Find best GT label for each pred
    ## best_iou_gt[p] = index of GT with highest IoU for pred p
    best_target = iou.argmax(axis=0)

    ap_lst = []
    f1_lst = []
    for IoU_threshold in IoU_thresholds:
        # whether IoU >= threshold
        valid = iou.max(axis=0) >= IoU_threshold

        # Greedy matching
        matched_target = -np.ones(num_target, dtype=int)
        best_target_sorted = best_target[order]
        valid_sorted = valid[order]

        first_hit = (matched_target[best_target_sorted] == -1) & valid_sorted
        matched_target[best_target_sorted[first_hit]] = 1

        TP = first_hit.astype(int)
        FP = 1 - TP

        # cumsum for PR curve
        tp_cum = np.cumsum(TP)
        fp_cum = np.cumsum(FP)

        recall = tp_cum / num_target
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

        # 101-point interpolation
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))

        if method == "interp":
            x = np.linspace(0, 1, 101)
            ap = np.trapz(np.interp(x, mrec, mpre), x)
        elif method == "searchsorted":
            q = np.zeros((101,))
            x = np.linspace(0, 1, 101)
            inds = np.searchsorted(recall, x, side="left")
            try:
                for ri, pi in enumerate(inds):
                    q[ri] = precision[pi]
            except:
                pass
            q_array = np.array(q)
            ap = np.mean(q_array[q_array > -1])
        elif method == "continous":
            i = np.where(mrec[1:] != mrec[:-1])[0]
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        else:
            raise ValueError("Unsupported value of 'method': {}".format(method))

        tp_count = tp_cum[-1]
        fp_count = fp_cum[-1]
        assert fp_count == (num_pred - tp_count)
        fn_count = num_target - tp_count

        precision_f1 = (
            tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
        )
        recall_f1 = (
            tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
        )
        f1 = (
            (2 * precision_f1 * recall_f1) / (precision_f1 + recall_f1)
            if (precision_f1 + recall_f1) > 0
            else 0.0
        )

        ap_lst.append(ap)
        f1_lst.append(f1)

    return ap_lst, f1_lst


def bootstrap_ci(per_case_values, n_boot=10000, ci=95, random_seed=319):
    """
    Compute bootstrap mean + confidence interval for per-case metrics.
    Args:
        per_case_values: (N,) array, one result of a metric per test case
        n_boot: number of bootstrap iterations
        ci: confidence interval percentage
    Returns:
        mean, lower_bound, upper_bound
    """
    np.random.seed(random_seed)
    metrics = np.asarray(per_case_values)
    n = len(metrics)

    boot_means = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        boot_means.append(metrics[idx].mean())

    boot_means = np.array(boot_means)
    alpha = (100 - ci) / 2
    lower = np.percentile(boot_means, alpha)
    upper = np.percentile(boot_means, 100 - alpha)

    return metrics.mean(), lower, upper


# if __name__ == "__main__":
#     import SimpleITK as sitk

#     pred_path = "./Logs/LNQ2023/only_P1/tversky_alpha_0.3_awce_beta_1.0_consis_weight_0.1_rampup_epoch_100_update_way_least_select_way_merge_num_prototype_2_memory_rate_0.999/fold_0/test_best/LNQ2023_0039.nii.gz"
#     target_path = "./Dataset/LNQ2023/labelsTs/LNQ2023_0039.nii.gz"

#     pred_array = sitk.GetArrayFromImage(sitk.ReadImage(pred_path))
#     target_array = sitk.GetArrayFromImage(sitk.ReadImage(target_path))

#     a = region_level_AP_no_conf(pred_array, target_array, 3, IoU_threshold=0.5, method="interp")
#     print(a)


if __name__ == "__main__":
    from wcode.utils.file_operations import open_json

    # json_path = "./Logs/LNQ2023Sparse/Hyper/tversky_alpha_0.3_awce_beta_1.0_consis_weight_1.2_rampup_epoch_100_update_way_least_select_way_merge_num_prototype_1_memory_rate_0.99/fold_0/test_best/summary.json"
    json_path = "./Logs/CTLymphNodes02Sparse/Ours/tversky_alpha_0.3_awce_beta_0.5_consis_weight_1.0_rampup_epoch_100_update_way_least_select_way_merge_num_prototype_3_memory_rate_0.999/summary_analysis.json"

    # per_case_metrics = open_json(json_path)["metric_per_case"]
    per_case_metrics = open_json(json_path)["z_metric_per_case"]

    dice_lst = []
    assd_lst = []
    ap_lst = []
    for case_name in per_case_metrics:
        dice_lst.append(per_case_metrics[case_name]["1"]["Dice"])
        assd_lst.append(per_case_metrics[case_name]["1"]["ASSD"])
        ap_lst.append(per_case_metrics[case_name]["1"]["AP@[0.05:0.05:0.50]"])

    dice_lst = np.array(dice_lst)
    assd_lst = np.array(assd_lst)
    ap_lst = np.array(ap_lst)

    dice_mean, dice_low, dice_high = bootstrap_ci(dice_lst, n_boot=10000, ci=95)
    assd_mean, assd_low, assd_high = bootstrap_ci(assd_lst, n_boot=10000, ci=95)
    ap_mean, ap_low, ap_high = bootstrap_ci(ap_lst, n_boot=10000, ci=95)

    print("DSC:", dice_mean, dice_low, dice_high)
    print("ASSD:", assd_mean, assd_low, assd_high)
    print("AP:", ap_mean, ap_low, ap_high)
