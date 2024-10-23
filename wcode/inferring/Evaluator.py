import os
import cv2
import multiprocessing
import numpy as np
import SimpleITK as sitk

from pprint import pprint
from time import sleep
from tqdm import tqdm

from wcode.inferring.utils.metrics import compute_tp_fp_fn_tn, region_or_label_to_mask
from wcode.utils.file_operations import save_json, open_yaml
from wcode.utils.json_export import recursive_fix_for_json_export
from wcode.utils.data_io import files_ending_for_sitk, files_ending_for_2d_img


class Evaluator:
    """
    The cases in ground_truth_folder should contain cases who have the same file names in prediction_folder
    """

    def __init__(
        self,
        prediction_folder,
        ground_truth_folder,
        dataset_yaml_or_its_path,
        num_processes=16,
    ):
        self.prediction_folder = prediction_folder
        self.ground_truth_folder = ground_truth_folder
        self.dataset_yaml = dataset_yaml_or_its_path
        if isinstance(self.dataset_yaml, str):
            self.dataset_yaml = open_yaml(dataset_yaml_or_its_path)
        self.value_fg_class = [i for i in self.dataset_yaml["labels"].values()][1:]
        self.files_ending = self.dataset_yaml["files_ending"]
        self.num_processes = num_processes

    def run(self):
        pred_file_set = set(
            [
                i
                for i in os.listdir(self.prediction_folder)
                if i.endswith(self.files_ending)
            ]
        )
        gt_file_set = set(os.listdir(self.ground_truth_folder))
        assert pred_file_set.issubset(
            gt_file_set
        ), "predictions should be contained by ground truth."

        r = []
        with multiprocessing.get_context("spawn").Pool(self.num_processes) as p:
            for pred in pred_file_set:
                prediction_file = os.path.join(self.prediction_folder, pred)
                reference_file = os.path.join(self.ground_truth_folder, pred)
                if self.files_ending in files_ending_for_sitk:
                    pred_obj = sitk.ReadImage(prediction_file)
                    gt_obj = sitk.ReadImage(reference_file)

                    pred_array = sitk.GetArrayFromImage(pred_obj)
                    gt_array = sitk.GetArrayFromImage(gt_obj)
                elif self.files_ending in files_ending_for_2d_img:
                    pred_img = cv2.imread(prediction_file)
                    # gt_array = cv2.imread(reference_file).transpose(2, 0, 1)
                    gt_img = cv2.imread(reference_file)
                    pred_array = np.zeros_like(pred_img)
                    gt_array = np.zeros_like(pred_img)
                    for label_value, pixel_value in enumerate(self.value_fg_class):
                        pred_array[pred_img == np.array(pixel_value)] = label_value + 1
                        gt_array[gt_img == np.array(pixel_value)] = label_value + 1
                    pred_array, gt_array = (
                        pred_array.transpose(2, 0, 1)[0],
                        gt_array.transpose(2, 0, 1)[0],
                    )

                r.append(
                    p.starmap_async(
                        Evaluator.compute_case_level_metrics,
                        ((self, pred_array, gt_array, pred),),
                    ),
                )
            remaining = list(range(len(pred_file_set)))
            # p is pretty nifti. If we kill workers they just respawn but don't do any work.
            # So we need to store the original pool of workers.
            workers = [j for j in p._pool]
            with tqdm(desc=None, total=len(pred_file_set), disable=False) as pbar:
                while len(remaining) > 0:
                    all_alive = all([j.is_alive() for j in workers])
                    if not all_alive:
                        raise RuntimeError
                    done = [i for i in remaining if r[i].ready()]
                    for _ in done:
                        pbar.update()
                    remaining = [i for i in remaining if i not in done]
                    sleep(0.1)
        results = [i.get()[0] for i in r]

        summary_case_level = {r[0]: r[1] for r in results}

        # mean metric per class
        metric_list = list(results[0][1][1].keys())
        means = {}
        for r in range(1, len(self.value_fg_class) + 1):
            means[r] = {}
            # compute the mean of some normal metrics
            for m in metric_list:
                means[r][m] = np.nanmean([i[1][r][m] for i in results])

            # compute dataset-level metrics
            # compute aggregated DSC
            tp_sum = np.sum([i[1][r]["TP"] for i in results])
            pred_sum = np.sum([i[1][r]["n_pred"] for i in results])
            gt_sum = np.sum([i[1][r]["n_gt"] for i in results])
            means[r]["DSC_agg"] = (2 * tp_sum) / (pred_sum + gt_sum)
        metric_list.append("DSC_agg")

        # foreground mean
        foreground_mean = {}
        for m in metric_list:
            values = []
            for k in means.keys():
                if k == 0 or k == "0":
                    continue
                values.append(means[k][m])
            foreground_mean[m] = np.mean(values)

        recursive_fix_for_json_export(summary_case_level)
        recursive_fix_for_json_export(means)
        recursive_fix_for_json_export(foreground_mean)
        summary = {
            "metric_per_case": summary_case_level,
            "mean": means,
            "foreground_mean": foreground_mean,
        }
        pprint(summary)
        save_json(summary, os.path.join(self.prediction_folder, "summary.json"))

    def compute_case_level_metrics(self, pred_array, gt_array, prediction_file):
        results = {}
        for r in range(1, len(self.value_fg_class) + 1):
            results[r] = {}
            pred_mask = region_or_label_to_mask(pred_array, r)
            gt_mask = region_or_label_to_mask(gt_array, r)
            # print(pred_mask.shape, gt_mask.shape)
            tp, fp, fn, tn = compute_tp_fp_fn_tn(pred_mask, gt_mask)
            if tp + fp + fn == 0:
                results[r]["Dice"] = np.nan
                results[r]["IoU"] = np.nan
            else:
                results[r]["Dice"] = 2 * tp / (2 * tp + fp + fn)
                results[r]["IoU"] = tp / (tp + fp + fn)

            results[r]["Accuracy"] = (tp + tn) / (tp + tn + fp + fn)

            if tp + fp == 0:
                results[r]["Precision"] = np.nan
            else:
                results[r]["Precision"] = tp / (tp + fp)

            if tp + fn == 0:
                results[r]["Recall"] = np.nan
            else:
                results[r]["Recall"] = tp / (tp + fn)

            if np.nan in [results[r]["Precision"], results[r]["Recall"]]:
                results[r]["F1 score"] = np.nan
            else:
                if results[r]["Precision"] + results[r]["Recall"] == 0:
                    results[r]["F1 score"] = np.nan
                else:
                    results[r]["F1 score"] = (
                        2 * results[r]["Precision"] * results[r]["Recall"]
                    ) / (results[r]["Precision"] + results[r]["Recall"])
            results[r]["FP"] = fp
            results[r]["TP"] = tp
            results[r]["FN"] = fn
            results[r]["TN"] = tn
            results[r]["n_pred"] = fp + tp
            results[r]["n_gt"] = fn + tp

        return prediction_file, results


if __name__ == "__main__":
    prediction_folder = "./Logs/MoNuSegFully/MoNuSegUpper_bound/fold_0/validation"
    ground_truth_folder = "./Dataset_preprocessed/MoNuSegFully/gt_segmentations"
    dataset_yaml = "./Dataset_preprocessed/MoNuSegFully/dataset.yaml"
    eva = Evaluator(
        prediction_folder,
        ground_truth_folder,
        dataset_yaml,
    )
    eva.run()
