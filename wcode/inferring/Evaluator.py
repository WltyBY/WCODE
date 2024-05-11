import os
import multiprocessing
import numpy as np
import SimpleITK as sitk

from pprint import pprint
from time import sleep
from tqdm import tqdm

from wcode.inferring.utils.metrics import compute_tp_fp_fn_tn
from wcode.utils.file_operations import save_json


class Evaluator():
    """
    The cases in ground_truth_folder should contain cases who have the same file names in prediction_folder
    """
    def __init__(self, prediction_folder, ground_truth_folder, metrics, foreground_classes, files_ending, num_processes=8):
        self.prediction_folder = prediction_folder
        self.ground_truth_folder = ground_truth_folder
        self.metrics = metrics
        self.num_classes = foreground_classes + 1
        self.files_ending = files_ending
        self.num_processes = num_processes

    def compute_metrics(self, smooth=1e-5):
        pred_file_set = set([i for i in os.listdir(self.prediction_folder) if i.endswith(self.files_ending)])
        gt_file_set = set(os.listdir(self.ground_truth_folder))
        assert pred_file_set.issubset(gt_file_set), "predictions should be contained by ground truth."

        r = []
        with multiprocessing.get_context("spawn").Pool(self.num_processes) as p:
            for pred in pred_file_set:
                r.append(
                    p.starmap_async(
                        Evaluator.run_case,
                        (
                            (
                                self,
                                pred,
                                smooth
                            ),
                        ),
                    )
                )
            remaining = list(range(len(pred_file_set)))
            # p is pretty nifti. If we kill workers they just respawn but don't do any work.
            # So we need to store the original pool of workers.
            workers = [j for j in p._pool]
            with tqdm(
                desc=None, total=len(pred_file_set), disable=False
            ) as pbar:
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

        summary = {r[0]:r[1] for r in results}
        
        DSC_lst = [summary[key]["DSC"] for key in summary.keys()]

        summary["statistics"] = {}
        summary["statistics"]["DSC"] = {}
        summary["statistics"]["DSC"]["mean"] = np.mean(np.stack(DSC_lst), axis=0).tolist()
        summary["statistics"]["DSC"]["std"] = np.std(np.stack(DSC_lst), axis=0).tolist()
        pprint(summary)

        save_json(summary, os.path.join(self.prediction_folder, "summary.json"))

    def run_case(self, file_name, smooth):
        pred_obj = sitk.ReadImage(os.path.join(self.prediction_folder, file_name))
        gt_obj = sitk.ReadImage(os.path.join(self.ground_truth_folder, file_name))

        pred_array = sitk.GetArrayFromImage(pred_obj)
        gt_array = sitk.GetArrayFromImage(gt_obj)

        tp, fp, fn, tn = compute_tp_fp_fn_tn(pred_array, gt_array, self.num_classes)
        DSC = ((2 * tp + smooth) / (2 * tp + fp + fn + smooth)).tolist()

        return file_name, {"DSC": DSC}


if __name__ == "__main__":
    prediction_folder = "/media/x/Wlty/LymphNodes/Predictions/SegRap2023/CE_output_1_fold4"
    ground_truth_folder = "/media/x/Wlty/LymphNodes/Dataset/SegRap2023/labels"
    eva = Evaluator(prediction_folder, ground_truth_folder, None, foreground_classes=2, files_ending=".nii.gz")
    eva.compute_metrics()