import os
import numpy as np
import SimpleITK as sitk

from pprint import pprint

from wcode.inferring.utils.metrics import compute_tp_fp_fn_tn
from wcode.utils.file_operations import save_json


class Evaluator():
    """
    The cases in ground_truth_folder should contain cases who have the same file names in prediction_folder
    """
    def __init__(self, prediction_folder, ground_truth_folder, metrics, foreground_classes, files_ending):
        self.prediction_folder = prediction_folder
        self.ground_truth_folder = ground_truth_folder
        self.metrics = metrics
        self.num_classes = foreground_classes + 1
        self.files_ending = files_ending

    def compute_metrics(self, smooth=1e-5):
        pred_file_set = set([i for i in os.listdir(self.prediction_folder) if i.endswith(self.files_ending)])
        gt_file_set = set(os.listdir(self.ground_truth_folder))
        assert pred_file_set.issubset(gt_file_set), "predictions should be contained by ground truth."

        summary = {}
        DSC_lst = []
        for pred in pred_file_set:
            print("Evaluating:", pred)
            summary[pred] = {}

            pred_obj = sitk.ReadImage(os.path.join(self.prediction_folder, pred))
            gt_obj = sitk.ReadImage(os.path.join(self.ground_truth_folder, pred))

            pred_array = sitk.GetArrayFromImage(pred_obj)
            gt_array = sitk.GetArrayFromImage(gt_obj)

            tp, fp, fn, tn = compute_tp_fp_fn_tn(pred_array, gt_array, self.num_classes)
            summary[pred]["DSC"] = ((2 * tp + smooth) / (2 * tp + fp + fn + smooth)).tolist()
            DSC_lst.append(summary[pred]["DSC"])
            
        summary["statistics"] = {}
        summary["statistics"]["DSC"] = {}
        summary["statistics"]["DSC"]["mean"] = np.mean(np.stack(DSC_lst), axis=0).tolist()
        summary["statistics"]["DSC"]["std"] = np.std(np.stack(DSC_lst), axis=0).tolist()
        pprint(summary)

        save_json(summary, os.path.join(self.prediction_folder, "summary.json"))


if __name__ == "__main__":
    prediction_folder = "/media/x/Wlty/LymphNodes/Predictions/SegRap2023/output_whole_Only_1"
    ground_truth_folder = "/media/x/Wlty/LymphNodes/Dataset/SegRap2023/labels"
    eva = Evaluator(prediction_folder, ground_truth_folder, None, foreground_classes=2, files_ending=".nii.gz")
    eva.compute_metrics()