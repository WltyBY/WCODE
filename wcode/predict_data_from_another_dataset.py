import os
import glob
import SimpleITK as sitk

from wcode.inferring.PatchBasedPredictor import PatchBasedPredictor
from wcode.inferring.Evaluator import Evaluator
from wcode.utils.file_operations import open_yaml

if __name__ == "__main__":
    original_img_folder = "./Dataset/HNTSMRG2024mid/images"
    prediction_save_folder = "./Predictions/HNTSMRG2024mid/fold3_best"
    gt_folder_path = "./Dataset_preprocessed/HNTSMRG2024mid/gt_segmentations"
    model_path = (
        "./Logs/HNTSMRG2024pre/HNTSMRG2024pre_oversample/fold_3/checkpoint_best.pth"
    )

    predict_configs = {
        "dataset_name": "HNTSMRG2024mid",
        "modality": [0],
        "fold": 3,
        "split": "val",
        "original_img_folder": original_img_folder,
        "predictions_save_folder": prediction_save_folder,
        "model_path": model_path,
        "device": {"gpu": [0]},
        "overwrite": True,
        "save_probabilities": False,
        "patch_size": [56, 224, 160],
        "tile_step_size": 0.5,
        "use_gaussian": True,
        "perform_everything_on_gpu": True,
        "use_mirroring": False,
        "allowed_mirroring_axes": None,
        "num_processes": 16,
    }
    config_dict = open_yaml("./Configs/HNTSMRG2024pre_oversample.yaml")
    config_dict["Inferring_settings"] = predict_configs
    predictor = PatchBasedPredictor(config_dict, allow_tqdm=True)
    evaluator = Evaluator(
        prediction_folder=prediction_save_folder,
        ground_truth_folder=gt_folder_path,
        dataset_yaml_or_its_path="./Dataset_preprocessed/HNTSMRG2024pre/dataset.yaml",
    )
    predictor.predict_from_file(
        predictor.original_img_folder,
        predictor.predictions_save_folder,
        predictor.modality,
        "3d",
    )
    evaluator.run()
