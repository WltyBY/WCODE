import os

from pprint import pprint

from wcode.inferring.PatchBasedPredictor import PatchBasedPredictor
from wcode.inferring.Evaluator import Evaluator
from wcode.utils.file_operations import open_yaml

if __name__ == "__main__":
    setting_file_name = "infer_config.yaml"
    settings_path = os.path.join("./Configs", setting_file_name)
    gt_folder_path = "./Dataset/HNTSMRG2024mid/labels"
    setting_yaml = open_yaml(settings_path)
    predictor = PatchBasedPredictor(settings_path, allow_tqdm=True, verbose=False)
    evaluator = Evaluator(
        prediction_folder=setting_yaml["Inferring_settings"]["predictions_save_folder"],
        ground_truth_folder=gt_folder_path,
        foreground_classes=setting_yaml["Network"]["out_channels"] - 1,
        files_ending=".nii.gz"
    )
    pprint(predictor.get_images_dict(predictor.original_img_folder, predictor.modality))
    predictor.predict_from_file(
        predictor.original_img_folder,
        predictor.predictions_save_folder,
        predictor.modality,
        "3d",
        predictor.save_probabilities,
    )
    evaluator.compute_metrics()
