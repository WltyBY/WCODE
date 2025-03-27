from wcode.inferring.PatchBasedPredictor import PatchBasedPredictor
from wcode.inferring.Evaluator import Evaluator
from wcode.utils.file_operations import open_yaml

if __name__ == "__main__":
    original_img_folder = "./Dataset/RAOSset3/images"
    prediction_save_folder = "./Predictions/RAOSset3/fold0"
    gt_folder_path = "./Dataset_preprocessed/RAOSset3/gt_segmentations"
    model_path = (
        "./Logs/RAOS/RAOS_Base/fold_0/checkpoint_final.pth"
    )

    predict_configs = {
        "dataset_name": "RAOSset3",
        "modality": [0],
        "fold": "0",
        "split": "val",
        "original_img_folder": original_img_folder,
        "predictions_save_folder": prediction_save_folder,
        "model_path": model_path,
        "device": {"gpu": [0]},
        "overwrite": True,
        "save_probabilities": False,
        "patch_size": [96, 112, 160],
        "tile_step_size": 0.5,
        "use_gaussian": True,
        "perform_everything_on_gpu": True,
        "use_mirroring": True,
        "allowed_mirroring_axes": [0, 1, 2],
        "num_processes": 16,
    }
    config_dict = open_yaml("./Configs/RAOS_Base.yaml")
    config_dict["Inferring_settings"] = predict_configs
    predictor = PatchBasedPredictor(config_dict, allow_tqdm=True)
    predictor.initialize()
    evaluator = Evaluator(
        prediction_folder=prediction_save_folder,
        ground_truth_folder=gt_folder_path,
        dataset_yaml_or_its_path="./Dataset_preprocessed/RAOS/dataset.yaml",
    )
    predictor.predict_from_file(
        predictor.original_img_folder,
        predictor.predictions_save_folder,
        predictor.modality,
        "3d",
    )
    evaluator.run()
