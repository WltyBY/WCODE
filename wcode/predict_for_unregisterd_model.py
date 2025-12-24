import os
import torch
import argparse

from wcode.inferring.PatchBasedPredictor import PatchBasedPredictor
from wcode.inferring.Evaluator import Evaluator
from wcode.utils.file_operations import open_yaml
from wcode.inferring.utils.load_pretrain_weight import load_pretrained_weights

# the architecture of model
from wcode.training.Trainers.Weakly.Incomplete_Learning.TestTrainer.models import (
    DIVNet_v4,
    DIVNet_v4_second
    # DIVNet_CBAM,
    # DIVNet_CrossAttention,
)
from wcode.training.Trainers.Weakly.Incomplete_Learning.ReCo_I2P.models import UsedVNet
from wcode.training.Trainers.Weakly.Incomplete_Learning.AIO2.models import MeanTeacher
from wcode.training.Trainers.Weakly.Incomplete_Learning.SASN_IL.models import MeanTeacher as SASNMT
from wcode.training.Trainers.Weakly.NLL.Coteaching.models import BiNet


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="SegRap2023", help="Name of dataset")
parser.add_argument(
    "--settiing_file",
    type=str,
    default="SegRap2023.yaml",
    help="Name of setting files, or its absolute path",
)
parser.add_argument(
    "--gpu",
    type=int,
    default=0,
    help="gpu index",
)

parser.add_argument(
    "-i",
    type=str,
    default=None,
    help="folder path of input images",
)
parser.add_argument(
    "--gt_path",
    type=str,
    default=None,
    help="Path of ground truth. If given, will do evaluation after prediction",
)

parser.add_argument(
    "-o",
    type=str,
    default=None,
    help="folder path of save path",
)
parser.add_argument(
    "-m",
    type=str,
    default=None,
    help="saving path of using model",
)
parser.add_argument("-f", type=str, default=None, help="fold, can be None")
parser.add_argument("-s", type=str, default=None, help="split of data, can be None")
parser.add_argument(
    "--data_dim",
    type=str,
    default="3d",
    help="dim of data, 2d or 3d",
)
parser.add_argument(
    "--save_probabilities",
    type=bool,
    default=False,
    help="Save the predicted probabilities or not.",
)
args = parser.parse_args()


if __name__ == "__main__":
    config_dict = open_yaml(os.path.join("./Configs", args.settiing_file))

    predict_configs = {
        "dataset_name": args.dataset,
        "modality": "all",
        "fold": args.f,
        "split": args.s,
        "original_img_folder": args.i,
        "predictions_save_folder": args.o,
        "model_path": args.m,
        "device": {"gpu": [args.gpu]},
        "overwrite": True,
        "save_probabilities": args.save_probabilities,
        "patch_size": config_dict["Training_settings"]["patch_size"],
        "tile_step_size": 0.5,
        "use_gaussian": True,
        "perform_everything_on_gpu": True,
        "use_mirroring": True,
        "allowed_mirroring_axes": [0, 1] if args.data_dim == "2d" else [0, 1, 2],
        "num_processes": 16,
    }

    config_dict["Inferring_settings"] = predict_configs
    predictor = PatchBasedPredictor(config_dict, allow_tqdm=True)
    if args.o is not None and not os.path.exists(args.o):
        os.makedirs(args.o)

    # model = MeanTeacher(config_dict["Network"])
    # model = SASNMT(config_dict["Network"])
    # model = DIVNet_v4_second(
    #     config_dict["Network"],
    #     num_prototype=2,
    #     memory_rate=0.999,
    #     update_way="least",
    #     select_way="merge",
    # )
    model = BiNet(config_dict["Network"])
    # model = UsedVNet(config_dict["Network"], num_prototype=1, memory_rate=0.99)

    load_pretrained_weights(model, args.m, load_all=True, verbose=True)
    model.to(predictor.device)
    model = torch.compile(model)
    predictor.manual_initialize(model, config_dict["Network"]["out_channels"])
    predictor.predict_from_file(
        predictor.original_img_folder,
        predictor.predictions_save_folder,
        predictor.modality,
        args.data_dim,
        args.save_probabilities,
    )
    if isinstance(args.gt_path, str):
        evaluator = Evaluator(
            prediction_folder=args.o,
            ground_truth_folder=args.gt_path,
            dataset_yaml_or_its_path=f"./Dataset_preprocessed/{args.dataset}/dataset.yaml",
        )
        evaluator.run()
