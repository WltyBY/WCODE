import os
import torch

from datetime import datetime
from time import time
from torch.amp import GradScaler

from wcode.training.loss.CompoundLoss import Tversky_and_CE_loss
from wcode.training.learning_rate.CosineAnnealingLR import CosineAnnealingLRScheduler
from wcode.utils.file_operations import open_yaml, copy_file_to_dstFolder, open_json
from wcode.inferring.utils.load_pretrain_weight import load_pretrained_weights
from wcode.training.Trainers.Fully.PatchBasedTrainer.PatchBasedTrainer import (
    PatchBasedTrainer,
)
from wcode.training.Trainers.Fully.TestTrainer.model import PlainQueryUNet
from wcode.inferring.PatchBasedPredictor import PatchBasedPredictor
from wcode.inferring.NaturalImagePredictor import NaturalImagePredictor
from wcode.inferring.Evaluator import Evaluator
from wcode.utils.data_io import file_endings_for_2d_img, file_endings_for_sitk


class TestTrainer(PatchBasedTrainer):
    def __init__(
        self,
        training_args,
        verbose: bool = True,
    ):
        self.verbose = verbose
        config_file_path = os.path.join("./Configs", training_args.setting)
        self.config_dict = open_yaml(config_file_path)

        self.get_train_settings(training_args)
        self.device = self.get_device()

        # Task-general params
        task_general_names = "BS_{}_GPU_NUM_{}_EPOCH_{}_SEED_{}_PRETRAINED_{}".format(
            self.batch_size,
            self.world_size,
            self.num_epoch,
            self.random_seed,
            self.pretrained_weight is not None,
        )

        # hyperparameter
        self.w_ce = training_args.w_ce
        self.w_dice = training_args.w_dice
        self.w_class = training_args.w_class
        hyperparams_name = "w_ce_{}_w_dice_{}_w_class_{}".format(
            self.w_ce, self.w_dice, self.w_class
        )

        self.allow_mirroring_axes_during_inference = None

        self.was_initialized = False
        self._best_ema = None

        timestamp = datetime.now()
        time_ = "Train_Log_%d_%d_%d_%02.0d_%02.0d_%02.0d" % (
            timestamp.year,
            timestamp.month,
            timestamp.day,
            timestamp.hour,
            timestamp.minute,
            timestamp.second,
        )
        assert self.method_name is not None
        self.logs_output_folder = os.path.join(
            "./Log",
            self.dataset_name,
            self.preprocess_config.upper() + "__" + self.method_name,
            task_general_names,
            hyperparams_name,
            "fold_" + self.fold,
        )
        if not os.path.exists(self.logs_output_folder):
            os.makedirs(self.logs_output_folder, exist_ok=True)

        self.log_file = os.path.join(self.logs_output_folder, time_ + ".txt")
        with open(self.log_file, "w"):
            pass

        self.print_to_log_file(
            f"Using device: {self.device} | DDP: {self.is_ddp} | rank {self.rank}/{self.world_size}"
        )

        # Save the config file and Trainer file to the logs folder
        config_and_code_save_path = os.path.join(
            self.logs_output_folder, "Config_and_code"
        )
        script_path = os.path.abspath(__file__)

        if self.is_main_process():
            os.makedirs(config_and_code_save_path, exist_ok=True)

            # copy the config file to the logs folder
            copy_file_to_dstFolder(config_file_path, config_and_code_save_path)

            # copy the trainer file to the logs folder
            copy_file_to_dstFolder(script_path, config_and_code_save_path)

            self.print_to_log_file(
                "Training logs will be saved in:", self.logs_output_folder
            )

        self.logger = self.get_logger()

        self.current_epoch = 0

        # checkpoint saving stuff
        self.save_every = 1
        self.disable_checkpointing = False

        self.grad_scaler = GradScaler() if self.device.type == "cuda" else None

        if self.continue_train:
            checkpoint = os.path.join(self.logs_output_folder, "checkpoint_latest.pth")
            self.sync_processes()
            if not os.path.isfile(checkpoint):
                raise FileNotFoundError(
                    f"Continue training was requested but checkpoint not found: {checkpoint}"
                )
            self.load_checkpoint(checkpoint)
        self.sync_processes()

    def initialize(self):
        if not self.was_initialized:
            self.init_random()
            self.setting_check()

            # build network
            self.network = self.get_networks(self.config_dict["Network"])
            if self.pretrained_weight is not None:
                if self.is_main_process():
                    self.print_to_log_file(
                        f"Loading pretrained weight from {self.pretrained_weight}"
                    )
                load_pretrained_weights(self.network, self.pretrained_weight)

            self.network.to(self.device)

            if self.is_ddp:
                self.network = self.convert_bn2syncbn(self.network)
                self.network = torch.nn.parallel.DistributedDataParallel(
                    self.network,
                    device_ids=[self.device.index],
                    output_device=self.device.index,
                )

            if self.is_main_process():
                self.print_to_log_file("Compiling network...")
            self.network = torch.compile(self.network)

            # optimizer & scheduler
            self.do_deep_supervision = self.config_dict["Network"]["deep_supervision"]
            self.optimizer, self.lr_scheduler = self.get_optimizers()

            # losses
            self.train_loss = Tversky_and_CE_loss(
                {
                    "batch_dice": True,
                    "alpha": 0.5,
                    "beta": 0.5,
                    "smooth": 1e-5,
                    "do_bg": False,
                    "ddp": self.is_ddp,
                    "apply_nonlin": True,
                },
                (
                    {"weight": (torch.tensor(self.w_class, device=self.device))}
                    if self.w_class
                    else {}
                ),
                weight_ce=self.w_ce,
                weight_tversky=self.w_dice,
                ignore_label=self.ignore_label,
            )

            self.val_loss = Tversky_and_CE_loss(
                {
                    "batch_dice": True,
                    "smooth": 1e-5,
                    "do_bg": True,
                    "ddp": self.is_ddp,
                    "apply_nonlin": True,
                },
                {},
                weight_ce=1,
                weight_tversky=1,
                ignore_label=self.ignore_label,
            )
            if self.do_deep_supervision:
                self.train_loss = self._build_deep_supervision_loss_object(
                    self.train_loss
                )
                self.val_loss = self._build_deep_supervision_loss_object(self.val_loss)

            self.was_initialized = True
        else:
            raise RuntimeError(
                "self.initialize() should only be called once. "
                "Or initialization was done before initialize method???"
            )

    def get_networks(self, network_settings):
        network = PlainQueryUNet(
            input_channels=network_settings["input_channels"],
            n_stages=network_settings["n_stages"],
            features_per_stage=network_settings["features_per_stage"],
            conv_op=torch.nn.Conv3d,
            kernel_sizes=network_settings["kernel_sizes"],
            strides=network_settings["strides"],
            n_conv_per_stage=network_settings["n_conv_per_stage"],
            num_classes=network_settings["num_classes"],
            n_conv_per_stage_decoder=network_settings["n_conv_per_stage_decoder"],
            query_dim=network_settings["query_dim"],
            num_transformer_layers=network_settings["num_transformer_layers"],
            num_heads=network_settings["num_heads"],
            conv_bias=network_settings["conv_bias"],
            norm_op=torch.nn.InstanceNorm3d,
            norm_op_kwargs=network_settings["norm_op_kwargs"],
            nonlin=torch.nn.LeakyReLU,  # network_settings['nonlin'], --- IGNORE ---
            nonlin_kwargs=network_settings["nonlin_kwargs"],
        )
        PlainQueryUNet.initialize(network)
        return network

    def get_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            self.base_lr,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        )
        lr_scheduler = CosineAnnealingLRScheduler(
            optimizer=optimizer, max_steps=self.num_epoch
        )
        return optimizer, lr_scheduler

    def perform_actual_validation(self, save_probabilities: bool = False):
        if not self.is_main_process():
            return

        self.print_to_log_file("----------Perform actual validation----------")

        # --------------------
        # Paths & basic setup
        # --------------------
        dataset_path = os.path.join("./Dataset_preprocessed", self.dataset_name)
        original_img_folder = os.path.join(
            dataset_path, f"preprocessed_datas_{self.preprocess_config}"
        )

        predictions_save_folder = os.path.join(self.logs_output_folder, "validation")
        final_ckpt_path = os.path.join(self.logs_output_folder, "checkpoint_final.pth")
        best_ckpt_path = os.path.join(self.logs_output_folder, "checkpoint_best.pth")

        # --------------------
        # Report best epoch (for logging only)
        # --------------------
        best_saved_model = torch.load(best_ckpt_path, weights_only=False)
        self.print_to_log_file(
            "Pseudo best model selected from epoch {}".format(
                best_saved_model["current_epoch"] - 1
            )
        )
        del best_saved_model

        # --------------------
        # Report which model is used for validation
        # --------------------
        self.print_to_log_file(
            f"Performing actual validation using FINAL checkpoint "
            f"(trained until epoch {self.current_epoch - 1})"
        )
        self.print_to_log_file(f"Checkpoint file: {final_ckpt_path}")

        # --------------------
        # Predictor configuration
        # --------------------
        if self.is_ddp:
            infer_device = torch.device("cuda:0")
            torch.cuda.set_device(infer_device)
        else:
            infer_device = self.device
        
        self.network = self.get_networks(self.config_dict["Network"])
        load_pretrained_weights(self.network, final_ckpt_path, load_all=True)
        self.network.to(infer_device)
        self.print_to_log_file("Compiling network for actual validation...")
        self.network = torch.compile(self.network)

        predict_configs = {
            "dataset_name": self.dataset_name,
            "modality": self.modality,
            "fold": self.fold,
            "split": "val",
            "original_img_folder": original_img_folder,
            "predictions_save_folder": predictions_save_folder,
            "model_path": final_ckpt_path,
            "device": infer_device,
            "overwrite": True,
            "patch_size": self.patch_size,
            "tile_step_size": 0.5,
            "use_gaussian": True,
            "perform_everything_on_gpu": True,
            "use_mirroring": True,
            "allowed_mirroring_axes": self.allow_mirroring_axes_during_inference,
            "num_workers": self.num_workers,
        }
        self.config_dict["Inferring_settings"] = predict_configs

        # --------------------
        # Build validation file list
        # --------------------
        dataset_split = open_json(
            os.path.join(
                "./Dataset_preprocessed", self.dataset_name, "dataset_split.json"
            )
        )

        data_path_list = [
            i
            for i in os.listdir(original_img_folder)
            if i.endswith(".npy") and not i.endswith("_seg.npy")
        ]

        val_ids = dataset_split["0" if self.fold == "all" else self.fold]["val"]
        validation_data_file = sorted(
            [f for f in data_path_list if f.split(".")[0] in val_ids]
        )

        validation_data_path = [
            os.path.join(original_img_folder, f) for f in validation_data_file
        ]
        validation_pkl_path = [
            os.path.join(original_img_folder, f.replace(".npy", ".pkl"))
            for f in validation_data_file
        ]
        predictions_save_path = [
            os.path.join(predictions_save_folder, f.replace(".npy", ""))
            for f in validation_data_file
        ]

        iter_lst = [
            {
                "data": data,
                "output_file": out,
                "data_properites": prop,
            }
            for data, out, prop in zip(
                validation_data_path,
                predictions_save_path,
                validation_pkl_path,
            )
        ]

        # --------------------
        # Run prediction
        # --------------------
        if self.dataset_yaml["files_ending"] in file_endings_for_sitk:
            predictor = PatchBasedPredictor(
                self.config_dict, allow_tqdm=True, verbose=False
            )
            predictor.manual_initialize(self.network, self.config_dict["Network"]["out_channels"]
            )
            self.print_to_log_file("Start predicting using PatchBasedPredictor.")
            start = time()
            predictor.predict_from_data_iterator(
                data_iterator=iter_lst,
                predict_way=self.preprocess_config,
                save_or_return_probabilities=save_probabilities,
            )
        elif self.dataset_yaml["files_ending"] in file_endings_for_2d_img:
            predictor = NaturalImagePredictor(
                self.config_dict, allow_tqdm=True, verbose=False
            )
            predictor.manual_initialize(
                self.network, self.config_dict["Network"]["out_channels"]
            )
            self.print_to_log_file("Start predicting using NaturalImagePredictor.")
            start = time()
            predictor.predict_from_data_iterator(
                data_iterator=iter_lst,
                save_vis_mask=True,
                save_or_return_probabilities=save_probabilities,
            )
        else:
            raise ValueError("Unsupported file extension.")

        self.print_to_log_file(f"Predicting ends. Cost: {time() - start:.2f}s")

        # --------------------
        # Evaluation
        # --------------------
        ground_truth_folder = os.path.join(dataset_path, "gt_segmentations")
        evaluator = Evaluator(
            predictions_save_folder,
            ground_truth_folder,
            dataset_yaml_or_its_path=self.dataset_yaml,
            num_workers=min(8, self.num_workers) if self.is_ddp else self.num_workers,
        )
        evaluator.run()
        self.print_to_log_file("Evaluating ends.")