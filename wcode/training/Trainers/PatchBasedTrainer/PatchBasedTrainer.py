import os
import sys
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn

from datetime import datetime
from time import time, sleep
from torch import autocast
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch._dynamo import OptimizedModule
from monai.transforms import (
    Compose,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    RandAdjustContrastd,
    ToTensor,
)

from wcode.net.VNet import VNet
from wcode.training.dataset.BasedDataset import BasedDataset
from wcode.training.loss.compound_loss import Tversky_and_CE_loss
from wcode.training.loss.deep_supervision import DeepSupervisionWeightedSummator
from wcode.training.logs_writer.logger import logger
from wcode.training.Collater import PatchBasedCollater
from wcode.training.learning_rate.PolyLRScheduler import PolyLRScheduler
from wcode.training.metrics import get_tp_fp_fn_tn
from wcode.utils.file_operations import open_yaml, copy_file_to_dstFolder
from wcode.utils.others import empty_cache, dummy_context
from wcode.utils.collate_outputs import collate_outputs
from wcode.inferring.PatchBasedPredictor import PatchBasedPredictor
from wcode.inferring.Evaluator import Evaluator


class PatchBasedTrainer(object):

    def __init__(self, config_file_path: str, fold: int):
        self.config_dict = open_yaml(config_file_path)
        del self.config_dict["Inferring_settings"]
        self.get_train_settings(self.config_dict["Training_settings"])
        self.fold = fold

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
        log_folder_name = self.method_name if self.method_name is not None else time_
        self.logs_output_folder = os.path.join(
            "./Logs",
            self.dataset_name,
            log_folder_name,
            "fold_" + str(self.fold),
        )
        config_and_code_save_path = os.path.join(
            self.logs_output_folder, "Config_and_code"
        )
        if not os.path.exists(config_and_code_save_path):
            os.makedirs(config_and_code_save_path)
        print("Training logs will be saved in:", self.logs_output_folder)

        # copy the config file to the logs folder
        copy_file_to_dstFolder(config_file_path, config_and_code_save_path)

        # copy the trainer file to the logs folder
        script_path = os.path.abspath(__file__)
        copy_file_to_dstFolder(script_path, config_and_code_save_path)

        self.log_file = os.path.join(self.logs_output_folder, time_ + ".txt")
        self.logger = logger()

        self.current_epoch = 0

        # checkpoint saving stuff
        self.save_every = 1
        self.disable_checkpointing = False

        self.device = self.get_device()
        self.grad_scaler = GradScaler() if self.device.type == "cuda" else None

        if self.checkpoint_path is not None:
            self.load_checkpoint(self.checkpoint_path)

    def get_train_settings(self, training_setting_dict):
        self.dataset_name = training_setting_dict["dataset_name"]
        self.modality = training_setting_dict["modality"]
        self.method_name = training_setting_dict["method_name"]
        self.device_dict = training_setting_dict["device"]
        self.num_epochs = training_setting_dict["epoch"]
        self.tr_iterations_per_epoch = training_setting_dict["tr_iterations_per_epoch"]
        self.val_iterations_per_epoch = training_setting_dict[
            "val_iterations_per_epoch"
        ]
        self.batch_size = training_setting_dict["batch_size"]
        self.patch_size = training_setting_dict["patch_size"]
        self.base_lr = training_setting_dict["base_lr"]
        self.weight_decay = training_setting_dict["weight_decay"]
        self.num_processes = training_setting_dict["num_processes"]
        self.deterministic = training_setting_dict["deterministic"]
        self.random_seed = training_setting_dict["seed"]
        self.oversample_rate = training_setting_dict["oversample_rate"]
        self.probabilistic_oversampling = training_setting_dict[
            "probabilistic_oversampling"
        ]
        self.ignore_label = training_setting_dict["ignore_label"]
        self.checkpoint_path = training_setting_dict["checkpoint"]

    def get_device(self):
        assert len(self.device_dict.keys()) == 1, "Device can only be GPU or CPU"

        if "gpu" in self.device_dict.keys():
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                str(i) for i in self.device_dict["gpu"]
            )
            # If os.environ['CUDA_VISIBLE_DEVICES'] are not used, some process with the same PID will run on another CUDA device.
            # For example, I have a device with 4 GPU. When I run on GPU0, there would be a process with the same PID on maybe GPU1 (a little gpu memory usage).
            # When use os.environ['CUDA_VISIBLE_DEVICES'] with just one GPU device, the device in torch must set to "cuda:0".
            if len(self.device_dict["gpu"]) == 1:
                device = torch.device(type="cuda", index=0)
            else:
                raise Exception("The number of gpu should >= 1.")
        elif "cpu" in self.device_dict.keys():
            device = torch.device(type="cpu")
        else:
            raise Exception("The device in training process can be gpu or cpu")

        print(f"Using device: {device}")
        return device

    def initialize(self):
        if not self.was_initialized:
            self.is_ddp = False
            self.init_random()
            self.network = self.get_networks(self.config_dict["Network"]).to(
                self.device
            )

            self.print_to_log_file("Compiling network...")
            self.network = torch.compile(self.network)

            self.do_deep_supervision = self.config_dict["Network"]["deep_supervision"]
            self.optimizer, self.lr_scheduler = self.get_optimizers()

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
                {},
                weight_ce=1,
                weight_tversky=1,
                ignore_label=self.ignore_label,
            )
            if self.do_deep_supervision:
                self.train_loss = self._build_deep_supervision_loss_object(
                    self.train_loss
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

            self.was_initialized = True
        else:
            raise Exception("Initialization was done before initialize method???")

    def _build_deep_supervision_loss_object(self, loss):
        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss.
        # When writing model's code, we assume that its multi-scales predictions range from high resolution to low resolution
        weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
        weights[-1] = 0

        # Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # Restructuring the loss
        loss = DeepSupervisionWeightedSummator(loss, weights)

        return loss

    def _get_deep_supervision_scales(self):
        deep_supervision_scales = len(self.config_dict["Network"]["pool_kernel_size"])
        return deep_supervision_scales

    def init_random(self):
        if self.deterministic:
            cudnn.benchmark = False
            cudnn.deterministic = True
        else:
            cudnn.benchmark = True
            cudnn.deterministic = False
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)

    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):
        timestamp = time()
        dt_object = datetime.fromtimestamp(timestamp)

        if add_timestamp:
            args = ("%s:" % dt_object, *args)

        successful = False
        max_attempts = 5
        ctr = 0
        while not successful and ctr < max_attempts:
            try:
                with open(self.log_file, "a+") as f:
                    for a in args:
                        f.write(str(a))
                        f.write(" ")
                    f.write("\n")
                successful = True
            except IOError:
                print(
                    "%s: failed to log: " % datetime.fromtimestamp(timestamp),
                    sys.exc_info(),
                )
                sleep(0.5)
                ctr += 1
        if also_print_to_console:
            print(*args)

    def get_networks(self, network_settings):
        return VNet(network_settings)

    def get_optimizers(self):
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            self.base_lr,
            weight_decay=self.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
        lr_scheduler = PolyLRScheduler(optimizer, self.base_lr, self.num_epochs)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=1e-5)
        return optimizer, lr_scheduler

    def get_train_and_val_transform(self):
        train_transform = Compose(
            [
                RandFlipd(keys=["image", "label"], prob=0.2),
                # RandRotate90d(keys=["image", "label"], prob=0.2, spatial_axes=(1, 2)),
                RandGaussianNoised(keys=["image"], prob=0.1),
                RandGaussianSmoothd(
                    ["image"], (0.5, 1.0), (0.5, 1.0), (0.5, 1.0), prob=0.2
                ),
                RandAdjustContrastd(
                    keys=["image"], prob=0.1, gamma=(0.7, 1.5), invert_image=True
                ),
                RandAdjustContrastd(
                    keys=["image"], prob=0.3, gamma=(0.7, 1.5), invert_image=False
                ),
                ToTensor(),
            ]
        )
        val_transform = Compose([ToTensor()])

        return train_transform, val_transform

    def get_train_and_val_dataset(self):
        train_dataset = BasedDataset(
            self.dataset_name,
            split="train",
            fold="fold" + str(self.fold),
            modality=self.modality,
        )

        val_dataset = BasedDataset(
            self.dataset_name,
            split="val",
            fold="fold" + str(self.fold),
            modality=self.modality,
        )
        return train_dataset, val_dataset

    def get_collator(self):
        train_transform, val_transform = self.get_train_and_val_transform()
        train_collator = PatchBasedCollater(
            self.patch_size,
            self.do_deep_supervision,
            self.config_dict["Network"]["pool_kernel_size"],
            self.oversample_rate,
            self.probabilistic_oversampling,
            train_transform,
        )
        val_collator = PatchBasedCollater(
            self.patch_size,
            False,
            None,
            self.oversample_rate,
            self.probabilistic_oversampling,
            val_transform,
        )
        return train_collator, val_collator

    def worker_init_fn(self, worker_id):
        random.seed(self.random_seed + worker_id)

    def get_train_and_val_dataloader(self):
        train_collator, val_collator = self.get_collator()
        train_dataset, val_dataset = self.get_train_and_val_dataset()
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_processes,
            pin_memory=True,
            persistent_workers=True,
            worker_init_fn=self.worker_init_fn,
            collate_fn=train_collator,
        )
        # shuffle is True here, because we expect patch based validation to be more comprehensive.
        # If the number of validation iteration is smaller than the valset, the validation
        # throughout the entire training process cannot cover all the images in the valset.
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_processes,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=val_collator,
        )
        return train_loader, val_loader

    def run_training(self):
        self.train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.epoch_start()

            self.train_epoch_start()
            train_outputs = []
            for batch_id in range(self.tr_iterations_per_epoch):
                try:
                    train_outputs.append(self.train_step(next(self.iter_train)))
                except StopIteration:
                    self.iter_train = iter(self.dataloader_train)
                    train_outputs.append(self.train_step(next(self.iter_train)))
            self.train_epoch_end(train_outputs)

            with torch.no_grad():
                self.validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.val_iterations_per_epoch):
                    try:
                        val_outputs.append(self.validation_step(next(self.iter_val)))
                    except StopIteration:
                        self.iter_val = iter(self.dataloader_val)
                        train_outputs.append(self.validation_step(next(self.iter_val)))
                self.validation_epoch_end(val_outputs)
            self.epoch_end(epoch)
        self.train_end()

    def train_start(self):
        if not self.was_initialized:
            self.initialize()

        empty_cache(self.device)

        self.dataloader_train, self.dataloader_val = self.get_train_and_val_dataloader()

    def train_end(self):
        self.save_checkpoint(
            os.path.join(self.logs_output_folder, "checkpoint_final.pth")
        )

        if os.path.isfile(
            os.path.join(self.logs_output_folder, "checkpoint_latest.pth")
        ):
            os.remove(os.path.join(self.logs_output_folder, "checkpoint_latest.pth"))

        empty_cache(self.device)
        self.print_to_log_file("Training done.")
        self.perform_actual_validation(save_probabilities=False)

    def epoch_start(self):
        self.logger.log("epoch_start_timestamps", time(), self.current_epoch)

    def epoch_end(self, epoch):
        self.logger.log("epoch_end_timestamps", time(), self.current_epoch)

        self.print_to_log_file(
            "train_loss", np.round(self.logger.logging["train_losses"][-1], decimals=4)
        )
        self.print_to_log_file(
            "val_loss", np.round(self.logger.logging["val_losses"][-1], decimals=4)
        )
        self.print_to_log_file(
            "Pseudo dice",
            [
                np.round(i, decimals=4)
                for i in self.logger.logging["dice_per_class"][-1]
            ],
        )
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.logging['epoch_end_timestamps'][-1] - self.logger.logging['epoch_start_timestamps'][-1], decimals=2)} s"
        )

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (
            self.num_epochs - 1
        ):
            self.save_checkpoint(
                os.path.join(self.logs_output_folder, "checkpoint_latest.pth")
            )

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if (
            self._best_ema is None
            or self.logger.logging["ema_fg_dice"][-1] > self._best_ema
        ):
            self.best_epoch = epoch
            self._best_ema = self.logger.logging["ema_fg_dice"][-1]
            self.print_to_log_file(
                f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}"
            )
            self.save_checkpoint(
                os.path.join(self.logs_output_folder, "checkpoint_best.pth")
            )

        self.logger.plot_progress_png(self.logs_output_folder)

        self.current_epoch += 1

    def train_epoch_start(self):
        self.iter_train = iter(self.dataloader_train)
        self.network.train()
        # self.lr_scheduler.step(self.current_epoch)
        # self.lr_scheduler.step()
        self.print_to_log_file("")
        self.print_to_log_file(f"Epoch {self.current_epoch}")
        self.print_to_log_file(
            f"learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}"
        )
        self.logger.log(
            "learning_rates", self.optimizer.param_groups[0]["lr"], self.current_epoch
        )

    def train_step(self, batch):
        # images in (b, c, (z,) y, x) and labels in (b, 1, (z,) y, x) or list object if do deep supervision
        images = batch["image"]
        labels = batch["label"]

        # to device
        images = images.to(self.device, non_blocking=True).as_tensor()
        if isinstance(labels, list):
            labels = [i.to(self.device, non_blocking=True) for i in labels]
        else:
            labels = labels.to(self.device, non_blocking=True)

        self.optimizer.zero_grad()
        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self.network(images)
            # Compute Loss
            l = self.train_loss(output, labels)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {"loss": l.detach().cpu().numpy()}

    def train_epoch_end(self, train_outputs):
        outputs = collate_outputs(train_outputs)
        self.lr_scheduler.step()
        loss_here = np.mean(outputs["loss"])

        self.logger.log("train_losses", loss_here, self.current_epoch)

    def validation_epoch_start(self):
        self.iter_val = iter(self.dataloader_val)
        self.network.eval()

    def validation_step(self, batch):
        # images in (b, c, (z,) y, x) and labels in (b, 1, (z,) y, x) or list object if do deep supervision
        images = batch["image"]
        labels = batch["label"]

        images = images.to(self.device, non_blocking=True).as_tensor()
        if isinstance(labels, list):
            labels = labels[0].to(self.device, non_blocking=True)
        else:
            labels = labels.to(self.device, non_blocking=True)

        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            outputs = self.network(images)
            del images
            l = self.val_loss(outputs, labels)

        # use the new name of outputs and labels, so that you only need to change the network inference process
        # during validation and the variable name assignment code below, without changing any evaluation code.
        output = outputs
        target = labels

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, len(output.shape)))

        # no need for softmax
        output_seg = output.argmax(1)[:, None]
        predicted_segmentation_onehot = torch.zeros(
            output.shape, device=output.device, dtype=torch.float32
        )
        predicted_segmentation_onehot.scatter_(1, output_seg, 1)
        del output_seg

        tp, fp, fn, _ = get_tp_fp_fn_tn(
            predicted_segmentation_onehot, target, axes=axes, mask=None
        )

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()

        tp_hard = tp_hard[1:]
        fp_hard = fp_hard[1:]
        fn_hard = fn_hard[1:]

        return {
            "loss": l.detach().cpu().numpy(),
            "tp_hard": tp_hard,
            "fp_hard": fp_hard,
            "fn_hard": fn_hard,
        }

    def validation_epoch_end(self, val_outputs):
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated["tp_hard"], 0)
        fp = np.sum(outputs_collated["fp_hard"], 0)
        fn = np.sum(outputs_collated["fn_hard"], 0)

        loss_here = np.mean(outputs_collated["loss"])

        global_dc_per_class = [
            i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]
        ]
        mean_fg_dice = np.nanmean(global_dc_per_class)
        self.logger.log("mean_fg_dice", mean_fg_dice, self.current_epoch)
        self.logger.log("dice_per_class", global_dc_per_class, self.current_epoch)
        self.logger.log("val_losses", loss_here, self.current_epoch)

    def save_checkpoint(self, filename: str) -> None:
        if not self.disable_checkpointing:
            mod = self.network
            if isinstance(mod, OptimizedModule):
                mod = mod._orig_mod

            checkpoint = {
                "network_weights": mod.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "grad_scaler_state": (
                    self.grad_scaler.state_dict()
                    if self.grad_scaler is not None
                    else None
                ),
                "logging": self.logger.get_checkpoint(),
                "_best_ema": self._best_ema,
                "current_epoch": self.current_epoch + 1,
                "LRScheduler_step": self.lr_scheduler.ctr,
            }
            torch.save(checkpoint, filename)
        else:
            self.print_to_log_file("No checkpoint written, checkpointing is disabled")

    def load_checkpoint(self, filename_or_checkpoint):
        self.print_to_log_file("Load checkpoint...")
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device)
        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        new_state_dict = {}
        for k, value in checkpoint["network_weights"].items():
            key = k
            if key not in self.network.state_dict().keys() and key.startswith(
                "module."
            ):
                key = key[7:]
            new_state_dict[key] = value

        self.current_epoch = checkpoint["current_epoch"]
        self.logger.load_checkpoint(checkpoint["logging"])
        self._best_ema = checkpoint["_best_ema"]

        if isinstance(self.network, OptimizedModule):
            self.network._orig_mod.load_state_dict(new_state_dict)
        else:
            self.network.load_state_dict(new_state_dict)

        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if self.grad_scaler is not None:
            if checkpoint["grad_scaler_state"] is not None:
                self.grad_scaler.load_state_dict(checkpoint["grad_scaler_state"])
        self.lr_scheduler.ctr = checkpoint["LRScheduler_step"]

    def perform_actual_validation(self, save_probabilities: bool = False):
        self.network.eval()

        dataset_path = os.path.join("./Dataset", self.dataset_name)
        original_img_folder = os.path.join(
            dataset_path,
            (
                "images"
                if os.path.isdir(os.path.join(dataset_path, "images"))
                else "imagesVal"
            ),
        )
        predictions_save_folder = os.path.join(self.logs_output_folder, "validation")
        self.print_to_log_file("Best Epoch:", self.best_epoch)
        model_path = os.path.join(self.logs_output_folder, "checkpoint_best.pth")
        predict_configs = {
            "dataset_name": self.dataset_name,
            "modality": self.modality,
            "fold": self.fold,
            "split": "val",
            "original_img_folder": original_img_folder,
            "predictions_save_folder": predictions_save_folder,
            "model_path": model_path,
            "device": self.device_dict,
            "overwrite": True,
            "save_probabilities": save_probabilities,
            "patch_size": self.patch_size,
            "tile_step_size": 0.5,
            "use_gaussian": True,
            "perform_everything_on_gpu": True,
            "use_mirroring": True,
            "allowed_mirroring_axes": [2],
            "num_processes": self.num_processes,
        }
        self.config_dict["Inferring_settings"] = predict_configs
        predictor = PatchBasedPredictor(
            self.config_dict, allow_tqdm=False, verbose=False
        )
        self.print_to_log_file("Start predicting.")
        start = time()
        predictor.predict_from_file(
            predictor.original_img_folder,
            predictor.predictions_save_folder,
            predictor.modality,
            predictor.save_probabilities,
        )
        self.print_to_log_file("Predicting ends. Cost: {}s".format(time() - start))

        ground_truth_folder = os.path.join(
            dataset_path,
            (
                "labels"
                if os.path.isdir(os.path.join(dataset_path, "labels"))
                else "labelsVal"
            ),
        )
        dataset_yaml = open_yaml(os.path.join(dataset_path, "dataset.yaml"))
        evaluator = Evaluator(
            predictions_save_folder,
            ground_truth_folder,
            None,
            foreground_classes=len(dataset_yaml["labels"]) - 1,
            files_ending=dataset_yaml["files_ending"],
            num_processes=self.num_processes,
        )
        evaluator.compute_metrics()
        self.print_to_log_file("Evaluating ends.")


if __name__ == "__main__":
    setting_file_name = "SegRap2023_test.yaml"
    settings_path = os.path.join("./Configs", setting_file_name)
    Trainer = PatchBasedTrainer(settings_path, 0)
    Trainer.run_training()
