import os
import torch
import numpy as np

from torch import autocast
from torch.amp import GradScaler
from datetime import datetime
from typing import Union

from wcode.training.Trainers.Fully.PatchBasedTrainer.PatchBasedTrainer import (
    PatchBasedTrainer,
)
from wcode.training.data_augmentation.compute_initial_patch_size import get_patch_size
from wcode.training.metrics import get_tp_fp_fn_tn
from wcode.utils.others import dummy_context
from wcode.utils.file_operations import open_yaml, copy_file_to_dstFolder
from wcode.training.logs_writer.logger_for_segmentation import logger


class PatchBased2DSliceTrainer(PatchBasedTrainer):
    def __init__(
        self,
        config_file_path: str,
        fold: int,
        w_ce: float,
        w_dice: float,
        w_class: Union[None, list],
        verbose: bool = False,
    ):
        # hyperparameter
        self.w_ce = w_ce
        self.w_dice = w_dice
        self.w_class = w_class
        hyperparams_name = "w_ce_{}_w_dice_{}_w_class_{}".format(
            self.w_ce, self.w_dice, self.w_class
        )

        self.verbose = verbose
        self.config_dict = open_yaml(config_file_path)
        if self.config_dict.__contains__("Inferring_settings"):
            del self.config_dict["Inferring_settings"]

        self.get_train_settings(self.config_dict["Training_settings"])
        self.fold = fold
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
        log_folder_name = self.method_name if self.method_name is not None else time_
        self.logs_output_folder = os.path.join(
            "./Logs",
            self.dataset_name,
            log_folder_name,
            hyperparams_name,
            "fold_" + self.fold,
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

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        patch_size = self.patch_size
        dim = len(patch_size)
        # todo rotation should be defined dynamically based on patch size (more isotropic patch sizes = more rotation)
        if dim == 2:
            do_dummy_2d_data_aug = False
            # todo revisit this parametrization
            if max(patch_size) / min(patch_size) > 1.5:
                rotation_for_DA = (-15.0 / 360 * 2.0 * np.pi, 15.0 / 360 * 2.0 * np.pi)
            else:
                rotation_for_DA = (
                    -180.0 / 360 * 2.0 * np.pi,
                    180.0 / 360 * 2.0 * np.pi,
                )
            mirror_axes = (0, 1)
        elif dim == 3:
            # todo this is not ideal. We could also have patch_size (64, 16, 128) in which case a full 180deg 2d rot would be bad
            # order of the axes is determined by spacing, not image size
            do_dummy_2d_data_aug = True
            if do_dummy_2d_data_aug:
                # why do we rotate 180 deg here all the time? We should also restrict it
                rotation_for_DA = (
                    -180.0 / 360 * 2.0 * np.pi,
                    180.0 / 360 * 2.0 * np.pi,
                )
            else:
                rotation_for_DA = (-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi)
            mirror_axes = (0, 1, 2)
        else:
            raise RuntimeError()

        # todo this function is stupid. It doesn't even use the correct scale range (we keep things as they were in the
        #  old nnunet for now)
        initial_patch_size = get_patch_size(
            patch_size[-dim:],
            rotation_for_DA,
            rotation_for_DA,
            rotation_for_DA,
            (0.85, 1.25),
        )
        if do_dummy_2d_data_aug:
            initial_patch_size[0] = patch_size[0]

        self.print_to_log_file(f"do_dummy_2d_data_aug: {do_dummy_2d_data_aug}")
        self.inference_allowed_mirroring_axes = (0, 1)

        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    def train_step(self, batch):
        # images in (b, c, (z,) y, x) and labels in (b, 1, (z,) y, x) or list object if do deep supervision
        images = batch["image"]
        labels = batch["label"]

        # to device
        images = torch.vstack(
            [images[i].permute(1, 0, 2, 3) for i in range(images.size()[0])]
        )
        images = images.to(self.device, non_blocking=True)
        if isinstance(labels, list):
            labels = [
                torch.vstack(
                    [label[i].permute(1, 0, 2, 3) for i in range(label.size()[0])]
                )
                for label in labels
            ]
            labels = [i.to(self.device, non_blocking=True) for i in labels]
        else:
            labels = torch.vstack(
                [labels[i].permute(1, 0, 2, 3) for i in range(labels.size()[0])]
            )
            labels = labels.to(self.device, non_blocking=True)

        self.optimizer.zero_grad()
        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            outputs = self.network(images)
            if isinstance(outputs, dict):
                outputs = outputs["pred"]
            # Compute Loss
            l = self.train_loss(outputs, labels)

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

    def validation_step(self, batch):
        # images in (b, c, (z,) y, x) and labels in (b, 1, (z,) y, x) or list object if do deep supervision
        images = batch["image"]
        labels = batch["label"]

        images = torch.vstack(
            [images[i].permute(1, 0, 2, 3) for i in range(images.size()[0])]
        )
        images = images.to(self.device, non_blocking=True)
        if isinstance(labels, list):
            labels = [
                torch.vstack(
                    [label[i].permute(1, 0, 2, 3) for i in range(label.size()[0])]
                )
                for label in labels
            ]
            labels = [i.to(self.device, non_blocking=True) for i in labels]

        else:
            labels = torch.vstack(
                [labels[i].permute(1, 0, 2, 3) for i in range(labels.size()[0])]
            )
            labels = labels.to(self.device, non_blocking=True)

        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            outputs = self.network(images)
            if isinstance(outputs, dict):
                outputs = outputs["pred"]
            del images
            l = self.val_loss(outputs, labels)

        # use the new name of outputs and labels, so that you only need to change the network inference process
        # during validation and the variable name assignment code below, without changing any evaluation code.
        if isinstance(outputs, (list, tuple)):
            output = outputs[0]
            target = labels[0]
        else:
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
