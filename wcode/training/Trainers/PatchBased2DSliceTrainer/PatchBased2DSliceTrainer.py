import os
import torch

from torch import autocast
from torch.amp import GradScaler
from datetime import datetime

from wcode.training.Trainers.PatchBasedTrainer.PatchBasedTrainer import (
    PatchBasedTrainer,
)
from wcode.training.metrics import get_tp_fp_fn_tn
from wcode.utils.others import dummy_context
from wcode.utils.file_operations import open_yaml, copy_file_to_dstFolder
from wcode.training.logs_writer.logger import logger


class PatchBased2DSliceTrainer(PatchBasedTrainer):
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

    def train_step(self, batch):
        # images in (b, c, (z,) y, x) and labels in (b, 1, (z,) y, x) or list object if do deep supervision
        images = batch["image"]
        labels = batch["label"]

        # to device
        images = images.to(self.device, non_blocking=True).as_tensor()
        images_2d = torch.vstack(
            [images[i].permute(1, 0, 2, 3) for i in range(images.size()[0])]
        )
        if isinstance(labels, list):
            labels = [i.to(self.device, non_blocking=True) for i in labels]
            labels_2d = [
                torch.vstack(
                    [label[i].permute(1, 0, 2, 3) for i in range(label.size()[0])]
                )
                for label in labels
            ]
        else:
            labels = labels.to(self.device, non_blocking=True)
            labels_2d = torch.vstack(
                [labels[i].permute(1, 0, 2, 3) for i in range(labels.size()[0])]
            )
        del images, labels

        self.optimizer.zero_grad()
        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            outputs = self.network(images_2d)
            if isinstance(outputs, dict):
                outputs = outputs["pred"]
            # Compute Loss
            l = self.train_loss(outputs, labels_2d)

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

        images = images.to(self.device, non_blocking=True).as_tensor()
        images_2d = torch.vstack(
            [images[i].permute(1, 0, 2, 3) for i in range(images.size()[0])]
        )
        if isinstance(labels, list):
            labels = [i.to(self.device, non_blocking=True) for i in labels]
            labels_2d = [
                torch.vstack(
                    [label[i].permute(1, 0, 2, 3) for i in range(label.size()[0])]
                )
                for label in labels
            ]
        else:
            labels = labels.to(self.device, non_blocking=True)
            labels_2d = torch.vstack(
                [labels[i].permute(1, 0, 2, 3) for i in range(labels.size()[0])]
            )
        del images, labels

        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            outputs = self.network(images_2d)
            if isinstance(outputs, dict):
                outputs = outputs["pred"]
            del images_2d
            l = self.val_loss(outputs, labels_2d)

        # use the new name of outputs and labels, so that you only need to change the network inference process
        # during validation and the variable name assignment code below, without changing any evaluation code.
        if isinstance(outputs, (list, tuple)):
            output = outputs[0]
            target = labels_2d[0]
        else:
            output = outputs
            target = labels_2d

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
