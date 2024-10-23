import os
import torch

from torch import autocast
from torch.amp import GradScaler
from datetime import datetime

from wcode.training.loss.compound_loss import (
    Tversky_and_CE_loss,
    Hinton_distillaton_loss,
)
from wcode.training.Trainers.PatchBasedTrainer.PatchBasedTrainer import (
    PatchBasedTrainer,
)
from wcode.training.metrics import get_tp_fp_fn_tn
from wcode.utils.others import dummy_context
from wcode.inferring.utils.load_pretrain_weight import load_pretrained_weights
from wcode.utils.file_operations import open_yaml, copy_file_to_dstFolder
from wcode.training.logs_writer.logger import logger


class HintonDistillTrainer(PatchBasedTrainer):
    # The setting of teacher model and the weight's path of it must be saved in the config yaml file!!!
    def __init__(
        self,
        config_file_path: str,
        fold: int,
        T: float,
        alpha: float,
        verbose: bool = False,
    ):
        """
        T is the temperature index
        alpha is a tradeoff between the loss of ground truth (1 - alpha) and soft label
        """
        self.T = T
        self.alpha = alpha
        self.verbose = verbose

        self.config_dict = open_yaml(config_file_path)
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
            "Temperature_{:.2f}_alpha_{:.2f}".format(self.T, self.alpha),
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

    def initialize(self):
        if not self.was_initialized:
            self.is_ddp = False
            self.init_random()
            self.setting_check()

            if len(self.config_dict["Network"]["kernel_size"][0]) == 3:
                self.preprocess_config = "3d"
            elif len(self.config_dict["Network"]["kernel_size"][0]) == 2:
                self.preprocess_config = "2d"
            else:
                raise Exception()

            self.network = self.get_networks(self.config_dict["Network"]).to(
                self.device
            )
            self.teacher_network = self.get_networks(
                self.config_dict["Teacher_Network"]
            )
            self.print_to_log_file("Load weights for the teacher model.")
            load_pretrained_weights(
                self.teacher_network,
                self.config_dict["Teacher_Network"]["weight_path"],
                verbose=False,
            )
            self.teacher_network.to(self.device)

            self.student_data_modality = self.config_dict["Network"]["modality"]
            self.teacher_data_modality = self.config_dict["Teacher_Network"]["modality"]

            self.print_to_log_file("Compiling network...")
            self.network = torch.compile(self.network)
            self.teacher_network = torch.compile(self.teacher_network)

            self.do_deep_supervision = self.config_dict["Network"]["deep_supervision"]
            self.optimizer, self.lr_scheduler = self.get_optimizers()

            self.train_loss = Hinton_distillaton_loss(
                {},
                temperature_index=self.T,
                weight_of_distill=self.alpha,
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
            raise Exception("Initialization was done before initialize method???")

    def train_start(self):
        super().train_start()
        self.teacher_network.eval()

    def obtain_data_for_corresponding_modality(self, data, modality):
        # data: [b, c, (z,) y, x]
        if modality is None or modality == "all":
            return data
        else:
            assert isinstance(modality, list)
            return data[:, modality]

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
            with torch.no_grad():
                teacher_output = self.teacher_network(
                    self.obtain_data_for_corresponding_modality(
                        images, self.teacher_data_modality
                    )
                )
            output = self.network(
                self.obtain_data_for_corresponding_modality(
                    images, self.student_data_modality
                )
            )
            """
            In the paper, Hinton is based on two assumptions to draw a conclusion that the distillation loss needs to be multiplied by t^2 to supplement the gradient:
            1. the temperature index is high compare to the output logits;
            2. the logits of student and teacher have zero mean;
            Print some information here and take a look
            """
            if self.verbose:
                print("Temperature index:", self.T)
                print(
                    "Logits MEAN: Student: {}, Teacher: {}".format(
                        torch.mean(output.detach()), torch.mean(teacher_output)
                    )
                )
                print(
                    "Logits MAX: Student: {}, Teacher: {}".format(
                        torch.max(output.detach()), torch.max(teacher_output)
                    )
                )
                print(
                    "Logits MIN: Student: {}, Teacher: {}".format(
                        torch.min(output.detach()), torch.min(teacher_output)
                    )
                )

            # Compute Loss
            l = self.train_loss(
                output[-len(self.pool_kernel_size) :],
                teacher_output[-len(self.pool_kernel_size) :],
                labels,
            )

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
        if isinstance(labels, list):
            labels = labels[0].to(self.device, non_blocking=True)
        else:
            labels = labels.to(self.device, non_blocking=True)

        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            outputs = self.network(
                self.obtain_data_for_corresponding_modality(
                    images, self.student_data_modality
                )
            )
            del images
            l = self.val_loss(outputs[-len(self.pool_kernel_size) :], labels)

        # use the new name of outputs and labels, so that you only need to change the network inference process
        # during validation and the variable name assignment code below, without changing any evaluation code.
        if isinstance(outputs[-len(self.pool_kernel_size) :], list):
            output = outputs[-len(self.pool_kernel_size) :][0]
            target = labels[0]
        else:
            output = outputs[-len(self.pool_kernel_size) :]
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