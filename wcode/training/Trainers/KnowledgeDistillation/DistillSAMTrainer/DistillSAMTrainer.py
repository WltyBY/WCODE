import os
import torch

from torch import nn
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
from wcode.net.Vision_Transformer.SAM.build_sam import sam_model_registry
from wcode.net.CNN.pooling import ConvDownPool
from wcode.net.CNN.baseblock_CNN import ResidualBlock
from wcode.training.learning_rate.PolyLRScheduler import PolyLRScheduler


class DistillSAMTrainer(PatchBasedTrainer):
    # The setting of SAM model and the weight's path of it must be written in the config yaml file!!!
    def __init__(
        self,
        config_file_path: str,
        fold: int,
        alpha: float,
        verbose: bool = False,
    ):
        """
        T is the temperature index
        alpha is a tradeoff between the loss of ground truth (1 - alpha) and soft label
        """
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
            "alpha_{:.2f}_{}".format(
                self.alpha, self.config_dict["SAM"]["sam_model_registry"]
            ),
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

            # self.embedding_layer = nn.Sequential(
            #     ConvDownPool(
            #         self.config_dict["Network"]["features"][-1] // 2,
            #         dim=len(self.config_dict["Network"]["kernel_size"][0]),
            #         pool_kernel_size=(2, 2),
            #     ),
            # ).to(self.device)
            self.embedding_layer = nn.Sequential(
                ResidualBlock(
                    self.config_dict["Network"]["features"][-1] // 2,
                    self.config_dict["Network"]["features"][-1] // 2,
                    dropout_p=0,
                    dim=len(self.config_dict["Network"]["kernel_size"][0]),
                    kernel_size=(3, 3),
                    padding_size=(1, 1),
                ),
                ConvDownPool(
                    self.config_dict["Network"]["features"][-1] // 2,
                    dim=len(self.config_dict["Network"]["kernel_size"][0]),
                    pool_kernel_size=(2, 2),
                ),
            ).to(self.device)
            self.network = self.get_networks(self.config_dict["Network"]).to(
                self.device
            )
            self.print_to_log_file("Initialize and load weights for the SAM.")
            self.SAM = sam_model_registry[
                self.config_dict["SAM"]["sam_model_registry"]
            ](self.config_dict["SAM"]["weight_path"])
            del self.SAM.prompt_encoder
            del self.SAM.mask_decoder
            self.SAM.to(self.device)

            self.print_to_log_file("Compiling network...")
            self.embedding_layer = torch.compile(self.embedding_layer)
            self.network = torch.compile(self.network)
            self.SAM = torch.compile(self.SAM)

            self.do_deep_supervision = self.config_dict["Network"]["deep_supervision"]
            self.optimizer, self.lr_scheduler = self.get_optimizers()

            self.img_embeding_loss = nn.MSELoss()
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

    def get_optimizers(self):
        optimizer = torch.optim.SGD(
            [*self.network.parameters(), *self.embedding_layer.parameters()],
            self.base_lr,
            weight_decay=self.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
        lr_scheduler = PolyLRScheduler(optimizer, self.base_lr, self.num_epochs)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=1e-5)
        return optimizer, lr_scheduler

    def train_start(self):
        super().train_start()
        self.SAM.eval()

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

        # 2, 1, 16, 256, 256
        # maybe using torchvision.utils.make_grid() also works, but our implementation below is already effective enough
        grid_id = int(images.size()[2] ** 0.5)
        row_lst = []
        for j in range(grid_id):
            start_id = j * grid_id
            row_lst.append(
                torch.cat(
                    [images[:, :, r] for r in range(start_id, start_id + grid_id)],
                    dim=2,
                )
            )
        images_SAM = torch.cat(row_lst, dim=3).repeat(1, 3, 1, 1)
        del images, labels, row_lst

        self.optimizer.zero_grad()
        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            with torch.no_grad():
                # 2, 256, 64, 64
                output_SAM = self.SAM.image_encoder(images_SAM)
            # maybe using torch.split() also works, but our implementation below is already effective enough
            grid_length = output_SAM.size()[2] // grid_id
            # rearrange to 2, 256, 16, 16, 16 -> 32, 256, 16, 16
            rearange_lst = []
            for i in range(grid_id):
                start_id = i * grid_length
                # (2, 256, 16, 16) * 16
                rearange_lst += [
                    output_SAM[:, :, start_id : start_id + grid_length, :][
                        :, :, :, r * grid_length : (r + 1) * grid_length
                    ]
                    for r in range(grid_id)
                ]
            # 2, 256, (16), 16, 16
            SAM_embedding = torch.stack(rearange_lst, dim=2)
            del rearange_lst
            # 32, 256, 16, 16
            SAM_embedding = torch.vstack(
                [
                    SAM_embedding[r].permute(1, 0, 2, 3)
                    for r in range(SAM_embedding.size()[0])
                ]
            )
            # 32, 512, 8, 8
            SAM_embedding = self.embedding_layer(SAM_embedding)
            output = self.network(images_2d)

            # 32, 512, 8, 8 for output[len(output)//2 - 1]
            # Compute Loss
            l = self.train_loss(
                output[-len(self.pool_kernel_size) :], labels_2d
            ) + self.alpha * self.img_embeding_loss(
                output[: -len(self.pool_kernel_size) + 1][
                    len(output[: -len(self.pool_kernel_size) + 1]) // 2 - 1
                ],
                SAM_embedding,
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
            del images_2d
            l = self.val_loss(outputs[-len(self.pool_kernel_size) :], labels_2d)

        # use the new name of outputs and labels, so that you only need to change the network inference process
        # during validation and the variable name assignment code below, without changing any evaluation code.
        if isinstance(outputs[-len(self.pool_kernel_size) :], list):
            output = outputs[-len(self.pool_kernel_size) :][0]
            target = labels_2d[0]
        else:
            output = outputs[-len(self.pool_kernel_size) :]
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
