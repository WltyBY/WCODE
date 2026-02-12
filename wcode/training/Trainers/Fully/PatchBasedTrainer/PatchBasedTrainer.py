import os
import torch
import random
import warnings
import numpy as np
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from datetime import datetime
from tqdm import tqdm
from time import time, sleep
from torch.amp import autocast
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from torch._dynamo import OptimizedModule
from torch.utils.data import DistributedSampler, RandomSampler
from typing import Tuple, Union, List

from wcode.training.data_augmentation.transformation_list import (
    Convert2DTo3DTransform,
    Convert3DTo2DTransform,
    SpatialTransform,
    RandomTransform,
    GaussianNoiseTransform,
    GaussianBlurTransform,
    MultiplicativeBrightnessTransform,
    ContrastTransform,
    BGContrast,
    SimulateLowResolutionTransform,
    GammaTransform,
    MirrorTransform,
    LabelValueTransform,
    DownsampleSegForDSTransform,
    ComposeTransforms,
    BasicTransform,
)
from wcode.training.data_augmentation.custom_transforms.scalar_type import RandomScalar
from wcode.training.data_augmentation.compute_initial_patch_size import get_patch_size
from wcode.preprocessing.resampling import ANISO_THRESHOLD
from wcode.net.build_network import build_network
from wcode.training.dataset.BaseDataset import PatchDataset
from wcode.training.loss.CompoundLoss import Tversky_and_CE_loss
from wcode.training.loss.deep_supervision import DeepSupervisionWeightedSummator
from wcode.training.logs_writer.logger_for_segmentation import logger
from wcode.training.dataloader.Collater import BasedCollater
from wcode.training.dataloader.InfiniteSampler import InfiniteSampler
from wcode.training.learning_rate.PolyLRScheduler import PolyLRScheduler
from wcode.training.metrics import get_tp_fp_fn_tn
from wcode.utils.file_operations import open_yaml, open_json, copy_file_to_dstFolder
from wcode.utils.others import empty_cache, dummy_context
from wcode.utils.collate_outputs import collate_outputs
from wcode.utils.data_io import file_endings_for_2d_img, file_endings_for_sitk
from wcode.inferring.PatchBasedPredictor import PatchBasedPredictor
from wcode.inferring.NaturalImagePredictor import NaturalImagePredictor
from wcode.inferring.Evaluator import Evaluator
from wcode.inferring.utils.load_pretrain_weight import load_pretrained_weights


class PatchBasedTrainer(object):
    def __init__(
        self,
        training_args,
        verbose: bool = False,
    ):
        self.training_args = training_args

        self.verbose = verbose
        config_file_path = os.path.join("./Configs", self.training_args.setting)
        self.config_dict = open_yaml(config_file_path)

        self.get_train_settings()
        self.device = self.get_device()
        
        # Task-general params
        task_general_names = "BS_{}_GPU_NUM_{}_SEED_{}_PRETRAINED_{}".format(
            self.batch_size, self.world_size, self.random_seed, self.pretrained_weight is not None
        )

        # hyperparameter
        self.w_ce = self.training_args.w_ce
        self.w_dice = self.training_args.w_dice
        self.w_class = self.training_args.w_class
        hyperparams_name = "w_ce_{}_w_dice_{}_w_class_{}".format(
            self.w_ce, self.w_dice, self.w_class
        )

        self.fold = self.training_args.fold
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

        if self.is_ddp:
            dist.barrier(device_ids=[self.device.index])

        self.logger = logger()

        self.current_epoch = 0

        # checkpoint saving stuff
        self.save_every = 1
        self.disable_checkpointing = False

        self.grad_scaler = GradScaler() if self.device.type == "cuda" else None

        if self.continue_train:
            checkpoint = os.path.join(self.logs_output_folder, "checkpoint_latest.pth")
            if not os.path.isfile(checkpoint):
                raise FileNotFoundError(
                    f"Continue training was requested but checkpoint not found: {checkpoint}"
                )
            self.load_checkpoint(checkpoint)

            if self.is_ddp:
                dist.barrier(device_ids=[self.device.index])

    def get_train_settings(self):
        # settings in training args
        self.dataset_name = self.training_args.dataset
        self.method_name = self.training_args.method_name
        self.batch_size = self.training_args.batch_size
        self.num_workers = self.training_args.num_workers
        self.random_seed = self.training_args.seed
        self.continue_train = self.training_args.continue_train
        self.pretrained_weight = self.training_args.pretrained_weight
        self.args_gpu = self.training_args.gpu

        # settings in config file
        self.modality = self.config_dict["Training_settings"]["modality"]
        self.num_epochs = self.config_dict["Training_settings"]["epoch"]
        self.tr_iterations_per_epoch = self.config_dict["Training_settings"][
            "tr_iterations_per_epoch"
        ]
        self.val_iterations_per_epoch = self.config_dict["Training_settings"][
            "val_iterations_per_epoch"
        ]
        self.patch_size = self.config_dict["Training_settings"]["patch_size"]
        self.base_lr = self.config_dict["Training_settings"]["base_lr"]
        self.weight_decay = self.config_dict["Training_settings"]["weight_decay"]
        self.deterministic = self.config_dict["Training_settings"]["deterministic"]
        self.oversample_rate = self.config_dict["Training_settings"]["oversample_rate"]
        self.probabilistic_oversampling = self.config_dict["Training_settings"][
            "probabilistic_oversampling"
        ]
        self.ignore_label = self.config_dict["Training_settings"]["ignore_label"]

        # Change parameter format for training purposes
        self.dataset_yaml = open_yaml(
            os.path.join("./Dataset_preprocessed", self.dataset_name, "dataset.yaml")
        )
        if self.modality == None or self.modality == "all":
            self.modality = [
                int(i) for i in range(len(self.dataset_yaml["channel_names"]))
            ]
        
        # decide preprocess config
        ks = self.config_dict["Network"]["kernel_size"][0]
        if len(ks) == 3:
            self.preprocess_config = "3d"
        elif len(ks) == 2:
            self.preprocess_config = "2d"
        else:
            raise ValueError("Unsupported kernel size dimension")

    def setting_check(self):
        if len(self.config_dict["Network"]["pool_kernel_size"]) == len(
            self.config_dict["Network"]["kernel_size"]
        ):
            raise ValueError(
                "The list of convolutional kernel sizes should be 1 smaller in length than the list of pooling kernels. "
                "Because the output with the lowest resolution does not require a pooling layer."
            )
        self.pool_kernel_size = self.config_dict["Network"]["pool_kernel_size"]

        if (
            self.config_dict["Network"].__contains__("activate")
            and self.config_dict["Network"]["activate"].lower() == "prelu"
            and self.weight_decay != 0
        ):
            warnings.warn(
                f"PReLU is used but weight_decay is set to {self.weight_decay}, which is not zero. "
                "This will push the learnable slope 'a' toward 0 and degrade performance. "
                "Consider setting weight_decay to 0 for PReLU parameters.",
                UserWarning,
            )

    def get_device(self):
        """
        args_gpu:
            None  -> use all visible GPUs
            []    -> CPU
            [0,2] -> only see GPU 0 and 2
        DDP is determined ONLY by torchrun (WORLD_SIZE > 1).
        """
        # limit visible GPUs if specified
        if self.args_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.args_gpu))

        use_cuda = torch.cuda.is_available() and self.args_gpu != []

        ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1 and use_cuda

        if ddp:
            local_rank = int(os.environ["LOCAL_RANK"])
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)

            if not dist.is_initialized():
                dist.init_process_group(backend="nccl", init_method="env://")

            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
            device = torch.device("cuda:0" if use_cuda else "cpu")

        self.is_ddp = ddp

        if self.is_ddp:
            dist.barrier(device_ids=[device.index])
        return device

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

    def is_main_process(self):
        return (not self.is_ddp) or self.rank == 0

    def _build_deep_supervision_loss_object(self, loss):
        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss.
        # When writing model's code, we assume that its multi-scales predictions range from high resolution to low resolution
        weights = np.array(
            [1.0 / (2**i) for i in range(len(deep_supervision_scales))],
            dtype=np.float64,
        )
        # Normalize weights so that they sum to 1
        weights /= weights.sum()
        assert len(weights) == len(
            deep_supervision_scales
        ), "Mismatch between deep supervision scales and loss weights"

        # Restructuring the loss
        loss = DeepSupervisionWeightedSummator(loss, weights)

        return loss

    def _get_deep_supervision_scales(self):
        if self.do_deep_supervision:
            deep_supervision_scales = [
                list(i.astype(np.float64))
                for i in 1.0
                / np.cumprod(
                    np.vstack(self.pool_kernel_size).astype(np.float64),
                    axis=0,
                )
            ]
        else:
            deep_supervision_scales = None  # for train and val_transforms
        return deep_supervision_scales

    def init_random(self):
        if self.deterministic:
            cudnn.benchmark = False
            cudnn.deterministic = True
        else:
            cudnn.benchmark = True
            cudnn.deterministic = False

        seed = self.random_seed
        if self.is_ddp:
            seed = seed + self.rank

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):
        timestamp = time()
        if add_timestamp:
            args = (f"{datetime.fromtimestamp(timestamp)}:", *args)

        for _ in range(5):
            try:
                with open(self.log_file, "a+") as f:
                    f.write(" ".join(map(str, args)) + "\n")
                break
            except IOError:
                sleep(0.5)

        if also_print_to_console:
            print(*args)

    def get_networks(self, network_settings):
        model = build_network(network_settings)
        if not self.is_ddp:
            return model

        has_bn = False
        for m in model.modules():
            if isinstance(
                m,
                (
                    torch.nn.BatchNorm1d,
                    torch.nn.BatchNorm2d,
                    torch.nn.BatchNorm3d,
                ),
            ):
                has_bn = True
                break

        if has_bn:
            if self.is_main_process():
                self.print_to_log_file(
                    "[DDP] BatchNorm detected → converting to SyncBatchNorm"
                )
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        else:
            if self.is_main_process():
                self.print_to_log_file(
                    "[DDP] No BatchNorm detected → skip SyncBatchNorm"
                )

        return model

    def get_optimizers(self):
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            self.base_lr,
            weight_decay=self.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
        lr_scheduler = PolyLRScheduler(optimizer=optimizer, max_steps=self.num_epochs)
        return optimizer, lr_scheduler

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        assert len(self.patch_size) in (2, 3), "patch_size must be 2D or 3D"
        patch_size = self.patch_size
        dim = len(patch_size)
        deg2rad = np.pi / 180.0

        # todo rotation should be defined dynamically based on patch size (more isotropic patch sizes = more rotation)
        if dim == 2:
            do_dummy_2d_data_aug = False
            # todo revisit this parametrization
            if max(patch_size) / min(patch_size) > 1.5:
                rotation_for_DA = (-15.0 * deg2rad, 15.0 * deg2rad)
            else:
                rotation_for_DA = (-180.0 * deg2rad, 180.0 * deg2rad)
            mirror_axes = (0, 1)
        elif dim == 3:
            # todo this is not ideal. We could also have patch_size (64, 16, 128) in which case a full 180deg 2d rot would be bad
            # order of the axes is determined by spacing, not image size
            do_dummy_2d_data_aug = (max(patch_size) / patch_size[0]) > ANISO_THRESHOLD
            if do_dummy_2d_data_aug:
                # why do we rotate 180 deg here all the time? We should also restrict it
                rotation_for_DA = (-180.0 * deg2rad, 180.0 * deg2rad)
            else:
                rotation_for_DA = (-30.0 * deg2rad, 30.0 * deg2rad)
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

        if self.is_main_process():
            self.print_to_log_file(f"do_dummy_2d_data_aug: {do_dummy_2d_data_aug}")
        self.allow_mirroring_axes_during_inference = mirror_axes

        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    @staticmethod
    def get_training_transforms(
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: RandomScalar,
        deep_supervision_scales: Union[List, Tuple, None],
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
    ) -> BasicTransform:
        transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None
        transforms.append(
            SpatialTransform(
                patch_size_spatial,
                patch_center_dist_from_border=0,
                random_crop=False,
                p_elastic_deform=0,
                p_rotation=0.2,
                rotation=rotation_for_DA,
                p_scaling=0.2,
                scaling=(0.7, 1.4),
                p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False,  # , mode_seg='nearest'
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        transforms.append(
            RandomTransform(
                GaussianNoiseTransform(
                    noise_variance=(0, 0.1), p_per_channel=1, synchronize_channels=True
                ),
                apply_probability=0.1,
            )
        )
        transforms.append(
            RandomTransform(
                GaussianBlurTransform(
                    blur_sigma=(0.5, 1.0),
                    synchronize_channels=False,
                    synchronize_axes=False,
                    p_per_channel=0.5,
                    benchmark=True,
                ),
                apply_probability=0.2,
            )
        )
        transforms.append(
            RandomTransform(
                MultiplicativeBrightnessTransform(
                    multiplier_range=BGContrast((0.75, 1.25)),
                    synchronize_channels=False,
                    p_per_channel=1,
                ),
                apply_probability=0.15,
            )
        )
        transforms.append(
            RandomTransform(
                ContrastTransform(
                    contrast_range=BGContrast((0.75, 1.25)),
                    preserve_range=True,
                    synchronize_channels=False,
                    p_per_channel=1,
                ),
                apply_probability=0.15,
            )
        )
        transforms.append(
            RandomTransform(
                SimulateLowResolutionTransform(
                    scale=(0.5, 1),
                    synchronize_channels=False,
                    synchronize_axes=True,
                    ignore_axes=ignore_axes,
                    allowed_channels=None,
                    p_per_channel=0.5,
                ),
                apply_probability=0.25,
            )
        )
        transforms.append(
            RandomTransform(
                GammaTransform(
                    gamma=BGContrast((0.7, 1.5)),
                    p_invert_image=1,
                    synchronize_channels=False,
                    p_per_channel=1,
                    p_retain_stats=1,
                ),
                apply_probability=0.1,
            )
        )
        transforms.append(
            RandomTransform(
                GammaTransform(
                    gamma=BGContrast((0.7, 1.5)),
                    p_invert_image=0,
                    synchronize_channels=False,
                    p_per_channel=1,
                    p_retain_stats=1,
                ),
                apply_probability=0.3,
            )
        )
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(MirrorTransform(allowed_axes=mirror_axes))

        transforms.append(LabelValueTransform(-1, 0))

        if deep_supervision_scales is not None:
            deep_supervision_scales = [[1] * len(patch_size)] + deep_supervision_scales
            transforms.append(
                DownsampleSegForDSTransform(ds_scales=deep_supervision_scales)
            )

        return ComposeTransforms(transforms)

    @staticmethod
    def get_validation_transforms(
        patch_size,
        deep_supervision_scales: Union[List, Tuple, None],
    ) -> BasicTransform:
        transforms = []
        transforms.append(LabelValueTransform(-1, 0))

        if deep_supervision_scales is not None:
            deep_supervision_scales = [[1] * len(patch_size)] + deep_supervision_scales
            transforms.append(
                DownsampleSegForDSTransform(ds_scales=deep_supervision_scales)
            )

        return ComposeTransforms(transforms)

    def get_train_and_val_dataset(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = (
            self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        )
        deep_supervision_scales = self._get_deep_supervision_scales()

        train_transform = self.get_training_transforms(
            self.patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
        )
        val_transform = self.get_validation_transforms(
            self.patch_size, deep_supervision_scales
        )

        train_dataset = PatchDataset(
            dataset_name=self.dataset_name,
            preprocess_config=self.preprocess_config,
            sample_patch_size=initial_patch_size,
            final_patch_size=self.patch_size,
            oversample_rate=self.oversample_rate,
            probabilistic_oversampling=self.probabilistic_oversampling,
            split="train",
            fold=self.fold,
            modality=self.modality,
            transform=train_transform,
        )

        val_dataset = PatchDataset(
            dataset_name=self.dataset_name,
            preprocess_config=self.preprocess_config,
            sample_patch_size=self.patch_size,
            final_patch_size=self.patch_size,
            oversample_rate=self.oversample_rate,
            probabilistic_oversampling=self.probabilistic_oversampling,
            split="val",
            fold=self.fold,
            modality=self.modality,
            transform=val_transform,
        )
        return train_dataset, val_dataset

    def worker_init_fn(self, worker_id):
        seed = self.random_seed
        if self.is_ddp:
            seed = seed + self.rank * self.num_workers
        seed = seed + worker_id

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def build_sampler(self, dataset):
        if self.is_ddp:
            base_sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                seed=self.random_seed,
                drop_last=True,
            )
        else:
            base_sampler = RandomSampler(dataset)
        return InfiniteSampler(base_sampler)

    def get_train_and_val_dataloader(self):
        train_dataset, val_dataset = self.get_train_and_val_dataset()

        train_sampler = self.build_sampler(train_dataset)
        # shuffle is True here, because we expect patch based validation to be more comprehensive.
        # If the number of validation iteration is smaller than the valset, the validation
        # throughout the entire training process cannot cover all the images in the val set.
        val_sampler = self.build_sampler(val_dataset)

        this_num_workers = self.num_workers // self.world_size

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            shuffle=False,
            num_workers=this_num_workers,
            collate_fn=BasedCollater(),
            pin_memory=self.device.type == "cuda",
            persistent_workers=True,
            worker_init_fn=self.worker_init_fn,
            drop_last=True,
            prefetch_factor=3,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=max(1, this_num_workers // 2),
            collate_fn=BasedCollater(),
            pin_memory=self.device.type == "cuda",
            persistent_workers=True,
            drop_last=True,
            prefetch_factor=2,
        )

        return train_loader, val_loader

    def run_training(self):
        try:
            self.train_start()

            for epoch in range(self.current_epoch, self.num_epochs):
                self.epoch_start(epoch)

                # ---------- train ----------
                self.train_epoch_start(epoch)
                train_outputs = []
                for _ in tqdm(
                    range(self.tr_iterations_per_epoch),
                    disable=(self.rank != 0) or self.verbose,
                ):
                    batch = next(self.train_iter)
                    train_outputs.append(self.train_step(batch))
                if self.is_ddp:
                    dist.barrier(device_ids=[self.device.index])
                self.train_epoch_end(train_outputs, epoch)

                # ---------- validate ----------
                with torch.no_grad():
                    self.validation_epoch_start()
                    val_outputs = []
                    for _ in tqdm(
                        range(self.val_iterations_per_epoch),
                        disable=(self.rank != 0) or self.verbose,
                    ):
                        batch = next(self.val_iter)
                        val_outputs.append(self.validation_step(batch))
                    if self.is_ddp:
                        dist.barrier(device_ids=[self.device.index])
                    self.validation_epoch_end(val_outputs, epoch)

                self.epoch_end(epoch)
            self.train_end()
        finally:
            if self.is_ddp and dist.is_initialized():
                dist.destroy_process_group()

    def train_start(self):
        if not self.was_initialized:
            self.initialize()

        empty_cache(self.device)

        self.dataloader_train, self.dataloader_val = self.get_train_and_val_dataloader()

        # check if final checkpoint already exists (training was already finished in a previous run), if yes, skip training and directly go to validation
        # must check after dataloader initialization, because we need to init self.allow_mirroring_axes_during_inference
        # through self.dataloader_train, self.dataloader_val = self.get_train_and_val_dataloader().
        final_ckpt = os.path.join(self.logs_output_folder, "checkpoint_final.pth")
        if os.path.isfile(final_ckpt):
            if self.is_main_process():
                self.print_to_log_file(
                    f"{final_ckpt} exists – training already finished."
                )
            self.current_epoch = self.num_epochs  # labeled as finished
            self.already_finish_training = True
        else:
            self.already_finish_training = False

    def train_end(self):
        if self.is_ddp:
            dist.barrier(device_ids=[self.device.index])

        # kill dataloader workers
        if hasattr(self, "dataloader_train"):
            if hasattr(self, "train_iter"):
                del self.train_iter
            if (
                hasattr(self, "dataloader_train")
                and self.dataloader_train._iterator is not None
            ):
                self.dataloader_train._iterator._shutdown_workers()
            del self.dataloader_train

        if hasattr(self, "dataloader_val"):
            if hasattr(self, "val_iter"):
                del self.val_iter
            if (
                hasattr(self, "dataloader_val")
                and self.dataloader_val._iterator is not None
            ):
                self.dataloader_val._iterator._shutdown_workers()
            del self.dataloader_val

        if self.is_main_process() and (not self.already_finish_training):
            # save final checkpoint
            self.save_checkpoint(
                os.path.join(self.logs_output_folder, "checkpoint_final.pth")
            )

            # del latest checkpoint
            latest_ckpt = os.path.join(self.logs_output_folder, "checkpoint_latest.pth")
            if os.path.isfile(latest_ckpt):
                os.remove(latest_ckpt)

        empty_cache(self.device)

        if self.is_main_process():
            self.print_to_log_file("Training done.")
            self.perform_actual_validation(save_probabilities=False)

    def epoch_start(self, epoch):
        if self.is_ddp:
            for loader in [self.dataloader_train, self.dataloader_val]:
                sampler = loader.sampler
                sampler.set_epoch(epoch)

        # only rank 0 or the main process
        if self.is_main_process():
            self.logger.log("epoch_start_timestamps", time(), epoch)

    def epoch_end(self, epoch):
        # only rank 0 should do logging / saving / printing
        if not self.is_main_process():
            self.current_epoch = epoch + 1
            return

        self.logger.log("epoch_end_timestamps", time(), epoch)

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
        if (epoch + 1) % self.save_every == 0 and epoch != (self.num_epochs - 1):
            self.save_checkpoint(
                os.path.join(self.logs_output_folder, "checkpoint_latest.pth")
            )

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if (
            self._best_ema is None
            or self.logger.logging["ema_fg_dice"][-1] > self._best_ema
        ):
            self._best_ema = self.logger.logging["ema_fg_dice"][-1]
            self.print_to_log_file(
                f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}"
            )
            self.save_checkpoint(
                os.path.join(self.logs_output_folder, "checkpoint_best.pth")
            )

        self.logger.plot_progress_png(self.logs_output_folder)

        self.current_epoch = epoch + 1

    def train_epoch_start(self, epoch):
        self.train_iter = iter(self.dataloader_train)
        self.network.train()
        self.lr_scheduler.step(epoch)

        if self.is_main_process():
            self.print_to_log_file("")
            self.print_to_log_file(f"Epoch {epoch}")
            self.print_to_log_file(
                f"learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}"
            )
            self.logger.log(
                "learning_rates", self.optimizer.param_groups[0]["lr"], epoch
            )

    def train_step(self, batch):
        # images in (b, c, (z,) y, x) and labels in (b, 1, (z,) y, x) or list object if do deep supervision
        images = batch["image"]
        labels = batch["label"]

        # to device
        images = images.to(self.device, non_blocking=True)
        if isinstance(labels, list):
            labels = [i.to(self.device, non_blocking=True) for i in labels]
        else:
            labels = labels.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            outputs = self.network(images)

            # Compute Loss
            l = self.train_loss(outputs["pred"], labels)

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

    def train_epoch_end(self, train_outputs, epoch):
        outputs = collate_outputs(train_outputs)

        local_loss_sum = float(np.sum(outputs["loss"]))
        local_count = len(outputs["loss"])

        if self.is_ddp:
            loss_sum = torch.tensor(local_loss_sum, device=self.device)
            count = torch.tensor(local_count, device=self.device, dtype=torch.long)

            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)

            loss_here = (loss_sum / count).item()
        else:
            loss_here = local_loss_sum / local_count

        if self.is_main_process():
            self.logger.log("train_losses", loss_here, epoch)

    def validation_epoch_start(self):
        self.val_iter = iter(self.dataloader_val)
        self.network.eval()

    def validation_step(self, batch):
        # images in (b, c, (z,) y, x) and labels in (b, 1, (z,) y, x) or list object if do deep supervision
        images = batch["image"]
        labels = batch["label"]

        images = images.to(self.device, non_blocking=True)
        if isinstance(labels, list):
            labels = [i.to(self.device, non_blocking=True) for i in labels]
        else:
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

        # (num_cls,)
        tp, fp, fn, _ = get_tp_fp_fn_tn(
            predicted_segmentation_onehot, target, axes=axes, mask=None
        )
        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()

        # omit the background, (num_cls - 1,)
        tp_hard = tp_hard[1:]
        fp_hard = fp_hard[1:]
        fn_hard = fn_hard[1:]

        return {
            "loss": l.detach().cpu().numpy(),
            "tp_hard": tp_hard,
            "fp_hard": fp_hard,
            "fn_hard": fn_hard,
        }

    def validation_epoch_end(self, val_outputs, epoch):
        outputs_collated = collate_outputs(val_outputs)

        # outputs_collated["tp_hard"]: (num_batch, num_cls - 1), tp: (num_cls - 1,)
        tp = np.sum(outputs_collated["tp_hard"], 0)
        fp = np.sum(outputs_collated["fp_hard"], 0)
        fn = np.sum(outputs_collated["fn_hard"], 0)
        local_loss_sum = float(np.sum(outputs_collated["loss"]))
        local_count = len(outputs_collated["loss"])

        if self.is_ddp:
            tp = torch.as_tensor(tp, device=self.device, dtype=torch.float64)
            fp = torch.as_tensor(fp, device=self.device, dtype=torch.float64)
            fn = torch.as_tensor(fn, device=self.device, dtype=torch.float64)
            loss_sum = torch.tensor(
                local_loss_sum, device=self.device, dtype=torch.float64
            )
            count = torch.tensor(local_count, device=self.device, dtype=torch.long)

            dist.all_reduce(tp, op=dist.ReduceOp.SUM)
            dist.all_reduce(fp, op=dist.ReduceOp.SUM)
            dist.all_reduce(fn, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)

            tp = tp.cpu().numpy()
            fp = fp.cpu().numpy()
            fn = fn.cpu().numpy()
            loss_here = (loss_sum / count).item()
        else:
            loss_here = local_loss_sum / local_count

        global_dc_per_class = [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]
        mean_fg_dice = np.nanmean(global_dc_per_class)

        if self.is_main_process():
            self.logger.log("mean_fg_dice", mean_fg_dice, epoch)
            self.logger.log("dice_per_class", global_dc_per_class, epoch)
            self.logger.log("val_losses", loss_here, epoch)

    def save_checkpoint(self, filename: str) -> None:
        # In DDP mode, only rank-0 saves the checkpoint to avoid concurrent writes
        if self.is_ddp and dist.get_rank() != 0:
            return

        # Skip saving if checkpointing is disabled
        if self.disable_checkpointing:
            self.print_to_log_file(
                "Checkpoint saving is disabled; no file will be written."
            )
            return

        # Extract the underlying model (remove DDP or OptimizedModule wrappers)
        mod = self.network
        while isinstance(
            mod, (OptimizedModule, torch.nn.parallel.DistributedDataParallel)
        ):
            mod = mod._orig_mod if isinstance(mod, OptimizedModule) else mod.module

        checkpoint = {
            "network_weights": mod.state_dict(),  # Keys without "module." prefix
            "optimizer_state": self.optimizer.state_dict(),
            "grad_scaler_state": (
                self.grad_scaler.state_dict() if self.grad_scaler else None
            ),
            "logging": self.logger.get_checkpoint(),
            "_best_ema": self._best_ema,
            # the correct epoch_id (next epoch) loaded for continual training.
            "current_epoch": self.current_epoch + 1,
            "LRScheduler_state": self.lr_scheduler.state_dict(),  # state from this epoch
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename_or_checkpoint):
        """Load checkpoint and automatically handle 'module.' prefix for DDP <-> single-GPU switching."""
        self.print_to_log_file("Loading checkpoint...")
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(
                filename_or_checkpoint, map_location=self.device, weights_only=False
            )

        # Remove 'module.' prefix if present (compatibility with DDP weights)
        new_state_dict = {}
        for k, v in checkpoint["network_weights"].items():
            if k.startswith("module."):
                k = k[7:]
            new_state_dict[k] = v

        self.current_epoch = checkpoint["current_epoch"]
        self.logger.load_checkpoint(checkpoint["logging"])
        self._best_ema = checkpoint["_best_ema"]

        # Load weights into the bare model (unwrap DDP or OptimizedModule)
        target_mod = self.network
        while isinstance(
            target_mod, (OptimizedModule, torch.nn.parallel.DistributedDataParallel)
        ):
            target_mod = (
                target_mod._orig_mod
                if isinstance(target_mod, OptimizedModule)
                else target_mod.module
            )
        target_mod.load_state_dict(new_state_dict)

        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if self.grad_scaler is not None:
            if checkpoint["grad_scaler_state"] is not None:
                self.grad_scaler.load_state_dict(checkpoint["grad_scaler_state"])
        self.lr_scheduler.load_state_dict(checkpoint["LRScheduler_state"])

        # Although 99.9999% the same, but check again for some unknown things.
        if self.lr_scheduler.last_epoch != self.current_epoch - 1:
            warnings.warn(
                f"[Warning] Expected last_epoch == current_epoch - 1, "
                f"but got last_epoch={self.lr_scheduler.last_epoch}, "
                f"current_epoch={self.current_epoch}",
                UserWarning,
            )

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
        start = time()
        if self.dataset_yaml["files_ending"] in file_endings_for_sitk:
            predictor = PatchBasedPredictor(
                self.config_dict, allow_tqdm=True, verbose=False
            )
            predictor.initialize()
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
            predictor.initialize()
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
