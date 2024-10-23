import os
import torch
import traceback
import itertools
import multiprocessing
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from time import sleep
from typing import Tuple, Union, List, Dict
from torch import nn

from wcode.net.build_network import build_network
from wcode.preprocessing.preprocessor import Preprocessor
from wcode.utils.file_operations import (
    open_yaml,
    open_json,
    open_pickle,
    check_workers_alive_and_busy,
)
from wcode.utils.others import empty_cache, dummy_context
from wcode.inferring.utils.sliding_window_prediction import (
    compute_gaussian,
    compute_steps_for_sliding_window,
)
from wcode.inferring.utils.padding import pad_nd_image
from wcode.inferring.utils.get_predictions import (
    convert_predicted_logits_to_segmentation_with_correct_shape,
    export_prediction_from_logits,
)
from wcode.inferring.utils.load_pretrain_weight import load_pretrained_weights
from wcode.inferring.utils.data_iter import preprocessing_iterator_fromfiles


class PatchBasedPredictor(object):
    def __init__(self, config_file_path_or_dict, allow_tqdm, verbose=False):
        if isinstance(config_file_path_or_dict, str):
            self.config_dict = open_yaml(config_file_path_or_dict)
        else:
            assert isinstance(config_file_path_or_dict, dict)
            self.config_dict = config_file_path_or_dict

        self.get_inferring_settings(self.config_dict["Inferring_settings"])

        self.device = self.get_device()
        self.allow_tqdm = allow_tqdm
        self.verbose = verbose

        self.dataset_yaml = open_yaml(
            os.path.join("./Dataset_preprocessed", self.dataset_name, "dataset.yaml")
        )
        self.dataset_yaml["channel_names"] = {
            key: value
            for key, value in self.dataset_yaml["channel_names"].items()
            if int(key) in self.modality
        }
        
        self.dataset_split = open_json(
            os.path.join(
                "./Dataset_preprocessed", self.dataset_name, "dataset_split.json"
            )
        )
        self.initialize()

    def get_inferring_settings(self, inferring_setting_dict):
        self.dataset_name = inferring_setting_dict["dataset_name"]
        self.modality = inferring_setting_dict["modality"]
        if self.modality == None or self.modality == "all":
            self.modality = [
                int(i) for i in range(len(self.dataset_yaml["channel_names"]))
            ]
        self.fold = inferring_setting_dict["fold"]
        self.split = inferring_setting_dict["split"]
        self.original_img_folder = inferring_setting_dict["original_img_folder"]
        self.predictions_save_folder = inferring_setting_dict["predictions_save_folder"]
        self.model_path = inferring_setting_dict["model_path"]
        self.device_dict = inferring_setting_dict["device"]
        self.overwrite = inferring_setting_dict["overwrite"]
        self.save_probabilities = inferring_setting_dict["save_probabilities"]
        self.patch_size = inferring_setting_dict["patch_size"]
        self.tile_step_size = inferring_setting_dict["tile_step_size"]
        self.use_gaussian = inferring_setting_dict["use_gaussian"]
        self.perform_everything_on_gpu = inferring_setting_dict[
            "perform_everything_on_gpu"
        ]
        self.use_mirroring = inferring_setting_dict["use_mirroring"]
        self.allowed_mirroring_axes = inferring_setting_dict["allowed_mirroring_axes"]
        self.num_processes = inferring_setting_dict["num_processes"]

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
        if not os.path.exists(self.predictions_save_folder):
            os.makedirs(self.predictions_save_folder)

        self.num_segmentation_heads = self.config_dict["Network"]["out_channels"]
        self.network = self.get_networks(self.config_dict["Network"])
        load_pretrained_weights(
            self.network, self.model_path, load_all=True, verbose=True
        )

        self.network.to(self.device)

        print("Compiling network...")
        self.network = torch.compile(self.network)

    def get_networks(self, network_settings: Dict):
        if "need_features" in network_settings.keys():
            network_settings["need_features"] = False
        if "weight_path" in network_settings.keys():
            del network_settings["weight_path"]

        return build_network(network_settings)

    def get_images_dict(
        self,
        source_img_folder: str,
        modality: Union[list],
        predictions_folder: str = None,
    ):
        # If having self.fold, decide the case list based on the dataset_split.json.
        # Or get all the cases in the source folder
        files_dict_of_lists = dict()
        files_ending = self.dataset_yaml["files_ending"]

        checking_length = len("_{:0>4d}".format(modality[0]) + files_ending)
        modalities_with_ending_lst = []
        for i in modality:
            modalities_with_ending_lst.append("_{:0>4d}".format(i) + files_ending)

        # list of all files in source folder
        file_lst = [
            i
            for i in os.listdir(source_img_folder)
            if i[-checking_length:] in modalities_with_ending_lst
        ]
        file_lst.sort()

        # get all the needed cases' name
        if self.split in ["val", "train"] and self.fold is not None:
            identifier = self.dataset_split["fold" + str(self.fold)][self.split]
        elif self.split == "test":
            identifier = self.dataset_split[self.split]
        elif self.split in ["val", "train"] and self.fold is None:
            raise Exception(
                "You should given the fold({}) or the split({}) correctly.".format(
                    self.fold, self.split
                )
            )
        else:
            name_length = len(self.dataset_name) + 5
            identifier = list(set([i[:name_length] for i in file_lst]))

        # get needed files' path
        for i in identifier:
            case_name_length = len(i)
            one_idx_all_files_lst = [
                os.path.join(source_img_folder, file)
                for file in file_lst
                if file[:case_name_length] == i
            ]
            one_idx_all_files_lst.sort()
            assert len(modality) == len(
                one_idx_all_files_lst
            ), "The number of modality({}) is not equal to the number of found files({}).".format(
                len(self.dataset_yaml["channel_names"]), len(one_idx_all_files_lst)
            )
            files_dict_of_lists[i] = one_idx_all_files_lst

        print(
            "There are {} files in the source folder with {} modality(ies).".format(
                len(file_lst), len(self.dataset_yaml["channel_names"])
            )
        )
        print(
            "The number of cases need to predict is {}.".format(
                len(files_dict_of_lists)
            )
        )

        # remove already predicted files form the lists
        if not self.overwrite:
            case_name_lst = list(files_dict_of_lists.keys())
            output_filename_truncated = [
                os.path.join(predictions_folder, i) for i in case_name_lst
            ]

            tmp = [
                os.path.isfile(i + self.dataset_yaml["files_ending"])
                for i in output_filename_truncated
            ]
            # save_probabilities means saving both hard labels and predicted probabilities
            if self.save_probabilities:
                tmp2 = [os.path.isfile(i + ".npy") for i in output_filename_truncated]
                tmp = [i and j for i, j in zip(tmp, tmp2)]
            existing_indexes = [i for i, j in enumerate(tmp) if j]

            # cases truly need to predict
            for index in existing_indexes:
                del files_dict_of_lists[case_name_lst[index]]
            print(
                f"overwrite was set to {self.overwrite}, so I am only working on cases that haven't been predicted yet. "
                f"That's {len(case_name_lst) - len(existing_indexes)} cases."
            )
        return files_dict_of_lists

    def _internal_get_sliding_window_slicers(self, image_size: Tuple[int, ...]):
        slicers = []
        if len(self.patch_size) < len(image_size):
            assert len(self.patch_size) == len(image_size) - 1, (
                "if tile_size has less entries than image_size, "
                "len(tile_size) "
                "must be one shorter than len(image_size) "
                "(only dimension "
                "discrepancy of 1 allowed)."
            )
            steps = compute_steps_for_sliding_window(
                image_size[1:], self.patch_size, self.tile_step_size
            )
            if self.verbose:
                print(
                    f"n_steps {image_size[0] * len(steps[0]) * len(steps[1])}, image size is"
                    f" {image_size}, tile_size {self.patch_size}, "
                    f"tile_step_size {self.tile_step_size}\nsteps:\n{steps}"
                )
            for d in range(image_size[0]):
                for sx in steps[0]:
                    for sy in steps[1]:
                        slicers.append(
                            tuple(
                                [
                                    slice(None),
                                    d,
                                    *[
                                        slice(si, si + ti)
                                        for si, ti in zip((sx, sy), self.patch_size)
                                    ],
                                ]
                            )
                        )
        else:
            steps = compute_steps_for_sliding_window(
                image_size, self.patch_size, self.tile_step_size
            )
            if self.verbose:
                print(
                    f"n_steps {np.prod([len(i) for i in steps])}, image size is {image_size}, tile_size {self.patch_size}, "
                    f"tile_step_size {self.tile_step_size}\nsteps:\n{steps}"
                )
            for sx in steps[0]:
                for sy in steps[1]:
                    for sz in steps[2]:
                        slicers.append(
                            tuple(
                                [
                                    slice(None),
                                    *[
                                        slice(si, si + ti)
                                        for si, ti in zip((sx, sy, sz), self.patch_size)
                                    ],
                                ]
                            )
                        )
        return slicers

    def _combine_network_outputs(self, x):
        if isinstance(x, dict):
            x = x["pred"]

        if isinstance(x, (list, tuple)):
            return x[0]
        else:
            return x

    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        prediction = self._combine_network_outputs(self.network(x))

        if mirror_axes is not None:
            # check for invalid numbers in mirror_axes
            # x should be 5d for 3d images and 4d for 2d. so the max value of mirror_axes cannot exceed len(x.shape) - 3
            assert (
                max(mirror_axes) <= len(x.shape) - 3
            ), "mirror_axes does not match the dimension of the input!"

            mirror_axes = [m + 2 for m in mirror_axes]
            axes_combinations = [
                c
                for i in range(len(mirror_axes))
                for c in itertools.combinations(mirror_axes, i + 1)
            ]
            for axes in axes_combinations:
                prediction += torch.flip(
                    self._combine_network_outputs(self.network(torch.flip(x, axes))),
                    axes,
                )
            prediction /= len(axes_combinations) + 1

        return prediction

    def predict_sliding_window_return_logits(self, input_image: torch.Tensor):
        assert (
            len(self.patch_size) == 3
        ), "The patch size in 3D prediction for 3D volume should have 3 elements."
        assert isinstance(input_image, torch.Tensor)

        self.network.eval()
        empty_cache(self.device)

        with torch.no_grad():
            with (
                torch.autocast(self.device.type, enabled=True)
                if self.device.type == "cuda"
                else dummy_context()
            ):
                assert (
                    len(input_image.shape) == 4
                ), "input_image must be a 4D np.ndarray or torch.Tensor (c, z, y, x)"

                if self.verbose:
                    print("Input shape:", input_image.shape)
                    print("step_size:", self.tile_step_size)
                    print(
                        "mirror_axes:",
                        self.allowed_mirroring_axes if self.use_mirroring else None,
                    )

                # if input_image is smaller than tile_size we need to pad it to tile_size.
                data, slicer_revert_padding = pad_nd_image(
                    input_image, self.patch_size, "constant", {"value": 0}, True, None
                )

                slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

                # preallocate results and num_predictions
                results_device = (
                    self.device
                    if self.perform_everything_on_gpu
                    else torch.device("cpu")
                )
                if self.verbose:
                    print("preallocating arrays")
                try:
                    data = data.to(self.device)
                    predicted_logits = torch.zeros(
                        (self.num_segmentation_heads, *data.shape[1:]),
                        dtype=torch.half,
                        device=results_device,
                    )
                    n_predictions = torch.zeros(
                        data.shape[1:], dtype=torch.half, device=results_device
                    )
                    if self.use_gaussian:
                        gaussian = compute_gaussian(
                            tuple(self.patch_size),
                            sigma_scale=1.0 / 8,
                            value_scaling_factor=1000,
                            device=results_device,
                        )
                except RuntimeError:
                    # sometimes the stuff is too large for GPUs. In that case fall back to CPU
                    results_device = torch.device("cpu")
                    data = data.to(results_device)
                    predicted_logits = torch.zeros(
                        (self.num_segmentation_heads, *data.shape[1:]),
                        dtype=torch.half,
                        device=results_device,
                    )
                    n_predictions = torch.zeros(
                        data.shape[1:], dtype=torch.half, device=results_device
                    )
                    if self.use_gaussian:
                        gaussian = compute_gaussian(
                            tuple(self.patch_size),
                            sigma_scale=1.0 / 8,
                            value_scaling_factor=1000,
                            device=results_device,
                        )
                finally:
                    empty_cache(self.device)

                print("running prediction")
                for sl in tqdm(slicers, disable=not self.allow_tqdm):
                    workon = data[sl][None]
                    workon = workon.to(self.device, non_blocking=False)

                    prediction = self._internal_maybe_mirror_and_predict(workon)[0].to(
                        results_device
                    )

                    predicted_logits[sl] += (
                        prediction * gaussian if self.use_gaussian else prediction
                    )
                    n_predictions[sl[1:]] += gaussian if self.use_gaussian else 1

                predicted_logits /= n_predictions
        empty_cache(self.device)
        return predicted_logits[tuple([slice(None), *slicer_revert_padding[1:]])]

    def predict_sliding_window_2d_slices_return_logits(self, input_image: torch.Tensor):
        assert (
            len(self.patch_size) == 3
        ), "The patch size in 3D prediction for 3D volume should have 3 elements."
        assert isinstance(input_image, torch.Tensor)

        self.network.eval()
        empty_cache(self.device)

        with torch.no_grad():
            with (
                torch.autocast(self.device.type, enabled=True)
                if self.device.type == "cuda"
                else dummy_context()
            ):
                assert (
                    len(input_image.shape) == 4
                ), "input_image must be a 4D np.ndarray or torch.Tensor (c, z, y, x)"

                if self.verbose:
                    print("Input shape:", input_image.shape)
                    print("step_size:", self.tile_step_size)
                    print(
                        "mirror_axes:",
                        self.allowed_mirroring_axes if self.use_mirroring else None,
                    )

                # if input_image is smaller than tile_size we need to pad it to tile_size.
                data, slicer_revert_padding = pad_nd_image(
                    input_image, self.patch_size, "constant", {"value": 0}, True, None
                )

                slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

                # preallocate results and num_predictions
                results_device = (
                    self.device
                    if self.perform_everything_on_gpu
                    else torch.device("cpu")
                )
                if self.verbose:
                    print("preallocating arrays")
                try:
                    data = data.to(self.device)
                    predicted_logits = torch.zeros(
                        (self.num_segmentation_heads, *data.shape[1:]),
                        dtype=torch.half,
                        device=results_device,
                    )
                    n_predictions = torch.zeros(
                        data.shape[1:], dtype=torch.half, device=results_device
                    )
                    if self.use_gaussian:
                        gaussian = compute_gaussian(
                            tuple(self.patch_size),
                            sigma_scale=1.0 / 8,
                            value_scaling_factor=1000,
                            device=results_device,
                        )
                except RuntimeError:
                    # sometimes the stuff is too large for GPUs. In that case fall back to CPU
                    results_device = torch.device("cpu")
                    data = data.to(results_device)
                    predicted_logits = torch.zeros(
                        (self.num_segmentation_heads, *data.shape[1:]),
                        dtype=torch.half,
                        device=results_device,
                    )
                    n_predictions = torch.zeros(
                        data.shape[1:], dtype=torch.half, device=results_device
                    )
                    if self.use_gaussian:
                        gaussian = compute_gaussian(
                            tuple(self.patch_size),
                            sigma_scale=1.0 / 8,
                            value_scaling_factor=1000,
                            device=results_device,
                        )
                finally:
                    empty_cache(self.device)

                print("running prediction")
                for sl in tqdm(slicers, disable=not self.allow_tqdm):
                    workon = data[sl].permute(1, 0, 2, 3)
                    workon = workon.to(self.device, non_blocking=False)

                    prediction = (
                        self._internal_maybe_mirror_and_predict(workon)
                        .to(results_device)
                        .permute(1, 0, 2, 3)
                    )

                    predicted_logits[sl] += (
                        prediction * gaussian if self.use_gaussian else prediction
                    )
                    n_predictions[sl[1:]] += gaussian if self.use_gaussian else 1

                predicted_logits /= n_predictions
        empty_cache(self.device)
        return predicted_logits[tuple([slice(None), *slicer_revert_padding[1:]])]

    def predict_logits_from_preprocessed_data(
        self, data: torch.Tensor, predict_way: str
    ) -> torch.Tensor:
        original_perform_everything_on_gpu = self.perform_everything_on_gpu
        with torch.no_grad():
            prediction = None
            if self.perform_everything_on_gpu:
                try:
                    data.to(self.device)
                    if predict_way == "3d":
                        prediction = self.predict_sliding_window_return_logits(data)
                    elif predict_way == "2d":
                        prediction = (
                            self.predict_sliding_window_2d_slices_return_logits(data)
                        )
                    else:
                        raise Exception('predict_way should be "2d" or "3d"')

                except RuntimeError:
                    print(
                        "Prediction with perform_everything_on_gpu=True failed due to insufficient GPU memory. "
                        "Falling back to perform_everything_on_gpu=False. Not a big deal, just slower..."
                    )
                    print("Error:")
                    traceback.print_exc()
                    prediction = None
                    self.perform_everything_on_gpu = False

            if prediction is None:
                if predict_way == "3d":
                    prediction = self.predict_sliding_window_return_logits(data)
                elif predict_way == "2d":
                    prediction = self.predict_sliding_window_2d_slices_return_logits(
                        data
                    )
                else:
                    raise Exception('predict_way should be "2d" or "3d"')

            print("Prediction done, transferring to CPU if needed")
            prediction = prediction.to("cpu")
            self.perform_everything_on_gpu = original_perform_everything_on_gpu

        return prediction

    # infer a single image
    def predict_single_npy_array(
        self,
        input_images_lst: np.ndarray,
        preprocess_config: str,
        save_or_return_probabilities: bool = False,
    ):
        """
        image_properties must only have a 'spacing' key!
        """
        ppa = Preprocessor(self.dataset_name, verbose=self.verbose)
        if self.verbose:
            print("preprocessing")
        # input_images_lst means one case with its all modalities
        data, _, property = ppa.run_case(input_images_lst, None, preprocess_config)

        data_lst = []
        count = 0
        for i in range(len(property["shapes"])):
            n_channel = property["shapes"][i][0]
            if i in self.modality:
                data_lst.append(data[list(range(count, count + n_channel))])
            count += n_channel
        data = torch.from_numpy(np.vstack(data_lst)).contiguous().float()

        if self.verbose:
            print("predicting")
        predicted_logits = self.predict_logits_from_preprocessed_data(
            data, preprocess_config
        )
        # data is on cpu here

        if self.verbose:
            print("resampling to original shape")

        ret = convert_predicted_logits_to_segmentation_with_correct_shape(
            predicted_logits,
            property,
            return_probabilities=save_or_return_probabilities,
        )

        if save_or_return_probabilities:
            return ret[0], ret[1]
        else:
            return ret

    # infer a list of images
    def _generate_data_iterator(self, images_dict, preprocess_config):
        return preprocessing_iterator_fromfiles(
            images_dict,
            self.modality,
            preprocess_config,
            self.predictions_save_folder,
            self.dataset_name,
            self.device.type == "cuda",
            self.num_processes,
            self.verbose,
        )

    def predict_from_data_iterator(
        self, data_iterator, predict_way, save_or_return_probabilities=False
    ):
        """
        each element returned by data_iterator must be a dict with 'data', 'output_file' and 'data_properites' keys!
        If 'output_file' is None, the result will be returned instead of written to a file
        """
        with multiprocessing.get_context("spawn").Pool(
            self.num_processes
        ) as export_pool:
            worker_list = [i for i in export_pool._pool]
            r = []
            for preprocessed in data_iterator:
                data = preprocessed["data"]
                properties = preprocessed["data_properites"]
                if isinstance(properties, str):
                    properties = open_pickle(properties)

                if isinstance(data, str):
                    data = np.load(data)
                    data_lst = []
                    count = 0
                    for i in range(len(properties["shapes"])):
                        n_channel = properties["shapes"][i][0]
                        if i in self.modality:
                            data_lst.append(data[list(range(count, count + n_channel))])
                        count += n_channel
                    data = torch.from_numpy(np.vstack(data_lst)).contiguous().float()

                ofile = preprocessed["output_file"]
                if ofile is not None:
                    print(f"\nPredicting {os.path.basename(ofile)}:")
                else:
                    print(f"\nPredicting image of shape {data.shape}:")

                print(f"perform_everything_on_gpu: {self.perform_everything_on_gpu}")

                # let's not get into a runaway situation where the GPU predicts so fast that the disk has to b swamped with
                # npy files
                proceed = not check_workers_alive_and_busy(
                    export_pool, worker_list, r, allowed_num_queued=2
                )
                while not proceed:
                    # print('sleeping')
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(
                        export_pool, worker_list, r, allowed_num_queued=2
                    )

                prediction = self.predict_logits_from_preprocessed_data(
                    data, predict_way
                ).cpu()

                if ofile is not None:
                    # this needs to go into background processes
                    # export_prediction_from_logits(prediction, properties, configuration_manager, plans_manager,
                    #                               dataset_json, ofile, save_probabilities)
                    print(
                        "sending off prediction to background worker for resampling and export"
                    )
                    r.append(
                        export_pool.starmap_async(
                            export_prediction_from_logits,
                            (
                                (
                                    prediction,
                                    properties,
                                    ofile,
                                    self.dataset_yaml,
                                    save_or_return_probabilities,
                                    self.num_processes,
                                ),
                            ),
                        )
                    )
                else:
                    print("sending off prediction to background worker for resampling")
                    r.append(
                        export_pool.starmap_async(
                            convert_predicted_logits_to_segmentation_with_correct_shape,
                            (
                                (
                                    prediction,
                                    properties,
                                    save_or_return_probabilities,
                                    self.num_processes,
                                ),
                            ),
                        )
                    )
                if ofile is not None:
                    print(f"done with {os.path.basename(ofile)}")
                else:
                    print(f"\nDone with image of shape {data.shape}:")
            if ofile is not None:
                ret = [i.get() for i in r]
            else:
                ret = [i.get()[0] for i in r]

        # clear lru cache
        compute_gaussian.cache_clear()
        # clear device cache
        empty_cache(self.device)

        return ret

    def predict_from_file(
        self,
        source_img_folder: str,
        output_img_folder: str,
        modality: List,
        preprocess_config: str,
        save_or_return_probabilities=False,
    ):
        images_dict = self.get_images_dict(
            source_img_folder, modality, output_img_folder
        )
        if len(images_dict) == 0:
            return

        data_iterator = self._generate_data_iterator(images_dict, preprocess_config)

        return self.predict_from_data_iterator(
            data_iterator, preprocess_config, save_or_return_probabilities
        )


if __name__ == "__main__":
    import SimpleITK as sitk
    from wcode.utils.file_operations import open_yaml

    predict_configs = {
        "dataset_name": "HNTSMRG2024mid",
        "modality": [0, 1, 2],
        "fold": 0,
        "split": "val",
        "original_img_folder": "./Dataset/HNTSMRG2024mid/images",
        "predictions_save_folder": "./Predictions/HNTSMRG2024mid",
        "model_path": "./Logs/HNTSMRG2024mid/HNTSMRG2024mid_oversample/fold_0/checkpoint_latest.pth",
        "device": {"gpu": [0]},
        "overwrite": True,
        "save_probabilities": False,
        "patch_size": [56, 224, 160],
        "tile_step_size": 0.5,
        "use_gaussian": True,
        "perform_everything_on_gpu": True,
        "use_mirroring": False,
        "allowed_mirroring_axes": None,
        "num_processes": 16,
    }
    config_dict = open_yaml("./Configs/HNTSMRG2024mid_oversample.yaml")
    config_dict["Inferring_settings"] = predict_configs
    pre = PatchBasedPredictor(config_dict, allow_tqdm=True)
    pred = pre.predict_single_npy_array(
        [
            "./Dataset/HNTSMRG2024mid/images/HNTSMRG2024mid_0003_0000.nii.gz",
            "./Dataset/HNTSMRG2024mid/images/HNTSMRG2024mid_0003_0001.nii.gz",
            "./Dataset/HNTSMRG2024mid/images/HNTSMRG2024mid_0003_0002.nii.gz",
        ],
        "3d",
    )
    pred_obj = sitk.GetImageFromArray(pred)
    pred_obj.CopyInformation(
        sitk.ReadImage(
            "./Dataset/HNTSMRG2024mid/images/HNTSMRG2024mid_0003_0002.nii.gz"
        )
    )
    sitk.WriteImage(pred_obj, "./prediction.nii.gz")

    # predict_configs = {
    #     "dataset_name": "HNTSMRG2024pre",
    #     "modality": [0],
    #     "fold": 0,
    #     "split": "val",
    #     "original_img_folder": "./Dataset/HNTSMRG2024pre/images",
    #     "predictions_save_folder": "./Predictions/HNTSMRG2024pre",
    #     "model_path": "./Logs/HNTSMRG2024pre/HNTSMRG2024pre_oversample/fold_0/checkpoint_final.pth",
    #     "device": {"gpu": [3]},
    #     "overwrite": True,
    #     "save_probabilities": False,
    #     "patch_size": [56, 224, 160],
    #     "tile_step_size": 0.5,
    #     "use_gaussian": True,
    #     "perform_everything_on_gpu": True,
    #     "use_mirroring": False,
    #     "allowed_mirroring_axes": None,
    #     "num_processes": 16,
    # }
    # config_dict = open_yaml("./Configs/HNTSMRG2024pre_oversample.yaml")
    # config_dict["Inferring_settings"] = predict_configs
    # pre = PatchBasedPredictor(config_dict, allow_tqdm=True)
    # pred = pre.predict_single_npy_array(
    #     ["./Dataset/HNTSMRG2024mid/images/HNTSMRG2024mid_0003_0000.nii.gz"], "3d"
    # )
    # pred_obj = sitk.GetImageFromArray(pred)
    # pred_obj.CopyInformation(
    #     sitk.ReadImage(
    #         "./Dataset/HNTSMRG2024mid/images/HNTSMRG2024mid_0003_0000.nii.gz"
    #     )
    # )
    # sitk.WriteImage(pred_obj, "./prediction.nii.gz")
