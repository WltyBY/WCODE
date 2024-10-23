import os
import cv2
import multiprocessing
import shutil
import random
import numpy as np

from time import sleep
from typing import Union, List
from tqdm import tqdm

from wcode.preprocessing.cropping import crop_to_mask
from wcode.preprocessing.normalizing import normalization_schemes_to_object
from wcode.preprocessing.resampling import (
    resample_npy_with_channels_on_spacing,
    compute_new_shape,
)
from wcode.utils.file_operations import open_yaml, save_pickle, open_json
from wcode.utils.data_io import (
    read_sitk_case,
    read_2d_img,
    files_ending_for_sitk,
    files_ending_for_2d_img,
    create_lists_from_splitted_dataset_folder,
)


class Preprocessor(object):
    def __init__(
        self, dataset_name: str = None, random_seed=319, verbose: bool = False
    ):
        self.verbose = verbose
        random.seed(random_seed)
        np.random.seed(random_seed)
        os.environ["PYTHONHASHSEED"] = str(random_seed)
        
        if dataset_name:
            self.dataset_figureprint = open_json(
                os.path.join(
                    "./Dataset_preprocessed", dataset_name, "dataset_figureprint.json"
                )
            )
            self.dataset_yaml = open_yaml(
                os.path.join("./Dataset_preprocessed", dataset_name, "dataset.yaml")
            )
            self.plans_json = open_json(
                os.path.join("./Dataset_preprocessed", dataset_name, "plans.json")
            )
        else:
            raise Exception(
                "You should provide dataset_name to get dataset_figureprint.json and dataset.yaml"
            )

        if self.dataset_yaml["files_ending"] in files_ending_for_sitk:
            self.img_Reader = read_sitk_case
            self.general_img_flag = False
        elif self.dataset_yaml["files_ending"] in files_ending_for_2d_img:
            self.img_Reader = read_2d_img
            self.general_img_flag = True
        else:
            raise Exception("Files' ending NOT SUPPORT!!!")

        self.dataset_name = dataset_name

    def run_case_npy(
        self,
        data: np.ndarray,
        seg: Union[np.ndarray, None],
        properties: dict,
        preprocess_config: str,
    ):
        data = np.copy(data)
        if seg is not None:
            assert (
                data.shape[1:] == seg.shape[1:]
            ), "Shape mismatch between image and segmentation. Please fix your dataset and make use of the --verify_dataset_integrity flag to ensure everything is correct"
            seg = np.copy(seg)

        # crop
        shape_before_cropping = data.shape[1:]
        data, seg, bbox = crop_to_mask(data, seg, {"threshold": 0})
        properties["shape_before_cropping"] = shape_before_cropping
        properties["bbox_used_for_cropping"] = bbox
        properties["shape_after_cropping_and_before_resampling"] = data.shape[1:]

        # normalize
        data = self._normalize(
            data,
            seg,
            self.plans_json,
            foreground_intensity_properties_per_channel=self.dataset_figureprint[
                "foreground_intensity_properties_per_channel"
            ],
            shapes=None,
        )

        # resample
        original_spacing = properties["spacing"]
        target_spacing = properties["target_spacing"]

        if preprocess_config == "2d":
            target_spacing[-1] = original_spacing[-1]

        old_shape = data.shape[1:]
        new_shape = compute_new_shape(old_shape, original_spacing, target_spacing)
        data = resample_npy_with_channels_on_spacing(
            data,
            original_spacing,
            target_spacing,
            channel_names=self.dataset_yaml["channel_names"].values(),
        )
        if seg is not None:
            seg = resample_npy_with_channels_on_spacing(
                seg, original_spacing, target_spacing, channel_names=["label"]
            )

        if self.verbose:
            print(
                f"old shape: {old_shape}, new_shape: {new_shape},"
                f"old_spacing: {original_spacing}, new_spacing: {target_spacing}"
            )

        return data, seg

    def run_2d_img_npy(
        self,
        data: np.ndarray,
        seg: Union[np.ndarray, None],
        properties: dict,
    ):
        # properties are the shapes of images from all modelities for 2d image.
        # properties: {"shapes": (c, h, w)}
        data = np.copy(data)
        if seg is not None:
            assert (
                data.shape[1:] == seg.shape[1:]
            ), "Shape mismatch between image and segmentation. Please fix your dataset and make use of the --verify_dataset_integrity flag to ensure everything is correct"
            seg = np.copy(seg)

        # normalize
        data = self._normalize(
            data,
            seg,
            self.plans_json,
            foreground_intensity_properties_per_channel=None,
            shapes=properties["shapes"],
        )

        # process seg, seg_new from c, h, w to h, w, c
        seg_new = np.zeros_like(seg.transpose(1, 2, 0))
        for i, key in enumerate(self.dataset_yaml["labels"].keys()):
            label = np.array(self.dataset_yaml["labels"][key])
            seg_new[seg.transpose(1, 2, 0) == label] = i

        return data, seg_new.transpose(2, 0, 1)[0][None]

    def run_case(
        self,
        image_files: List[str],
        seg_file: Union[str, None],
        preprocess_config: str,
    ):
        """
        seg file can be none (test cases)

        order of operations is: transpose -> crop -> resample
        so when we export we need to run the following order: resample -> crop -> transpose (we could also run
        transpose at a different place, but reverting the order of operations done during preprocessing seems cleaner)
        """
        # load image(s)
        data, data_properties = self.img_Reader(image_files)
        if not self.general_img_flag:
            data_properties["target_spacing"] = self.plans_json["target_spacing"]

        # if possible, load seg
        if seg_file is not None:
            if not self.general_img_flag:
                seg, _ = self.img_Reader(seg_file)
            else:
                seg = cv2.cvtColor(
                    cv2.imread(seg_file[0]), cv2.COLOR_BGR2RGB
                ).transpose(2, 0, 1)
        else:
            seg = None

        if not self.general_img_flag:
            data, seg = self.run_case_npy(data, seg, data_properties, preprocess_config)
        else:
            data, seg = self.run_2d_img_npy(data, seg, data_properties)

        return data, seg, data_properties

    def run_case_save(
        self,
        output_filename_truncated: str,
        image_files: List[str],
        seg_file: str,
        preprocess_config: str,
    ):
        data, seg, properties = self.run_case(image_files, seg_file, preprocess_config)
        properties["oversample"] = self._sample_foreground_locations(
            seg, output_filename_truncated
        )

        np.save(output_filename_truncated + ".npy", arr=data)
        np.save(output_filename_truncated + "_seg.npy", arr=seg)
        save_pickle(properties, output_filename_truncated + ".pkl")

    def _normalize(
        self,
        data: np.ndarray,
        seg: np.ndarray,
        dataset_configuration: dict,
        foreground_intensity_properties_per_channel: dict,
        shapes: dict,
    ) -> np.ndarray:
        if (foreground_intensity_properties_per_channel is None) and (shapes is None):
            raise ValueError("Only one of them can be None")
        if (foreground_intensity_properties_per_channel is not None) and (
            shapes is not None
        ):
            raise ValueError("Only one of them can not be None")

        if foreground_intensity_properties_per_channel is not None:
            for c in range(data.shape[0]):
                scheme = dataset_configuration["normalization_schemes"][c]
                normalizer_class = normalization_schemes_to_object[scheme]
                if normalizer_class is None:
                    raise RuntimeError(
                        "Unable to locate class '%s' for normalization" % scheme
                    )

                normalizer = normalizer_class(
                    use_mask_for_norm=self.plans_json["use_mask_for_norm"][c],
                    intensityproperties=foreground_intensity_properties_per_channel[
                        str(c)
                    ],
                )
                data[c] = normalizer.run(data[c], seg[0])
        elif shapes is not None:
            channel_lst = []
            for shape in shapes:
                channel_lst.append(shape[0])
            cumsum_id = np.cumsum(channel_lst)
            for c in range(data.shape[0]):
                normalizer_id = (cumsum_id > c).tolist().index(True)
                scheme = dataset_configuration["normalization_schemes"][normalizer_id]

                normalizer_class = normalization_schemes_to_object[scheme]
                if normalizer_class is None:
                    raise RuntimeError(
                        "Unable to locate class '%s' for normalization" % scheme
                    )

                normalizer = normalizer_class(
                    use_mask_for_norm=self.plans_json["use_mask_for_norm"][
                        normalizer_id
                    ],
                    intensityproperties={},
                )
                data[c] = normalizer.run(data[c], seg[0])

        return data

    def _sample_foreground_locations(
        self, seg: np.ndarray, name, num_samples=10000, seed: int = 319
    ):
        # seg in (c, z, y, x)
        # At least 1% of the class voxels need to be selected, otherwise it may be too
        # sparse
        min_percent_coverage = 0.01
        selected_dict = {}

        for i in range(1, len(self.dataset_yaml["labels"])):
            all_locs = np.argwhere(seg[0] != i)
            if len(all_locs) != 0:
                target_num_samples = min(num_samples, len(all_locs))
                target_num_samples = max(
                    target_num_samples,
                    int(np.ceil(len(all_locs) * min_percent_coverage)),
                )

                rndst = np.random.RandomState(seed)
                selected = all_locs[
                    rndst.choice(len(all_locs), target_num_samples, replace=False)
                ]
                selected_dict[i] = selected
            else:
                selected_dict[i] = "No_fg_point"

        return selected_dict

    def run(
        self,
        preprocess_config,
        num_processes: int = 16,
    ):
        raw_data_folder = os.path.join("./Dataset", self.dataset_name)
        assert os.path.isdir(
            raw_data_folder
        ), "The requested dataset could not be found in Dataset folder"

        if self.verbose:
            print(f"Preprocessing the following configuration: ")
            print("-----Normalization-----")
            print(
                self.dataset_figureprint["foreground_intensity_properties_per_channel"]
            )
            print("-------Resample-------")
            print("Target Spacing:", self.plans_json["target_spacing"])

        data_split_json = os.path.join(
            "./Dataset_preprocessed", self.dataset_name, "dataset_split.json"
        )
        assert os.path.isfile(data_split_json), (
            "Expected plans file (%s) not found. Do dataset analysis first."
            % data_split_json
        )
        data_split_dict = open_json(data_split_json)
        identifiers = (
            data_split_dict["fold0"]["train"] + data_split_dict["fold0"]["val"]
        )

        files_ending = self.dataset_yaml["files_ending"]

        if self.general_img_flag:
            preprocess_config = "2d"

        print("Proprecessing in", preprocess_config, "configuration...")

        # make save folder for preprocessed images and seg
        output_directory = os.path.join(
            "./Dataset_preprocessed",
            self.dataset_name,
            "preprocessed_datas_" + preprocess_config,
        )
        if os.path.isdir(output_directory):
            shutil.rmtree(output_directory, ignore_errors=True)
        os.makedirs(output_directory, exist_ok=True)

        output_filenames_truncated = [
            os.path.join(output_directory, i) for i in identifiers
        ]

        folder_lst = [
            i
            for i in os.listdir(raw_data_folder)
            if not os.path.isfile(os.path.join(raw_data_folder, i))
        ]

        if {"images", "labels"}.issubset(folder_lst) and len(folder_lst) == 2:
            # list of lists with image filenames
            image_fnames = create_lists_from_splitted_dataset_folder(
                os.path.join("./Dataset", self.dataset_name, "images"),
                files_ending,
                identifiers,
            )
            # list of segmentation filenames
            seg_fnames = [
                [
                    os.path.join(
                        "./Dataset", self.dataset_name, "labels", i + files_ending
                    )
                ]
                for i in identifiers
            ]

            gt_segmentation_path = os.path.join(
                "./Dataset_preprocessed", self.dataset_name, "gt_segmentations"
            )
            if os.path.isdir(gt_segmentation_path):
                shutil.rmtree(gt_segmentation_path, ignore_errors=True)
            os.makedirs(gt_segmentation_path, exist_ok=True)
            shutil.copytree(
                os.path.join("./Dataset", self.dataset_name, "labels"),
                gt_segmentation_path,
                dirs_exist_ok=True,
            )
        elif {"imagesTr", "labelsTr", "imagesVal", "labelsVal"}.issubset(folder_lst):
            image_fnames_train = create_lists_from_splitted_dataset_folder(
                os.path.join("./Dataset", self.dataset_name, "imagesTr"),
                files_ending,
                data_split_dict["fold0"]["train"],
            )
            image_fnames_val = create_lists_from_splitted_dataset_folder(
                os.path.join("./Dataset", self.dataset_name, "imagesVal"),
                files_ending,
                data_split_dict["fold0"]["val"],
            )
            seg_fnames_train = [
                [
                    os.path.join(
                        "./Dataset", self.dataset_name, "labelsTr", i + files_ending
                    )
                ]
                for i in data_split_dict["fold0"]["train"]
            ]
            seg_fnames_val = [
                [
                    os.path.join(
                        "./Dataset", self.dataset_name, "labelsVal", i + files_ending
                    )
                ]
                for i in data_split_dict["fold0"]["val"]
            ]
            image_fnames = image_fnames_train + image_fnames_val
            seg_fnames = seg_fnames_train + seg_fnames_val

            gt_segmentation_path = os.path.join(
                "./Dataset_preprocessed", self.dataset_name, "gt_segmentations"
            )
            if os.path.isdir(gt_segmentation_path):
                shutil.rmtree(gt_segmentation_path, ignore_errors=True)
            os.makedirs(gt_segmentation_path, exist_ok=True)
            shutil.copytree(
                os.path.join("./Dataset", self.dataset_name, "labelsTr"),
                gt_segmentation_path,
                dirs_exist_ok=True,
            )
            shutil.copytree(
                os.path.join("./Dataset", self.dataset_name, "labelsVal"),
                gt_segmentation_path,
                dirs_exist_ok=True,
            )

        # multiprocessing magic.
        r = []
        with multiprocessing.get_context("spawn").Pool(num_processes) as p:
            for outfile, infiles, segfiles in zip(
                output_filenames_truncated, image_fnames, seg_fnames
            ):
                # print(outfile, infiles, segfiles)
                r.append(
                    p.starmap_async(
                        self.run_case_save,
                        ((outfile, infiles, segfiles, preprocess_config),),
                    )
                )
            remaining = list(range(len(output_filenames_truncated)))
            # p is pretty nifti. If we kill workers they just respawn but don't do any work.
            # So we need to store the original pool of workers.
            workers = [j for j in p._pool]
            with tqdm(
                desc=None, total=len(output_filenames_truncated), disable=self.verbose
            ) as pbar:
                while len(remaining) > 0:
                    all_alive = all([j.is_alive() for j in workers])
                    if not all_alive:
                        raise RuntimeError(
                            "Some background worker is 6 feet under. Yuck."
                        )
                    done = [i for i in remaining if r[i].ready()]
                    for _ in done:
                        pbar.update()
                    remaining = [i for i in remaining if i not in done]
                    sleep(0.1)

        print("Now checking files...")
        preprocessed_files = os.listdir(output_directory)
        if len(identifiers) * 3 != len(preprocessed_files):
            print("Maybe something wrong during preprocessing.")
            print(
                "There should be {} files, but find {} files.".format(
                    len(identifiers) * 3, len(preprocessed_files)
                )
            )
            preprocessed_files_set = set(
                [file.split(".")[0] for file in preprocessed_files if "seg" not in file]
            )
            error_cases = set(identifiers) - preprocessed_files_set
            for case in error_cases:
                print("Error for:", case)
        else:
            print("Judging by the number of files, the process is correct.")
