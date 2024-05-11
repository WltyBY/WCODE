import os
import multiprocessing
import shutil
import random
import numpy as np

from time import sleep
from typing import Union, List
from tqdm import tqdm

from wcode.preprocessing.cropping import crop_to_mask
from wcode.preprocessing.normalizing import find_normalizer
from wcode.preprocessing.resampling import (
    resample_npy_with_channels_on_spacing,
    compute_new_shape,
)
from wcode.utils.file_operations import open_yaml, save_pickle, open_json
from wcode.utils.data_io import (
    read_sitk_case,
    create_lists_from_splitted_dataset_folder,
)


class Preprocessor(object):
    def __init__(self, dataset_name: str = None, verbose: bool = False):
        self.verbose = verbose

        if dataset_name:
            self.dataset_figureprint = open_json(
                os.path.join(
                    "./Dataset_preprocessed", dataset_name, "dataset_figureprint.json"
                )
            )
            self.dataset_yaml = open_yaml(
                os.path.join("./Dataset", dataset_name, "dataset.yaml")
            )
            self.plans_json = open_json(
                os.path.join("./Dataset_preprocessed", dataset_name, "plans.json")
            )
        else:
            raise Exception(
                "You should provide dataset_name to get dataset_figureprint.json and dataset.yaml"
            )
        self.dataset_name = dataset_name

    def run_case_npy(
        self,
        data: np.ndarray,
        seg: Union[np.ndarray, None],
        properties: dict,
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
            self.dataset_yaml,
            self.dataset_figureprint["foreground_intensity_properties_per_channel"],
        )

        # resample
        original_spacing = properties["spacing"]
        target_spacing = properties["target_spacing"]

        old_shape = data.shape[1:]
        new_shape = compute_new_shape(old_shape, original_spacing, target_spacing)
        data = resample_npy_with_channels_on_spacing(
            data, original_spacing, target_spacing, is_seg=False
        )
        if seg is not None:
            seg = resample_npy_with_channels_on_spacing(
                seg, original_spacing, target_spacing, is_seg=True
            )

        if self.verbose:
            print(
                f"old shape: {old_shape}, new_shape: {new_shape},"
                f"old_spacing: {original_spacing}, new_spacing: {target_spacing}"
            )

        return data, seg

    def run_case(
        self,
        image_files: List[str],
        seg_file: Union[str, None],
    ):
        """
        seg file can be none (test cases)

        order of operations is: transpose -> crop -> resample
        so when we export we need to run the following order: resample -> crop -> transpose (we could also run
        transpose at a different place, but reverting the order of operations done during preprocessing seems cleaner)
        """
        # load image(s)
        data, data_properties = read_sitk_case(image_files)
        data_properties["target_spacing"] = self.plans_json["target_spacing"]

        # if possible, load seg
        if seg_file is not None:
            seg, _ = read_sitk_case(seg_file)
        else:
            seg = None

        data, seg = self.run_case_npy(data, seg, data_properties)

        return data, seg, data_properties

    def run_case_save(
        self,
        output_filename_truncated: str,
        image_files: List[str],
        seg_file: str,
    ):
        data, seg, properties = self.run_case(image_files, seg_file)        
        properties["oversample"] = self._sample_foreground_locations(seg, output_filename_truncated)

        np.savez(output_filename_truncated + ".npz", data=data, seg=seg)
        save_pickle(properties, output_filename_truncated + ".pkl")

    def _normalize(
        self,
        data: np.ndarray,
        seg: np.ndarray,
        dataset_configuration: dict,
        foreground_intensity_properties_per_channel: dict,
    ) -> np.ndarray:
        for c in range(data.shape[0]):
            scheme = list(dataset_configuration.keys())[c]
            normalizer_class = find_normalizer(scheme)
            if normalizer_class is None:
                raise RuntimeError(
                    "Unable to locate class '%s' for normalization" % scheme
                )

            normalizer = normalizer_class(
                use_mask_for_norm=self.plans_json["use_mask_for_norm"][c],
                intensityproperties=foreground_intensity_properties_per_channel[str(c)],
            )
            data[c] = normalizer.run(data[c], seg[0])

        return data

    @staticmethod
    def _sample_foreground_locations(
        seg: np.ndarray, name, num_samples=10000, seed: int = 319
    ):
        # seg in (c, z, y, x)
        # At least 1% of the class voxels need to be selected, otherwise it may be too
        # sparse
        min_percent_coverage = 0.01

        all_locs = np.argwhere(seg[0] != 0)
        target_num_samples = min(num_samples, len(all_locs))
        target_num_samples = max(
            target_num_samples, int(np.ceil(len(all_locs) * min_percent_coverage))
        )

        # Perhaps this can accelerate the sampling process?
        # When the number of elements increases, the time consumption of np.random.choice remains unchanged,
        # while the time cost increases using random.sample.
        if target_num_samples / len(all_locs) > 0.1:
            rndst = np.random.RandomState(seed)
            selected = all_locs[
                rndst.choice(len(all_locs), target_num_samples, replace=False)
            ]
        else:
            random.seed(seed)
            selected = all_locs[random.sample(range(len(all_locs)), target_num_samples)]

        return selected

    def run(
        self,
        num_processes: int = 8,
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

        # make save folder for preprocessed images and seg
        output_directory = os.path.join(
            "./Dataset_preprocessed", self.dataset_name, "preprocessed_datas"
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

        # multiprocessing magic.
        r = []
        with multiprocessing.get_context("spawn").Pool(num_processes) as p:
            for outfile, infiles, segfiles in zip(
                output_filenames_truncated, image_fnames, seg_fnames
            ):
                # print(outfile, infiles, segfiles)
                r.append(
                    p.starmap_async(self.run_case_save, ((outfile, infiles, segfiles),))
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
