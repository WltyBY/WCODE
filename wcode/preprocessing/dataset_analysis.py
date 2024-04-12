import multiprocessing
import os
import numpy as np

from tqdm import tqdm
from time import sleep
from typing import List, Tuple

from wcode.preprocessing.cropping import crop_to_mask
from wcode.preprocessing.resampling import get_lowres_axis, whether_anisotropy
from wcode.preprocessing.normalizing import find_normalizer
from wcode.utils.file_operations import open_yaml, save_yaml, open_json, save_json
from wcode.utils.data_io import get_all_img_and_label_path, read_sitk_case
from wcode.utils.dataset_split import dataset_split


class DatasetFingerprintExtractor(object):
    def __init__(
        self,
        dataset_name,
        split_rate: List[int] = [7, 1, 2],
        verbose: bool = False,
    ):
        # if False, show progress bar
        self.verbose = verbose

        if not dataset_name:
            raise Exception(
                "You should provide dataset_name to get dataset_figureprint.json and dataset.yaml"
            )
        self.dataset_name = dataset_name
        self.input_folder = os.path.join("./Dataset", dataset_name)
        self.dataset_json = open_yaml(os.path.join(self.input_folder, "dataset.yaml"))
        self.dataset, self.whehter_need_to_split = get_all_img_and_label_path(
            dataset_name, self.dataset_json["files_ending"]
        )

        if self.whehter_need_to_split:
            self.train_and_val_set, self.fold_cases_dict = dataset_split(
                list(self.dataset.keys()), split_rate=split_rate
            )
            self.fold_cases_dict["all_fold_is_the_same"] = False
        else:
            self.train_and_val_set = list(self.dataset["train"].keys()) + list(
                self.dataset["val"].keys()
            )
            self.fold_cases_dict = {}
            for i in range(5):
                self.fold_cases_dict["fold" + str(i)] = {}
                self.fold_cases_dict["fold" + str(i)]["train"] = sorted(
                    list(self.dataset["train"].keys())
                )
                self.fold_cases_dict["fold" + str(i)]["val"] = sorted(
                    list(self.dataset["val"].keys())
                )
            self.fold_cases_dict["test"] = sorted(list(self.dataset["test"].keys()))
            self.fold_cases_dict["all_fold_is_the_same"] = True

        # We don't want to use all foreground voxels because that can accumulate a lot of data (out of memory). It is
        # also not critically important to get all pixels as long as there are enough. Let's use 10e7 voxels in total
        # (for the entire dataset)
        self.num_foreground_voxels_for_intensitystats = 10e7

    @staticmethod
    def collect_foreground_intensities(
        segmentation: np.ndarray,
        images: np.ndarray,
        seed: int = 319,
        num_samples: int = 10000,
    ):
        """
        images=image with multiple channels = shape (c, x, y(, z))
        """
        assert images.ndim == 4
        assert segmentation.ndim == 4

        assert not np.any(
            np.isnan(segmentation)
        ), "Segmentation contains NaN values. grrrr.... :-("
        assert not np.any(np.isnan(images)), "Images contains NaN values. grrrr.... :-("

        rs = np.random.RandomState(seed)

        intensities_per_channel = []
        # we don't use the intensity_statistics_per_channel at all, it's just something that might be nice to have
        intensity_statistics_per_channel = []

        # segmentation is 4d: 1,x,y,z. We need to remove the empty dimension for the following code to work
        foreground_mask = segmentation[0] > 0

        for i in range(len(images)):
            foreground_pixels = images[i][foreground_mask]
            num_fg = len(foreground_pixels)
            # sample with replacement so that we don't get issues with cases that have less than num_samples
            # foreground_pixels. We could also just sample less in those cases but that would than cause these
            # training cases to be underrepresented
            intensities_per_channel.append(
                rs.choice(foreground_pixels, num_samples, replace=True)
                if num_fg > 0
                else []
            )
            intensity_statistics_per_channel.append(
                {
                    "mean": np.mean(foreground_pixels) if num_fg > 0 else np.nan,
                    "median": np.median(foreground_pixels) if num_fg > 0 else np.nan,
                    "min": np.min(foreground_pixels) if num_fg > 0 else np.nan,
                    "max": np.max(foreground_pixels) if num_fg > 0 else np.nan,
                    "percentile_99_5": (
                        np.percentile(foreground_pixels, 99.5) if num_fg > 0 else np.nan
                    ),
                    "percentile_00_5": (
                        np.percentile(foreground_pixels, 0.5) if num_fg > 0 else np.nan
                    ),
                }
            )

        return intensities_per_channel, intensity_statistics_per_channel

    @staticmethod
    def analyze_case(
        image_files: List[str],
        segmentation_file: str,
        num_samples: int = 10000,
    ):
        images, properties_image = read_sitk_case(image_files)
        segmentation, properties_seg = read_sitk_case(segmentation_file)

        data_cropped, seg_cropped, bbox = crop_to_mask(images, segmentation)

        foreground_intensities_per_channel, foreground_intensity_stats_per_channel = (
            DatasetFingerprintExtractor.collect_foreground_intensities(
                seg_cropped, data_cropped, num_samples=num_samples
            )
        )

        spacing = properties_image["spacing"]

        shape_before_crop = images.shape[1:]
        shape_after_crop = data_cropped.shape[1:]
        relative_size_after_cropping = np.prod(shape_after_crop) / np.prod(
            shape_before_crop
        )

        # print(shape_after_crop, shape_before_crop, relative_size_after_cropping)

        return (
            shape_after_crop,
            spacing,
            foreground_intensities_per_channel,
            foreground_intensity_stats_per_channel,
            relative_size_after_cropping,
        )

    @staticmethod
    def find_target_spacing(spacings):
        anisotropy_flag = False
        dim = len(spacings[0])
        spacings_array = np.array(spacings)
        worst_axis_lst = [0 for _ in range(dim)]

        for spacing in spacings:
            if not anisotropy_flag:
                anisotropy_flag = whether_anisotropy(spacing)
            worst_axis_lst[get_lowres_axis(spacing)] += 1

        if anisotropy_flag:
            target_spacing = [np.nan for _ in range(dim)]
            worst_axis = np.argmax(worst_axis_lst)
            worst_axis_spacing = np.percentile(spacings_array[:, worst_axis], 10)
            target_spacing[worst_axis] = worst_axis_spacing
            for axis in range(dim):
                if axis == worst_axis:
                    continue
                target_spacing[axis] = np.median(spacings_array[:, axis])
        else:
            target_spacing = [np.nan for _ in range(dim)]
            for axis in range(dim):
                target_spacing[axis] = np.median(spacings_array[:, axis])

        if True in np.isnan(target_spacing):
            raise Exception("There is Nan in target_spacing during calculating.")

        return target_spacing

    def save_dataset_split_json(self, save_path):
        output = {}
        for i in range(5):
            fold = "fold" + str(i)
            output[fold] = {}
            output[fold]["train"] = [
                "{}-{}".format(self.dataset_name, case)
                for case in self.fold_cases_dict[fold]["train"]
            ]
            output[fold]["train"].sort()

            output[fold]["val"] = [
                "{}-{}".format(self.dataset_name, case)
                for case in self.fold_cases_dict[fold]["val"]
            ]
            output[fold]["val"].sort()

        output["test"] = [
            "{}-{}".format(self.dataset_name, case)
            for case in self.fold_cases_dict["test"]
        ]
        if output["test"]:
            output["test"].sort()

        output["all_fold_is_the_same"] = self.fold_cases_dict["all_fold_is_the_same"]

        save_json(output, save_path)

        return output

    def determine_normalization_scheme_and_whether_mask_is_used_for_norm(
        self,
    ) -> Tuple[List[str], List[bool]]:
        if "channel_names" not in self.dataset_json.keys():
            print('WARNING: "channel_names" should be in dataset.yaml.')
        modalities = self.dataset_json["channel_names"]
        normalization_schemes = [find_normalizer(m) for m in modalities.values()]
        if self.median_relative_size_after_cropping < (3 / 4.0):
            use_nonzero_mask_for_norm = [
                i.leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true
                for i in normalization_schemes
            ]
        else:
            use_nonzero_mask_for_norm = [False] * len(normalization_schemes)
            assert all([i in (True, False) for i in use_nonzero_mask_for_norm]), (
                "use_nonzero_mask_for_norm must be " "True or False and cannot be None"
            )
        normalization_schemes = [i.__name__ for i in normalization_schemes]
        return normalization_schemes, use_nonzero_mask_for_norm

    def run(self, num_processes: int = 8, overwrite_existing: bool = False) -> dict:
        preprocessed_output_folder = os.path.join(
            "./Dataset_preprocessed", self.dataset_name
        )
        if not os.path.isdir(preprocessed_output_folder):
            os.mkdir(preprocessed_output_folder)

        dataset_split_file = os.path.join(
            preprocessed_output_folder, "dataset_split.json"
        )
        properties_file = os.path.join(
            preprocessed_output_folder, "dataset_figureprint.json"
        )
        plans_file = os.path.join(preprocessed_output_folder, "plans.json")

        if not os.path.isfile(properties_file) or overwrite_existing:
            # determine how many foreground voxels we need to sample per training case
            num_foreground_samples_per_case = int(
                self.num_foreground_voxels_for_intensitystats // len(self.dataset)
            )

            r = []
            with multiprocessing.get_context("spawn").Pool(num_processes) as p:
                for k in self.train_and_val_set:
                    r.append(
                        p.starmap_async(
                            DatasetFingerprintExtractor.analyze_case,
                            (
                                (
                                    self.dataset[k]["image"],
                                    self.dataset[k]["label"],
                                    num_foreground_samples_per_case,
                                ),
                            ),
                        )
                    )
                remaining = list(range(len(self.train_and_val_set)))
                # p is pretty nifti. If we kill workers they just respawn but don't do any work.
                # So we need to store the original pool of workers.
                workers = [j for j in p._pool]
                with tqdm(
                    desc=None, total=len(self.train_and_val_set), disable=self.verbose
                ) as pbar:
                    while len(remaining) > 0:
                        all_alive = all([j.is_alive() for j in workers])
                        if not all_alive:
                            raise RuntimeError(
                                "Some background worker is 6 feet under. Yuck. \n"
                                "OK jokes aside.\n"
                                "One of your background processes is missing. This could be because of "
                                "an error (look for an error message) or because it was killed "
                                "by your OS due to running out of RAM. If you don't see "
                                "an error message, out of RAM is likely the problem. In that case "
                                "reducing the number of workers might help"
                            )
                        done = [i for i in remaining if r[i].ready()]
                        for _ in done:
                            pbar.update()
                        remaining = [i for i in remaining if i not in done]
                        sleep(0.1)

            # results = ptqdm(DatasetFingerprintExtractor.analyze_case,
            #                 (training_images_per_case, training_labels_per_case),
            #                 processes=self.num_processes, zipped=True, reader_writer_class=reader_writer_class,
            #                 num_samples=num_foreground_samples_per_case, disable=self.verbose)
            results = [i.get()[0] for i in r]

            shapes_after_crop = [r[0] for r in results]
            spacings = [r[1] for r in results]
            foreground_intensities_per_channel = [
                np.concatenate([r[2][i] for r in results])
                for i in range(len(results[0][2]))
            ]
            # we drop this so that the json file is somewhat human readable
            # foreground_intensity_stats_by_case_and_modality = [r[3] for r in results]
            self.median_relative_size_after_cropping = np.median(
                [r[4] for r in results], 0
            )

            num_channels = len(
                self.dataset_json["channel_names"].keys()
                if "channel_names" in self.dataset_json.keys()
                else self.dataset_json["modality"].keys()
            )
            intensity_statistics_per_channel = {}
            for i in range(num_channels):
                intensity_statistics_per_channel[i] = {
                    "mean": float(np.mean(foreground_intensities_per_channel[i])),
                    "median": float(np.median(foreground_intensities_per_channel[i])),
                    "std": float(np.std(foreground_intensities_per_channel[i])),
                    "min": float(np.min(foreground_intensities_per_channel[i])),
                    "max": float(np.max(foreground_intensities_per_channel[i])),
                    "percentile_99_5": float(
                        np.percentile(foreground_intensities_per_channel[i], 99.5)
                    ),
                    "percentile_00_5": float(
                        np.percentile(foreground_intensities_per_channel[i], 0.5)
                    ),
                }

            fingerprint = {
                "spacings": spacings,
                "shapes_after_crop": shapes_after_crop,
                "foreground_intensity_properties_per_channel": intensity_statistics_per_channel,
                "median_relative_size_after_cropping": self.median_relative_size_after_cropping,
            }

            normalization_schemes, use_nonzero_mask_for_norm = (
                self.determine_normalization_scheme_and_whether_mask_is_used_for_norm()
            )
            plans = {
                "target_spacing": self.find_target_spacing(spacings),
                "normalization_schemes": normalization_schemes,
                "use_mask_for_norm": use_nonzero_mask_for_norm,
            }

            try:
                save_json(fingerprint, properties_file)
                save_json(plans, plans_file)
                data_split = self.save_dataset_split_json(dataset_split_file)
            except Exception as e:
                if os.path.isfile(properties_file):
                    os.remove(properties_file)
                if os.path.isfile(plans_file):
                    os.remove(plans_file)
                if os.path.isfile(dataset_split_file):
                    os.remove(dataset_split_file)
                raise e
        else:
            print("Dataset has already been analyzed.")
            fingerprint = open_json(properties_file)
            plans = open_json(plans_file)
            data_split = open_json(dataset_split_file)

        return fingerprint, data_split
