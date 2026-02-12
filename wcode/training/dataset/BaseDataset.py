import os
import torch
import numpy as np

from torch.utils.data import Dataset
from typing import List, Tuple, Union

from wcode.training.dataloader.Patchsampler import Patchsampler
from wcode.utils.file_operations import open_json, open_yaml, open_pickle
from wcode.utils.data_io import file_endings_for_2d_img, file_endings_for_sitk


class PatchDataset(Dataset):
    def __init__(
        self,
        dataset_name,
        preprocess_config,
        sample_patch_size: List,
        final_patch_size: List,
        oversample_rate: Union[float, List[float], Tuple[float]],
        probabilistic_oversampling: bool,
        split="train",
        fold="0",
        modality=None,
        transform=None,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.modality = modality
        self.transform = transform
        assert isinstance(self.modality, list)

        # get the split of dataset
        split_json_path = os.path.join(
            "./Dataset_preprocessed", self.dataset_name, "dataset_split.json"
        )
        if not os.path.isfile(split_json_path):
            raise Exception(
                "dataset_split.json is needed when generating Dataset object."
            )
        self.split_json = open_json(split_json_path)

        # get some meta info of dataset
        dataset_yaml_path = os.path.join(
            "./Dataset_preprocessed", self.dataset_name, "dataset.yaml"
        )
        if not os.path.isfile(dataset_yaml_path):
            raise Exception("dataset.yaml is needed when generating Dataset object.")
        self.dataset_yaml = open_yaml(dataset_yaml_path)

        # determine the image reading method
        if self.dataset_yaml["files_ending"] in file_endings_for_sitk:
            self.general_img_flag = False
        elif self.dataset_yaml["files_ending"] in file_endings_for_2d_img:
            self.general_img_flag = True
        else:
            raise Exception("Files' ending NOT SUPPORT!!!")

        # get the preprocessed dataset path
        self.preprocessed_dataset_folder_path = os.path.join(
            "./Dataset_preprocessed",
            self.dataset_name,
            "preprocessed_datas_" + preprocess_config,
        )

        # get the ids of current split and fold
        if fold != "all":
            if self.split in ["train", "val"]:
                self.ids = self.split_json[fold][self.split]
            elif self.split == "test":
                self.ids = self.split_json[self.split]
            else:
                raise Exception('Para:split should be "train", "val" or "test".')
        else:
            if self.split in ["train", "val"]:
                self.ids = self.split_json["0"]["train"] + self.split_json["0"]["val"]
            elif self.split == "test":
                self.ids = self.split_json[self.split]
            else:
                raise Exception('Para:split should be "train", "val" or "test".')

        self.ids.sort()

        # print(
        #     "{} dataset has {} samples.".format(split.upper(), len(self.ids))
        # )

        # initialize Patchsampler
        self.Patchsampler = Patchsampler(
            sample_patch_size=sample_patch_size,
            final_patch_size=final_patch_size,
            oversample_rate=oversample_rate,
            probabilistic_oversampling=probabilistic_oversampling,
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # get image, label and property dict
        case = self.ids[idx]
        data = np.load(
            os.path.join(self.preprocessed_dataset_folder_path, case + ".npy")
        )
        seg = np.load(
            os.path.join(self.preprocessed_dataset_folder_path, case + "_seg" + ".npy")
        )
        data_dict = {"data": data, "seg": seg}
        property_dict = open_pickle(
            os.path.join(self.preprocessed_dataset_folder_path, case + ".pkl")
        )
        fg_locations = property_dict["oversample"]

        if self.general_img_flag:
            data_lst = []
            count = 0
            for i in range(len(property_dict["shapes"])):
                n_channel = property_dict["shapes"][i][0]
                if i in self.modality:
                    data_lst.append(
                        data_dict["data"][list(range(count, count + n_channel))]
                    )
                count += n_channel
            sample = {"image": np.vstack(data_lst), "label": data_dict["seg"]}
        else:
            sample = {
                "image": data_dict["data"][self.modality],
                "label": data_dict["seg"],
            }

        # sample a patch
        sample["image"], sample["label"] = self.Patchsampler.sample_a_patch(
            sample["image"], sample["label"], fg_locations
        )

        # apply data augmentation if available
        if self.transform is not None:
            sample["image"] = torch.from_numpy(sample["image"]).float()
            sample["label"] = torch.from_numpy(sample["label"]).to(torch.int16)
            sample = self.transform(**sample)

        sample["idx"] = case
        sample["property"] = property_dict

        return sample
