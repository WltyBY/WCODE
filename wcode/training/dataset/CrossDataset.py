import os
import torch
import numpy as np

from torch.utils.data import Dataset

from wcode.utils.file_operations import open_json, open_yaml, open_pickle
from wcode.utils.data_io import files_ending_for_2d_img, files_ending_for_sitk


class CrossDataset(Dataset):
    def __init__(
        self,
        dataset_dict: dict,
        preprocess_config,
        split="train",
        fold="fold0",
        modality=None,
        transform=None,
    ):
        """
        images' id number should be the same as it in segs' dataset.
        Inputs:
            dataset_dict: {"image":["LNQ2023Sparse"], "seg":["LNQ2023Sparse", "LNQ2023SparseCoronal", "LNQ2023SparseTranseverse"]}
        """
        self.dataset_dict = dataset_dict
        self.dataset_name_image_lst: list = dataset_dict["image"]
        assert len(self.dataset_name_image_lst) == 1
        self.dataset_name_label_lst: list = dataset_dict["seg"]

        self.split = split
        self.modality = modality
        self.transform = transform
        assert isinstance(self.modality, list)

        # check split file
        split_json_path = os.path.join(
            "./Dataset_preprocessed",
            self.dataset_name_image_lst[0],
            "dataset_split.json",
        )
        if not os.path.isfile(split_json_path):
            raise Exception(
                "dataset_split.json is needed when generating Dataset object."
            )
        self.split_json = open_json(split_json_path)

        dataset_yaml_path = os.path.join(
            "./Dataset_preprocessed", self.dataset_name_image_lst[0], "dataset.yaml"
        )
        if not os.path.isfile(dataset_yaml_path):
            raise Exception("dataset.yaml is needed when generating Dataset object.")
        self.dataset_yaml = open_yaml(dataset_yaml_path)

        if self.dataset_yaml["files_ending"] in files_ending_for_sitk:
            self.general_img_flag = False
        elif self.dataset_yaml["files_ending"] in files_ending_for_2d_img:
            self.general_img_flag = True
        else:
            raise Exception("Files' ending NOT SUPPORT!!!")

        # get needed image folder
        self.preprocessed_image_folder = os.path.join(
            "./Dataset_preprocessed",
            self.dataset_name_image_lst[0],
            "preprocessed_datas_" + preprocess_config,
        )
        # get needed seg folder
        self.preprocessed_label_folder = []
        for dataset in self.dataset_name_label_lst:
            self.preprocessed_label_folder.append(
                os.path.join(
                    "./Dataset_preprocessed",
                    dataset,
                    "preprocessed_datas_" + preprocess_config,
                )
            )

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

        print("{} dataset has {} samples".format(split.upper(), len(self.ids)))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        case = self.ids[idx]
        id_number: str = case.split("_")[-1]
        data = np.load(os.path.join(self.preprocessed_image_folder, case + ".npy"))
        seg = []
        for seg_folder, dataset in zip(
            self.preprocessed_label_folder, self.dataset_name_label_lst
        ):
            seg.append(
                np.load(
                    os.path.join(
                        seg_folder, dataset + "_" + id_number + "_seg" + ".npy"
                    )
                )
            )
        seg = np.vstack(seg)
        property_dict = open_pickle(
            os.path.join(self.preprocessed_image_folder, case + ".pkl")
        )
        data_dict = {"data": data, "seg": seg}

        if self.general_img_flag:
            data_lst = []
            for i in range(len(property_dict["shapes"])):
                count = 0
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
        if self.transform is not None:
            sample["image"] = torch.from_numpy(sample["image"]).float()
            sample["label"] = torch.from_numpy(sample["label"]).to(torch.int16)
            output_dict = {"image": [], "label": []}
            for i in range(sample["image"].shape[0]):
                sample_data = {"image": sample["image"][i], "label": sample["label"][i]}
                sample_data = self.transform(**sample_data)
                output_dict["image"].append(sample_data["image"])
                output_dict["label"].append(sample_data["label"])
            sample["image"] = torch.stack(output_dict["image"])
            sample["label"] = torch.stack(output_dict["label"])

        sample["idx"] = case
        sample["property"] = property_dict
        return sample
