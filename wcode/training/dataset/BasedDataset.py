import os
import numpy as np

from torch.utils.data import Dataset

from wcode.utils.file_operations import open_json, open_yaml, open_pickle


class BasedDataset(Dataset):

    def __init__(
        self,
        dataset_name,
        split="train",
        fold="fold0",
        modality=None,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.modality = modality

        split_json_path = os.path.join(
            "./Dataset_preprocessed", self.dataset_name, "dataset_split.json"
        )
        if not os.path.isfile(split_json_path):
            raise Exception(
                "dataset_split.json is needed when generating Dataset object."
            )
        self.split_json = open_json(split_json_path)

        dataset_yaml_path = os.path.join("./Dataset", self.dataset_name, "dataset.yaml")
        if not os.path.isfile(dataset_yaml_path):
            raise Exception("dataset.yaml is needed when generating Dataset object.")
        self.dataset_yaml = open_yaml(dataset_yaml_path)

        self.preprocessed_dataset_folder_path = os.path.join(
            "./Dataset_preprocessed", self.dataset_name, "preprocessed_datas"
        )

        if self.split in ["train", "val"]:
            self.ids = self.split_json[fold][self.split]
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
        data_dict = np.load(
            os.path.join(self.preprocessed_dataset_folder_path, case + ".npz")
        )
        property_dict = open_pickle(
            os.path.join(self.preprocessed_dataset_folder_path, case + ".pkl")
        )

        if self.modality is None or self.modality == "all":
            sample = {"image": data_dict["data"], "label": data_dict["seg"]}
        else:
            assert isinstance(self.modality, list)
            sample = {"image": data_dict["data"][self.modality], "label": data_dict["seg"]}

        sample["idx"] = case
        sample["property"] = property_dict

        return sample
