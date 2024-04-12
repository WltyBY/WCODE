import os
import re
import nibabel
import SimpleITK as sitk
import numpy as np

from typing import List


def get_all_img_and_label_path(dataset_name, files_ending):
    dataset_folder_path = os.path.join("./Dataset", dataset_name)
    folder_lst = [
        i
        for i in os.listdir(dataset_folder_path)
        if not os.path.isfile(os.path.join(dataset_folder_path, i))
    ]

    if {"images", "labels"}.issubset(folder_lst) and len(folder_lst) == 2:
        images_folder = os.path.join(dataset_folder_path, "images")
        images_name_lst = [
            i for i in os.listdir(images_folder) if i.endswith(files_ending)
        ]
        images_name_lst.sort()

        labels_folder = os.path.join(dataset_folder_path, "labels")
        labels_name_lst = [
            i for i in os.listdir(labels_folder) if i.endswith(files_ending)
        ]

        output_dict = dict()
        for img in images_name_lst:
            case_id = img.split("-")[1].split("_")[0]
            if case_id not in output_dict.keys():
                output_dict[case_id] = {}
                output_dict[case_id]["image"] = []
                output_dict[case_id]["label"] = False
            label = "{}-{}{}".format(dataset_name, case_id, files_ending)

            output_dict[case_id]["image"].append(os.path.join(images_folder, img))
            if not output_dict[case_id]["label"]:
                output_dict[case_id]["label"] = [os.path.join(labels_folder, label)]
        whehter_need_to_split = True

        return output_dict, whehter_need_to_split
    elif {"imagesTr", "labelsTr", "imagesVal", "labelsVal"}.issubset(folder_lst):
        train_images_folder = os.path.join(dataset_folder_path, "imagesTr")
        train_images_name_lst = [
            i for i in os.listdir(train_images_folder) if i.endswith(files_ending)
        ]
        train_images_name_lst.sort()

        train_labels_folder = os.path.join(dataset_folder_path, "labelsTr")
        train_labels_name_lst = [
            i for i in os.listdir(train_labels_folder) if i.endswith(files_ending)
        ]

        val_images_folder = os.path.join(dataset_folder_path, "imagesVal")
        val_images_name_lst = [
            i for i in os.listdir(val_images_folder) if i.endswith(files_ending)
        ]
        val_images_name_lst.sort()

        val_labels_folder = os.path.join(dataset_folder_path, "labelsVal")
        val_labels_name_lst = [
            i for i in os.listdir(val_labels_folder) if i.endswith(files_ending)
        ]

        names_dict = {
            "train": {
                "images": [train_images_folder, train_images_name_lst],
                "labels": [train_labels_folder, train_labels_name_lst],
            },
            "val": {
                "images": [val_images_folder, val_images_name_lst],
                "labels": [val_labels_folder, val_labels_name_lst],
            },
            "test": {},
        }

        if {"imagesTs"}.issubset(folder_lst):
            test_images_folder = os.path.join(dataset_folder_path, "imagesTs")
            test_images_name_lst = [
                i for i in os.listdir(test_images_folder) if i.endswith(files_ending)
            ]
            test_images_name_lst.sort()
            names_dict["test"]["images"] = [test_images_folder, test_images_name_lst]

            if {"labelsTs"}.issubset(folder_lst):
                have_test_labels = True
                test_labels_folder = os.path.join(dataset_folder_path, "labelsTs")
                test_labels_name_lst = [
                    i
                    for i in os.listdir(test_labels_folder)
                    if i.endswith(files_ending)
                ]

                names_dict["test"]["labels"] = [
                    test_labels_folder,
                    test_labels_name_lst,
                ]
            else:
                have_test_labels = False
        else:
            have_test_labels = False
            del names_dict["test"]

        output_dict = dict()
        for dataset_class in names_dict.keys():
            output_dict[dataset_class] = {}
            for img in names_dict[dataset_class]["images"][1]:
                case_id = img.split("-")[1].split("_")[0]
                if case_id not in output_dict[dataset_class].keys():
                    output_dict[dataset_class][case_id] = {}
                    output_dict[dataset_class][case_id]["image"] = []

                label = "{}-{}{}".format(dataset_name, case_id, files_ending)

                output_dict[dataset_class][case_id]["image"].append(
                    os.path.join(names_dict[dataset_class]["images"][0], img)
                )

                if not have_test_labels and dataset_class == "test":
                    pass
                else:
                    output_dict[dataset_class][case_id]["label"] = [
                        os.path.join(names_dict[dataset_class]["labels"][0], label)
                    ]
        whehter_need_to_split = False

        return output_dict, whehter_need_to_split
    else:
        raise Exception("dataset format error.")


def check_all_same_array(input_list):
    # compare all entries to the first
    for i in input_list[1:]:
        if not all([a == b for a, b in zip(i.shape, input_list[0].shape)]):
            return False
        all_same = np.allclose(i, input_list[0])
        if not all_same:
            return False
    return True


def check_all_same(input_list):
    # compare all entries to the first
    for i in input_list[1:]:
        if not len(i) == len(input_list[0]):
            return False
        all_same = all(i[j] == input_list[0][j] for j in range(len(i)))
        if not all_same:
            return False
    return True


def read_sitk_case(file_paths):
    """
    file_paths: pathes of different modalities of the same sample
    """
    images = []
    spacings = []
    directions = []
    origins = []

    for f in file_paths:
        sitk_obj = sitk.ReadImage(f)
        data_array = sitk.GetArrayFromImage(sitk_obj)
        assert len(data_array.shape) == 3, "only 3d images are supported"
        # spacing in x, y, z
        spacings.append(sitk_obj.GetSpacing())
        # images' shape in z, y, x
        images.append(data_array[None])
        origins.append(sitk_obj.GetOrigin())
        directions.append(sitk_obj.GetDirection())

    if not check_all_same([i.shape for i in images]):
        print("ERROR! Not all input images have the same shape!")
        print("Shapes:")
        print([i.shape for i in images])
        print("Image files:")
        print(file_paths)
        raise RuntimeError()
    if not check_all_same(origins):
        print("ERROR! Not all input images have the same origins!")
        print("Origins:")
        print(origins)
        print("Image files:")
        print(file_paths)
        raise RuntimeError()
    if not check_all_same(directions):
        print("ERROR! Not all input images have the same directions!")
        print("Directions:")
        print(directions)
        print("Image files:")
        print(file_paths)
        raise RuntimeError()
    if not check_all_same(spacings):
        print(
            "ERROR! Not all input images have the same spacings! This might be caused by them not "
            "having the same affine"
        )
        print("spacings_for_nnunet:")
        print(spacings)
        print("Image files:")
        print(file_paths)
        raise RuntimeError()

    stacked_images = np.vstack(images)
    dict = {
        "spacing": spacings[0],
        "direction": directions[0],
        "origin": origins[0],
    }

    return stacked_images.astype(np.float32), dict


def create_lists_from_splitted_dataset_folder(
    folder_path: str, file_ending: str, identifiers: List[str] = None
) -> List[List[str]]:
    if identifiers is None:
        raise Exception("Identifiers is needed here.")
    files = [i for i in os.listdir(folder_path) if i.endswith(file_ending)]
    list_of_lists = []
    for f in identifiers:
        p = re.compile(re.escape(f) + r"_\d\d\d\d" + re.escape(file_ending))
        list_of_lists.append(
            sorted([os.path.join(folder_path, i) for i in files if p.fullmatch(i)])
        )
    return list_of_lists


if __name__ == "__main__":
    output_dict, whehter_need_to_split = get_all_img_and_label_path("test1", ".nii.gz")
    print(output_dict, whehter_need_to_split)

    output_dict, whehter_need_to_split = get_all_img_and_label_path("test2", ".nii.gz")
    print(output_dict, whehter_need_to_split)

    output_dict, whehter_need_to_split = get_all_img_and_label_path("test3", ".nii.gz")
    print(output_dict, whehter_need_to_split)

    npy_array, dict = read_sitk_case(
        ["./Dataset/RADCURE/images/RADCURE-0005_0000.nii.gz"]
    )
    print(npy_array.shape)
    print(dict)
