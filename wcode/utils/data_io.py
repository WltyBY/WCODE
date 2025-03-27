import os
import re
import cv2
import SimpleITK as sitk
import numpy as np

from typing import List

files_ending_for_sitk = {".nii.gz", ".mhd", ".nii", ".nrrd"}
files_ending_for_2d_img = {".tif", ".jpg", ".png"}


def get_all_img_and_label_path(dataset_name, files_ending, channel_names):
    dataset_folder_path = os.path.join("./Dataset", dataset_name)
    folder_lst = [
        i
        for i in os.listdir(dataset_folder_path)
        if not os.path.isfile(os.path.join(dataset_folder_path, i))
    ]
    files_ending_lst = [
        "{:0>4s}".format(channel_names) + files_ending
        for channel_names in channel_names.keys()
    ]
    length_ends = len(files_ending_lst[0])

    if {"images", "labels"}.issubset(folder_lst):
        images_folder = os.path.join(dataset_folder_path, "images")
        images_name_lst = [
            i for i in os.listdir(images_folder) if i[-length_ends:] in files_ending_lst
        ]
        images_name_lst.sort()

        labels_folder = os.path.join(dataset_folder_path, "labels")
        labels_name_lst = [
            i for i in os.listdir(labels_folder) if i.endswith(files_ending)
        ]

        output_dict = dict()
        for img in images_name_lst:
            case_id = img.split("_")[1]
            if case_id not in output_dict.keys():
                output_dict[case_id] = {}
                output_dict[case_id]["image"] = []
                output_dict[case_id]["label"] = False
            label = "{}_{}{}".format(dataset_name, case_id, files_ending)

            output_dict[case_id]["image"].append(os.path.join(images_folder, img))
            if not output_dict[case_id]["label"]:
                output_dict[case_id]["label"] = [os.path.join(labels_folder, label)]
        whether_need_to_split = True

        return output_dict, whether_need_to_split

    elif {"imagesTr", "labelsTr", "imagesVal", "labelsVal"}.issubset(folder_lst):
        train_images_folder = os.path.join(dataset_folder_path, "imagesTr")
        train_images_name_lst = [
            i
            for i in os.listdir(train_images_folder)
            if i[-length_ends:] in files_ending_lst
        ]
        train_images_name_lst.sort()

        train_labels_folder = os.path.join(dataset_folder_path, "labelsTr")
        train_labels_name_lst = [
            i for i in os.listdir(train_labels_folder) if i.endswith(files_ending)
        ]

        val_images_folder = os.path.join(dataset_folder_path, "imagesVal")
        val_images_name_lst = [
            i
            for i in os.listdir(val_images_folder)
            if i[-length_ends:] in files_ending_lst
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
                i
                for i in os.listdir(test_images_folder)
                if i[-length_ends:] in files_ending_lst
            ]
            test_images_name_lst.sort()
            names_dict["test"]["images"] = [test_images_folder, test_images_name_lst]

            if {"labelsTs"}.issubset(folder_lst):
                have_test_labels = True
                test_labels_folder = os.path.join(dataset_folder_path, "labelsTs")
                test_labels_name_lst = [
                    i
                    for i in os.listdir(test_labels_folder)
                    if i[-length_ends:] in files_ending_lst
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
                case_id = img.split("_")[1]
                if case_id not in output_dict[dataset_class].keys():
                    output_dict[dataset_class][case_id] = {}
                    output_dict[dataset_class][case_id]["image"] = []

                label = "{}_{}{}".format(dataset_name, case_id, files_ending)

                output_dict[dataset_class][case_id]["image"].append(
                    os.path.join(names_dict[dataset_class]["images"][0], img)
                )

                if not have_test_labels and dataset_class == "test":
                    pass
                else:
                    output_dict[dataset_class][case_id]["label"] = [
                        os.path.join(names_dict[dataset_class]["labels"][0], label)
                    ]
        whether_need_to_split = False

        return output_dict, whether_need_to_split
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
        if os.path.isfile(f):
            sitk_obj = sitk.ReadImage(f)       
        elif os.path.isdir(f):
            reader = sitk.ImageSeriesReader();
            seriesIDs = reader.GetGDCMSeriesIDs(f)
            dcm_series = reader.GetGDCMSeriesFileNames(f, seriesIDs[0])
            reader.SetFileNames(dcm_series)
            sitk_obj =reader.Execute()
        else:
            raise Exception(f)
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
    '''
    The origin order of all thing by SimpleITK is in x, y, z. You can check this by using .GetSize(), .GetSpacing().
    This two is mostly used in our program. However, after transform the SimpleITK.SimpleITK.Image to np.NDarray using 
    sitk.GetArrayFromImage(), the axis order of the array is changed to z, y, x. So, when use spacing and NDarray,
    don't forget to change the order of spacing.
    '''
    dict = {
        "spacing": spacings[0],
        "direction": directions[0],
        "origin": origins[0],
        "shapes": [i.shape for i in images],
    }

    return stacked_images.astype(np.float32), dict


def read_2d_img(file_paths):
    """
    file_paths: pathes of different modalities of the same sample
    All the 2d images' shapes are in C H W
    """
    images = []
    shapes = []
    for f in file_paths:
        img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        if check_whether_all_channels_the_same(img):
            images.append(img[0][None])
            shapes.append(img[0][None].shape)
        else:
            images.append(img)
            shapes.append(img.shape)
    stacked_images = np.vstack(images)
    
    return stacked_images.astype(np.float32), {"shapes": shapes}


def check_whether_all_channels_the_same(img):
    # img in c, h, w for 2D img
    flag = True
    compare_img = img[0]
    for channel in range(1, img.shape[0]):
        if not np.array_equal(compare_img, img[channel]):
            flag = False
            break
        else:
            compare_img = img[channel]
    return flag


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
