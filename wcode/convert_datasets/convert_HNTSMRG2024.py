import os
import re
import shutil
import multiprocessing
import numpy as np
import SimpleITK as sitk

from tqdm import tqdm
from time import sleep
from scipy import ndimage

from wcode.utils.file_operations import save_csv, save_yaml
from wcode.utils.NDarray_operations import (
    get_largest_k_components,
    get_ND_bounding_box,
    crop_ND_volume_with_bounding_box,
)

# mid or pre
dataset_kind = "mid"
num_processes = 16
if dataset_kind == "pre":
    files_lst = {"data": ["_preRT_T2.nii.gz"], "mask": ["_preRT_mask.nii.gz"]}
elif dataset_kind == "mid":
    files_lst = {
        "data": [
            "_midRT_T2.nii.gz",
            "_preRT_T2_registered.nii.gz",
            "_preRT_mask_registered.nii.gz",
        ],
        "mask": ["_midRT_mask.nii.gz"],
    }

dataset_info = {
    "channel_names": {
        str(i): file.split(".")[0] for i, file in enumerate(files_lst["data"])
    },
    "labels": {"background": 0, "GTVp": 1, "GTVnd": 2},
    "files_ending": ".nii.gz",
}


def crop(img_array_lst, channel_names, seg_array, origin, spacing):
    assert isinstance(img_array_lst, list), "type error."

    bbmin = []
    bbmax = []
    for img_array, channel_name in zip(img_array_lst, channel_names.values()):
        input_shape = img_array.shape
        if any(
            [True if s in channel_name else False for s in ["mask", "label", "seg"]]
        ):
            mask = np.asarray(img_array > 0)
            bbmin_now, bbmax_now = get_ND_bounding_box(mask, margin=[2, 2, 2])
        else:
            mask = np.asarray(img_array > 60)
            se = np.ones([3, 3, 3])
            mask = ndimage.binary_closing(mask, se, iterations=2)
            bbmin_now, bbmax_now = get_ND_bounding_box(mask, margin=[2, 0, 0])

            mask_findhead = mask[input_shape[0] // 2 :]
            bbmin_head, bbmax_head = get_ND_bounding_box(mask_findhead)
            bbmin_now[1:], bbmax_now[1:] = bbmin_head[1:], bbmax_head[1:]

        if not bbmin:
            bbmin, bbmax = bbmin_now, bbmax_now
        else:
            for i in range(len(bbmin)):
                bbmin[i] = max(min(bbmin[i], bbmin_now[i]), 0)
                bbmax[i] = min(max(bbmax[i], bbmax_now[i]), input_shape[i])

    img_crop = []
    for img in img_array_lst:
        img_crop.append(crop_ND_volume_with_bounding_box(img, bbmin, bbmax))
    seg_crop = crop_ND_volume_with_bounding_box(seg_array, bbmin, bbmax)
    new_origin = [origin[i] + bbmin[::-1][i] * spacing[i] for i in range(len(origin))]

    return img_crop, seg_crop, new_origin


def run_case(case, datafolder, imgsavefolder, segsavefolder, dataset_name):
    modality_count = 0
    img_array_lst = []
    img_save_path = []
    get_info_flag = True
    for file_name in files_lst["data"]:
        file_name = case + file_name
        save_file_name = dataset_name + "_{:0>4s}_{:0>4d}.nii.gz".format(
            case, modality_count
        )

        modality_count += 1
        src_path = os.path.join(datafolder, case, dataset_kind + "RT", file_name)
        if get_info_flag:
            img_obj = sitk.ReadImage(src_path)
            origin_shape = sitk.GetArrayFromImage(img_obj).shape
            spacing = img_obj.GetSpacing()
            direction = img_obj.GetDirection()
            origin = img_obj.GetOrigin()
            get_info_flag = False
        img_save_path.append(os.path.join(imgsavefolder, save_file_name))
        img_array_lst.append(sitk.GetArrayFromImage(sitk.ReadImage(src_path)))

    file_name = case + files_lst["mask"][0]
    save_file_name = dataset_name + "_{:0>4s}.nii.gz".format(case)
    src_path = os.path.join(datafolder, case, dataset_kind + "RT", file_name)
    seg_save_path = os.path.join(segsavefolder, save_file_name)
    seg_array = sitk.GetArrayFromImage(sitk.ReadImage(src_path))
    bool_re = np.in1d([1, 2], seg_array)

    img_crop, seg_crop, new_origin = crop(
        img_array_lst, dataset_info["channel_names"], seg_array, origin, spacing
    )
    for img, save_path in zip(img_crop, img_save_path):
        img_obj = sitk.GetImageFromArray(img)
        img_obj.SetDirection(direction)
        img_obj.SetOrigin(new_origin)
        img_obj.SetSpacing(spacing)
        sitk.WriteImage(img_obj, save_path)
    seg_obj = sitk.GetImageFromArray(seg_crop)
    seg_obj.SetDirection(direction)
    seg_obj.SetOrigin(new_origin)
    seg_obj.SetSpacing(spacing)
    sitk.WriteImage(seg_obj, seg_save_path)

    info = [
        file_name,
        str(bool_re[0]),
        str(bool_re[1]),
        origin_shape,
        img_crop[0].shape,
    ]
    return info


if __name__ == "__main__":
    dataset_name = "HNTSMRG2024" + dataset_kind
    data_folder = "./Dataset/HNTSMRG2024_raw/HNTSMRG24_train"
    save_folder = os.path.join("./Dataset", dataset_name)

    img_save_folder = os.path.join(save_folder, "images")
    seg_save_folder = os.path.join(save_folder, "labels")

    if os.path.isdir(img_save_folder):
        shutil.rmtree(img_save_folder, ignore_errors=True)
    os.makedirs(img_save_folder, exist_ok=True)
    if os.path.isdir(seg_save_folder):
        shutil.rmtree(seg_save_folder, ignore_errors=True)
    os.makedirs(seg_save_folder, exist_ok=True)

    save_yaml(dataset_info, os.path.join(save_folder, "dataset.yaml"))
    save_csv(
        [
            [
                "Case",
                "GTVp",
                "GTVnd",
                "origin_shape",
                "crop_shape",
            ]
        ],
        os.path.join(save_folder, "Data_info.csv"),
        mode="w",
    )

    case_lst = [i for i in os.listdir(data_folder)]
    case_lst.sort(key=lambda l: int(re.findall("\d+", l)[0]))
    r = []
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        for case in case_lst:
            r.append(
                p.starmap_async(
                    run_case,
                    (
                        (
                            case,
                            data_folder,
                            img_save_folder,
                            seg_save_folder,
                            dataset_name,
                        ),
                    ),
                )
            )
        remaining = list(range(len(case_lst)))
        workers = [j for j in p._pool]
        with tqdm(desc=None, total=len(case_lst), disable=False) as pbar:
            while len(remaining) > 0:
                all_alive = all([j.is_alive() for j in workers])
                if not all_alive:
                    raise RuntimeError("Some background worker is 6 feet under. Yuck.")
                done = [i for i in remaining if r[i].ready()]
                for _ in done:
                    pbar.update()
                remaining = [i for i in remaining if i not in done]
                sleep(0.1)
    results = [i.get()[0] for i in r]

    save_csv(
        results,
        os.path.join(save_folder, "Data_info.csv"),
        mode="a",
    )

    print("Now checking files...")
    preprocessed_files = os.listdir(img_save_folder)
    if len(case_lst) * len(files_lst["data"]) != len(preprocessed_files):
        print("Maybe something wrong during preprocessing.")
        print(
            "There should be {} files, but find {} files.".format(
                len(case_lst) * len(files_lst["data"]), len(preprocessed_files)
            )
        )
        preprocessed_files_set = set(
            [str(int(file.split("_")[1])) for file in preprocessed_files]
        )
        error_cases = set(case_lst) - preprocessed_files_set
        for case in error_cases:
            print("Error for:", case)
    else:
        print("Judging by the number of files, the process is correct.")
