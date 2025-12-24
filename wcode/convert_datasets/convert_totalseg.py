import os
import shutil
import multiprocessing

import nibabel as nib
import pandas as pd
import numpy as np
import SimpleITK as sitk

from tqdm import tqdm
from time import sleep

WANTED_SEG_DICT = {
    "thorax": {
        "lung_left": {
            "seg_value": 1,
            "contained_seg": ["lung_upper_lobe_left", "lung_lower_lobe_left"],
        },
        "lung_right": {
            "seg_value": 2,
            "contained_seg": [
                "lung_lower_lobe_right",
                "lung_middle_lobe_right",
                "lung_upper_lobe_right",
            ],
        },
        "heart": {
            "seg_value": 3,
            "contained_seg": ["heart"],
        },
        "esophagus": {
            "seg_value": 4,
            "contained_seg": ["esophagus"],
        },
        "trachea": {
            "seg_value": 5,
            "contained_seg": ["trachea"],
        },
        "spinal_cord": {
            "seg_value": 6,
            "contained_seg": ["spinal_cord"],
        },
        "rib_left": {
            "seg_value": 7,
            "contained_seg": ["rib_left_{}".format(i) for i in range(1, 13)],
        },
        "rib_right": {
            "seg_value": 8,
            "contained_seg": ["rib_right_{}".format(i) for i in range(1, 13)],
        },
        "thyroid_gland": {
            "seg_value": 9,
            "contained_seg": ["thyroid_gland"],
        },
        "stomach": {
            "seg_value": 10,
            "contained_seg": ["stomach"],
        },
        "humerus_left": {
            "seg_value": 11,
            "contained_seg": ["humerus_left"],
        },
        "humerus_right": {
            "seg_value": 12,
            "contained_seg": ["humerus_right"],
        },
        "aorta": {
            "seg_value": 13,
            "contained_seg": ["aorta"],
        },
        "vertebrae_T": {
            "seg_value": 14,
            "contained_seg": ["vertebrae_T{}".format(i) for i in range(1, 13)],
        },
        "sternum": {
            "seg_value": 15,
            "contained_seg": ["sternum"],
        },
    },
}


def process(case, dataset_name, raw_folder, img_save_folder, seg_save_folder, GET_DATASET):
    img_path = os.path.join(raw_folder, case, "ct.nii.gz")
    case_name = "{}_{}".format(dataset_name, case[1:])

    try:
        img_obj = sitk.ReadImage(img_path)
        img_array = sitk.GetArrayFromImage(img_obj)
    except:
        img = nib.load(img_path)
        qform = img.get_qform()
        img.set_qform(qform)
        sfrom = img.get_sform()
        img.set_sform(sfrom)
        nib.save(img, img_path)

        img_obj = sitk.ReadImage(img_path)
        img_array = sitk.GetArrayFromImage(img_obj)

    sitk.WriteImage(img_obj, os.path.join(img_save_folder, case_name + "_0000.nii.gz"))

    # get the wanted segmentation
    seg_new = np.zeros_like(img_array)
    for class_name in WANTED_SEG_DICT[GET_DATASET].keys():
        seg_for_class = np.zeros_like(img_array)
        for file_name in WANTED_SEG_DICT[GET_DATASET][class_name]["contained_seg"]:
            file_path = os.path.join(
                raw_folder, case, "segmentations", file_name + ".nii.gz"
            )
            try:
                seg_for_class_obj = sitk.ReadImage(file_path)
                seg_for_class_array = sitk.GetArrayFromImage(seg_for_class_obj)
            except:
                seg = nib.load(file_path)
                qform = seg.get_qform()
                seg.set_qform(qform)
                sfrom = seg.get_sform()
                seg.set_sform(sfrom)
                nib.save(seg, file_path)

                seg_for_class_obj = sitk.ReadImage(file_path)
                seg_for_class_array = sitk.GetArrayFromImage(seg_for_class_obj)
            seg_for_class[seg_for_class_array != 0] = 1
        # if np.sum(seg_new[seg_for_class != 0]) != 0:
        #     print(
        #         "There may be intersections between annotated categories. case: {}, class: {}, pixels: {}.".format(
        #             case, class_name, np.sum(seg_new[seg_for_class != 0])
        #         )
        #     )
        seg_new[seg_for_class != 0] = WANTED_SEG_DICT[GET_DATASET][class_name][
            "seg_value"
        ]
    seg_new_obj = sitk.GetImageFromArray(seg_new.astype(np.uint8))
    seg_new_obj.CopyInformation(img_obj)
    sitk.WriteImage(seg_new_obj, os.path.join(seg_save_folder, case_name + ".nii.gz"))


if __name__ == "__main__":
    NUM_PROCESS = 16
    GET_DATASET = "thorax"
    Dataset_name = "Totalseg" + GET_DATASET.capitalize()

    raw_dataset_folder = "./Dataset/Total"
    save_folder = "./Dataset/" + Dataset_name
    meta_info_path = raw_dataset_folder + "/meta.csv"

    data = pd.read_csv(meta_info_path, sep=";", header=0)
    case_lst = []
    for i in range(len(data)):
        if not isinstance(data.iloc[i]["study_type"], str):
            continue
        if (
            GET_DATASET in data.iloc[i]["study_type"]
            and data.iloc[i]["split"] == "train"
        ):
            case_lst.append(data.iloc[i]["image_id"])

    img_save_folder = os.path.join(save_folder, "images")
    if os.path.isdir(img_save_folder):
        shutil.rmtree(img_save_folder, ignore_errors=True)
    os.makedirs(img_save_folder, exist_ok=True)

    seg_save_folder = os.path.join(save_folder, "labels")
    if os.path.isdir(seg_save_folder):
        shutil.rmtree(seg_save_folder, ignore_errors=True)
    os.makedirs(seg_save_folder, exist_ok=True)

    r = []
    with multiprocessing.get_context("spawn").Pool(NUM_PROCESS) as p:
        for case in case_lst:
            r.append(
                p.starmap_async(
                    process,
                    (
                        (
                            case,
                            Dataset_name,
                            raw_dataset_folder,
                            img_save_folder,
                            seg_save_folder,
                            GET_DATASET,
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
                    raise RuntimeError(
                        "Some background worker is 6 feet under. Yuck."
                    )
                done = [i for i in remaining if r[i].ready()]
                for _ in done:
                    pbar.update()
                remaining = [i for i in remaining if i not in done]
                sleep(0.1)
