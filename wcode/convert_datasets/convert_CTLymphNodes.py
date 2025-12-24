import os
import copy
import math
import random
import shutil
import multiprocessing
import numpy as np
import SimpleITK as sitk

from tqdm import tqdm
from time import sleep

from wcode.utils.NDarray_operations import (
    get_largest_k_components,
    get_ND_bounding_box,
    crop_ND_volume_with_bounding_box,
)
from wcode.inferring.PatchBasedPredictor import PatchBasedPredictor
from wcode.utils.file_operations import open_yaml


def refine_and_delete_instance(seg, remain_ratio=0.5):
    unique = np.unique(seg)
    unique = unique[unique != 0]
    assert len(unique) == np.max(unique), "Discontinuous instance annotation order!"

    for idx in unique:
        seg_per_idx = copy.deepcopy(seg)
        seg_per_idx[seg_per_idx != idx] = 0
        bbmin, bbmax = get_ND_bounding_box(seg_per_idx)
        if np.any((np.array(bbmax) - np.array(bbmin)) <= 1):
            seg[seg == idx] = 0

    if 0 < remain_ratio < 1:
        refine_unique = np.unique(seg[seg != 0])
        voxel_lst = []
        for i in list(refine_unique):
            voxel_lst.append((seg == i).sum())
        weights = np.array(voxel_lst) / np.sum(voxel_lst)
        MAX = len(refine_unique)
        num_element = math.ceil(MAX * remain_ratio)

        remain_idx = np.random.choice(
            refine_unique, p=weights, size=num_element, replace=False
        )

        del_idx = list(set(refine_unique) - set(remain_idx))

        for idx in del_idx:
            seg[seg == idx] = 0
            
    # get sparse seg
    new_seg = np.zeros_like(seg)
    remain_unique = np.unique(seg[seg != 0])
    for idx in list(remain_unique):
        instance_label = np.zeros_like(seg)
        instance_label[seg == idx] = 1

        z, *_ = np.where(instance_label)
        slice_loc = (z.max() + z.min()) // 2
        new_seg[slice_loc][instance_label[slice_loc] != 0] = idx

        transposed_new_seg = np.transpose(new_seg, (1, 0, 2))
        transposed_instance_label = np.transpose(instance_label, (1, 0, 2))
        z, *_ = np.where(transposed_instance_label)
        slice_loc = (z.max() + z.min()) // 2
        transposed_new_seg[slice_loc][transposed_instance_label[slice_loc] != 0] = idx
        new_seg = np.transpose(transposed_new_seg, (1, 0, 2))

    seg[seg != 0] = 1
    new_seg[new_seg != 0] = 1
    return seg.astype(np.uint8), new_seg.astype(np.uint8)


def get_predictor(
    model_path,
    GPU_ID,
    config_dict,
    data_dim,
    num_processors,
):
    predict_configs = {
        "dataset_name": "TotalsegThorax",
        "modality": "all",
        "fold": None,
        "split": None,
        "original_img_folder": None,
        "predictions_save_folder": None,
        "model_path": model_path,
        "device": {"gpu": [GPU_ID]},
        "overwrite": True,
        "save_probabilities": False,
        "patch_size": config_dict["Training_settings"]["patch_size"],
        "tile_step_size": 0.5,
        "use_gaussian": True,
        "perform_everything_on_gpu": True,
        "use_mirroring": True,
        "allowed_mirroring_axes": [0, 1] if data_dim == "2d" else [0, 1, 2],
        "num_processes": num_processors,
    }
    config_dict["Inferring_settings"] = predict_configs
    p = PatchBasedPredictor(config_dict, allow_tqdm=True)
    p.initialize()
    return p


def process(
    img_path,
    seg_path,
    img_save_path,
    seg_save_path,
    sparse_img_save_path,
    sparse_seg_save_path,
    vis_save_path,
    totalseg_pred,
    remain_instance_ratio,
    if_vis,
):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(img_path)
    reader.SetFileNames(dicom_names)
    image_obj = reader.Execute()
    seg_obj = sitk.ReadImage(seg_path)

    img_array = sitk.GetArrayFromImage(image_obj)
    seg_array = sitk.GetArrayFromImage(seg_obj)

    # Get lung mask
    lung_mask = np.zeros_like(totalseg_pred)
    ## refine lung_mask
    lung_mask_mid = np.zeros_like(totalseg_pred)
    lung_mask_mid[totalseg_pred == 1] = 1
    lung_mask_mid = get_largest_k_components(lung_mask_mid)
    lung_mask[lung_mask_mid == 1] = 1

    lung_mask_mid = np.zeros_like(totalseg_pred)
    lung_mask_mid[totalseg_pred == 2] = 1
    lung_mask_mid = get_largest_k_components(lung_mask_mid)
    lung_mask[lung_mask_mid == 1] = 1

    # get bbox
    bbmin = [0, 0, 0]
    bbmax = [0, 0, 0]
    bbmin_img, bbmax_img = get_ND_bounding_box(lung_mask, margin=[0, 5, 5])
    # print(bbmin_img, bbmax_img)
    bbmin_seg, bbmax_seg = get_ND_bounding_box(seg_array, margin=[5, 5, 5])
    # print(bbmin_seg, bbmax_seg)
    for i in range(len(bbmin_img)):
        bbmin[i] = min(bbmin_img[i], bbmin_seg[i])
        bbmax[i] = max(bbmax_img[i], bbmax_seg[i])

    origin = image_obj.GetOrigin()
    spacing = image_obj.GetSpacing()
    origin_output = tuple(
        [origin[i] + spacing[i] * bbmin[::-1][i] for i in range(len(bbmin))]
    )
    img_output = crop_ND_volume_with_bounding_box(img_array, bbmin, bbmax)
    seg_output = crop_ND_volume_with_bounding_box(seg_array, bbmin, bbmax)

    seg_output, sparse_seg_output = refine_and_delete_instance(
        seg_output, remain_instance_ratio
    )

    img_output = sitk.GetImageFromArray(img_output)
    img_output.SetOrigin(origin_output)
    img_output.SetSpacing(spacing)
    img_output.SetDirection(image_obj.GetDirection())
    sitk.WriteImage(img_output, img_save_path)
    sitk.WriteImage(img_output, sparse_img_save_path)

    seg_output = sitk.GetImageFromArray(seg_output)
    seg_output.SetOrigin(origin_output)
    seg_output.SetSpacing(spacing)
    seg_output.SetDirection(image_obj.GetDirection())
    sitk.WriteImage(seg_output, seg_save_path)

    sparse_seg_output = sitk.GetImageFromArray(sparse_seg_output)
    sparse_seg_output.SetOrigin(origin_output)
    sparse_seg_output.SetSpacing(spacing)
    sparse_seg_output.SetDirection(image_obj.GetDirection())
    sitk.WriteImage(sparse_seg_output, sparse_seg_save_path)

    if if_vis:
        lung_output = crop_ND_volume_with_bounding_box(lung_mask, bbmin, bbmax)
        lung_output = sitk.GetImageFromArray(lung_output)
        lung_output.SetOrigin(origin_output)
        lung_output.SetSpacing(spacing)
        lung_output.SetDirection(image_obj.GetDirection())
        sitk.WriteImage(lung_output, vis_save_path)


if __name__ == "__main__":
    NUM_PROCESS = 8
    GPU_ID = 0
    SEED = 319
    IF_VIS = True
    REMAIN_RATIO = 0.2
    Dataset_name = "CTLymphNodes02"
    Sparse_Dataset_name = Dataset_name + "Sparse"

    np.random.seed(SEED)
    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)

    Totalsegmentor_training_config_dict = open_yaml("./Configs/TotalsegThorax.yaml")
    predictor = get_predictor(
        model_path="./Logs/TotalsegThorax/TotalsegThorax_fully/w_ce_1.0_w_dice_1.0_w_class_None/fold_0/checkpoint_final.pth",
        GPU_ID=GPU_ID,
        config_dict=Totalsegmentor_training_config_dict,
        data_dim="3d",
        num_processors=NUM_PROCESS,
    )

    data_folder = "./Dataset/CT_Lymph_Nodes/CT_Lymph_Nodes"
    seg_folder = "./Dataset/CT_Lymph_Nodes/CT_NIH_Annotations/NIH"

    data_save_folder = "./Dataset/{}".format(Dataset_name)
    img_save_folder = os.path.join(data_save_folder, "images")
    seg_save_folder = os.path.join(data_save_folder, "labels")
    if os.path.isdir(img_save_folder):
        shutil.rmtree(img_save_folder, ignore_errors=True)
    os.makedirs(img_save_folder, exist_ok=True)
    if os.path.isdir(seg_save_folder):
        shutil.rmtree(seg_save_folder, ignore_errors=True)
    os.makedirs(seg_save_folder, exist_ok=True)

    sparse_data_save_folder = "./Dataset/{}".format(Sparse_Dataset_name)
    sparse_img_save_folder = os.path.join(sparse_data_save_folder, "images")
    sparse_seg_save_folder = os.path.join(sparse_data_save_folder, "labels")
    if os.path.isdir(sparse_img_save_folder):
        shutil.rmtree(sparse_img_save_folder, ignore_errors=True)
    os.makedirs(sparse_img_save_folder, exist_ok=True)
    if os.path.isdir(sparse_seg_save_folder):
        shutil.rmtree(sparse_seg_save_folder, ignore_errors=True)
    os.makedirs(sparse_seg_save_folder, exist_ok=True)

    if IF_VIS:
        lung_save_folder = os.path.join(data_save_folder, "vis_lung")
        if os.path.isdir(lung_save_folder):
            shutil.rmtree(lung_save_folder, ignore_errors=True)
        os.makedirs(lung_save_folder, exist_ok=True)

    case_lst = [i for i in os.listdir(data_folder) if "MED" in i and "043" not in i]
    print("Will process {} cases.".format(len(case_lst)))
    r = []
    with multiprocessing.get_context("spawn").Pool(NUM_PROCESS) as p:
        for case in case_lst:
            case_id = int(case.split("_")[-1])
            if case_id == 43:
                continue
            print("Processing:", case)
            seg_path = os.path.join(seg_folder, "Pat" + str(case_id))
            seg_file_lst = os.listdir(seg_path)
            assert (
                len(seg_file_lst) == 1
            ), "The number of Files(or Folders) should be 1 in this folder."
            seg_file_path = os.path.join(seg_path, seg_file_lst[0])

            data_path = os.path.join(data_folder, case)
            data_file_lst = os.listdir(data_path)
            assert (
                len(data_file_lst) == 1
            ), "The number of Files(or Folders) should be 1 in this folder."
            data_path = os.path.join(data_path, data_file_lst[0])
            data_file_lst = os.listdir(data_path)
            for file in data_file_lst:
                if "mediastinallymphnodes" in file:
                    img_dcm_path = os.path.join(data_path, file)
                elif "Lymph node segmentations" in file:
                    pass
                else:
                    raise FileNotFoundError

            img_save_path = os.path.join(
                img_save_folder, "{}_{:0>4d}_0000.nii.gz".format(Dataset_name, case_id)
            )
            seg_save_path = os.path.join(
                seg_save_folder, "{}_{:0>4d}.nii.gz".format(Dataset_name, case_id)
            )
            sparse_img_save_path = os.path.join(
                sparse_img_save_folder,
                "{}_{:0>4d}_0000.nii.gz".format(Sparse_Dataset_name, case_id),
            )
            sparse_seg_save_path = os.path.join(
                sparse_seg_save_folder,
                "{}_{:0>4d}.nii.gz".format(Sparse_Dataset_name, case_id),
            )

            if IF_VIS:
                lung_save_path = os.path.join(
                    lung_save_folder, "{}_{:0>4d}.nii.gz".format(Dataset_name, case_id)
                )

            totalseg_pred = predictor.predict_single_npy_array([img_dcm_path], "3d")

            r.append(
                p.starmap_async(
                    process,
                    (
                        (
                            img_dcm_path,
                            seg_file_path,
                            img_save_path,
                            seg_save_path,
                            sparse_img_save_path,
                            sparse_seg_save_path,
                            lung_save_path,
                            totalseg_pred,
                            REMAIN_RATIO,
                            IF_VIS,
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
