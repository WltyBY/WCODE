import os
import shutil
import multiprocessing

import numpy as np
import SimpleITK as sitk

from scipy import ndimage
from time import sleep
from tqdm import tqdm


def process(file, file_folder, save_folder):
    obj = sitk.ReadImage(os.path.join(file_folder, file))
    array = sitk.GetArrayFromImage(obj)
    dim = len(array.shape)

    s = ndimage.generate_binary_structure(dim, connectivity=1)
    labeled_array, numpatches = ndimage.label(array, s)

    new_seg = np.zeros_like(array)
    for idx in range(1, numpatches + 1):
        instance_label = np.zeros_like(array)
        instance_label[labeled_array == idx] = 1

        z, *_ = np.where(instance_label)
        slice_loc = (z.max() + z.min()) // 2
        new_seg[slice_loc][instance_label[slice_loc] != 0] = idx

        transposed_new_seg = np.transpose(new_seg, (1, 0, 2))
        transposed_instance_label = np.transpose(instance_label, (1, 0, 2))

        z, *_ = np.where(transposed_instance_label)
        slice_loc = (z.max() + z.min()) // 2
        transposed_new_seg[slice_loc][transposed_instance_label[slice_loc] != 0] = idx
        new_seg = np.transpose(transposed_new_seg, (1, 0, 2))
    new_seg[new_seg != 0] = 1
    
    out_obj = sitk.GetImageFromArray(new_seg)
    out_obj.CopyInformation(obj)
    sitk.WriteImage(
        out_obj, os.path.join(save_folder, file.replace("LNQ2023", "LNQ2023Sparse"))
    )


if __name__ == "__main__":
    NUM_PROCESS = 8

    data_folder = "./Dataset/LNQ2023"
    save_folder = "./Dataset/LNQ2023Sparse"

    # file need to rename and copy
    folder_lst = ["imagesTr", "imagesVal", "imagesTs", "labelsVal", "labelsTs"]
    for folder in folder_lst:
        folder_process = os.path.join(data_folder, folder)
        save_folder_process = os.path.join(save_folder, folder)
        if os.path.isdir(save_folder_process):
            shutil.rmtree(save_folder_process, ignore_errors=True)
        os.makedirs(save_folder_process, exist_ok=True)

        for f in os.listdir(folder_process):
            shutil.copyfile(
                os.path.join(folder_process, f),
                os.path.join(
                    save_folder_process, f.replace("LNQ2023", "LNQ2023Sparse")
                ),
            )

    mask_folder = os.path.join(data_folder, "labelsTr")
    save_mask_folder = os.path.join(save_folder, "labelsTr")
    if os.path.isdir(save_mask_folder):
        shutil.rmtree(save_mask_folder, ignore_errors=True)
    os.makedirs(save_mask_folder, exist_ok=True)

    r = []
    with multiprocessing.get_context("spawn").Pool(NUM_PROCESS) as p:
        for case in os.listdir(mask_folder):
            r.append(p.starmap_async(process, ((case, mask_folder, save_mask_folder),)))

        remaining = list(range(len(os.listdir(mask_folder))))
        workers = [j for j in p._pool]
        with tqdm(desc=None, total=len(os.listdir(mask_folder)), disable=False) as pbar:
            while len(remaining) > 0:
                all_alive = all([j.is_alive() for j in workers])
                if not all_alive:
                    raise RuntimeError("Some background worker is 6 feet under. Yuck.")
                done = [i for i in remaining if r[i].ready()]
                for _ in done:
                    pbar.update()
                remaining = [i for i in remaining if i not in done]
                sleep(0.1)
