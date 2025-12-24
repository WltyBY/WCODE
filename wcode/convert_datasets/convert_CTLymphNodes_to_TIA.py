import os
import math
import shutil
import multiprocessing
import numpy as np
import SimpleITK as sitk

from tqdm import tqdm
from time import sleep
from scipy import ndimage


def refine_and_delete_instance(seg):
    dim = len(seg.shape)
    num_slice = seg.shape[0]

    new_seg = np.zeros_like(seg)
    for slice in range(num_slice):
        s = ndimage.generate_binary_structure(dim - 1, connectivity=1)
        labeled_array, numpatches = ndimage.label(seg[slice], s)

        refine_unique = np.unique(labeled_array[labeled_array != 0])
        if len(refine_unique) == 0:
            continue

        sizes = ndimage.sum(seg[slice], labeled_array, range(1, numpatches + 1))
        max_label = np.argmax(sizes) + 1

        new_seg[slice][labeled_array == max_label] = 1

    new_seg[new_seg != 0] = 1
    return new_seg.astype(np.uint8)


def process(
    img_path,
    seg_path,
    img_save_path,
    seg_save_path,
):
    image_obj = sitk.ReadImage(img_path)
    seg_obj = sitk.ReadImage(seg_path)

    img_array = sitk.GetArrayFromImage(image_obj)
    seg_array = sitk.GetArrayFromImage(seg_obj)

    seg_output = refine_and_delete_instance(seg_array)

    img_output = sitk.GetImageFromArray(img_array)
    img_output.CopyInformation(image_obj)
    sitk.WriteImage(img_output, img_save_path)

    seg_output = sitk.GetImageFromArray(seg_output)
    seg_output.CopyInformation(seg_obj)
    sitk.WriteImage(seg_output, seg_save_path)


if __name__ == "__main__":
    NUM_PROCESS = 8
    Dataset_name = "CTLymphNodes02TIA"

    data_folder = "./Dataset/CTLymphNodes/images"
    seg_folder = "./Dataset/CTLymphNodes/labels"

    data_save_folder = "./Dataset/{}".format(Dataset_name)
    img_save_folder = os.path.join(data_save_folder, "images")
    seg_save_folder = os.path.join(data_save_folder, "labels")
    if os.path.isdir(img_save_folder):
        shutil.rmtree(img_save_folder, ignore_errors=True)
    os.makedirs(img_save_folder, exist_ok=True)
    if os.path.isdir(seg_save_folder):
        shutil.rmtree(seg_save_folder, ignore_errors=True)
    os.makedirs(seg_save_folder, exist_ok=True)

    case_lst = [i for i in os.listdir(data_folder)]
    print("Will process {} cases.".format(len(case_lst)))
    r = []
    with multiprocessing.get_context("spawn").Pool(NUM_PROCESS) as p:
        for case in case_lst:
            img_path = os.path.join(data_folder, case)
            seg_path = os.path.join(seg_folder, case.replace("_0000.nii.gz", ".nii.gz"))
            case_id = int(case.split("_")[1])

            img_save_path = os.path.join(
                img_save_folder, "{}_{:0>4d}_0000.nii.gz".format(Dataset_name, case_id)
            )
            seg_save_path = os.path.join(
                seg_save_folder, "{}_{:0>4d}.nii.gz".format(Dataset_name, case_id)
            )

            r.append(
                p.starmap_async(
                    process,
                    (
                        (
                            img_path,
                            seg_path,
                            img_save_path,
                            seg_save_path,
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
