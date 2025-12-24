import os, shutil
import multiprocessing
import numpy as np
import SimpleITK as sitk

from time import sleep
from tqdm import tqdm
from scipy import ndimage
from skimage.segmentation import clear_border

from wcode.utils.ML_utils.ScribbleGenerator import ScribbleGenerator
from wcode.utils.NDarray_operations import get_largest_k_components, get_ND_bounding_box

"""
NOTE: Before running this script, please run convert_LNQ2023.py or convert_LNQ2023_using_totalseg.py first. This script is running on the preprocessed datasets.
But better run script using totalsegmentator.

You can get the dataset on https://www.cancerimagingarchive.net/collection/mediastinal-lymph-node-seg/
Data Citation:
Idris, T., Somarouthu, S., Jacene, H., LaCasce, A., Ziegler, E., Pieper, S., Khajavi, R., Dorent, R., Pujol, S., Kikinis, R., & Harris, G. (2024). 
Mediastinal Lymph Node Quantification (LNQ): Segmentation of Heterogeneous CT Data (Version 1) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/QVAZ-JA09

Challenge paper:
@article{dorent2024lnq,
  title={LNQ 2023 challenge: Benchmark of weakly-supervised techniques for mediastinal lymph node quantification},
  author={Dorent, Reuben and Khajavi, Roya and Idris, Tagwa and Ziegler, Erik and Somarouthu, Bhanusupriya and Jacene, Heather and LaCasce, Ann and Deissler, Jonathan and Ehrhardt, Jan and Engelson, Sofija and others},
  journal={arXiv preprint arXiv:2408.10069},
  year={2024}
}
"""


def process(img_file_path, seg_file_path, ignore_class, seed, seg_save_path):
    img_obj = sitk.ReadImage(img_file_path)
    seg_obj = sitk.ReadImage(seg_file_path)
    img_array = sitk.GetArrayFromImage(img_obj)
    seg_array = sitk.GetArrayFromImage(seg_obj)

    seg_array[seg_array != 1] = 0
    seg_scribble = np.ones_like(seg_array) * ignore_class

    # generate a circle around the labeled lymph node as background
    core_dilation = np.ones([3, 3, 3])
    dilated_seg = ndimage.binary_dilation(
        seg_array, core_dilation, iterations=1
    ).astype(np.uint8)

    scribble_bg_mask = (dilated_seg - seg_array).astype(bool)  # 0
    scribble_fg_mask = seg_array == 1  # 1
    lungmask = get_lungmask(img_array).astype(bool)  # 0
    # label_obj = sitk.GetImageFromArray(lungmask.astype(np.uint8))
    # label_obj.CopyInformation(img_obj)
    # sitk.WriteImage(
    #     label_obj,
    #     os.path.join("./Predictions/lung_mask", os.path.basename(img_file_path)),
    # )

    bbmin, bbmax = get_ND_bounding_box(lungmask, margin=[-5, 0, 0])
    arounding_mask = np.ones_like(seg_array)
    arounding_mask[bbmin[0] : bbmax[0], :, :] = 0  # 0

    seg_scribble[arounding_mask.astype(bool)] = 0
    seg_scribble[lungmask] = 0
    seg_scribble[scribble_bg_mask] = 0
    seg_scribble[scribble_fg_mask] = 1

    # label_obj = sitk.GetImageFromArray(seg_scribble.astype(np.uint8))
    # label_obj.CopyInformation(img_obj)
    # sitk.WriteImage(
    #     label_obj,
    #     os.path.join("./Predictions/final_mask", os.path.basename(img_file_path)),
    # )

    sg = ScribbleGenerator(ignore_class_id=ignore_class, dim=2, seed=seed)
    slices_lst = []
    for i in range(seg_array.shape[0]):
        slices_lst.append(sg.generate_scribble(seg_scribble[i], [1, 2])[None])

    scibble = np.vstack(slices_lst, dtype=np.uint8)
    scibble[scribble_bg_mask] = 0
    scibble_obj = sitk.GetImageFromArray(scibble)
    scibble_obj.CopyInformation(img_obj)
    sitk.WriteImage(scibble_obj, seg_save_path)


def get_lungmask(img_array):
    lung_mask = np.zeros_like(img_array)
    # intensity smaller than -300 as air
    lung_mask[img_array <= -300] = 1

    struct = np.ones([3, 3, 3])
    lung_mask = ndimage.binary_erosion(
        lung_mask, structure=struct, iterations=1
    ).astype("uint8")

    lung_mask = clear_border(lung_mask)

    struct = np.ones([3, 3, 3])
    lung_mask = ndimage.binary_dilation(
        lung_mask, structure=struct, iterations=1
    ).astype("uint8")

    struct = np.ones([7, 7, 7])
    lung_mask = ndimage.binary_closing(
        lung_mask, structure=struct, iterations=1
    ).astype("uint8")

    lung_mask = ndimage.binary_fill_holes(lung_mask, struct)

    components = get_largest_k_components(lung_mask, k=2)
    assert isinstance(components, list)

    label = np.zeros_like(img_array, dtype=np.uint8)
    if len(components) > 1:
        size_max = np.sum(components[0])
        size_min = np.sum(components[1])
        if size_max * 0.3 < size_min:
            label[components[0] == 1] = 1
            label[components[1] == 1] = 1
        else:
            label[components[0] == 1] = 1
    else:
        label[components[0] == 1] = 1

    return label.astype("uint8")


if __name__ == "__main__":
    SEED = 319
    NUM_PROCESS = 16
    IGNORE_CLASS = 2
    dataset_folder = "./Dataset/LNQ2023"
    save_folder = "./Dataset/LNQ2023Scribble"
    img_folder = ["imagesTr", "imagesVal", "imagesTs"]
    seg_folder = ["labelsTr", "labelsVal", "labelsTs"]

    for i in range(len(img_folder)):
        img_save_folder = os.path.join(save_folder, img_folder[i])
        if os.path.isdir(img_save_folder):
            shutil.rmtree(img_save_folder, ignore_errors=True)
        os.makedirs(img_save_folder, exist_ok=True)

        seg_save_folder = os.path.join(save_folder, seg_folder[i])
        if os.path.isdir(seg_save_folder):
            shutil.rmtree(seg_save_folder, ignore_errors=True)
        os.makedirs(seg_save_folder, exist_ok=True)

        img_lst = os.listdir(os.path.join(dataset_folder, img_folder[i]))

        if "Tr" in img_folder[i]:
            # for img_file in img_lst:
            #     case_id = img_file.split("_")[1]
            #     shutil.copyfile(
            #         os.path.join(dataset_folder, img_folder[i], img_file),
            #         os.path.join(
            #             img_save_folder, img_file.replace("LNQ", "LNQScribble")
            #         ),
            #     )
            #     seg_file = "LNQ_" + case_id + ".nrrd"
            #     process(
            #         os.path.join(dataset_folder, img_folder[i], img_file),
            #         os.path.join(dataset_folder, seg_folder[i], seg_file),
            #         2,
            #         SEED,
            #         os.path.join(
            #             seg_save_folder,
            #             seg_file.replace("LNQ", "LNQScribble"),
            #         ),
            #     )
            r = []
            with multiprocessing.get_context("spawn").Pool(NUM_PROCESS) as p:
                for img_file in img_lst:
                    case_id = img_file.split("_")[1]
                    shutil.copyfile(
                        os.path.join(dataset_folder, img_folder[i], img_file),
                        os.path.join(
                            img_save_folder,
                            img_file.replace("LNQ2023", "LNQ2023Scribble"),
                        ),
                    )
                    seg_file = "LNQ2023_" + case_id + ".nrrd"
                    r.append(
                        p.starmap_async(
                            process,
                            (
                                (
                                    os.path.join(
                                        dataset_folder, img_folder[i], img_file
                                    ),
                                    os.path.join(
                                        dataset_folder, seg_folder[i], seg_file
                                    ),
                                    IGNORE_CLASS,
                                    SEED,
                                    os.path.join(
                                        seg_save_folder,
                                        seg_file.replace("LNQ2023", "LNQ2023Scribble"),
                                    ),
                                ),
                            ),
                        )
                    )
                remaining = list(range(len(img_lst)))
                workers = [j for j in p._pool]
                with tqdm(desc=None, total=len(img_lst), disable=False) as pbar:
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
        else:
            for img_file in img_lst:
                shutil.copyfile(
                    os.path.join(dataset_folder, img_folder[i], img_file),
                    os.path.join(
                        img_save_folder, img_file.replace("LNQ2023", "LNQ2023Scribble")
                    ),
                )
                case_id = img_file.split("_")[1]
                seg_file = "LNQ2023_" + case_id + ".nrrd"
                shutil.copyfile(
                    os.path.join(dataset_folder, seg_folder[i], seg_file),
                    os.path.join(
                        seg_save_folder, seg_file.replace("LNQ2023", "LNQ2023Scribble")
                    ),
                )
