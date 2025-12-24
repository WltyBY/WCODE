import os
import glob
import shutil
import multiprocessing
import numpy as np
import SimpleITK as sitk

from time import sleep
from tqdm import tqdm
from skimage.segmentation import clear_border
from scipy import ndimage

from wcode.utils.NDarray_operations import (
    get_largest_k_components,
    get_ND_bounding_box,
    crop_ND_volume_with_bounding_box,
)

"""
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


def crop_ct_scan(input_img, input_seg):
    """
    Crop a CT scan based on the bounding box of the human region.
    """
    img = sitk.GetArrayFromImage(input_img)
    seg = sitk.GetArrayFromImage(input_seg)

    mask = np.asarray(img > -600)
    se = np.ones([3, 3, 3])
    mask = ndimage.binary_opening(mask, se, iterations=2)
    mask = get_largest_k_components(mask, 1)
    bbmin, bbmax = get_ND_bounding_box(mask, margin=[5, 10, 10])

    origin = input_img.GetOrigin()
    spacing = input_img.GetSpacing()
    new_origin = tuple(
        [origin[i] + spacing[i] * bbmin[::-1][i] for i in range(len(bbmin))]
    )

    img_sub = crop_ND_volume_with_bounding_box(img, bbmin, bbmax)
    img_sub_obj = sitk.GetImageFromArray(img_sub)
    img_sub_obj.SetOrigin(new_origin)
    img_sub_obj.SetSpacing(spacing)
    img_sub_obj.SetDirection(input_img.GetDirection())

    seg_sub = crop_ND_volume_with_bounding_box(seg, bbmin, bbmax)
    seg_sub_obj = sitk.GetImageFromArray(seg_sub)
    seg_sub_obj.SetOrigin(new_origin)
    seg_sub_obj.SetSpacing(spacing)
    seg_sub_obj.SetDirection(input_img.GetDirection())

    return img_sub_obj, seg_sub_obj


def get_human_region_mask(img):
    """
    Get the mask of human region in CT volumes
    """
    dim = len(img.shape)
    if dim == 4:
        img = img[0]
    mask = np.asarray(img > -600)
    se = np.ones([3, 3, 3])
    mask = ndimage.binary_opening(mask, se, iterations=2)
    mask = get_largest_k_components(mask, 1)
    mask_close = ndimage.binary_closing(mask, se, iterations=2)

    D, H, W = mask.shape
    for d in [1, 2, D - 3, D - 2]:
        mask_close[d] = mask[d]
    for d in range(0, D, 2):
        mask_close[d, 2:-2, 2:-2] = np.ones((H - 4, W - 4))

    # get background component
    bg = np.zeros_like(mask)
    bgs = get_largest_k_components(1 - mask_close, 10)
    for bgi in bgs:
        indices = np.where(bgi)
        if bgi.sum() < 1000:
            break
        if (
            indices[0].min() == 0
            or indices[1].min() == 0
            or indices[2].min() == 0
            or indices[0].max() == D - 1
            or indices[1].max() == H - 1
            or indices[2].max() == W - 1
        ):
            bg = bg + bgi
    fg = 1 - bg

    fg = ndimage.binary_opening(fg, se, iterations=1)
    fg = get_largest_k_components(fg, 1)
    if dim == 4:
        fg = np.expand_dims(fg, 0)
    fg = np.asarray(fg, np.uint8)
    return fg


def get_lungmask(img_array):
    coarse_body_array = np.zeros_like(img_array)
    # intensity smaller than -300 as air
    coarse_body_array[img_array >= -300] = 1

    # get mask of body
    bodymask_array = get_human_region_mask(img_array)

    # bodymask_array - threshold to get the region of air in body.
    lungmask_array = bodymask_array - coarse_body_array
    lungmask_array[lungmask_array != 1] = 0

    # refine mask of lung
    struct = np.ones([3, 3, 3])
    lungmask_array = ndimage.binary_erosion(
        lungmask_array, structure=struct, iterations=1
    ).astype("uint8")
    lungmask_array = clear_border(lungmask_array)
    struct = np.ones([3, 3, 3])
    lungmask_array = ndimage.binary_dilation(
        lungmask_array, structure=struct, iterations=1
    ).astype("uint8")

    struct = np.ones([7, 7, 7])
    lungmask_array = ndimage.binary_closing(
        lungmask_array, structure=struct, iterations=1
    ).astype("uint8")

    lungmask_array = ndimage.binary_fill_holes(lungmask_array, struct)

    components = get_largest_k_components(lungmask_array, k=2)
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


def crop_to_lung_area(file_path, img_save_path, seg_path, seg_save_path):
    vol = sitk.ReadImage(file_path)
    seg = sitk.ReadImage(seg_path)
    crop_to_body, seg_crop_to_body = crop_ct_scan(vol, seg)
    # sitk.WriteImage(crop_to_body, "/home/wlty/disk2/nnUNet_data/nnUNet_raw/Dataset110_CTLymphNodes/body.nii.gz")
    img_array = sitk.GetArrayFromImage(crop_to_body)
    seg_array = sitk.GetArrayFromImage(seg_crop_to_body)

    lungmask_array = get_lungmask(img_array)
    # mask_obj = sitk.GetImageFromArray(mask)
    # mask_obj.CopyInformation(crop_to_body)
    # sitk.WriteImage(mask_obj, "/home/wlty/disk2/nnUNet_data/nnUNet_raw/Dataset110_CTLymphNodes/mask.nii.gz")

    bbmin = [0, 0, 0]
    bbmax = [0, 0, 0]
    bbmin_img, bbmax_img = get_ND_bounding_box(lungmask_array, margin=[5, 10, 10])
    # print(bbmin_img, bbmax_img)
    bbmin_seg, bbmax_seg = get_ND_bounding_box(seg_array, margin=(5, 10, 10))
    # print(bbmin_seg, bbmax_seg)
    for i in range(len(bbmin_img)):
        bbmin[i] = min(bbmin_img[i], bbmin_seg[i])
        bbmax[i] = max(bbmax_img[i], bbmax_seg[i])

    crop_shape = img_array.shape
    center = np.array(crop_shape) // 2

    for i in range(1, len(bbmin)):
        if bbmin[i] < center[i] < bbmax[i]:
            if (center[i] - bbmin[i]) / (bbmax[i] - center[i]) >= 2:
                bbmax[i] = crop_shape[i] - bbmin[i]
            elif (center[i] - bbmin[i]) / (bbmax[i] - center[i]) <= 0.5:
                bbmin[i] = crop_shape[i] - bbmax[i]
        elif bbmin[i] >= center[i]:
            bbmin[i] = crop_shape[i] - bbmax[i]
        elif bbmax[i] <= center[i]:
            bbmax[i] = crop_shape[i] - bbmin[i]

    origin = vol.GetOrigin()
    spacing = vol.GetSpacing()
    origin_output = tuple(
        [origin[i] + spacing[i] * bbmin[::-1][i] for i in range(len(bbmin))]
    )
    img_output = crop_ND_volume_with_bounding_box(
        sitk.GetArrayFromImage(crop_to_body), bbmin, bbmax
    )
    seg_output = crop_ND_volume_with_bounding_box(seg_array, bbmin, bbmax)

    img_output = sitk.GetImageFromArray(img_output)
    img_output.SetOrigin(origin_output)
    img_output.SetSpacing(spacing)
    img_output.SetDirection(vol.GetDirection())
    sitk.WriteImage(img_output, img_save_path)

    seg_output = sitk.GetImageFromArray(seg_output)
    seg_output.SetOrigin(origin_output)
    seg_output.SetSpacing(spacing)
    seg_output.SetDirection(vol.GetDirection())
    sitk.WriteImage(seg_output, seg_save_path)


if __name__ == "__main__":
    NUM_PROCESS = 16
    data_folder_path = "./Dataset/LNQ2023_raw"
    data_save_folder = "./Dataset/LNQ2023"
    split = ["train", "val", "test"]

    for s in split:
        if s == "train":
            img_save_folder = os.path.join(data_save_folder, "imagesTr")
            seg_save_folder = os.path.join(data_save_folder, "labelsTr")
        elif s == "val":
            img_save_folder = os.path.join(data_save_folder, "imagesVal")
            seg_save_folder = os.path.join(data_save_folder, "labelsVal")
        elif s == "test":
            img_save_folder = os.path.join(data_save_folder, "imagesTs")
            seg_save_folder = os.path.join(data_save_folder, "labelsTs")
        else:
            raise ValueError

        data_split_folder = os.path.join(data_folder_path, s)
        if os.path.isdir(img_save_folder):
            shutil.rmtree(img_save_folder, ignore_errors=True)
        os.makedirs(img_save_folder, exist_ok=True)
        if os.path.isdir(seg_save_folder):
            shutil.rmtree(seg_save_folder, ignore_errors=True)
        os.makedirs(seg_save_folder, exist_ok=True)

        img_file_lst = glob.glob(os.path.join(data_split_folder, "*-ct.nrrd"))
        r = []
        with multiprocessing.get_context("spawn").Pool(NUM_PROCESS) as p:
            for file_path in img_file_lst:
                idx = int(os.path.basename(file_path).split("-")[2])
                save_img_file_name = "LNQ2023_{:0>4d}_0000.nrrd".format(idx)
                save_seg_file_name = "LNQ2023_{:0>4d}.nrrd".format(idx)
                img_save_path = os.path.join(img_save_folder, save_img_file_name)
                seg_save_path = os.path.join(seg_save_folder, save_seg_file_name)

                r.append(
                    p.starmap_async(
                        crop_to_lung_area,
                        (
                            (
                                file_path,
                                img_save_path,
                                file_path.replace("ct", "seg"),
                                seg_save_path,
                            ),
                        ),
                    )
                )
            remaining = list(range(len(img_file_lst)))
            workers = [j for j in p._pool]
            with tqdm(desc=None, total=len(img_file_lst), disable=False) as pbar:
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
