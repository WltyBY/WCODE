import os
import glob
import shutil
import scipy
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
from wcode.inferring.PatchBasedPredictor import PatchBasedPredictor
from wcode.utils.file_operations import open_yaml
from wcode.utils.ML_utils.ScribbleGenerator import ScribbleGenerator


"""
NOTE: Before running this script, you should train a segmentation model on Totalsegmentatorv2 dataset using convert_totalseg.py at the same folder.
And set GET_DATASET param in convert_totalseg.py to "thorax". Then train a fully-supervised model in this framework.

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


def process(
    file_path,
    img_save_path,
    seg_path,
    seg_save_path,
    totalseg_pred,
    lung_save_path,
    ignore_class,
    seed,
    split,
    if_vis=False,
):
    img_obj = sitk.ReadImage(file_path)
    seg_obj = sitk.ReadImage(seg_path)
    img_array = sitk.GetArrayFromImage(img_obj)
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

    origin = img_obj.GetOrigin()
    spacing = img_obj.GetSpacing()
    origin_output = tuple(
        [origin[i] + spacing[i] * bbmin[::-1][i] for i in range(len(bbmin))]
    )
    img_cropped = crop_ND_volume_with_bounding_box(img_array, bbmin, bbmax)
    seg_cropped = crop_ND_volume_with_bounding_box(seg_array, bbmin, bbmax)
    totalseg_pred_cropped = crop_ND_volume_with_bounding_box(
        totalseg_pred, bbmin, bbmax
    )
    lung_mask_cropped = crop_ND_volume_with_bounding_box(lung_mask, bbmin, bbmax)

    if split == "train":
        # generate a circle around the labeled lymph node as background
        core_dilation = np.ones([3, 3, 3])
        dilated_seg = ndimage.binary_dilation(seg_cropped, core_dilation, iterations=1)

        # Get some background components
        scribble_bg_mask = (dilated_seg - seg_cropped).astype(bool)  # 0
        convex_hull_of_the_lung, _ = flood_fill_hull(lung_mask_cropped)
        bg_components = np.zeros_like(seg_cropped)
        bg_components[convex_hull_of_the_lung == 0] = 1
        bg_components[totalseg_pred_cropped != 0] = 1

        seg_scribble = np.ones_like(seg_cropped) * ignore_class
        seg_scribble[seg_cropped == 1] = 1
        seg_scribble[bg_components == 1] = 0

        sg = ScribbleGenerator(ignore_class_id=ignore_class, dim=2, seed=seed)
        slices_lst = []
        for i in range(seg_cropped.shape[0]):
            slices_lst.append(sg.generate_scribble(seg_scribble[i], [1, 2])[None])
        scibble = np.vstack(slices_lst, dtype=np.uint8)

        scibble[scribble_bg_mask] = 0

        img_output = sitk.GetImageFromArray(img_cropped)
        img_output.SetOrigin(origin_output)
        img_output.SetSpacing(spacing)
        img_output.SetDirection(img_obj.GetDirection())
        sitk.WriteImage(img_output, img_save_path)

        scibble_obj = sitk.GetImageFromArray(scibble)
        scibble_obj.SetOrigin(origin_output)
        scibble_obj.SetSpacing(spacing)
        scibble_obj.SetDirection(img_obj.GetDirection())
        sitk.WriteImage(scibble_obj, seg_save_path)

        if if_vis:
            before_scribble_obj = sitk.GetImageFromArray(seg_scribble)
            before_scribble_obj.SetOrigin(origin_output)
            before_scribble_obj.SetSpacing(spacing)
            before_scribble_obj.SetDirection(img_obj.GetDirection())
            sitk.WriteImage(before_scribble_obj, lung_save_path)
    else:
        img_output = sitk.GetImageFromArray(img_cropped)
        img_output.SetOrigin(origin_output)
        img_output.SetSpacing(spacing)
        img_output.SetDirection(img_obj.GetDirection())
        sitk.WriteImage(img_output, img_save_path)

        seg_output = sitk.GetImageFromArray(seg_cropped)
        seg_output.SetOrigin(origin_output)
        seg_output.SetSpacing(spacing)
        seg_output.SetDirection(img_obj.GetDirection())
        sitk.WriteImage(seg_output, seg_save_path)


def flood_fill_hull(image: np.ndarray) -> tuple:
    """
    Source: https://stackoverflow.com/questions/46310603/how-to-compute-convex-hull-image-volume-in-3d-numpy-arrays
    """
    points = np.transpose(np.where(image))
    hull = scipy.spatial.ConvexHull(points)
    deln = scipy.spatial.Delaunay(points[hull.vertices])
    idx = np.stack(np.indices(image.shape), axis=-1)
    out_idx = np.nonzero(deln.find_simplex(idx) + 1)
    out_img = np.zeros(image.shape)
    out_img[out_idx] = 1
    return out_img, hull


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


if __name__ == "__main__":
    NUM_PROCESS = 16
    GPU_ID = 3
    IF_VIS = True
    SEED = 319
    IGNORE_CLASS = 2

    Totalsegmentor_training_config_dict = open_yaml("./Configs/TotalsegThorax.yaml")
    predictor = get_predictor(
        model_path="./Logs/TotalsegThorax/TotalsegThorax_fully/w_ce_1.0_w_dice_1.0_w_class_None/fold_0/checkpoint_final.pth",
        GPU_ID=GPU_ID,
        config_dict=Totalsegmentor_training_config_dict,
        data_dim="3d",
        num_processors=NUM_PROCESS,
    )

    data_folder_path = "./Dataset/LNQ2023_raw"
    data_save_folder = "./Dataset/LNQ2023Scribble"
    split = ["train", "val", "test"]

    lung_save_folder = os.path.join(data_save_folder, "vis")
    if os.path.isdir(lung_save_folder):
        shutil.rmtree(lung_save_folder, ignore_errors=True)
    os.makedirs(lung_save_folder, exist_ok=True)

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
        p = multiprocessing.get_context("spawn").Pool(NUM_PROCESS)
        
        for file_path in img_file_lst:
            print("Processing:", file_path)
            idx = int(os.path.basename(file_path).split("-")[2])
            save_img_file_name = "LNQ2023Scribble_{:0>4d}_0000.nrrd".format(idx)
            save_seg_file_name = "LNQ2023Scribble_{:0>4d}.nrrd".format(idx)
            img_save_path = os.path.join(img_save_folder, save_img_file_name)
            seg_save_path = os.path.join(seg_save_folder, save_seg_file_name)
            if IF_VIS:
                lung_save_path = os.path.join(lung_save_folder, save_seg_file_name)

            totalseg_pred = predictor.predict_single_npy_array([file_path], "3d")

            r.append(
                p.starmap_async(
                    process,
                    (
                        (
                            file_path,
                            img_save_path,
                            file_path.replace("ct", "seg"),
                            seg_save_path,
                            totalseg_pred,
                            lung_save_path,
                            IGNORE_CLASS,
                            SEED,
                            s,
                            IF_VIS,
                        ),
                    ),
                )
            )
        p.close()
        p.join()
