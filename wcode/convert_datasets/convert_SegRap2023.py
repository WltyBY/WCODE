import os
import shutil
import multiprocessing
import SimpleITK as sitk
import numpy as np

from time import sleep
from tqdm import tqdm

from wcode.preprocessing.cropping import create_mask_base_on_human_body, crop_to_mask
from wcode.utils.data_io import read_sitk_case

"""
A dataset from https://segrap2023.grand-challenge.org/
@misc{luo2023segrap2023benchmarkorgansatriskgross,
      title={SegRap2023: A Benchmark of Organs-at-Risk and Gross Tumor Volume Segmentation for Radiotherapy Planning of Nasopharyngeal Carcinoma}, 
      author={Xiangde Luo and Jia Fu and Yunxin Zhong and Shuolin Liu and Bing Han and Mehdi Astaraki and Simone Bendazzoli and Iuliana Toma-Dasu and Yiwen Ye and Ziyang Chen and Yong Xia and Yanzhou Su and Jin Ye and Junjun He and Zhaohu Xing and Hongqiu Wang and Lei Zhu and Kaixiang Yang and Xin Fang and Zhiwei Wang and Chan Woong Lee and Sang Joon Park and Jaehee Chun and Constantin Ulrich and Klaus H. Maier-Hein and Nchongmaje Ndipenoch and Alina Miron and Yongmin Li and Yimeng Zhang and Yu Chen and Lu Bai and Jinlong Huang and Chengyang An and Lisheng Wang and Kaiwen Huang and Yunqi Gu and Tao Zhou and Mu Zhou and Shichuan Zhang and Wenjun Liao and Guotai Wang and Shaoting Zhang},
      year={2023},
      eprint={2312.09576},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2312.09576}, 
}
"""


def get_seg(each_seg_path_lst):
    obj = sitk.ReadImage(each_seg_path_lst[0])
    array = sitk.GetArrayFromImage(obj)

    seg_array = np.zeros_like(array)
    for i in range(len(each_seg_path_lst)):
        itk_obj = sitk.ReadImage(each_seg_path_lst[i])
        itk_array = sitk.GetArrayFromImage(itk_obj)
        seg_array[itk_array != 0] = i + 1

    return seg_array[None].astype(np.uint8)


def set_itk_obj_info(itk_obj, info_dict):
    itk_obj.SetDirection(info_dict["direction"])
    itk_obj.SetSpacing(info_dict["spacing"])
    itk_obj.SetOrigin(info_dict["origin"])


def run_a_case(case, data_folder, img_save_folder, seg_save_folder):
    case_id = int(case.split("_")[-1])
    data_path = os.path.join(data_folder, case)

    img_path = os.path.join(data_path, "image.nii.gz")
    img_save_path = os.path.join(
        img_save_folder, "SegRap2023_{:0>4d}_0000.nii.gz".format(case_id)
    )

    contrast_img_path = os.path.join(data_path, "image_contrast.nii.gz")
    contrast_img_save_path = os.path.join(
        img_save_folder, "SegRap2023_{:0>4d}_0001.nii.gz".format(case_id)
    )

    GTVnd_path = os.path.join(data_path, "GTVnd.nii.gz")
    GTVp_path = os.path.join(data_path, "GTVp.nii.gz")
    seg_save_path = os.path.join(
        seg_save_folder, "SegRap2023_{:0>4d}.nii.gz".format(case_id)
    )

    ###############___get the class of label you want here___##############
    seg_array = get_seg([GTVnd_path])

    img_array, info_dict = read_sitk_case([img_path, contrast_img_path])
    data_cropped, seg_cropped, bbox = crop_to_mask(
        img_array, seg_array, create_mask=create_mask_base_on_human_body
    )
    info_dict["origin"] = [
        info_dict["origin"][i] + bbox[0][::-1][i] * info_dict["spacing"][i]
        for i in range(len(info_dict["origin"]))
    ]

    img_cropped_obj = sitk.GetImageFromArray(data_cropped[0])
    contrast_img_cropped_obj = sitk.GetImageFromArray(data_cropped[1])
    seg_obj = sitk.GetImageFromArray(seg_cropped[0])

    set_itk_obj_info(img_cropped_obj, info_dict)
    set_itk_obj_info(contrast_img_cropped_obj, info_dict)
    set_itk_obj_info(seg_obj, info_dict)

    sitk.WriteImage(img_cropped_obj, img_save_path)
    sitk.WriteImage(contrast_img_cropped_obj, contrast_img_save_path)
    sitk.WriteImage(seg_obj, seg_save_path)


if __name__ == "__main__":
    num_processes = 8
    data_folder = "./Dataset/SegRap2023_raw/SegRap2023_Training_Set_120cases"
    save_folder = "./Dataset/SegRap2023"

    img_save_folder = os.path.join(save_folder, "images")
    seg_save_folder = os.path.join(save_folder, "labels")

    if os.path.isdir(img_save_folder):
        shutil.rmtree(img_save_folder, ignore_errors=True)
    os.makedirs(img_save_folder, exist_ok=True)
    if os.path.isdir(seg_save_folder):
        shutil.rmtree(seg_save_folder, ignore_errors=True)
    os.makedirs(seg_save_folder, exist_ok=True)

    case_lst = [i for i in os.listdir(data_folder) if "segrap" in i]
    r = []
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        for case in case_lst:
            r.append(
                p.starmap_async(
                    run_a_case, ((case, data_folder, img_save_folder, seg_save_folder),)
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
