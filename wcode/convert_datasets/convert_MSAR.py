import os
import shutil
import numpy as np
import SimpleITK as sitk

from pprint import pprint


if __name__ == "__main__":
    raw_folder = "./Dataset/MSAR_raw"
    # pre (1), mid (2)
    stage_name: str = "pre"
    # head, headandneck, neck
    used_regions: list = ["head"]
    # 1-GTVp, 2-GTVnd
    used_class: list = [1]
    saved_class_id = [i for i in range(1, len(used_class) + 1)]

    patient_lst = [i for i in os.listdir(raw_folder) if "patient" in i]
    summary = {"stages": {}, "regions": {}, "cases": {}}

    dataset_name = "MSAR{}".format(stage_name)
    save_folder = "./Dataset/{}".format(dataset_name)
    img_save_folder = os.path.join(save_folder, "images")
    seg_save_folder = os.path.join(save_folder, "labels")
    if os.path.isdir(img_save_folder):
        shutil.rmtree(img_save_folder, ignore_errors=True)
    os.makedirs(img_save_folder, exist_ok=True)
    if os.path.isdir(seg_save_folder):
        shutil.rmtree(seg_save_folder, ignore_errors=True)
    os.makedirs(seg_save_folder, exist_ok=True)

    for patient in patient_lst:
        stages_lst = os.listdir(os.path.join(raw_folder, patient))
        for stages in stages_lst:
            stage_id = stages.split("_")[0]
            if stage_id not in summary["stages"].keys():
                summary["stages"][stage_id] = {}

            if stage_id not in summary["cases"].keys():
                summary["cases"][stage_id] = 1
            else:
                summary["cases"][stage_id] += 1

            img_lst = os.listdir(os.path.join(raw_folder, patient, stages))
            for img in img_lst:
                if not img.endswith("_0000.nii.gz"):
                    continue
                img_region = img.split("_")[0].split("-")[-1]

                if img_region not in ["head", "headandneck", "neck"]:
                    raise Exception(os.path.join(raw_folder, patient, stages, img))

                if img_region not in summary["stages"][stage_id].keys():
                    summary["stages"][stage_id][img_region] = 1
                else:
                    summary["stages"][stage_id][img_region] += 1

                if img_region not in summary["regions"].keys():
                    summary["regions"][img_region] = 1
                else:
                    summary["regions"][img_region] += 1

                if (
                    (stage_name == "pre" and stage_id == "1")
                    or (stage_name == "mid" and stage_id == "2")
                ) and img_region in used_regions:
                    num_file = len(os.listdir(img_save_folder))
                    seg = img.replace("_0000.nii.gz", ".nii.gz")

                    img_obj = sitk.ReadImage(
                        os.path.join(raw_folder, patient, stages, img)
                    )
                    seg_obj = sitk.ReadImage(
                        os.path.join(raw_folder, patient, stages, seg)
                    )

                    img_array = sitk.GetArrayFromImage(img_obj)
                    seg_array = sitk.GetArrayFromImage(seg_obj)

                    seg_new = np.zeros_like(seg_array)
                    for class_id, saved_id in zip(used_class, saved_class_id):
                        seg_new[seg_array == class_id] = saved_id

                    img_save = sitk.GetImageFromArray(img_array)
                    img_save.SetDirection(img_obj.GetDirection())
                    img_save.SetOrigin(img_obj.GetOrigin())
                    img_save.SetSpacing(img_obj.GetSpacing())
                    sitk.WriteImage(
                        img_save,
                        os.path.join(
                            img_save_folder,
                            "{}_{:0>4d}_0000.nii.gz".format(dataset_name, num_file),
                        ),
                    )

                    seg_save = sitk.GetImageFromArray(seg_new)
                    seg_save.SetDirection(img_obj.GetDirection())
                    seg_save.SetOrigin(img_obj.GetOrigin())
                    seg_save.SetSpacing(img_obj.GetSpacing())
                    sitk.WriteImage(
                        seg_save,
                        os.path.join(
                            seg_save_folder,
                            "{}_{:0>4d}.nii.gz".format(dataset_name, num_file),
                        ),
                    )

    pprint(summary)
    print(
        "Processed files: img-{}, seg-{}.".format(
            len(os.listdir(img_save_folder)), len(os.listdir(seg_save_folder))
        )
    )
    print("ID of used class to ID of saved class.")
    print(used_class)
    print(saved_class_id)
