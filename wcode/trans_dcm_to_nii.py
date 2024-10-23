import os
import shutil
from wcode.utils.file_operations import dicom2nifti

if __name__ == "__main__":
    data_path = "./配对MRI"
    img_save_folder = "./Liao_data"
    if os.path.isdir(img_save_folder):
        shutil.rmtree(img_save_folder, ignore_errors=True)
    os.makedirs(img_save_folder, exist_ok=True)

    for img_folder in os.listdir(data_path):
        dcm_path = os.path.join(data_path, img_folder)
        print(img_folder)
        case_name = img_folder.split("_")[0].split("^")[0]
        save_folder = os.path.join(img_save_folder, case_name)
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder, exist_ok=True)

        save_img_name = (
            img_folder.split("_")[3]
            + "_"
            +img_folder.split("_")[-4]
            + "_"
            + img_folder.split("_")[-3]
            + "_"
            + img_folder.split("_")[-1]
        ).replace(".", "_")
        dicom2nifti(dcm_path, os.path.join(save_folder, save_img_name + ".nii.gz"))

    print("Original number of images:", len(os.listdir(data_path)))
    count = 0
    for case in os.listdir(img_save_folder):
        count += len(os.listdir(os.path.join(img_save_folder, case)))
    print("Number of images after transition:", count)
