import os
import shutil
import SimpleITK as sitk

if __name__ == "__main__":
    dataset_identity = "RAOS{}".format("set3")
    dataset_folder = "./Dataset/RAOS-Real/MissingOrgans(Set3)"
    split_lst = ["Ts"]
    file_ending = ".nii.gz"

    save_folder = "./Dataset/" + dataset_identity

    count = 0
    for split in split_lst:
        img_folder = os.path.join(dataset_folder, "images" + split)
        seg_folder = os.path.join(dataset_folder, "labels" + split)
        img_save_folder = os.path.join(save_folder, "images" + split)
        seg_save_folder = os.path.join(save_folder, "labels" + split)

        if os.path.isdir(img_save_folder):
            shutil.rmtree(img_save_folder, ignore_errors=True)
        os.makedirs(img_save_folder, exist_ok=True)
        if os.path.isdir(seg_save_folder):
            shutil.rmtree(seg_save_folder, ignore_errors=True)
        os.makedirs(seg_save_folder, exist_ok=True)

        img_lst = os.listdir(img_folder)
        img_lst.sort()
        for i in img_lst:
            case_name = "{}_{:0>4}".format(dataset_identity, count)
            count += 1

            if split == "Ts":
                img_path = os.path.join(img_folder, i)
                seg_path = os.path.join(seg_folder, i.split("_")[0] + file_ending)
            else:
                img_path = os.path.join(img_folder, i)
                seg_path = os.path.join(seg_folder, i)
                
            img_save_path = os.path.join(
                img_save_folder, case_name + "_0000" + file_ending
            )
            seg_save_path = os.path.join(seg_save_folder, case_name + file_ending)
            shutil.copyfile(img_path, img_save_path)
            shutil.copyfile(seg_path, seg_save_path)
