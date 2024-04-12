import os 

if __name__ == "__main__":
    img_path = "./Dataset/SegRap2023/images"
    seg_path = "./Dataset/SegRap2023/labels"
    img_lst = os.listdir(img_path)
    seg_lst = os.listdir(seg_path)

    for img in img_lst:
        case_lst = img.split('_')
        new_name = "{}-{}_{}".format(case_lst[0], case_lst[1], case_lst[2])
        src_img_path = os.path.join(img_path, img)
        dst_img_path = os.path.join(img_path, new_name)
        os.rename(src_img_path, dst_img_path)

    for seg in seg_lst:
        case_lst = seg.split('_')
        src_seg_path = os.path.join(seg_path, "{}_{}".format(case_lst[0], case_lst[1]))
        dst_seg_path = os.path.join(seg_path, "{}-{}".format(case_lst[0], case_lst[1]))
        os.rename(src_seg_path, dst_seg_path)
