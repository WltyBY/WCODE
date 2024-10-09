# This script will convert a WSI(1000, 1000) to converted one(1000, 1000)
import os
import random
import cv2
import shutil
import numpy as np
import xml.etree.ElementTree as ET

from skimage import draw
from shapely.geometry import Polygon

from wcode.utils.file_operations import save_csv, save_yaml
from wcode.utils.NDarray_operations import create_h_component_image

# Processing flag: Fully, Point-level, Partial Point-level, Partial Instance-level
PROCESSING_FLAG = "Point-level"
# If PROCESSING_FLAG is Partial Point-level or Partial Instance-level, REMAIN_RATE will be used.
REMAIN_RATE = 0.2

# Althougn 37 in the origin downloading train set, 7 of their segmentations are in poor quality.
# So just use the following 30 cases.
train_set_lst = [
    "TCGA-38-6178-01Z-00-DX1.tif",
    "TCGA-G9-6356-01Z-00-DX1.tif",
    "TCGA-HE-7129-01Z-00-DX1.tif",
    "TCGA-HE-7128-01Z-00-DX1.tif",
    "TCGA-21-5784-01Z-00-DX1.tif",
    "TCGA-B0-5698-01Z-00-DX1.tif",
    "TCGA-E2-A14V-01Z-00-DX1.tif",
    "TCGA-DK-A2I6-01A-01-TS1.tif",
    "TCGA-RD-A8N9-01A-01-TS1.tif",
    "TCGA-B0-5710-01Z-00-DX1.tif",
    "TCGA-CH-5767-01Z-00-DX1.tif",
    "TCGA-A7-A13E-01Z-00-DX1.tif",
    "TCGA-G9-6336-01Z-00-DX1.tif",
    "TCGA-18-5592-01Z-00-DX1.tif",
    "TCGA-E2-A1B5-01Z-00-DX1.tif",
    "TCGA-AR-A1AS-01Z-00-DX1.tif",
    "TCGA-A7-A13F-01Z-00-DX1.tif",
    "TCGA-G9-6362-01Z-00-DX1.tif",
    "TCGA-G9-6363-01Z-00-DX1.tif",
    "TCGA-G9-6348-01Z-00-DX1.tif",
    "TCGA-50-5931-01Z-00-DX1.tif",
    "TCGA-21-5786-01Z-00-DX1.tif",
    "TCGA-49-4488-01Z-00-DX1.tif",
    "TCGA-AY-A8YK-01A-01-TS1.tif",
    "TCGA-NH-A8F7-01A-01-TS1.tif",
    "TCGA-G2-A2EK-01A-02-TSB.tif",
    "TCGA-B0-5711-01Z-00-DX1.tif",
    "TCGA-HE-7130-01Z-00-DX1.tif",
    "TCGA-AR-A1AK-01Z-00-DX1.tif",
    "TCGA-KB-A93J-01A-01-TS1.tif",
]

def gen_colormap(shape, seed):
    np.random.seed(seed)
    color_val = np.random.randint(0, 255, shape)
    return color_val


if __name__ == "__main__":
    random_seed = 319
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)

    Dataset_name = (
        "MoNuSeg" + PROCESSING_FLAG + str(REMAIN_RATE)
        if "Partial" in PROCESSING_FLAG
        else "MoNuSeg" + PROCESSING_FLAG
    )
    data_folder = ".\Dataset\MoNuSeg_raw"
    img_data_folder = os.path.join(
        data_folder, "MoNuSeg 2018 Training Data", "Tissue Images"
    )
    seg_data_folder = os.path.join(
        data_folder, "MoNuSeg 2018 Training Data", "Annotations"
    )
    test_data_folder = os.path.join(data_folder, "MoNuSegTestData")

    save_folder = "./Dataset/MoNuSeg" + PROCESSING_FLAG
    if PROCESSING_FLAG in ["Partial Point-level", "Partial Instance-level"]:
        save_folder = "./Dataset/MoNuSeg" + PROCESSING_FLAG + str(REMAIN_RATE)

    imgTr_save_folder = os.path.join(save_folder, "imagesTr")
    segTr_save_folder = os.path.join(save_folder, "labelsTr")
    if os.path.isdir(imgTr_save_folder):
        shutil.rmtree(imgTr_save_folder, ignore_errors=True)
    os.makedirs(imgTr_save_folder, exist_ok=True)
    if os.path.isdir(segTr_save_folder):
        shutil.rmtree(segTr_save_folder, ignore_errors=True)
    os.makedirs(segTr_save_folder, exist_ok=True)

    imgVal_save_folder = os.path.join(save_folder, "imagesVal")
    segVal_save_folder = os.path.join(save_folder, "labelsVal")
    if os.path.isdir(imgVal_save_folder):
        shutil.rmtree(imgVal_save_folder, ignore_errors=True)
    os.makedirs(imgVal_save_folder, exist_ok=True)
    if os.path.isdir(segVal_save_folder):
        shutil.rmtree(segVal_save_folder, ignore_errors=True)
    os.makedirs(segVal_save_folder, exist_ok=True)

    imgTs_save_folder = os.path.join(save_folder, "imagesTs")
    segTs_save_folder = os.path.join(save_folder, "labelsTs")
    if os.path.isdir(imgTs_save_folder):
        shutil.rmtree(imgTs_save_folder, ignore_errors=True)
    os.makedirs(imgTs_save_folder, exist_ok=True)
    if os.path.isdir(segTs_save_folder):
        shutil.rmtree(segTs_save_folder, ignore_errors=True)
    os.makedirs(segTs_save_folder, exist_ok=True)

    vis_save_folder = os.path.join(save_folder, "visualization")
    if os.path.isdir(vis_save_folder):
        shutil.rmtree(vis_save_folder, ignore_errors=True)
    os.makedirs(vis_save_folder, exist_ok=True)

    save_csv(
        [
            [
                "Origin_name",
                "Name",
                "Origin_cases",
                "Remain_cases",
                "Each_case_pixels",
            ]
        ],
        os.path.join(save_folder, "Data_info.csv"),
        mode="w",
    )

    img_lst = [
        i
        for i in os.listdir(img_data_folder)
        if i.endswith(".tif") and i in train_set_lst
    ]
    Train_img_set = np.random.choice(img_lst, int(len(img_lst) * 0.8), replace=False)
    Val_img_set = np.setdiff1d(img_lst, Train_img_set)
    Test_img_set = [i for i in os.listdir(test_data_folder) if i.endswith(".tif")]

    print(
        "There are",
        len(Train_img_set),
        "images for training. And",
        len(Val_img_set),
        "images for validation.",
    )

    # Training
    count = 0
    to_csv_info = []
    save_csv(
        [["Training"]],
        os.path.join(save_folder, "Data_info.csv"),
        mode="a",
    )
    for img_name in Train_img_set:
        print("Processing:", img_name)
        img_path = os.path.join(img_data_folder, img_name)
        seg_path = os.path.join(seg_data_folder, img_name.replace(".tif", ".xml"))

        img = cv2.imread(img_path)
        shutil.copy(
            img_path,
            os.path.join(
                imgTr_save_folder, Dataset_name + "_{:0>4d}_0000.png".format(count)
            ),
        )
        h_component_image = create_h_component_image(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        )
        cv2.imwrite(
            os.path.join(
                imgTr_save_folder, Dataset_name + "_{:0>4d}_0001.png".format(count)
            ),
            cv2.cvtColor(h_component_image, cv2.COLOR_RGB2BGR),
        )
        tree = ET.parse(seg_path)
        root = tree.getroot()

        for k in range(len(root)):
            for child in root[k]:
                pixel_count = []
                save_fig = False
                for x in child:
                    if x.tag == "Attribute":
                        regions = []
                        binary_mask = np.zeros_like(img)
                        vis_mask = np.zeros_like(img)
                        isinstance_vis_mask = np.zeros_like(img)

                    if x.tag == "Region":
                        save_fig = True
                        vertices = x[1]
                        coords = np.zeros((len(vertices), 2))
                        for i, vertex in enumerate(vertices):
                            coords[i][0] = vertex.attrib["X"]
                            coords[i][1] = vertex.attrib["Y"]
                        if len(coords) <= 3:
                            print(
                                "The cell case (ID:{}) in {} has less than 3 pixel being labeled. Skip.".format(
                                    x.attrib["Id"], img_name
                                )
                            )
                            pixel_count.append(len(coords))
                            continue
                        regions.append(coords)
                        ori_num = len(regions)

                if save_fig:
                    if PROCESSING_FLAG in [
                        "Partial Point-level",
                        "Partial Instance-level",
                    ]:
                        number_sample_case = int(len(regions) * REMAIN_RATE)
                        candidate_id = np.random.choice(
                            range(len(regions)),
                            number_sample_case,
                            replace=False,
                        )
                        candidate_lst = []
                        for id in candidate_id:
                            candidate_lst.append(regions[id])
                        regions = candidate_lst
                        del candidate_lst

                    color_map = gen_colormap((len(regions), 3), seed=random_seed)

                    for region, color in zip(regions, color_map):
                        poly = Polygon(region)

                        vertex_row_coords = region[:, 0]
                        vertex_col_coords = region[:, 1]
                        fill_row_coords, fill_col_coords = draw.polygon(
                            vertex_col_coords, vertex_row_coords, binary_mask.shape
                        )
                        pixel_count.append(len(fill_col_coords))
                        if PROCESSING_FLAG in ["Point-level", "Partial Point-level"]:
                            array_idx = np.random.choice(range(len(fill_row_coords)), 1)
                            binary_mask[
                                fill_row_coords[array_idx],
                                fill_col_coords[array_idx],
                            ] = 255
                            cv2.circle(
                                vis_mask,
                                [
                                    int(fill_col_coords[array_idx][0]),
                                    int(fill_row_coords[array_idx][0]),
                                ],
                                4,
                                [0, 255, 255],
                                -1,
                            )
                        else:
                            binary_mask[fill_row_coords, fill_col_coords] = 255
                            vis_mask[fill_row_coords, fill_col_coords] = [0, 255, 255]
                        isinstance_vis_mask[fill_row_coords, fill_col_coords] = color

                    alpha = 0.5
                    beta = 1 - alpha
                    gamma = 0
                    vis_image = cv2.addWeighted(vis_mask, alpha, img, beta, gamma)
                    vis_path = os.path.join(
                        vis_save_folder, Dataset_name + "_{:0>4d}.png".format(count)
                    )
                    cv2.imwrite(vis_path, vis_image)

                    isinstance_vis_image = cv2.addWeighted(
                        isinstance_vis_mask, alpha, img, beta, gamma
                    )
                    isinstance_vis_path = os.path.join(
                        vis_save_folder,
                        Dataset_name + "_{:0>4d}_instance.png".format(count),
                    )
                    cv2.imwrite(isinstance_vis_path, isinstance_vis_image)

                    isinstance_vis_binary_mask = np.zeros_like(isinstance_vis_mask)
                    isinstance_vis_binary_mask[
                        np.sum(isinstance_vis_mask, axis=2) > 0
                    ] = [255, 255, 255]
                    isinstance_vis_path = os.path.join(
                        vis_save_folder,
                        Dataset_name + "_{:0>4d}_binary.png".format(count),
                    )
                    cv2.imwrite(isinstance_vis_path, isinstance_vis_binary_mask)

                    mask_path = os.path.join(
                        segTr_save_folder, Dataset_name + "_{:0>4d}.png".format(count)
                    )
                    cv2.imwrite(mask_path, binary_mask)
                    to_csv_info.append(
                        [
                            img_name,
                            Dataset_name + "_{:0>4d}.png".format(count),
                            ori_num,
                            len(pixel_count),
                            *pixel_count,
                        ]
                    )
        count = count + 1
    save_csv(to_csv_info, os.path.join(save_folder, "Data_info.csv"), mode="a")

    # Validation
    to_csv_info = []
    save_csv(
        [["Validation"]],
        os.path.join(save_folder, "Data_info.csv"),
        mode="a",
    )
    for img_name in Val_img_set:
        print("Processing:", img_name)
        img_path = os.path.join(img_data_folder, img_name)
        seg_path = os.path.join(seg_data_folder, img_name.replace(".tif", ".xml"))

        img = cv2.imread(img_path)
        shutil.copy(
            img_path,
            os.path.join(
                imgVal_save_folder, Dataset_name + "_{:0>4d}_0000.png".format(count)
            ),
        )
        h_component_image = create_h_component_image(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        )
        cv2.imwrite(
            os.path.join(
                imgVal_save_folder, Dataset_name + "_{:0>4d}_0001.png".format(count)
            ),
            cv2.cvtColor(h_component_image, cv2.COLOR_RGB2BGR),
        )
        tree = ET.parse(seg_path)
        root = tree.getroot()

        for k in range(len(root)):
            for child in root[k]:
                pixel_count = []
                save_fig = False
                for x in child:
                    if x.tag == "Attribute":
                        regions = []
                        binary_mask = np.zeros_like(img)
                        vis_mask = np.zeros_like(img)
                        isinstance_vis_mask = np.zeros_like(img)

                    if x.tag == "Region":
                        save_fig = True
                        vertices = x[1]
                        coords = np.zeros((len(vertices), 2))
                        for i, vertex in enumerate(vertices):
                            coords[i][0] = vertex.attrib["X"]
                            coords[i][1] = vertex.attrib["Y"]
                        if len(coords) <= 3:
                            print(
                                "The cell case (ID:{}) in {} has less than 3 pixel being labeled. Skip.".format(
                                    x.attrib["Id"], img_name
                                )
                            )
                            pixel_count.append(len(coords))
                            continue
                        regions.append(coords)
                        ori_num = len(regions)

                if save_fig:
                    color_map = gen_colormap((len(regions), 3), seed=random_seed)

                    for region, color in zip(regions, color_map):
                        poly = Polygon(region)

                        vertex_row_coords = region[:, 0]
                        vertex_col_coords = region[:, 1]
                        fill_row_coords, fill_col_coords = draw.polygon(
                            vertex_col_coords, vertex_row_coords, binary_mask.shape
                        )
                        pixel_count.append(len(fill_col_coords))
                        binary_mask[fill_row_coords, fill_col_coords] = 255
                        vis_mask[fill_row_coords, fill_col_coords] = [0, 255, 255]
                        isinstance_vis_mask[fill_row_coords, fill_col_coords] = color

                    alpha = 0.5
                    beta = 1 - alpha
                    gamma = 0
                    vis_image = cv2.addWeighted(vis_mask, alpha, img, beta, gamma)
                    vis_path = os.path.join(
                        vis_save_folder, Dataset_name + "_{:0>4d}.png".format(count)
                    )
                    cv2.imwrite(vis_path, vis_image)

                    isinstance_vis_image = cv2.addWeighted(
                        isinstance_vis_mask, alpha, img, beta, gamma
                    )
                    isinstance_vis_path = os.path.join(
                        vis_save_folder,
                        Dataset_name + "_{:0>4d}_instance.png".format(count),
                    )
                    cv2.imwrite(isinstance_vis_path, isinstance_vis_image)

                    isinstance_vis_binary_mask = np.zeros_like(isinstance_vis_mask)
                    isinstance_vis_binary_mask[
                        np.sum(isinstance_vis_mask, axis=2) > 0
                    ] = [255, 255, 255]
                    isinstance_vis_path = os.path.join(
                        vis_save_folder,
                        Dataset_name + "_{:0>4d}_binary.png".format(count),
                    )
                    cv2.imwrite(isinstance_vis_path, isinstance_vis_binary_mask)

                    mask_path = os.path.join(
                        segVal_save_folder, Dataset_name + "_{:0>4d}.png".format(count)
                    )
                    cv2.imwrite(mask_path, binary_mask)
                    to_csv_info.append(
                        [
                            img_name,
                            Dataset_name + "_{:0>4d}.png".format(count),
                            ori_num,
                            len(pixel_count),
                            *pixel_count,
                        ]
                    )
        count = count + 1
    save_csv(to_csv_info, os.path.join(save_folder, "Data_info.csv"), mode="a")

    # Testing
    to_csv_info = []
    save_csv(
        [["Testing"]],
        os.path.join(save_folder, "Data_info.csv"),
        mode="a",
    )
    for img_name in Test_img_set:
        print("Processing:", img_name)
        img_path = os.path.join(test_data_folder, img_name)
        seg_path = os.path.join(test_data_folder, img_name.replace(".tif", ".xml"))

        img = cv2.imread(img_path)
        shutil.copy(
            img_path,
            os.path.join(
                imgTs_save_folder, Dataset_name + "_{:0>4d}_0000.png".format(count)
            ),
        )
        h_component_image = create_h_component_image(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        )
        cv2.imwrite(
            os.path.join(
                imgTs_save_folder, Dataset_name + "_{:0>4d}_0001.png".format(count)
            ),
            cv2.cvtColor(h_component_image, cv2.COLOR_RGB2BGR),
        )
        tree = ET.parse(seg_path)
        root = tree.getroot()

        for k in range(len(root)):
            for child in root[k]:
                pixel_count = []
                save_fig = False
                for x in child:
                    if x.tag == "Attribute":
                        regions = []
                        binary_mask = np.zeros_like(img)
                        vis_mask = np.zeros_like(img)
                        isinstance_vis_mask = np.zeros_like(img)

                    if x.tag == "Region":
                        save_fig = True
                        vertices = x[1]
                        coords = np.zeros((len(vertices), 2))
                        for i, vertex in enumerate(vertices):
                            coords[i][0] = vertex.attrib["X"]
                            coords[i][1] = vertex.attrib["Y"]
                        if len(coords) <= 3:
                            print(
                                "The cell case (ID:{}) in {} has less than 3 pixel being labeled. Skip.".format(
                                    x.attrib["Id"], img_name
                                )
                            )
                            pixel_count.append(len(coords))
                            continue
                        regions.append(coords)
                        ori_num = len(regions)

                if save_fig:
                    color_map = gen_colormap((len(regions), 3), seed=random_seed)

                    for region, color in zip(regions, color_map):
                        poly = Polygon(region)

                        vertex_row_coords = region[:, 0]
                        vertex_col_coords = region[:, 1]
                        fill_row_coords, fill_col_coords = draw.polygon(
                            vertex_col_coords, vertex_row_coords, binary_mask.shape
                        )
                        pixel_count.append(len(fill_col_coords))
                        binary_mask[fill_row_coords, fill_col_coords] = 255
                        vis_mask[fill_row_coords, fill_col_coords] = [0, 255, 255]
                        isinstance_vis_mask[fill_row_coords, fill_col_coords] = color
                    alpha = 0.5
                    beta = 1 - alpha
                    gamma = 0
                    vis_image = cv2.addWeighted(vis_mask, alpha, img, beta, gamma)
                    vis_path = os.path.join(
                        vis_save_folder, Dataset_name + "_{:0>4d}.png".format(count)
                    )
                    cv2.imwrite(vis_path, vis_image)

                    isinstance_vis_image = cv2.addWeighted(
                        isinstance_vis_mask, alpha, img, beta, gamma
                    )
                    isinstance_vis_path = os.path.join(
                        vis_save_folder,
                        Dataset_name + "_{:0>4d}_instance.png".format(count),
                    )
                    cv2.imwrite(isinstance_vis_path, isinstance_vis_image)

                    isinstance_vis_binary_mask = np.zeros_like(isinstance_vis_mask)
                    isinstance_vis_binary_mask[
                        np.sum(isinstance_vis_mask, axis=2) > 0
                    ] = [255, 255, 255]
                    isinstance_vis_path = os.path.join(
                        vis_save_folder,
                        Dataset_name + "_{:0>4d}_binary.png".format(count),
                    )
                    cv2.imwrite(isinstance_vis_path, isinstance_vis_binary_mask)

                    mask_path = os.path.join(
                        segTs_save_folder, Dataset_name + "_{:0>4d}.png".format(count)
                    )
                    cv2.imwrite(mask_path, binary_mask)
                    to_csv_info.append(
                        [
                            img_name,
                            Dataset_name + "_{:0>4d}.png".format(count),
                            ori_num,
                            len(pixel_count),
                            *pixel_count,
                        ]
                    )
        count = count + 1
    save_csv(to_csv_info, os.path.join(save_folder, "Data_info.csv"), mode="a")
