import os
import cv2
import torch
import shutil
import numpy as np

from typing import Union

from wcode.net.Vision_Transformer.SAM.build_sam import sam_model_registry
from wcode.net.Vision_Transformer.SAM.predictor import SamPredictor
from wcode.net.Vision_Transformer.SAM.automatic_mask_generator import (
    SamAutomaticMaskGenerator,
)
from wcode.net.Vision_Transformer.SAM.model.Sam import Sam

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# vit_h, vit_l, vit_b
MODEL_REIGIST = "vit_b"
model_weight_dict = {
    "vit_b": "./Dataset/SAM/sam_vit_b_01ec64.pth",
    "vit_l": "./Dataset/SAM/sam_vit_l_0b3195.pth",
    "vit_h": "./Dataset/SAM/sam_vit_h_4b8939.pth",
}
MODEL_WEIGHT = model_weight_dict[MODEL_REIGIST]

# Sam, SamPredictor, SamAutomaticMaskGenerator
PREDICTOR = "Sam"

# normal, sliding
PREDICT_WAY = "normal"
STEP = 1
PATCHSIZE = [250, 250]
LOW_GPU_VRAM = True

# Point-level, Partial Point-level0.5, Partial Point-level0.2
dataset_class = "Partial Point-level0.2"
SPLIT = "Tr"
assert SPLIT != "Ts"
WSI_FILE = "./Dataset/MoNuSeg{}/images".format(dataset_class) + SPLIT
WSI_PROMPT_FILE = "./Dataset/MoNuSeg{}/labels".format(dataset_class) + SPLIT
GT_FILE = "./Dataset/MoNuSegFully/labels" + SPLIT
SAVE_PATH = os.path.join(
    "./Predictions",
    "MoNuSeg",
    dataset_class,
    PREDICTOR,
    PREDICT_WAY + "_" + MODEL_REIGIST + "_" + SPLIT,
)
if os.path.isdir(SAVE_PATH):
    shutil.rmtree(SAVE_PATH, ignore_errors=True)
os.makedirs(SAVE_PATH, exist_ok=True)


def masked_prompt_to_SAM_prompt(masked_prompt_NDarray):
    fg_idx = np.where(masked_prompt_NDarray[0] == 255)
    if len(fg_idx[0]) > 0:
        point_prompt = []
        for h, w in zip(fg_idx[0], fg_idx[1]):
            point_prompt.append(np.array([h, w]))
        prompt = np.vstack(point_prompt)[None]
        prompt_label = np.ones([1, prompt.shape[1]])
    elif len(fg_idx[0]) == 0:
        prompt = None
        prompt_label = None
    else:
        raise RuntimeError("Why len(fg_idx) can smaller than 0 or something else?")

    # 1, N, 2 and 1, N
    return prompt, prompt_label


def generate_sliding_window(WSI: np.ndarray, prompt_masked: Union[np.ndarray, None]):
    step_length = np.array(PATCHSIZE) * STEP
    shape = WSI.shape[1:]
    mesh = np.meshgrid(
        *[np.arange(0, shape[i], step_length[i]) for i in range(len(shape))]
    )
    mesh = [mesh[i].flatten() for i in range(len(shape))]
    output_lst = []
    for i in range(len(mesh[0])):
        bbmin = np.array([ori[i] for ori in mesh])
        bbmax = bbmin + np.array(PATCHSIZE)
        img = WSI[:, bbmin[0] : bbmax[0], bbmin[1] : bbmax[1]]
        if prompt_masked is not None:
            prompt, prompt_label = masked_prompt_to_SAM_prompt(
                prompt_masked[:, bbmin[0] : bbmax[0], bbmin[1] : bbmax[1]]
            )
            output_lst.append(
                {
                    "image": torch.tensor(img),
                    "original_size": img.shape[1:],
                    "point_coords": (
                        torch.tensor(prompt) if prompt is not None else None
                    ),
                    "point_labels": (
                        torch.tensor(prompt_label) if prompt is not None else None
                    ),
                }
            )
        else:
            output_lst.append(
                {
                    "image": torch.tensor(img),
                    "original_size": img.shape[1:],
                    "point_coords": None,
                    "point_labels": None,
                }
            )

    return output_lst, mesh


def run_a_WSI_Sam(
    sam_model: Sam,
    WSI: np.ndarray,
    prompt_masked: np.ndarray,
    predict_way=PREDICT_WAY,
):
    WSI = WSI.transpose(2, 0, 1)
    prompt_masked = prompt_masked.transpose(2, 0, 1)
    if predict_way == "normal":
        prompt, prompt_label = masked_prompt_to_SAM_prompt(prompt_masked)
        data = [
            {
                "image": torch.tensor(WSI),
                "original_size": WSI.shape[1:],
                "point_coords": torch.tensor(prompt),
                "point_labels": torch.tensor(prompt_label),
            }
        ]
        with torch.no_grad():
            outputs = sam_model(data, False)
        output = outputs[0]
        predict = (
            (output["masks"][0].repeat(3, 1, 1).int() * 255)
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
        )
    elif predict_way == "sliding":
        data, mesh = generate_sliding_window(WSI, prompt_masked)
        predict = np.zeros_like(WSI)
        if LOW_GPU_VRAM:
            pred_lst = []
            for input_dict in data:
                if input_dict["point_coords"] is None:
                    del input_dict["point_coords"]
                if input_dict["point_labels"] is None:
                    del input_dict["point_labels"]
                with torch.no_grad():
                    outputs = sam_model([input_dict], False)
                output = outputs[0]
                pred_lst.append(
                    (output["masks"][0].repeat(3, 1, 1).int() * 255).cpu().numpy()
                )
            assert len(mesh[0]) == len(pred_lst)
            # print(mesh[0])
            for idx, pred in enumerate(pred_lst):
                predict[
                    :,
                    mesh[0][idx] : mesh[0][idx] + PATCHSIZE[0],
                    mesh[1][idx] : mesh[1][idx] + PATCHSIZE[1],
                ] = pred
        else:
            for idx in range(len(data)):
                if data[idx]["point_coords"] is None:
                    del data[idx]["point_coords"]
                if data[idx]["point_labels"] is None:
                    del data[idx]["point_labels"]

            with torch.no_grad():
                outputs = sam_model(data, False)
            assert len(mesh[0]) == len(outputs)
            for idx, output in enumerate(outputs):
                pred = (output["masks"][0].repeat(3, 1, 1).int() * 255).cpu().numpy()
                predict[
                    :,
                    mesh[0][idx] : mesh[0][idx] + PATCHSIZE[0],
                    mesh[1][idx] : mesh[1][idx] + PATCHSIZE[1],
                ] = pred
        predict = predict.transpose(1, 2, 0)
    else:
        raise Exception(predict_way, "is not supported.")

    return predict


def run_a_WSI_Sam_Predictor(
    sam_predictor: SamPredictor,
    WSI: np.ndarray,
    prompt_masked: np.ndarray,
    predict_way=PREDICT_WAY,
):
    WSI = WSI.transpose(2, 0, 1)
    prompt_masked = prompt_masked.transpose(2, 0, 1)
    if predict_way == "normal":
        prompt, prompt_label = masked_prompt_to_SAM_prompt(prompt_masked)
        with torch.no_grad():
            sam_predictor.set_image(WSI.transpose(1, 2, 0))
            output, _, _ = sam_predictor.predict(
                point_coords=prompt[0],
                point_labels=prompt_label[0],
                multimask_output=False,
            )
        predict = (np.tile(output, (3, 1, 1)).astype(np.uint8) * 255).transpose(1, 2, 0)
    elif predict_way == "sliding":
        data, mesh = generate_sliding_window(WSI, prompt_masked)
        predict = np.zeros_like(WSI)
        pred_lst = []
        for input_dict in data:
            with torch.no_grad():
                sam_predictor.set_image(input_dict["image"].numpy().transpose(1, 2, 0))
                output, _, _ = sam_predictor.predict(
                    point_coords=(
                        input_dict["point_coords"][0].numpy()
                        if input_dict["point_coords"] is not None
                        else None
                    ),
                    point_labels=(
                        input_dict["point_labels"][0].numpy()
                        if input_dict["point_coords"] is not None
                        else None
                    ),
                    multimask_output=False,
                )
            pred_lst.append(np.tile(output, (3, 1, 1)).astype(np.uint8) * 255)
        assert len(mesh[0]) == len(pred_lst)
        # print(mesh[0])
        for idx, pred in enumerate(pred_lst):
            predict[
                :,
                mesh[0][idx] : mesh[0][idx] + PATCHSIZE[0],
                mesh[1][idx] : mesh[1][idx] + PATCHSIZE[1],
            ] = pred
        predict = predict.transpose(1, 2, 0)
    else:
        raise Exception(predict_way, "is not supported.")

    return predict


def gen_colormap(shape, seed=319):
    np.random.seed(seed)
    color_val = np.random.randint(0, 255, shape)
    return color_val


def get_anns(WSI, anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)

    seg_mask = np.zeros((*WSI.shape[:2], 1))
    vis_color_mask = np.zeros_like(WSI)
    color_map = gen_colormap((len(sorted_anns), 3))
    bg_flag = True
    for ann, color in zip(sorted_anns, color_map):
        m = ann["segmentation"]
        if bg_flag:
            bg_flag = False
        else:
            seg_mask[m] = 1
            vis_color_mask[m] = color
    alpha = 0.5
    beta = 1 - alpha
    gamma = 0

    return seg_mask, cv2.addWeighted(vis_color_mask, alpha, WSI, beta, gamma)


def run_a_WSI_Sam_AutomaticMaskGenerator(
    Sam_automaticMaskGenerator: SamAutomaticMaskGenerator,
    WSI: np.ndarray,
    prompt_masked: np.ndarray = None,
    predict_way=PREDICT_WAY,
):
    WSI = WSI.transpose(2, 0, 1)
    prompt_masked = None
    if predict_way == "normal":
        with torch.no_grad():
            curr_anns = Sam_automaticMaskGenerator.generate(WSI.transpose(1, 2, 0))
        predict, vis_img = get_anns(WSI.transpose(1, 2, 0), curr_anns)
        # predict = (np.tile(output, (3, 1, 1)).astype(np.uint8) * 255).transpose(1, 2, 0)
    elif predict_way == "sliding":
        data, mesh = generate_sliding_window(WSI, prompt_masked)
        predict = np.zeros_like(WSI.transpose(1, 2, 0))
        vis_img = np.zeros_like(WSI.transpose(1, 2, 0))
        vis_lst = []
        pred_lst = []
        for input_dict in data:
            with torch.no_grad():
                curr_anns = Sam_automaticMaskGenerator.generate(
                    input_dict["image"].numpy().transpose(1, 2, 0)
                )
            pred, vis = get_anns(
                input_dict["image"].numpy().transpose(1, 2, 0), curr_anns
            )
            pred_lst.append(pred)
            vis_lst.append(vis)
        assert len(mesh[0]) == len(pred_lst) == len(vis_lst)

        for x, y, pred, vis in zip(mesh[0], mesh[1], pred_lst, vis_lst):
            predict[
                x : x + PATCHSIZE[0],
                y : y + PATCHSIZE[1],
                :,
            ] = pred
            vis_img[
                x : x + PATCHSIZE[0],
                y : y + PATCHSIZE[1],
                :,
            ] = vis
    else:
        raise Exception(predict_way, "is not supported.")
    predict = predict.astype(np.uint8) * 255
    return predict, vis_img


def evaluate(prediction_folder, ground_truth_folder):
    file_lst = os.listdir(prediction_folder)
    tp = []
    dsc = []
    for file_name in file_lst:
        pred_obj = cv2.imread(os.path.join(prediction_folder, file_name)).transpose(
            2, 0, 1
        )
        gt_odj = cv2.imread(os.path.join(ground_truth_folder, file_name)).transpose(
            2, 0, 1
        )
        pred_array = pred_obj[0] / 255
        gt_array = gt_odj[0] / 255

        tp = (gt_array * pred_array).sum()
        fp = ((1 - gt_array) * pred_array).sum()
        fn = (gt_array * (1 - pred_array)).sum()
        dsc.append((2 * tp + 1e-5) / (2 * tp + fp + fn + 1e-5))
    print("DSC: {}+{}".format(np.mean(dsc), np.std(dsc)))


if __name__ == "__main__":
    WSI_lst = [i for i in os.listdir(WSI_FILE) if i.endswith("_0000.png")]
    prompt_lst = os.listdir(WSI_PROMPT_FILE)

    if PREDICTOR == "Sam":
        SAM = sam_model_registry[MODEL_REIGIST](checkpoint=MODEL_WEIGHT).to(device)
        Runner = run_a_WSI_Sam
    elif PREDICTOR == "SamPredictor":
        SAM = SamPredictor(sam_model_registry[MODEL_REIGIST](checkpoint=MODEL_WEIGHT).to(device))
        Runner = run_a_WSI_Sam_Predictor
    elif PREDICTOR == "SamAutomaticMaskGenerator":
        vis_folder = SAVE_PATH + "_vis"
        if os.path.isdir(vis_folder):
            shutil.rmtree(vis_folder, ignore_errors=True)
        os.makedirs(vis_folder, exist_ok=True)
        SAM = SamAutomaticMaskGenerator(
            sam_model_registry[MODEL_REIGIST](checkpoint=MODEL_WEIGHT).to(device)
        )
        Runner = run_a_WSI_Sam_AutomaticMaskGenerator
    else:
        raise Exception("Not Supported PREDICTOR!!!")

    for img_path, prompt_path in zip(WSI_lst, prompt_lst):
        case_name = "MoNuSegFully_" + img_path.split("_")[1]
        print("Predict:", case_name)
        # C, H, W
        img_path = os.path.join(WSI_FILE, img_path)
        prompt_path = os.path.join(WSI_PROMPT_FILE, prompt_path)
        img_array = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        prompt_array = cv2.imread(prompt_path)

        if PREDICTOR in ["Sam", "SamPredictor"]:
            prediction = Runner(SAM, img_array, prompt_array)
            print("Save to:", os.path.join(SAVE_PATH, case_name + ".png"))
            cv2.imwrite(
                os.path.join(SAVE_PATH, case_name + ".png"),
                (prediction).astype(np.uint8),
            )
        elif PREDICTOR == "SamAutomaticMaskGenerator":
            prediction, vis_img = Runner(SAM, img_array, prompt_array)
            print("Save to:", os.path.join(SAVE_PATH, case_name + ".png"))
            cv2.imwrite(
                os.path.join(SAVE_PATH, case_name + ".png"),
                (prediction).astype(np.uint8),
            )
            print("Save to:", os.path.join(vis_folder, case_name + ".png"))
            cv2.imwrite(
                os.path.join(vis_folder, case_name + ".png"),
                (vis_img).astype(np.uint8),
            )

    evaluate(SAVE_PATH, GT_FILE)
