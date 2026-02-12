import torch
import numpy as np

from typing import Union, List, Tuple

from wcode.preprocessing.resampling import resample_ND_data_to_given_shape
from wcode.utils.file_operations import open_yaml, save_pickle, save_itk


def apply_inference_nonlin(logits, nonlinear_func="softmax"):
    """
    logits has to have shape (c, (z,) y, x) where c is the number of classes/regions
    """
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)

    with torch.no_grad():
        # softmax etc is not implemented for half
        logits = logits.float()

        if nonlinear_func.lower() == "sigmoid":
            probabilities = torch.sigmoid(logits)
        elif nonlinear_func.lower() == "softmax":
            probabilities = torch.softmax(logits, dim=0)
        else:
            raise Exception("sigmoid or softmax")

    return probabilities


def convert_probabilities_to_segmentation(predicted_probabilities):
    """
    assumes that inference_nonlinearity was already applied!

    predicted_probabilities has to have shape (c, (z,) y, x) where c is the number of classes/regions
    """
    if not isinstance(predicted_probabilities, (np.ndarray, torch.Tensor)):
        raise RuntimeError(
            f"Unexpected input type. Expected np.ndarray or torch.Tensor,"
            f" got {type(predicted_probabilities)}"
        )

    segmentation = predicted_probabilities.argmax(0)

    return segmentation


def bounding_box_to_slice(bbox):
    # bbox is in [bbmin, bbmax], and bbmin and bbmax are all list object.
    return tuple(slice(*i) for i in zip(bbox[0], bbox[1]))


def revert_cropping_on_probabilities(
    predicted_probabilities: Union[torch.Tensor, np.ndarray],
    bbox: List[List[int]],
    original_shape: Union[List[int], Tuple[int, ...]],
):
    # revert cropping
    probs_reverted_cropping = (
        np.zeros(
            (predicted_probabilities.shape[0], *original_shape),
            dtype=predicted_probabilities.dtype,
        )
        if isinstance(predicted_probabilities, np.ndarray)
        else torch.zeros(
            (predicted_probabilities.shape[0], *original_shape),
            dtype=predicted_probabilities.dtype,
        )
    )
    slicer = bounding_box_to_slice(bbox)
    probs_reverted_cropping[tuple([slice(None)] + list(slicer))] = (
        predicted_probabilities
    )

    return probs_reverted_cropping


def convert_predicted_logits_to_segmentation_with_correct_shape(
    predicted_logits: Union[torch.Tensor, np.ndarray],
    properties_dict: dict,
    return_probabilities: bool = False,
    num_threads_torch: int = 8,
) -> np.ndarray:
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    # resample to original shape (revert resampling)
    predicted_logits = (
        predicted_logits.type(torch.float32)
        if isinstance(predicted_logits, torch.Tensor)
        else predicted_logits.astype("float32")
    )
    predicted_logits_lst = []
    for i in range(predicted_logits.shape[0]):
        predicted_logits_lst.append(
            resample_ND_data_to_given_shape(
                predicted_logits[i],
                properties_dict["shape_after_cropping_and_before_resampling"],
                current_spacing=properties_dict["target_spacing"],
                is_seg=False,
            )[None]
        )
    predicted_logits = torch.from_numpy(np.vstack(predicted_logits_lst)).float()
    # return value of resampling_fn_probabilities can be ndarray or Tensor but that doesnt matter because
    # apply_inference_nonlin will covnert to torch
    predicted_probabilities = apply_inference_nonlin(predicted_logits)
    del predicted_logits
    segmentation = convert_probabilities_to_segmentation(predicted_probabilities)

    # segmentation may be torch.Tensor but we continue with numpy
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()

    # put segmentation in bbox (revert cropping)
    # properties_dict["shape_before_cropping"] is in [(z,) y, x]
    segmentation_reverted_cropping = np.zeros(
        properties_dict["shape_before_cropping"],
        dtype=np.uint8,
    )
    # properties_dict["bbox_used_for_cropping"] is in [bbmin, bbmax],
    # and bbmin and bbmax are all list object
    slicer = bounding_box_to_slice(properties_dict["bbox_used_for_cropping"])
    segmentation_reverted_cropping[slicer] = segmentation
    del segmentation

    if return_probabilities:
        # revert cropping
        predicted_probabilities = revert_cropping_on_probabilities(
            predicted_probabilities,
            properties_dict["bbox_used_for_cropping"],
            properties_dict["shape_before_cropping"],
        )
        predicted_probabilities = predicted_probabilities.cpu().numpy()
        torch.set_num_threads(old_threads)
        return segmentation_reverted_cropping, predicted_probabilities
    else:
        torch.set_num_threads(old_threads)
        return segmentation_reverted_cropping


def export_prediction_from_logits(
    predicted_logits: Union[torch.Tensor, np.ndarray],
    properties_dict: dict,
    ofile: str,
    dataset_yaml_dict_or_file_path: Union[dict, str],
    return_probabilities: bool = False,
    num_threads_torch: int = 8,
):
    if isinstance(dataset_yaml_dict_or_file_path, str):
        dataset_yaml_dict_or_file_path = open_yaml(dataset_yaml_dict_or_file_path)

    ret = convert_predicted_logits_to_segmentation_with_correct_shape(
        predicted_logits,
        properties_dict,
        return_probabilities,
        num_threads_torch,
    )
    del predicted_logits

    # save
    if return_probabilities:
        segmentation_final, probabilities_final = ret
        np.savez_compressed(ofile + ".npz", probabilities=probabilities_final)
        save_pickle(properties_dict, ofile + ".pkl")
        del probabilities_final, ret
    else:
        segmentation_final = ret
        del ret

    save_itk(
        segmentation_final,
        properties_dict,
        ofile + dataset_yaml_dict_or_file_path["files_ending"],
    )


def export_original_logits(
    predicted_logits: Union[torch.Tensor, np.ndarray],
    properties_dict: dict,
    ofile: str,
    dataset_yaml_dict_or_file_path: Union[dict, str],
    num_threads_torch: int = 8,
):
    if isinstance(dataset_yaml_dict_or_file_path, str):
        dataset_yaml_dict_or_file_path = open_yaml(dataset_yaml_dict_or_file_path)

    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    # resample to original shape (revert resampling)
    predicted_logits = (
        predicted_logits.type(torch.float32)
        if isinstance(predicted_logits, torch.Tensor)
        else predicted_logits.astype("float32")
    )
    predicted_logits_lst = []
    for i in range(predicted_logits.shape[0]):
        predicted_logits_lst.append(
            resample_ND_data_to_given_shape(
                predicted_logits[i],
                properties_dict["shape_after_cropping_and_before_resampling"],
                current_spacing=properties_dict["target_spacing"],
                is_seg=False,
            )[None]
        )
    predicted_logits = torch.from_numpy(np.vstack(predicted_logits_lst)).float()

    if isinstance(predicted_logits, torch.Tensor):
        predicted_logits = predicted_logits.cpu().numpy()

    # put predicted_logits in bbox (revert cropping)
    # properties_dict["shape_before_cropping"] is in [(z,) y, x]
    predicted_logits_reverted_cropping = np.zeros(
        properties_dict["shape_before_cropping"],
        dtype=np.float32,
    )
    # properties_dict["bbox_used_for_cropping"] is in [bbmin, bbmax],
    # and bbmin and bbmax are all list object
    slicer = bounding_box_to_slice(properties_dict["bbox_used_for_cropping"])
    predicted_logits_reverted_cropping[slicer] = predicted_logits
    del predicted_logits

    torch.set_num_threads(old_threads)

    # save
    save_itk(
        predicted_logits_reverted_cropping,
        properties_dict,
        ofile + dataset_yaml_dict_or_file_path["files_ending"],
    )
