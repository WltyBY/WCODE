import numpy as np

from numpy import number
from typing import Type
from abc import ABC, abstractmethod


class ImageNormalization(ABC):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = None

    def __init__(
        self,
        use_mask_for_norm: bool = None,
        intensityproperties: dict = None,
        target_dtype: Type[number] = np.float32,
    ):
        assert use_mask_for_norm is None or isinstance(use_mask_for_norm, bool)
        self.use_mask_for_norm = use_mask_for_norm

        assert isinstance(intensityproperties, dict)
        self.intensityproperties = intensityproperties
        self.target_dtype = target_dtype

    @abstractmethod
    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        """
        Image and seg must have the same shape. Seg is not always used
        """
        pass


class CTNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        assert (
            self.intensityproperties is not None
        ), "CTNormalization requires intensity properties"
        mean_intensity = self.intensityproperties["mean"]
        std_intensity = self.intensityproperties["std"]
        lower_bound = self.intensityproperties["percentile_00_5"]
        upper_bound = self.intensityproperties["percentile_99_5"]

        image = image.astype(self.target_dtype, copy=False)
        np.clip(image, lower_bound, upper_bound, out=image)
        image -= mean_intensity
        image /= max(std_intensity, 1e-8)
        return image


class ZScoreNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = True

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        """
        here seg is used to store the zero valued region. The value for that region in the segmentation is -1 by
        default.
        """
        image = image.astype(self.target_dtype)
        if self.use_mask_for_norm is not None and self.use_mask_for_norm:
            # negative values in the segmentation encode the 'outside' region (think zero values around the brain as
            # in BraTS). We want to run the normalization only in the brain region, so we need to mask the image.
            # The default nnU-net sets use_mask_for_norm to True if cropping to the nonzero region substantially
            # reduced the image size.
            mask = seg >= 0
            mean = image[mask].mean()
            std = image[mask].std()
            image[mask] = (image[mask] - mean) / (max(std, 1e-8))
        else:
            mean = image.mean()
            std = image.std()
            image = (image - mean) / (max(std, 1e-8))
        return image
    

class GeneralNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        """
        here seg is used to store the zero valued region. The value for that region in the segmentation is -1 by
        default.
        """
        image = image.astype(self.target_dtype)
        mean = image.mean()
        std = image.std()
        image = (image - mean) / (max(std, 1e-8))
        return image
    

class NoNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        return image.astype(self.target_dtype, copy=False)


channel_name_to_normalization_mapping = {
    'CT': CTNormalization,
    'label': NoNormalization
}

normalization_schemes_to_object = {
    CTNormalization.__name__: CTNormalization,
    ZScoreNormalization.__name__: ZScoreNormalization,
    GeneralNormalization.__name__: GeneralNormalization,
    NoNormalization.__name__: NoNormalization,
}


def find_normalizer(scheme):
    if "CT" in scheme:
        scheme = "CT"
    elif any(True if s in scheme else False for s in ["mask", "label", "seg"]):
        scheme = "label"
    norm_scheme = channel_name_to_normalization_mapping.get(scheme)
    if norm_scheme is None:
        norm_scheme = ZScoreNormalization
    # print('Using %s for image normalization' % norm_scheme.__name__)
    return norm_scheme


