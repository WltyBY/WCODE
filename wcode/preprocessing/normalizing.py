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

        image = image.astype(self.target_dtype, copy=False).clip(
            min=lower_bound, max=upper_bound
        )
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
        image = image.astype(self.target_dtype, copy=False)
        if self.use_mask_for_norm is not None and self.use_mask_for_norm:
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
        mean_intensity = self.intensityproperties["mean"]
        std_intensity = self.intensityproperties["std"]
        
        image = image.astype(self.target_dtype, copy=False)
        image -= mean_intensity
        image /= max(std_intensity, 1e-8)
        return image


class ZeroOneNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        """
        Rescale image to [0, 1]
        """
        image = image.astype(self.target_dtype, copy=False)
        image -= image.min()
        image /= np.clip(image.max, a_min=1e-8, a_max=None)
        return image


class NoNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        return image.astype(self.target_dtype, copy=False)


channel_name_to_normalization_mapping = {
    "CT": CTNormalization,
    "label": NoNormalization,
}

normalization_schemes_to_object = {
    CTNormalization.__name__: CTNormalization,
    ZScoreNormalization.__name__: ZScoreNormalization,
    GeneralNormalization.__name__: GeneralNormalization,
    ZeroOneNormalization.__name__: ZeroOneNormalization,
    NoNormalization.__name__: NoNormalization,
}


def find_normalizer(scheme):
    if "CT" in scheme:
        scheme = "CT"
    elif any(True if s in scheme.lower() else False for s in ["mask", "label", "seg"]):
        scheme = "label"
    norm_scheme = channel_name_to_normalization_mapping.get(scheme)
    if norm_scheme is None:
        norm_scheme = ZScoreNormalization
    # print('Using %s for image normalization' % norm_scheme.__name__)
    return norm_scheme
