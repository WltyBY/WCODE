import random
import numpy as np

from typing import List, Tuple, Union


class Patchsampler(object):
    def __init__(
        self,
        sample_patch_size: List,
        final_patch_size: List,
        oversample_rate: Union[float, List[float], Tuple[float]],
        probabilistic_oversampling: bool,
    ):
        self.sample_patch_size = np.array(sample_patch_size)
        self.half_sample_patch_size = self.sample_patch_size // 2
        self.final_patch_size = np.array(final_patch_size)
        self.need_to_pad = (self.sample_patch_size - self.final_patch_size).astype(int)

        self.oversample_rate = oversample_rate
        self.probabilistic_oversampling = probabilistic_oversampling

        if isinstance(self.oversample_rate, float):
            assert (
                0 <= self.oversample_rate <= 1
            ), "oversample_rate should be in [0, 1]."
        elif isinstance(self.oversample_rate, (list, tuple)):
            assert (
                all([i >= 0 for i in self.oversample_rate])
                and sum(self.oversample_rate) == 1
            ), "Each element should be in range [0, 1] and their sum should be equal to 1."
        else:
            raise TypeError("Unsupported type for oversample_rate.")

        if self.probabilistic_oversampling is False:
            assert isinstance(
                self.oversample_rate, float
            ), "When probabilistic_oversampling is False, we do oversample on case-level which means we would sample location on all available foreground locations, and then do randomly sample."
        elif self.probabilistic_oversampling is True:
            assert isinstance(
                self.oversample_rate, (list, tuple, dict)
            ), "When probabilistic_oversampling is True, we do oversample on case-level which means we would randomly do oversample based on the given oversample_rate for each class."
        else:
            raise TypeError(
                "probabilistic_oversampling should be a bool variable. Type:",
                type(self.probabilistic_oversampling),
            )

    def sample_a_patch(
        self,
        image: np.ndarray,
        label: np.ndarray,
        fg_location: dict,
    ):
        """
        Oversampler processes data for one case,
        and image and label should be in (c, (z,) y, x)
        fg_loacation: dict, key is class_id, value is np.ndarray of shape (n, dim) or "NoPoint"
        """
        # -------- sampling strategy decision --------
        oversample_flag, class_location = self._decide_oversample(fg_location)

        # -------- patch sampling --------
        bbmin, bbmax = self._sample_bbox(
            image.shape[1:], oversample_flag, class_location
        )

        image_patch, label_patch = self._crop_and_pad(image, label, bbmin, bbmax)

        return image_patch, label_patch

    def _decide_oversample(self, fg_location: dict):
        # return: oversample_flag, available class_locations
        if self.probabilistic_oversampling:
            if isinstance(self.oversample_rate, dict):
                class_ids = list(self.oversample_rate.keys())
                probs = list(self.oversample_rate.values())
            else:
                class_ids = [i for i in range(len(self.oversample_rate))]
                probs = self.oversample_rate

            sampled_class = random.choices(class_ids, weights=probs, k=1)[0]

            if sampled_class != 0 and fg_location[sampled_class] != "NoPoint":
                return True, fg_location[sampled_class]
            return False, None
        else:
            if random.random() < self.oversample_rate:
                candidates = []
                for k, v in fg_location.items():
                    if isinstance(v, str):
                        assert v == "NoPoint", "Unsupported string in fg_location."
                        continue
                    if k != 0:
                        candidates.append(v)
                if len(candidates) > 0:
                    return True, np.vstack(candidates)
            return False, None

    def _sample_bbox(self, shape, oversample_flag, class_location):
        dim = len(shape)
        need_to_pad = self.need_to_pad.copy()

        for d in range(dim):
            if need_to_pad[d] + shape[d] < self.sample_patch_size[d]:
                need_to_pad[d] = self.sample_patch_size[d] - shape[d]

        # get the range of the centered pixel of one patch
        lbs = [(-need_to_pad[i] // 2) for i in range(dim)]
        ubs = [
            shape[i]
            + need_to_pad[i] // 2
            + need_to_pad[i] % 2
            - self.sample_patch_size[i]
            for i in range(dim)
        ]

        if not oversample_flag:
            bbmin = np.array(
                [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
            )
        else:
            centered_point = class_location[np.random.choice(len(class_location))]
            bbmin = np.array(
                [
                    max(lbs[i], centered_point[i] - self.sample_patch_size[i] // 2)
                    for i in range(dim)
                ]
            )

        bbmax = bbmin + self.sample_patch_size
        return bbmin, bbmax

    def _crop_and_pad(
        self, image: np.ndarray, label: np.ndarray, bbmin: np.ndarray, bbmax: np.ndarray
    ):
        """
        image in (c, (z,) y, x)
        label in (c, (z,) y, x)
        """
        shape = image.shape[1:]
        dim = len(shape)

        valid_bbmin = np.clip(bbmin, a_min=0, a_max=None)
        valid_bbmax = np.minimum(shape, bbmax)

        if dim == 2:
            image = image[
                :, valid_bbmin[0] : valid_bbmax[0], valid_bbmin[1] : valid_bbmax[1]
            ]
            label = label[
                :, valid_bbmin[0] : valid_bbmax[0], valid_bbmin[1] : valid_bbmax[1]
            ]
        else:
            image = image[
                :,
                valid_bbmin[0] : valid_bbmax[0],
                valid_bbmin[1] : valid_bbmax[1],
                valid_bbmin[2] : valid_bbmax[2],
            ]
            label = label[
                :,
                valid_bbmin[0] : valid_bbmax[0],
                valid_bbmin[1] : valid_bbmax[1],
                valid_bbmin[2] : valid_bbmax[2],
            ]

        padding = [(-min(0, bbmin[i]), max(bbmax[i] - shape[i], 0)) for i in range(dim)]
        padding = [(0, 0), *padding]

        image = np.pad(image, padding, "constant", constant_values=0)
        label = np.pad(label, padding, "constant", constant_values=-1)
        return image, label
