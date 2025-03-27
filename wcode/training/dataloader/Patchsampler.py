import random
import numpy as np

from typing import List


class Patchsampler(object):
    def __init__(
        self,
        sample_patch_size,
        final_patch_size,
        oversample_foreground_percent,
        probabilistic_oversampling,
    ):
        self.sample_patch_size = np.array(sample_patch_size)
        self.half_sample_patch_size = self.sample_patch_size // 2
        self.final_patch_size = np.array(final_patch_size)
        self.need_to_pad = (self.sample_patch_size - self.final_patch_size).astype(int)

        self.oversample_foreground_percent = oversample_foreground_percent
        self.probabilistic_oversampling = probabilistic_oversampling

        if isinstance(self.oversample_foreground_percent, (float, int)):
            assert (
                0 <= self.oversample_foreground_percent <= 1
            ), "oversample_foreground_percent should be in [0, 1]."
        elif isinstance(self.oversample_foreground_percent, (list, tuple)):
            assert (
                all([i >= 0 for i in self.oversample_foreground_percent])
                and sum(self.oversample_foreground_percent) == 1
            ), "Each element and their sum should be in range [0, 1] and equal to 1."
        else:
            raise TypeError("Unsupported type for oversample_foreground_percent.")

        if self.probabilistic_oversampling is False:
            assert isinstance(
                self.oversample_foreground_percent, float
            ), "When probabilistic_oversampling is False, we do oversample on batch-level which means we would select (batchsize * oversample_foreground_percent) cases to do oversample and remains do randomly sample."
        elif self.probabilistic_oversampling is True:
            assert isinstance(
                self.oversample_foreground_percent, (list, tuple)
            ), "When probabilistic_oversampling is True, we do oversample on case-level which means we would randomly do oversample based on the given oversample_foreground_percent for each class and each case."
        else:
            raise TypeError(
                "probabilistic_oversampling should be a bool variable. Type:",
                type(self.probabilistic_oversampling),
            )

    def run(
        self,
        image_lst: List,
        label_lst: List,
        oversample_fg_candidate: List,
        batchsize: int,
    ):
        """
        Oversampler processes data from dataloader,
        and the elements of image_lst and label_lst should be in (c, (z,) y, x)
        """
        assert len(image_lst) == len(
            label_lst
        ), "image_lst and label_lst should be in the same length."

        data_all = []
        seg_all = []

        a = batchsize // len(image_lst)
        b = batchsize % len(image_lst)
        idx_lst = [i for i in range(len(image_lst))] * a + [i for i in range(b)]

        if not self.probabilistic_oversampling:
            num_oversample = round(self.oversample_foreground_percent * batchsize)
            sampled_idx = np.random.choice(
                range(len(idx_lst)), num_oversample, replace=False
            )
            oversample_idx = np.array([False for _ in range(len(idx_lst))])
            oversample_idx[sampled_idx] = True
        else:
            oversample_idx = [None for _ in range(len(idx_lst))]

        for i, idx in enumerate(idx_lst):
            data = image_lst[idx]
            seg = label_lst[idx]
            fg_location = oversample_fg_candidate[idx]

            shape = data.shape[1:]
            dim = len(shape)

            oversample_flag, class_location = self._get_class_location(
                fg_location, oversample_idx[i]
            )
            bbmin, bbmax = self._get_centered_point(
                shape, oversample_flag, class_location
            )

            valid_bbmin = np.clip(bbmin, a_min=0, a_max=None)
            valid_bbmax = np.minimum(shape, bbmax)

            data = self._bbmin_and_bbmax_crop(data, dim, valid_bbmin, valid_bbmax)
            seg = self._bbmin_and_bbmax_crop(seg, dim, valid_bbmin, valid_bbmax)
            padding = [
                (-min(0, bbmin[i]), max(bbmax[i] - shape[i], 0)) for i in range(dim)
            ]
            padding = [(0, 0), *padding]
            data_all.append(np.pad(data, padding, "constant", constant_values=0))
            seg_all.append(np.pad(seg, padding, "constant", constant_values=-1))
        return np.stack(data_all), np.stack(seg_all), idx_lst

    def _get_class_location(self, foreground_locations, oversample):
        """
        return: oversample_flag, class_location
        """
        if self.probabilistic_oversampling and oversample is None:
            sampled_class = random.choices(
                [i for i in range(len(self.oversample_foreground_percent))],
                weights=self.oversample_foreground_percent,
                k=1,
            )[0]
            if sampled_class != 0:
                if foreground_locations[sampled_class] == "No_fg_point":
                    return False, None
                elif isinstance(foreground_locations[sampled_class], str):
                    raise ValueError("Why having other string???")
                else:
                    return True, foreground_locations[sampled_class]
            else:
                return False, None
        elif (not self.probabilistic_oversampling) and oversample in [True, False]:
            if oversample:
                candidate = []
                for key in foreground_locations.keys():
                    if key != 0 and not isinstance(foreground_locations[key], str):
                        candidate.append(foreground_locations[key])
                if candidate == []:
                    return False, None
                else:
                    return True, np.vstack(candidate)
            else:
                return False, None
        else:
            raise Exception(
                "Something might be wrong!!! self.probabilistic_oversampling: {}, oversample: {}({})".format(
                    self.probabilistic_oversampling, oversample, type(oversample)
                )
            )

    def _get_centered_point(self, shape, oversample_flag, class_location):
        # shape: spatial shape
        dim = len(shape)
        need_to_pad = self.need_to_pad.copy()

        for d in range(dim):
            # if data.shape + need_to_pad is still < sampled patch size, we need to pad more!
            # We pad on both sides always.
            if need_to_pad[d] + shape[d] < self.sample_patch_size[d]:
                need_to_pad[d] = self.sample_patch_size[d] - shape[d]

        # Get the range of the centered pixel of one patch
        lbs = [(-need_to_pad[i] // 2) for i in range(dim)]
        ubs = [
            shape[i]
            + need_to_pad[i] // 2
            + need_to_pad[i] % 2
            - self.sample_patch_size[i]
            for i in range(dim)
        ]

        if not oversample_flag:
            bbox_lbs = np.array(
                [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
            )
        else:
            # oversample
            assert class_location is not None
            centered_point = class_location[np.random.choice(len(class_location))]
            bbox_lbs = [
                max(lbs[i], centered_point[i] - self.sample_patch_size[i] // 2)
                for i in range(dim)
            ]

        bbox_ubs = [bbox_lbs[i] + self.sample_patch_size[i] for i in range(dim)]

        return bbox_lbs, bbox_ubs

    def _bbmin_and_bbmax_crop(self, data, dim, bbmin, bbmax):
        if dim == 2:
            return data[:, bbmin[0] : bbmax[0], bbmin[1] : bbmax[1]]
        elif dim == 3:
            return data[
                :, bbmin[0] : bbmax[0], bbmin[1] : bbmax[1], bbmin[2] : bbmax[2]
            ]
