import random
import numpy as np

from typing import List


class Patchsampler(object):
    def __init__(
        self,
        patch_size,
        oversample_foreground_percent,
        probabilistic_oversampling,
    ):
        self.patch_size = np.array(patch_size)
        self.half_patch_size = self.patch_size // 2
        self.oversample_foreground_percent = oversample_foreground_percent
        self.probabilistic_oversampling = probabilistic_oversampling

        if isinstance(self.oversample_foreground_percent, float):
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
            raise TypeError("probabilistic_oversampling should be a bool variable.")
        
    def run(
        self,
        image_lst: List,
        label_lst: List,
        oversample_fg_candidate: List,
        batchsize: int,
    ):
        """
        Oversampler processes data from dataloader,
        so the elements of image_lst and label_lst should be in (c, (z,) y, x)
        """
        assert len(image_lst) == len(
            label_lst
        ), "image_lst and label_lst should be in the same length."

        a = batchsize // len(image_lst)
        b = batchsize % len(image_lst)
        idx_lst = [i for i in range(len(image_lst))] * a + [i for i in range(b)]
        idx_lst_passed = idx_lst

        if self.probabilistic_oversampling:
            # Determine whether to oversample based on probability
            image_sampled = []
            label_sampled = []
            image_iter = iter(image_lst)
            label_iter = iter(label_lst)
            OFC_iter = iter(oversample_fg_candidate)
            for _ in range(batchsize):
                sampled_class = random.choices(
                    [i for i in range(len(self.oversample_foreground_percent))],
                    weights=self.oversample_foreground_percent,
                    k=1,
                )[0]
                try:
                    if sampled_class != 0:
                        # do oversample
                        image_patch, label_patch = self._oversample(
                            next(image_iter),
                            next(label_iter),
                            next(OFC_iter)[sampled_class],
                        )
                    else:
                        image_patch, label_patch = self._normal_sample(
                            next(image_iter), next(label_iter)
                        )
                        _ = next(OFC_iter)
                    # add one more dim: batchsize
                    image_sampled.append(image_patch[None])
                    label_sampled.append(label_patch[None])
                except StopIteration:
                    image_iter = iter(image_lst)
                    label_iter = iter(label_lst)
                    OFC_iter = iter(oversample_fg_candidate)
                    if sampled_class != 0:
                        # do oversample
                        image_patch, label_patch = self._oversample(
                            next(image_iter),
                            next(label_iter),
                            next(OFC_iter)[sampled_class],
                        )
                    else:
                        image_patch, label_patch = self._normal_sample(
                            next(image_iter), next(label_iter)
                        )
                        _ = next(OFC_iter)
                    # add one more dim: batchsize
                    image_sampled.append(image_patch[None])
                    label_sampled.append(label_patch[None])
        else:
            num_oversample = round(self.oversample_foreground_percent * batchsize)

            # len(image_lst) <= batchsize
            sampled_idx = np.random.choice(
                range(len(idx_lst)), num_oversample, replace=False
            )
            oversample_idx = np.array([False for _ in range(len(idx_lst))])
            oversample_idx[sampled_idx] = True

            image_sampled = []
            label_sampled = []
            idx_lst = iter(idx_lst)
            oversample_idx = iter(oversample_idx)
            for _ in range(batchsize):
                try:
                    idx = next(idx_lst)
                    oversample_flag = next(oversample_idx)
                    if oversample_flag:
                        # do oversample
                        candidate = []
                        for i in oversample_fg_candidate[idx].keys():
                            if (
                                i != 0
                                and not isinstance(oversample_fg_candidate[idx][i], str)
                            ):
                                candidate.append(oversample_fg_candidate[idx][i])
                        candidate = np.vstack(candidate)

                        image_patch, label_patch = self._oversample(
                            image_lst[idx],
                            label_lst[idx],
                            candidate,
                        )
                    else:
                        image_patch, label_patch = self._normal_sample(
                            image_lst[idx], label_lst[idx]
                        )
                    image_sampled.append(image_patch[None])
                    label_sampled.append(label_patch[None])
                except StopIteration:
                    idx_lst = iter(idx_lst)
                    oversample_idx = iter(oversample_idx)

                    idx = next(idx_lst)
                    oversample_flag = next(oversample_idx)
                    if oversample_flag:
                        # do oversample
                        candidate = []
                        for i in oversample_fg_candidate[idx].keys():
                            if (
                                i != 0
                                and not isinstance(oversample_fg_candidate[idx][i], str)
                            ):
                                candidate.append(oversample_fg_candidate[idx][i])
                        candidate = np.vstack(candidate)

                        image_patch, label_patch = self._oversample(
                            image_lst[idx],
                            label_lst[idx],
                            candidate,
                        )
                    else:
                        image_patch, label_patch = self._normal_sample(
                            image_lst[idx], label_lst[idx]
                        )
                    image_sampled.append(image_patch[None])
                    label_sampled.append(label_patch[None])

        return np.vstack(image_sampled), np.vstack(label_sampled), idx_lst_passed

    def _oversample(self, image, label, oversample_fg_candidate):
        # image and label in (c, (z,) y, x)
        shape = np.array(image.shape[1:])
        dim = len(self.half_patch_size)
        assert len(shape) == dim, "shape:{}, dim:{}".format(shape, dim)
        if (
            isinstance(oversample_fg_candidate, str)
            and oversample_fg_candidate == "No_fg_point"
        ):
            # case with all pixels are the background.
            index_range_low, index_range_up = self._get_index_range(dim, shape, False)
        else:
            index_range_low, index_range_up = self._get_index_range(dim, shape, True)
        assert np.all(
            index_range_low < index_range_up
        ), "The patch size is too big! Patch Size:{}, Low Index:{}, Up Index:{}".format(
            self.patch_size, index_range_low, index_range_up
        )

        # get the foreground voxels' index
        # fg_index = zip(*np.where(label[0] > 0))
        if (
            isinstance(oversample_fg_candidate, str)
            and oversample_fg_candidate == "No_fg_point"
        ):
            # case with all pixels are the background.
            sampled_voxel = [
                np.random.randint(index_range_low[i], index_range_up[i])
                for i in range(len(index_range_low))
            ]
        else:
            candidate_fg_index = [
                tuple(fg)
                for fg in oversample_fg_candidate
                if tuple(fg) < tuple(index_range_up)
                and tuple(fg) > tuple(index_range_low)
            ]
            sampled_voxel = random.sample(candidate_fg_index, 1)

        bbox_lbs = np.array(sampled_voxel)[0] - self.half_patch_size
        bbox_ubs = bbox_lbs + self.patch_size

        # bbmin whether smaller than the lowest index of image
        underflow_flag = [0 > bbox_lbs[i] for i in range(dim)]
        # bbmax whether larger than the highest index of image
        overflow_flag = [shape[i] < bbox_ubs[i] for i in range(dim)]
        # True in it means the direction of this axis need to pad

        if np.any(underflow_flag):
            # refine bbmin
            for i in range(len(underflow_flag)):
                if underflow_flag[i]:
                    bbox_ubs[i] = bbox_ubs[i] - bbox_lbs[i]
                    bbox_lbs[i] = 0

        if np.any(overflow_flag):
            # refine bbmax
            for i in range(len(overflow_flag)):
                if overflow_flag[i]:
                    bbox_lbs[i] = bbox_lbs[i] - (bbox_ubs[i] - shape[i])
                    bbox_ubs[i] = shape[i]
        
        data = self._bbmin_and_bbmax_crop(image, dim, bbox_lbs, bbox_ubs)
        seg = self._bbmin_and_bbmax_crop(label, dim, bbox_lbs, bbox_ubs)

        return data, seg

    def _normal_sample(self, image, label):
        shape = np.array(image.shape[1:])
        dim = len(self.half_patch_size)
        assert len(shape) == dim
        index_range_low, index_range_up = self._get_index_range(dim, shape, False)

        sampled_voxel = [
            np.random.randint(index_range_low[i], index_range_up[i])
            for i in range(len(index_range_low))
        ]

        bbox_lbs = np.array(sampled_voxel) - self.half_patch_size
        bbox_ubs = bbox_lbs + self.patch_size
        data = self._bbmin_and_bbmax_crop(image, dim, bbox_lbs, bbox_ubs)
        seg = self._bbmin_and_bbmax_crop(label, dim, bbox_lbs, bbox_ubs)

        return data, seg

    def _get_index_range(self, dim, shape, oversample_flag):
        if dim == 2:
            index_range_low = self.half_patch_size
            index_range_up = shape - self.patch_size + self.half_patch_size
        elif dim == 3:
            if oversample_flag:
                # oversample, only limit Z-axis range here
                index_range_low = np.array([self.half_patch_size[0], 0, 0])
                index_range_up = np.array(
                    [
                        shape[0] - self.patch_size[0] + self.half_patch_size[0],
                        shape[1],
                        shape[2],
                    ]
                )
            else:
                # do normal sample, limit every axis range
                index_range_low = np.array(
                    [self.half_patch_size[i] for i in range(dim)]
                )
                index_range_up = np.array(
                    [
                        shape[i] - self.patch_size[i] + self.half_patch_size[i]
                        for i in range(dim)
                    ]
                )
        else:
            raise Exception("Oversampler can only process 2d and 3d data.")

        return index_range_low, index_range_up

    def _bbmin_and_bbmax_crop(self, data, dim, bbmin, bbmax):
        if dim == 2:
            return data[:, bbmin[0] : bbmax[0], bbmin[1] : bbmax[1]]
        elif dim == 3:
            return data[
                :, bbmin[0] : bbmax[0], bbmin[1] : bbmax[1], bbmin[2] : bbmax[2]
            ]
