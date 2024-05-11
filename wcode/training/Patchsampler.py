import random
import numpy as np


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
        assert (
            0 <= self.oversample_foreground_percent <= 1
        ), "oversample_foreground_percent should be in [0, 1]."
        self.probabilistic_oversampling = probabilistic_oversampling

    def run(self, image_lst, label_lst, oversample_fg_candidate):
        """
        Oversampler processes data from dataloader,
        so the element of image_lst and label_lst is in (c, (z,) y, x)
        """
        batch_size = len(image_lst)
        assert batch_size == len(
            label_lst
        ), "image_lst and label_lst should be in the same length."

        if self.probabilistic_oversampling:
            # Determine whether to oversample based on probability
            image_sampled = []
            label_sampled = []
            for i in range(batch_size):
                if random.random() <= self.oversample_foreground_percent:
                    # do oversample
                    image_patch, label_patch = self._oversample(image_lst[i], label_lst[i], oversample_fg_candidate[i])
                else:
                    image_patch, label_patch = self._normal_sample(image_lst[i], label_lst[i])
                image_sampled.append(image_patch[None])
                label_sampled.append(label_patch[None])
        else:
            num_oversample = round(self.oversample_foreground_percent * batch_size)
            idx_lst = [i for i in range(batch_size)]
            sampled_idx = random.sample(idx_lst, num_oversample)
            remain_idx = np.setdiff1d(idx_lst, sampled_idx)
            image_sampled = []
            label_sampled = []
            for i in range(batch_size):
                if i in sampled_idx:
                    # do oversample
                    image_patch, label_patch = self._oversample(image_lst[i], label_lst[i], oversample_fg_candidate[i])
                elif i in remain_idx:
                    image_patch, label_patch = self._normal_sample(image_lst[i], label_lst[i])
                else:
                    raise Exception("Maybe something wrong.")
                image_sampled.append(image_patch[None])
                label_sampled.append(label_patch[None])

        return np.vstack(image_sampled), np.vstack(label_sampled)

    def _oversample(self, image, label, oversample_fg_candidate):
        # image and label in (c, (z,) y, x)
        shape = np.array(image.shape[1:])
        dim = len(self.half_patch_size)
        assert len(shape) == dim, "shape:{}, dim:{}".format(shape, dim)

        index_range_low, index_range_up = self._get_index_range(dim, shape, True)
        assert np.all(
            index_range_low < index_range_up
        ), "The patch size is too big! Patch Size:{}, Low Index:{}, Up Index:{}".format(
            self.patch_size, index_range_low, index_range_up
        )

        # get the foreground voxels' index
        # fg_index = zip(*np.where(label[0] > 0))
        candidate_fg_index = [
            tuple(fg)
            for fg in oversample_fg_candidate
            if tuple(fg) < tuple(index_range_up) and tuple(fg) > tuple(index_range_low)
        ]
        sampled_voxel = random.sample(candidate_fg_index, 1)

        bbox_lbs = np.array(sampled_voxel)[0] - self.half_patch_size
        bbox_ubs = bbox_lbs + self.patch_size

        # bbmin whether smaller than the lowest index of image
        underflow_flag = [0 > bbox_lbs[i] for i in range(dim)]
        # bbmax whether larger than the highest index of image
        overflow_flag = [shape[i] < bbox_ubs[i] for i in range(dim)]
        # True in it means the direction of this axis need to pad

        if np.any(underflow_flag + overflow_flag):
            valid_bbmin = [max(0, bbox_lbs[i]) for i in range(dim)]
            valid_bbmax = [min(shape[i], bbox_ubs[i]) for i in range(dim)]

            valid_image = self._bbmin_and_bbmax_crop(
                image, dim, valid_bbmin, valid_bbmax
            )
            valid_label = self._bbmin_and_bbmax_crop(
                label, dim, valid_bbmin, valid_bbmax
            )
            # padding_size
            padding = [
                (-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0))
                for i in range(dim)
            ]
            data_per_channel = []
            for i in range(image.shape[0]):
                pad_value = np.percentile(valid_image[i], 0.1)
                data_per_channel.append(
                    np.pad(
                        valid_image[i],
                        padding,
                        "constant",
                        constant_values=pad_value,
                    )[None]
                )
            data = np.vstack(data_per_channel)
            seg = np.pad(valid_label, ((0, 0), *padding), "constant", constant_values=0)
        else:
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
            index_range_low = np.array([0, 0])
            index_range_up = shape
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
