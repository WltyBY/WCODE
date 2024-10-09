import time
import torch
import numpy as np
import torch.nn.functional as F

from wcode.training.Patchsampler import Patchsampler


class PatchBasedCollater(object):
    """
    This is for PatchBaseTrainer to do patch sample and transform in dataloader
    """

    def __init__(
        self,
        batchsize,
        patch_size,
        do_deep_supervision,
        pool_kernel_size_lst,
        modality,
        oversample_foreground_percent: float,
        probabilistic_oversampling: bool,
        transform,
    ):
        self.batchsize = batchsize
        self.do_deep_supervision = do_deep_supervision
        self.pool_kernel_size_lst = pool_kernel_size_lst
        # modality is the name of channels
        self.modality = modality
        self.transform = transform
        self.Patchsampler = Patchsampler(
            patch_size, oversample_foreground_percent, probabilistic_oversampling
        )

    def __call__(self, data):
        """
        Dataloader will give a list object to collate_fn method. Each element in it is the return of the __getitem__
        method in your dataset.
        image and label is (c, z, y, x), so it is needed to add one more channel to make the shape is (b, c, z, y, x)
        """
        # We manually pass in batchsize separately, because if the number of batchsize larger than the size of dataset,
        # the length of data here is smaller than the batchsize we want.
        # You may meet this situation when we do a patch-based training on Whole Slide Image (WSI). There are 30 WSIs, for
        # example, we set batchsize to 32. So in Patchsampler, we need to sample two more patches.
        output_dict = {key: [] for key in data[0].keys()}

        image_lst = []
        label_lst = []
        oversample_lst = []
        for i in range(len(data)):
            image_lst.append(data[i]["image"])
            label_lst.append(data[i]["label"])
            oversample_lst.append(data[i]["property"]["oversample"])

        # For natural images
        if data[0]["property"].__contains__("shapes"):
            channels = [data[i]["property"]["shapes"][0] for i in range(len(data))]
        else:
            channels = None

        # sample patch
        # image: (b, c, patchsize), label: (b, 1, patchsize)
        # start = time.time()
        image, label, idx_lst = self.Patchsampler.run(
            image_lst, label_lst, oversample_lst, self.batchsize
        )
        # print("Patch Sample: {}s".format(time.time() - start))

        # start = time.time()
        # do transform
        find_seg = np.array(
            [True if s in self.modality else False for s in ["mask", "label", "seg"]]
        )
        if any(find_seg):
            if channels:
                # natural image, seg_idx above is not the actual seg_idx for it.
                seg_idx = []
                count = 0
                # if is_seg, the para num_channel is not used, because we set the channel of seg to 1 during preprocess.
                # So num_channel is fake for seg here.
                for is_seg, num_channel in zip(find_seg, channels):
                    if is_seg:
                        seg_idx += [count]
                        count += 1
                    else:
                        count += num_channel
            else:
                # medical image
                seg_idx = np.where(find_seg)[0]
            segs_out = image[:, seg_idx]
            image = np.delete(image, seg_idx, axis=1)
            label = np.stack([label, segs_out], axis=1)

        # in monai, most of the pre-/post-processing transforms expect: (num_channels, spatial_dim_1[, spatial_dim_2, ...]),
        for i, idx in zip(list(range(image.shape[0])), idx_lst):
            img_count = 1
            seg_count = 1
            # the input shape for augmetation is c, (z,) y, x
            sample_data = {"image": image[i], "label": label[i]}
            if self.transform is not None:
                sample_data = self.transform(sample_data)

            if any(find_seg):
                image_add_input_seg = []
                for j, is_seg in enumerate(find_seg):
                    if is_seg:
                        seg_count_now = seg_count + 1
                        image_add_input_seg.append(
                            sample_data["label"][seg_count:seg_count_now]
                        )
                        seg_count = seg_count_now
                    else:
                        img_count_now = (
                            img_count + channels[j] if channels else img_count + 1
                        )
                        image_add_input_seg.append(
                            sample_data["image"][img_count:img_count_now]
                        )
                        img_count = img_count_now
                output_dict["image"].append(torch.vstack(image_add_input_seg)[None])
                output_dict["label"].append(sample_data["label"][[0]][None])
            else:
                # add one more batchsize channel in image and label
                output_dict["image"].append(sample_data["image"][None])
                output_dict["label"].append(sample_data["label"][None])
            output_dict["idx"].append(data[idx]["idx"])
            output_dict["property"].append(data[idx]["property"])

        # we use the vstack method in torch here because the image and label
        # are Tensors after transform.
        output_dict["image"] = torch.vstack(output_dict["image"])
        output_dict["label"] = torch.vstack(output_dict["label"])
        # print("Augmentation: {}s".format(time.time() - start))

        # start = time.time()
        # generate deep supervision labels. Resolution from high to low.
        if self.do_deep_supervision:
            axes = list(range(2, len(output_dict["label"].shape)))
            deep_supervision_labels = [
                torch.Tensor(output_dict["label"]),
            ]
            origin_shape = output_dict["label"].shape[2:]
            downsample_ratio = np.array([1, 1, 1])
            for pool_kernel_size in self.pool_kernel_size_lst:
                if not isinstance(pool_kernel_size, (tuple, list)):
                    pool_kernel_size = [pool_kernel_size] * len(axes)
                else:
                    assert len(pool_kernel_size) == len(axes), (
                        f"If pool_kernel_size is a tuple for each resolution (one downsampling factor "
                        f"for each axis) then the number of entried in that tuple (here "
                        f"{len(pool_kernel_size)}) must be the same as the number of axes (here {len(axes)})."
                    )

                downsample_ratio = downsample_ratio * np.array(pool_kernel_size)
                if all([i == 1 for i in downsample_ratio]):
                    deep_supervision_labels.append(output_dict["label"])
                else:
                    new_shape = origin_shape // downsample_ratio
                    deep_supervision_labels.append(
                        F.interpolate(
                            deep_supervision_labels[0],
                            size=[int(i) for i in new_shape],
                            mode="nearest",
                        )
                    )
            output_dict["label"] = deep_supervision_labels
        # print("Deep_supervision: {}s".format(time.time() - start))

        return output_dict
