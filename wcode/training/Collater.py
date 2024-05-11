import torch
import torch.nn.functional as F

from wcode.training.Patchsampler import Patchsampler


class PatchBasedCollater(object):
    """
    This is for PatchBaseTrainer to do patch sample and transform in dataloader
    """

    def __init__(
        self,
        patch_size,
        do_deep_supervision,
        pool_kernel_size_lst,
        oversample_foreground_percent: float,
        probabilistic_oversampling: bool,
        transform,
    ):
        self.do_deep_supervision = do_deep_supervision
        self.pool_kernel_size_lst = pool_kernel_size_lst
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
        batch_size = len(data)
        output_dict = {key: [] for key in data[0].keys()}

        image_lst = []
        label_lst = []
        oversample_lst = []
        for i in range(batch_size):
            image_lst.append(data[i]["image"])
            label_lst.append(data[i]["label"])
            oversample_lst.append(data[i]["property"]["oversample"])

        # sample patch
        # image: (b, c, patchsize), label: (b, 1, patchsize)
        # start = time.time()
        image, label = self.Patchsampler.run(image_lst, label_lst, oversample_lst)
        # print("Patch Sample: {}s".format(time.time() - start))

        # start = time.time()
        # do transform
        # in monai, most of the pre-/post-processing transforms expect: (num_channels, spatial_dim_1[, spatial_dim_2, ...]),
        for i in range(batch_size):
            sample_data = {"image": image[i], "label": label[i]}
            if self.transform is not None:
                sample_data = self.transform(sample_data)
            # add one more batchsize channel in image and label
            output_dict["image"].append(sample_data["image"][None])
            output_dict["label"].append(sample_data["label"][None])
            output_dict["idx"].append(data[i]["idx"])
            output_dict["property"].append(data[i]["property"])

        # we use the vstack method in torch here because the image and label
        # are Tensors after transform.
        output_dict["image"] = torch.vstack(output_dict["image"])
        output_dict["label"] = torch.vstack(output_dict["label"])
        # print("Augmentation: {}s".format(time.time() - start))

        # generate deep supervision labels
        axes = list(range(2, len(output_dict["label"].shape)))
        if self.do_deep_supervision:
            deep_supervision_labels = [
                output_dict["label"],
            ]
            for pool_kernel_size in self.pool_kernel_size_lst:
                upper_resolution_label = deep_supervision_labels[-1]

                if not isinstance(pool_kernel_size, (tuple, list)):
                    pool_kernel_size = [pool_kernel_size] * len(axes)
                else:
                    assert len(pool_kernel_size) == len(axes), (
                        f"If pool_kernel_size is a tuple for each resolution (one downsampling factor "
                        f"for each axis) then the number of entried in that tuple (here "
                        f"{len(pool_kernel_size)}) must be the same as the number of axes (here {len(axes)})."
                    )

                if all([i == 1 for i in pool_kernel_size]):
                    deep_supervision_labels.append(upper_resolution_label)
                else:
                    old_shape = upper_resolution_label.shape[2:]
                    new_shape = [
                        round(old_shape[i] / pool_kernel_size[i])
                        for i in range(len(old_shape))
                    ]
                    deep_supervision_labels.append(
                        F.interpolate(
                            upper_resolution_label, size=new_shape, mode="nearest"
                        )
                    )
            output_dict["label"] = deep_supervision_labels

        return output_dict
