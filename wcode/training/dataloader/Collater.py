import torch


class BasedCollater(object):
    def __init__(
        self,
    ):
        pass

    def __call__(self, data):
        """
        When each element of the Dataset is a Dict, the original DataLoader recursively packs all data in the collate function,
        and will throw an error if the shapes are different. Here, we rewrite the data, only batch-level packing the image and label tensors,
        and packing all other parameters into lists.
        """
        output_dict = {key: [] for key in data[0].keys()}
        image_lst = []
        label_lst = []
        property_lst = []
        idx_lst = []
        for i in range(len(data)):
            image_lst.append(data[i]["image"])
            label_lst.append(data[i]["label"])
            property_lst.append(data[i]["property"])
            idx_lst.append(data[i]["idx"])
        output_dict["property"] = property_lst
        output_dict["idx"] = idx_lst

        image_all = torch.stack(image_lst)
        if isinstance(label_lst[0], list):
            # for deep supervision
            label_all = [
                torch.stack([s[i] for s in label_lst]) for i in range(len(label_lst[0]))
            ]
        else:
            label_all = torch.stack(label_lst)
            
        output_dict["image"] = image_all
        output_dict["label"] = label_all

        return output_dict
