import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class ReCo_ContrastiveLoss(nn.Module):
    """
    @inproceedings{liu2022reco,
        title={Bootstrapping Semantic Segmentation with Regional Contrast},
        author={Liu, Shikun and Zhi, Shuaifeng and Johns, Edward and Davison, Andrew J},
        booktitle={International Conference on Learning Representations},
        year={2022},
    }
    """

    def __init__(
        self,
        tau=0.5,
        strong_threshold=1.0,
        num_queries=256,
        num_negatives=256,
        do_bg=True,
        apply_nonlin=True,
    ):
        super(ReCo_ContrastiveLoss, self).__init__()
        self.tau = tau
        self.strong_threshold = strong_threshold
        self.num_queries = num_queries
        self.num_negatives = num_negatives
        self.do_bg = do_bg
        self.apply_nonlin = apply_nonlin

    def forward(
        self, pred: torch.Tensor, label: torch.Tensor, proj_feature: torch.Tensor
    ):
        """
        pred is [b, c, z, x, y] predicted one_hot label, where c is the number of class
        label is [b, 1, z, x, y] ground truth with class label 0, 1, 2, 3, ...
        proj_feature in [b, m, z, x, y], where m is the dismension of feature in latent space
        """
        device = proj_feature.device
        shp_x, shp_y = pred.shape, label.shape
        dim = len(shp_x[2:])
        # num_feat is m
        num_feat = proj_feature.shape[1]

        if self.apply_nonlin:
            pred = torch.softmax(pred, dim=1)
        num_segments = pred.shape[1]

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                # if y in b, z, y, x, change to [b, 1, z, y, x]
                label = label.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(shp_x, shp_y)]):
                # if this is the case then gt is probably already a one hot encoding
                label_onehot = label
            else:
                gt = label.long()
                label_onehot = torch.zeros(shp_x, device=pred.device)
                label_onehot.scatter_(1, gt, 1)

        if self.do_bg:
            mask = label >= 0
        else:
            mask = label > 0
        # valid_pixel in [b, c, z, x, y]
        valid_pixel = label_onehot * mask

        # 2D: [b, m, h, w] -> [b, h, w, m], 3D: [b, m, z, y, x] -> [b, z, y, x, m]
        proj_feature = (
            proj_feature.permute(0, 2, 3, 1)
            if dim == 2
            else proj_feature.permute(0, 2, 3, 4, 1)
        )

        # compute prototype (class mean representation) for each class across all valid pixels
        seg_proto_list = []
        seg_feat_all_list = []
        seg_feat_hard_list = []
        seg_num_list = []
        for i in range(num_segments):
            # select binary mask for i-th class, valid_pixel_segin [b, z, y, x]
            valid_pixel_seg = valid_pixel[:, i]

            # not all classes would be available in a mini-batch
            if valid_pixel_seg.sum() == 0:
                continue

            prob_seg = pred[:, i, :, :]
            # select hard queries, rep_mask_hard in [b, z, y, x]
            rep_mask_hard = (prob_seg < self.strong_threshold) * valid_pixel_seg.bool()

            # each prototype in [1, m]
            seg_proto_list.append(
                torch.mean(proj_feature[valid_pixel_seg.bool()], dim=0, keepdim=True)
            )
            # each element in [num_of_pixels_belong_to_this_class, m]
            seg_feat_all_list.append(proj_feature[valid_pixel_seg.bool()])
            # each element in [num_of_hard_pixels_belong_to_this_class, m]
            seg_feat_hard_list.append(proj_feature[rep_mask_hard])
            # each element is a scalar num_of_pixels_belong_to_this_class
            seg_num_list.append(int(valid_pixel_seg.sum().item()))

        # compute regional contrastive loss
        if len(seg_num_list) <= 1:
            # in some rare cases, a small mini-batch might only contain 1 or no semantic class
            return torch.tensor(0.0)
        else:
            reco_loss = torch.tensor(0.0)
            # seg_proto in c, m
            seg_proto = torch.cat(seg_proto_list)
            valid_seg = len(seg_num_list)
            seg_len = torch.arange(valid_seg)

            for i in range(valid_seg):
                # sample hard queries
                if len(seg_feat_hard_list[i]) > 0:
                    seg_hard_idx = torch.randint(
                        len(seg_feat_hard_list[i]), size=(self.num_queries,)
                    )
                    anchor_feat_hard = seg_feat_hard_list[i][seg_hard_idx]
                    anchor_feat = anchor_feat_hard
                else:
                    # in some rare cases, all queries in the current query class are easy
                    continue

                # apply negative key sampling (with no gradients)
                with torch.no_grad():
                    # generate index mask for the current query class; e.g. [0, 1, 2] -> [1, 2, 0] -> [2, 0, 1]
                    seg_mask = torch.cat(([seg_len[i:], seg_len[:i]]))

                    # compute similarity for each negative segment prototype (semantic class relation graph)
                    # proto_sim in [c-1]
                    proto_sim = torch.cosine_similarity(
                        seg_proto[seg_mask[0]].unsqueeze(0),
                        seg_proto[seg_mask[1:]],
                        dim=1,
                    )
                    proto_prob = torch.softmax(proto_sim / self.tau, dim=0)

                    # sampling negative keys based on the generated distribution [num_queries x num_negatives]
                    negative_dist = torch.distributions.categorical.Categorical(
                        probs=proto_prob
                    )
                    samp_class = negative_dist.sample(
                        sample_shape=[self.num_queries, self.num_negatives]
                    )
                    samp_num = torch.stack(
                        [(samp_class == c).sum(1) for c in range(len(proto_prob))],
                        dim=1, 
                    )

                    # sample negative indices from each negative class
                    negative_num_list = seg_num_list[i + 1 :] + seg_num_list[:i]
                    negative_index = self.negative_index_sampler(
                        samp_num, negative_num_list
                    )

                    # index negative keys (from other classes)
                    negative_feat_all = torch.cat(
                        seg_feat_all_list[i + 1 :] + seg_feat_all_list[:i]
                    )
                    # negative_feat_all[negative_index] in [num_of_neg_samples, proj_feature m]
                    # reshape num_of_neg_samples to [self.num_queries, self.num_negatives]
                    negative_feat = negative_feat_all[negative_index].reshape(
                        self.num_queries, self.num_negatives, num_feat
                    )

                    # combine positive and negative keys: keys = [positive key | negative keys] with 1 + num_negative dim
                    positive_feat = (
                        seg_proto[i]
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .repeat(self.num_queries, 1, 1)
                    )
                    # one positive_feat and self.num_negatives negative_feat
                    all_feat = torch.cat((positive_feat, negative_feat), dim=1)

                seg_logits = torch.cosine_similarity(
                    anchor_feat.unsqueeze(1), all_feat, dim=2
                )
                reco_loss = reco_loss + F.cross_entropy(
                    seg_logits / self.tau,
                    torch.zeros(self.num_queries).long().to(device),
                )
        return reco_loss / valid_seg

    def negative_index_sampler(self, samp_num, seg_num_list):
        negative_index = []
        for i in range(samp_num.shape[0]):
            for j in range(samp_num.shape[1]):
                negative_index += np.random.randint(
                    low=sum(seg_num_list[:j]),
                    high=sum(seg_num_list[: j + 1]),
                    size=int(samp_num[i, j]),
                ).tolist()
        return negative_index


if __name__ == "__main__":
    loss = ReCo_ContrastiveLoss(
        tau=0.5, strong_threshold=0.95, num_queries=256, num_negatives=512
    )
    pred = torch.rand(10, 3, 512, 512)
    label = torch.randint(0, 2, (10, 3, 512, 512))
    proj_feature = torch.rand(10, 256, 512, 512)
    print(loss(pred, label, proj_feature))
