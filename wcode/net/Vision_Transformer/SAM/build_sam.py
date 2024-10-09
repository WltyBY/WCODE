# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from functools import partial

from wcode.net.Vision_Transformer.SAM.model.ImageEncoderViT import ImageEncoderViT
from wcode.net.Vision_Transformer.SAM.model.mask_decoder import MaskDecoder
from wcode.net.Vision_Transformer.SAM.model.prompt_encoder import PromptEncoder
from wcode.net.Vision_Transformer.SAM.model.Sam import Sam
from wcode.net.Vision_Transformer.SAM.model.transformer import TwoWayTransformer


def build_sam_vit_h(
    image_size=1024,
    num_classes=3,
    pixel_mean=[123.675, 116.28, 103.53],
    pixel_std=[58.395, 57.12, 57.375],
    whether_normalize=True,
    checkpoint=None,
):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        num_classes=num_classes,
        image_size=image_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
        whether_normalize=whether_normalize,
        checkpoint=checkpoint,
    )


def build_sam_vit_l(
    image_size=1024,
    num_classes=3,
    pixel_mean=[123.675, 116.28, 103.53],
    pixel_std=[58.395, 57.12, 57.375],
    whether_normalize=True,
    checkpoint=None,
):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        num_classes=num_classes,
        image_size=image_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
        whether_normalize=whether_normalize,
        checkpoint=checkpoint,
    )


def build_sam_vit_b(
    image_size=1024,
    num_classes=3,
    pixel_mean=[123.675, 116.28, 103.53],
    pixel_std=[58.395, 57.12, 57.375],
    whether_normalize=True,
    checkpoint=None,
):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        num_classes=num_classes,
        image_size=image_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
        whether_normalize=whether_normalize,
        checkpoint=checkpoint,
    )


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    num_classes=3,
    image_size=1024,
    pixel_mean=[123.675, 116.28, 103.53],
    pixel_std=[58.395, 57.12, 57.375],
    whether_normalize=True,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = image_size
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=num_classes,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
        whether_normalize=whether_normalize,
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict)
    return sam


if __name__ == "__main__":
    import time

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("-----------------------SAM_b-----------------------")
    checkpoint_b = "./Dataset/SAM/sam_vit_b_01ec64.pth"
    SAM_b = build_sam_vit_b(checkpoint=checkpoint_b)
    del SAM_b.prompt_encoder
    del SAM_b.mask_decoder
    SAM_b.to(device)

    step = 16
    begin = time.time()
    for _ in range(step):
        inputs = torch.rand((1, 3, 1024, 1024)).to(device)
        with torch.no_grad():
            outputs = SAM_b.image_encoder(inputs)
    print("Time:", (time.time() - begin) / step)
    print("Outputs:")
    if isinstance(outputs, (list, tuple)):
        for output in outputs:
            print(output.shape)
    else:
        print(outputs.shape)
    total = sum(p.numel() for p in SAM_b.image_encoder.parameters())
    print("Total params: %.3fM" % (total / 1e6))
    del SAM_b, inputs, outputs

    print("----------------------")
    SAM_b = build_sam_vit_b(checkpoint=checkpoint_b)
    SAM_b.to(device)
    inputs = torch.rand((3, 250, 250)).to(device)
    outputs = SAM_b([{"image": inputs, "original_size": inputs.size()[1:]}], True)
    for output in outputs:
        print(output["masks"].shape)
        print(output["iou_predictions"])
        print(output["low_res_logits"].shape)
