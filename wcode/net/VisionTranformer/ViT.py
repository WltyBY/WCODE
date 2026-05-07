import torch
import torch.nn as nn

from typing import List, Union, Dict

from wcode.net.VisionTranformer.modules.baseblocks import (
    PatchEmbedding,
    MHSATransformerBlock,
    MHSCATransformerBlock,
    Learnable1DPositionalEmbedding,
)


class ViTCLS(nn.Module):
    """
    Vision Transformer model for image classification.
    """

    def __init__(self, params: Dict):
        """
        in_channels: number of input channels
        num_classes: number of classes for classification
        image_size: input image size (square)
        patch_size: patch size
        embed_dim: embedding dimension
        depth: number of Transformer encoder blocks
        num_heads: number of attention heads
        mlp_ratio: ratio of MLP hidden dimension to embed_dim
        dropout_rate: dropout probability inside embeddings and Transformer
        PE_dropout_rate: dropout probability after position embedding
        """
        super().__init__()
        assert isinstance(params["image_size"], list), "image_size must be a list"
        dim = len(params["image_size"])
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            params["image_size"],
            params["patch_size"],
            params["in_channels"],
            params["embed_dim"],
            dim,
        )
        n_patches = self.patch_embed.n_patches

        # Learnable class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, params["embed_dim"]))
        # Learnable position embedding
        self.pos_embed = Learnable1DPositionalEmbedding(
            n_patches + 1, params["embed_dim"], params["PE_dropout_rate"]
        )

        self.MHSA_blocks = nn.ModuleList(
            [
                MHSATransformerBlock(
                    params["embed_dim"],
                    params["num_heads"],
                    params["mlp_ratio"],
                    params["dropout_rate"],
                )
                for _ in range(params["depth"])
            ]
        )

        # Final layer norm
        self.norm = nn.LayerNorm(params["embed_dim"])

        # Classification head
        self.cls_head = nn.Linear(params["embed_dim"], params["out_channels"])

        # Weight initialization
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        """
        Input x: B, C, (Z,) Y, X
        Output logits: B, out_channels
        """
        B = x.shape[0]
        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, n_patches+1, embed_dim)

        # Add position embedding
        x = self.pos_embed(x)  # (B, n_patches+1, embed_dim)

        # Pass through Transformer blocks
        features = []
        for block in self.MHSA_blocks:
            x = block(x)
            features.append(
                x[:, 1:]  # Exclude class token
                .transpose(1, 2)
                .reshape(B, -1, *self.patch_embed.n_patches_per_dim)
            )

        # Final layer norm
        x = self.norm(x)

        # Extract class token output
        cls_token_final = x[:, 0]  # (B, embed_dim)

        # Classification head
        logits = self.cls_head(cls_token_final)
        return {"pred": logits, "features": features}


class ViTSEG(nn.Module):
    """
    Vision Transformer for Semantic Segmentation without any resolution change.
    Encoder: multi-stage with self-attention.
    Decoder: symmetric stages with cross‑attention skip connections.
    """

    def __init__(
        self,
        params: Dict,
    ):
        super().__init__()
        assert isinstance(params["image_size"], list), "image_size must be a list"
        dim = len(params["image_size"])

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            params["image_size"],
            params["patch_size"],
            params["in_channels"],
            params["embed_dim"],
            dim,
        )
        n_patches = self.patch_embed.n_patches

        # Learnable class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, params["embed_dim"]))
        # Learnable position embedding
        self.pos_embed = Learnable1DPositionalEmbedding(
            n_patches + 1, params["embed_dim"], params["PE_dropout_rate"]
        )

        # Encoder stages
        self.MHSA_blocks = nn.ModuleList(
            [
                MHSATransformerBlock(
                    params["embed_dim"],
                    params["num_heads"],
                    params["mlp_ratio"],
                    params["dropout_rate"],
                )
                for _ in range(params["depth"])
            ]
        )

        # Decoder stages
        self.MHSCA_blocks = nn.ModuleList(
            [
                MHSCATransformerBlock(
                    params["embed_dim"],
                    params["num_heads"],
                    params["mlp_ratio"],
                    params["dropout_rate"],
                )
                for _ in range(params["depth"] - 1)
            ]
        )

        # segmentation head
        if dim == 2:
            ConvLayer, ConvTransLayer, NormLayer = (
                nn.Conv2d,
                nn.ConvTranspose2d,
                nn.BatchNorm2d,
            )
        elif dim == 3:
            ConvLayer, ConvTransLayer, NormLayer = (
                nn.Conv3d,
                nn.ConvTranspose3d,
                nn.InstanceNorm3d,
            )
        else:
            raise ValueError(f"Unsupported dimensionality: {dim}")

        self.deep_supervision = params["deep_supervision"]
        self.need_features = params["need_features"]
        if self.deep_supervision:
            self.prediction_head = nn.ModuleList()
            # we will not do deep supervision on the prediction of bottleneck output feature
            # the prediction_heads are from low(near bottleneck) to high resolution.
            for _ in range(len(self.MHSCA_blocks)):
                self.prediction_head.append(
                    nn.Sequential(
                        ConvTransLayer(
                            params["embed_dim"],
                            params["embed_dim"] // 16,
                            kernel_size=params["patch_size"],
                            stride=params["patch_size"],
                        ),
                        NormLayer(params["embed_dim"] // 16),
                        nn.LeakyReLU(inplace=True),
                        ConvLayer(
                            params["embed_dim"] // 16,
                            params["out_channels"],
                            kernel_size=1,
                        ),
                    )
                )
        else:
            self.prediction_head = nn.Sequential(
                ConvTransLayer(
                    params["embed_dim"],
                    params["embed_dim"] // 16,
                    kernel_size=params["patch_size"],
                    stride=params["patch_size"],
                ),
                NormLayer(params["embed_dim"] // 16),
                nn.LeakyReLU(inplace=True),
                ConvLayer(
                    params["embed_dim"] // 16, params["out_channels"], kernel_size=1
                ),
            )

        # Weight initialization
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        """
        Input x: B, C, (Z,) Y, X
        Output logits: B, out_channels, (Z,) Y, X
        """
        B = x.shape[0]
        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, n_patches+1, embed_dim)

        # Add position embedding
        x = self.pos_embed(x)  # (B, n_patches+1, embed_dim)

        # Encoding
        encode_features = []
        for block in self.MHSA_blocks:
            x = block(x)
            encode_features.append(x)

        # Decoding with skip connections
        skip_features = encode_features[::-1]  # Reverse for skip connections
        x = skip_features.pop()  # Start decoding from the last encoder output
        assert len(self.MHSCA_blocks) == len(skip_features)
        decode_features = []
        for i, block in enumerate(self.MHSCA_blocks):
            x = block(x, skip_features[i])  # Cross-attention with skip connection
            # (B, N_q, C) -> (B, C, (Z,) Y, X) for prediction
            decode_features.append(
                x[:, 1:]
                .transpose(1, 2)
                .reshape(B, -1, *self.patch_embed.n_patches_per_dim)
            )

        if self.deep_supervision:
            outputs = []
            for i, head in enumerate(self.prediction_head):
                outputs.append(head(decode_features[i]))
            # we assume that the multi-level prediction ranking ranges from high resolution,
            # to low(near bottleneck) resolution
            if self.need_features:
                for i in range(len(encode_features)):
                    encode_features[i] = (
                        encode_features[i][:, 1:]
                        .transpose(1, 2)
                        .reshape(B, -1, *self.patch_embed.n_patches_per_dim)
                    )
                return {
                    "pred": outputs[::-1],
                    "CLS_Token": x[:, 0],  # (B, embed_dim)
                    "feature": encode_features + decode_features,
                }
            else:
                return {"pred": outputs[::-1], "CLS_Token": x[:, 0]}
        else:
            outputs = self.prediction_head(decode_features[-1])
            # we assume that the multi-level prediction ranking ranges from high resolution,
            # to low(near bottleneck) resolution
            if self.need_features:
                for i in range(len(encode_features)):
                    encode_features[i] = (
                        encode_features[i][:, 1:]
                        .transpose(1, 2)
                        .reshape(B, -1, *self.patch_embed.n_patches_per_dim)
                    )
                return {
                    "pred": outputs,
                    "CLS_Token": x[:, 0],
                    "feature": encode_features + decode_features,
                }
            else:
                return {"pred": outputs, "CLS_Token": x[:, 0]}


if __name__ == "__main__":
    import time

    from wcode.utils.file_operations import open_yaml

    data = open_yaml("./wcode/net/examples/ViT.yaml")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_dict = {"ViTCLS": ViTCLS, "ViTSEG": ViTSEG}

    for model in data.keys():
        print(f"----------{model}----------")
        net = model_dict[model[:6]](data[model]).to(device).eval()
        inputs = torch.rand(
            (1, data[model]["in_channels"], *data[model]["image_size"])
        ).to(device)
        begin = time.time()
        with torch.no_grad():
            outputs = net(inputs)
        print("Time:", time.time() - begin)

        for k, v in outputs.items():
            if isinstance(v, list):
                print(f"{k}: {[x.shape for x in v]}")
            else:
                print(f"{k}: {v.shape}")

        total = sum(p.numel() for p in net.parameters())
        print("Total params: %.3fM" % (total / 1e6))
