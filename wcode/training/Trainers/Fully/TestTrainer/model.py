import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple, Optional, Type
from torch.utils.checkpoint import checkpoint
import pydoc
from dynamic_network_architectures.building_blocks.helper import (
    convert_conv_op_to_dim,
    get_matching_convtransp,
    maybe_convert_scalar_to_list,
)
from typing import Tuple, Union, List, Optional, Dict, Any
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd


class PositionalEncoding3D(nn.Module):

    def __init__(self, d_model: int, max_spatial: int = 512):
        super().__init__()
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_dims = x.shape[2:]
        device = x.device
        dtype = x.dtype
        C = x.shape[1]

        encodings = []
        dim_per_axis = C // len(spatial_dims)
        remainder = C % len(spatial_dims)

        for i, size in enumerate(spatial_dims):
            extra = 1 if i < remainder else 0
            d = dim_per_axis + extra
            pe = torch.zeros(size, d, device=device, dtype=dtype)
            position = torch.arange(0, size, dtype=dtype, device=device).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d, 2, dtype=dtype, device=device) * -(math.log(10000.0) / d)
            )
            pe[:, 0::2] = torch.sin(position * div_term[:pe[:, 0::2].shape[1]])
            pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
            encodings.append(pe)

        if len(spatial_dims) == 3:
            D, H, W = spatial_dims
            pe_d, pe_h, pe_w = encodings
            pos = torch.cat([
                pe_d.unsqueeze(1).unsqueeze(1).expand(-1, H, W, -1).reshape(-1, pe_d.shape[1]),
                pe_h.unsqueeze(0).unsqueeze(2).expand(D, -1, W, -1).reshape(-1, pe_h.shape[1]),
                pe_w.unsqueeze(0).unsqueeze(0).expand(D, H, -1, -1).reshape(-1, pe_w.shape[1]),
            ], dim=-1)
        else:  # 2D
            H, W = spatial_dims
            pe_h, pe_w = encodings
            pos = torch.cat([
                pe_h.unsqueeze(1).expand(-1, W, -1).reshape(-1, pe_h.shape[1]),
                pe_w.unsqueeze(0).expand(H, -1, -1).reshape(-1, pe_w.shape[1]),
            ], dim=-1)

        if pos.shape[1] > C:
            pos = pos[:, :C]
        elif pos.shape[1] < C:
            pad = torch.zeros(pos.shape[0], C - pos.shape[1], device=device, dtype=dtype)
            pos = torch.cat([pos, pad], dim=1)

        return pos.unsqueeze(0)  # (1, N, C)


class QueryTransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 8,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, queries, memory, memory_pos=None):
        q = self.norm1(queries)
        k = memory + memory_pos if memory_pos is not None else memory
        queries = queries + self.cross_attn(q, k, memory)[0]

        q = self.norm2(queries)
        queries = queries + self.self_attn(q, q, q)[0]

        queries = queries + self.ffn(self.norm3(queries))
        return queries


class QueryTransformerDecoder(nn.Module):
    """Stack of Transformer decoder layers with optional gradient checkpointing."""

    def __init__(self, d_model=64, nhead=8, num_layers=3,
                 dim_feedforward=256, dropout=0.0, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.layers = nn.ModuleList([
            QueryTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, queries, memory, memory_pos=None):
        for layer in self.layers:
            if self.use_checkpoint:
                queries = checkpoint(
                    layer,
                    queries,
                    memory,
                    memory_pos if memory_pos is not None else None,
                    use_reentrant=False
                )
            else:
                queries = layer(queries, memory, memory_pos)
        return queries


# =====================================================================
#                     HIERARCHICAL QUERY HEAD & MODEL
# =====================================================================

class HierarchicalQueryHead(nn.Module):
    """
    Two‑level hierarchical query head with task‑level conditioning.

    Key changes:
    - Feature downsampling before Transformer (optional).
    - Mask generation via direct dot product (L2‑normalized features & queries).
    - No intermediate `mask_embed` MLP.
    """

    def __init__(
        self,
        num_classes: int,
        task_total_number: int,
        feature_channels: int,
        query_dim: int = 64,
        num_transformer_layers: int = 3,
        num_heads: int = 8,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        spatial_downsample: int = 1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.task_total_number = task_total_number
        self.query_dim = query_dim
        self.spatial_downsample = spatial_downsample

        if spatial_downsample > 1:
            self.downsample = nn.Conv3d(
                feature_channels, feature_channels,
                kernel_size=spatial_downsample, stride=spatial_downsample, bias=False
            )
        else:
            self.downsample = nn.Identity()

        self.feature_proj = nn.Sequential(
            nn.Conv3d(feature_channels, query_dim, 1, bias=False),
            nn.GroupNorm(min(32, query_dim), query_dim),
            nn.ReLU(inplace=True),
        )
        self.pos_encoding = PositionalEncoding3D(query_dim)

        self.task_class_queries = nn.Parameter(
            torch.randn(task_total_number, num_classes, query_dim) * 0.02
        )

        self.transformer_decoder = QueryTransformerDecoder(
            d_model=query_dim,
            nhead=num_heads,
            num_layers=num_transformer_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def forward(self, features: torch.Tensor, dataset_id: torch.Tensor) -> torch.Tensor:
        B = features.shape[0]
        spatial_shape_orig = features.shape[2:]

        # 1. Downsample if needed
        feat = self.downsample(features)                     # (B, C', D', H', W')
        feat_proj = self.feature_proj(feat)                  # (B, qd, D', H', W')

        N = int(np.prod(feat_proj.shape[2:]))
        memory = feat_proj.flatten(2).permute(0, 2, 1)      # (B, N, qd)
        memory_pos = self.pos_encoding(feat_proj)            # (1, N, qd)

        # 2. Retrieve task‑specific queries
        queries = self.task_class_queries[dataset_id.long()] # (B, K, qd)

        # 3. Transformer refinement
        refined = self.transformer_decoder(queries, memory, memory_pos)  # (B, K, qd)

        logits_low = torch.bmm(refined, memory.transpose(1, 2))
        logits_low = logits_low.view(B, self.num_classes, *feat_proj.shape[2:])

        # 5. Upsample if downsampled
        if self.spatial_downsample > 1 and logits_low.shape[2:] != spatial_shape_orig:
            logits = F.interpolate(logits_low, size=spatial_shape_orig,
                                   mode='trilinear', align_corners=False)
        else:
            logits = logits_low

        return logits


class HierarchicalQueryUNet(nn.Module):
    """
    PlainConvUNet encoder + lightweight FPN decoder + Hierarchical Query Head.
    """

    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_classes: int,
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
        task_total_number: int,
        query_dim: int = 64,
        num_transformer_layers: int = 3,
        num_heads: int = 8,
        spatial_downsample: int = 1,   # Pass through to query head
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        nonlin_first: bool = False,
    ):
        super().__init__()

        dim = convert_conv_op_to_dim(conv_op)
        self.dim = dim
        self.num_classes = num_classes
        self.task_total_number = task_total_number

        # Standardise lists
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        kernel_sizes = [maybe_convert_scalar_to_list(conv_op, k) for k in kernel_sizes]
        strides = [maybe_convert_scalar_to_list(conv_op, s) for s in strides]

        self.n_stages = n_stages
        self.features_per_stage = list(features_per_stage)
        self.strides = strides

        # Encoder
        self.encoder = PlainConvEncoder(
            input_channels, n_stages, features_per_stage,
            conv_op, kernel_sizes, strides, n_conv_per_stage,
            conv_bias, norm_op, norm_op_kwargs,
            dropout_op, dropout_op_kwargs,
            nonlin, nonlin_kwargs,
            return_skips=True,
            nonlin_first=nonlin_first,
        )

        # Decoder
        transpconv_op = get_matching_convtransp(conv_op=conv_op)
        self.decoder_transpconvs = nn.ModuleList()
        self.decoder_stages = nn.ModuleList()
        for u in range(n_stages - 1):
            in_features = features_per_stage[-1] if u == 0 else features_per_stage[-(u + 1)]
            out_features = features_per_stage[-(u + 2)]
            stride = strides[-(u + 1)]
            self.decoder_transpconvs.append(
                transpconv_op(in_features, out_features, stride, stride, bias=conv_bias)
            )
            self.decoder_stages.append(
                StackedConvBlocks(
                    n_conv_per_stage_decoder[u], conv_op,
                    out_features * 2, out_features,
                    kernel_sizes[-(u + 2)], 1,
                    conv_bias, norm_op, norm_op_kwargs,
                    dropout_op, dropout_op_kwargs,
                    nonlin, nonlin_kwargs,
                    nonlin_first=nonlin_first,
                )
            )

        self.decoder_output_channels = features_per_stage[0]

        self.query_head = HierarchicalQueryHead(
            num_classes=num_classes,
            task_total_number=task_total_number,
            feature_channels=self.decoder_output_channels,
            query_dim=query_dim,
            num_transformer_layers=num_transformer_layers,
            num_heads=num_heads,
            dim_feedforward=query_dim * 4,
            dropout=0.0,
            spatial_downsample=spatial_downsample,
        )


        print(f"[HierarchicalQueryUNet] input_channels={input_channels}, "
              f"num_classes={num_classes}, task_total_number={task_total_number}, "
              f"query_dim={query_dim}, spatial_downsample={spatial_downsample}")

    def forward(self, x: torch.Tensor, dataset_id: torch.Tensor):
        skips = self.encoder(x)
        x = skips[-1]
        decoder_features = []
        for u in range(len(self.decoder_transpconvs)):
            x = self.decoder_transpconvs[u](x)
            skip = skips[-(u + 2)]
            x = torch.cat([x, skip], dim=1)
            x = self.decoder_stages[u](x)
            decoder_features.append(x)

        logits = self.query_head(x, dataset_id)
        return logits

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == self.dim
        total = self.encoder.compute_conv_feature_map_size(input_size)
        spatial_sizes = [list(input_size)]
        for s in range(self.n_stages):
            new_size = [spatial_sizes[-1][d] // self.strides[s][d] for d in range(self.dim)]
            spatial_sizes.append(new_size)
        for u in range(self.n_stages - 1):
            decoder_spatial = spatial_sizes[-(u + 2)]
            decoder_numel = int(np.prod(decoder_spatial))
            features = self.features_per_stage[-(u + 2)]
            total += features * decoder_numel * 2
        full_res_numel = int(np.prod(input_size))
        total += self.query_head.query_dim * full_res_numel
        total += self.num_classes * full_res_numel
        return total

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        for m in module.modules():
            if isinstance(m, nn.MultiheadAttention):
                nn.init.xavier_uniform_(m.in_proj_weight)
                if m.in_proj_bias is not None:
                    nn.init.constant_(m.in_proj_bias, 0)
                nn.init.xavier_uniform_(m.out_proj.weight)
                if m.out_proj.bias is not None:
                    nn.init.constant_(m.out_proj.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# =====================================================================
#                     NON‑HIERARCHICAL (PLAIN) QUERY HEAD & MODEL
# =====================================================================

class PlainQueryHead(nn.Module):
    """
    Flat query mask head – no task/dataset conditioning.

    Same corrections as HierarchicalQueryHead.
    """

    def __init__(
        self,
        num_classes: int,
        feature_channels: int,
        query_dim: int = 64,
        num_transformer_layers: int = 3,
        num_heads: int = 8,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        spatial_downsample: int = 1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.query_dim = query_dim
        self.spatial_downsample = spatial_downsample

        if spatial_downsample > 1:
            self.downsample = nn.Conv3d(
                feature_channels, feature_channels,
                kernel_size=spatial_downsample, stride=spatial_downsample, bias=False
            )
        else:
            self.downsample = nn.Identity()

        self.feature_proj = nn.Sequential(
            nn.Conv3d(feature_channels, query_dim, 1, bias=False),
            nn.GroupNorm(min(32, query_dim), query_dim),
            nn.ReLU(inplace=True),
        )
        self.pos_encoding = PositionalEncoding3D(query_dim)

        self.class_queries = nn.Parameter(
            torch.randn(num_classes, query_dim) * 0.02
        )

        self.transformer_decoder = QueryTransformerDecoder(
            d_model=query_dim,
            nhead=num_heads,
            num_layers=num_transformer_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B = features.shape[0]
        spatial_shape_orig = features.shape[2:]

        feat = self.downsample(features)
        feat_proj = self.feature_proj(feat)
        N = int(np.prod(feat_proj.shape[2:]))
        memory = feat_proj.flatten(2).permute(0, 2, 1)
        memory_pos = self.pos_encoding(feat_proj)

        queries = self.class_queries.unsqueeze(0).expand(B, -1, -1)
        refined = self.transformer_decoder(queries, memory, memory_pos)

        logits_low = torch.bmm(refined, memory.transpose(1, 2))
        logits_low = logits_low.view(B, self.num_classes, *feat_proj.shape[2:])

        if self.spatial_downsample > 1 and logits_low.shape[2:] != spatial_shape_orig:
            logits = F.interpolate(logits_low, size=spatial_shape_orig,
                                   mode='trilinear', align_corners=False)
        else:
            logits = logits_low

        return logits


class PlainQueryUNet(nn.Module):
    """
    PlainConvUNet encoder + FPN decoder + PlainQueryHead.
    """

    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_classes: int,
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
        query_dim: int = 64,
        num_transformer_layers: int = 3,
        num_heads: int = 8,
        spatial_downsample: int = 1,
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        nonlin_first: bool = False,
    ):
        super().__init__()

        dim = convert_conv_op_to_dim(conv_op)
        self.dim = dim
        self.num_classes = num_classes

        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        kernel_sizes = [maybe_convert_scalar_to_list(conv_op, k) for k in kernel_sizes]
        strides = [maybe_convert_scalar_to_list(conv_op, s) for s in strides]

        self.n_stages = n_stages
        self.features_per_stage = list(features_per_stage)
        self.strides = strides

        self.encoder = PlainConvEncoder(
            input_channels, n_stages, features_per_stage,
            conv_op, kernel_sizes, strides, n_conv_per_stage,
            conv_bias, norm_op, norm_op_kwargs,
            dropout_op, dropout_op_kwargs,
            nonlin, nonlin_kwargs,
            return_skips=True,
            nonlin_first=nonlin_first,
        )

        transpconv_op = get_matching_convtransp(conv_op=conv_op)
        self.decoder_transpconvs = nn.ModuleList()
        self.decoder_stages = nn.ModuleList()
        for u in range(n_stages - 1):
            in_features = features_per_stage[-1] if u == 0 else features_per_stage[-(u + 1)]
            out_features = features_per_stage[-(u + 2)]
            stride = strides[-(u + 1)]
            self.decoder_transpconvs.append(
                transpconv_op(in_features, out_features, stride, stride, bias=conv_bias)
            )
            self.decoder_stages.append(
                StackedConvBlocks(
                    n_conv_per_stage_decoder[u], conv_op,
                    out_features * 2, out_features,
                    kernel_sizes[-(u + 2)], 1,
                    conv_bias, norm_op, norm_op_kwargs,
                    dropout_op, dropout_op_kwargs,
                    nonlin, nonlin_kwargs,
                    nonlin_first=nonlin_first,
                )
            )

        self.decoder_output_channels = features_per_stage[0]

        self.query_head = PlainQueryHead(
            num_classes=num_classes,
            feature_channels=self.decoder_output_channels,
            query_dim=query_dim,
            num_transformer_layers=num_transformer_layers,
            num_heads=num_heads,
            dim_feedforward=query_dim * 4,
            dropout=0.0,
            spatial_downsample=spatial_downsample,
        )

        print(f"[PlainQueryUNet] input_channels={input_channels}, "
              f"num_classes={num_classes}, query_dim={query_dim}, "
              f"spatial_downsample={spatial_downsample}")

    def forward(self, x: torch.Tensor):
        skips = self.encoder(x)
        x = skips[-1]
        decoder_features = []
        for u in range(len(self.decoder_transpconvs)):
            x = self.decoder_transpconvs[u](x)
            skip = skips[-(u + 2)]
            x = torch.cat([x, skip], dim=1)
            x = self.decoder_stages[u](x)
            decoder_features.append(x)

        logits = self.query_head(x)
        
        return {"pred": logits}

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == self.dim
        total = self.encoder.compute_conv_feature_map_size(input_size)
        spatial_sizes = [list(input_size)]
        for s in range(self.n_stages):
            new_size = [spatial_sizes[-1][d] // self.strides[s][d] for d in range(self.dim)]
            spatial_sizes.append(new_size)
        for u in range(self.n_stages - 1):
            decoder_spatial = spatial_sizes[-(u + 2)]
            decoder_numel = int(np.prod(decoder_spatial))
            features = self.features_per_stage[-(u + 2)]
            total += features * decoder_numel * 2
        full_res_numel = int(np.prod(input_size))
        total += self.query_head.query_dim * full_res_numel
        total += self.num_classes * full_res_numel
        return total

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        for m in module.modules():
            if isinstance(m, nn.MultiheadAttention):
                nn.init.xavier_uniform_(m.in_proj_weight)
                if m.in_proj_bias is not None:
                    nn.init.constant_(m.in_proj_bias, 0)
                nn.init.xavier_uniform_(m.out_proj.weight)
                if m.out_proj.bias is not None:
                    nn.init.constant_(m.out_proj.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    