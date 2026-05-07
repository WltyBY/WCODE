import torch
import torch.nn as nn
import numpy as np

from typing import List, Union


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        image_size: List[int],
        patch_size: Union[int, List[int]] = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        dim: int = 3,
    ):
        """
        embed_dim: dimension of the output embedding. For standard ViT,
            this is typically 768 for ViT-Base, 1024 for ViT-Large and 1280 for ViT-Huge.
        dim: dimensionality of the input (2D or 3D).
        """
        super().__init__()
        assert dim in [2, 3], "Dimension must be either 2 or 3."
        assert isinstance(
            patch_size, (int, list)
        ), "Patch size must be an integer or a list of integers."

        if isinstance(patch_size, int):
            patch_size = [patch_size] * dim
        else:
            assert len(patch_size) == dim, f"Patch size list must have {dim} elements."
        assert isinstance(image_size, list), "Image size must be a list of integers."
        assert len(image_size) == dim, f"Image size list must have {dim} elements."

        self.n_patches_per_dim = [image_size[i] // patch_size[i] for i in range(dim)]
        self.n_patches = np.prod(self.n_patches_per_dim)

        ConvLayer = nn.Conv3d if dim == 3 else nn.Conv2d
        self.proj = ConvLayer(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        """
        Input x: B, c, (z,), y, x
        Output: B, n_patches, embed_dim
        """
        x = self.proj(x)  # B, embed_dim, (z//PaSize), (y//PaSize), (x//PaSize)
        x = x.flatten(2).transpose(
            1, 2
        )  # B, embed_dim, n_patches -> B, n_patches, embed_dim

        return x


class Learnable1DPositionalEmbedding(nn.Module):
    def __init__(self, n_patches, embed_dim, dropout_rate=0.1):
        super().__init__()
        # Here n_patches already counts the class token
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
        self.dropout = nn.Dropout(p=dropout_rate)

        # initialize the positional embeddings with a truncated normal distribution
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        """
        Input x: B, n_patches + 1, embed_dim
        Output: B, n_patches + 1, embed_dim
        """
        x = x + self.pos_embed
        x = self.dropout(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi‑Head self‑Attention module.
    """

    def __init__(self, in_channels, num_heads, dropout_rate=0.0):
        super().__init__()
        assert in_channels % num_heads == 0, f"embed_dim: {in_channels} must be divisible by num_heads: {num_heads}"
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim**-0.5  # scaling factor

        # Linear layer to compute Q, K, V
        self.qkv = nn.Linear(in_channels, in_channels * 3)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(in_channels, in_channels)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Input x: B, n_patches + 1, embed_dim
        Output: B, n_patches + 1, embed_dim
        """
        B, N, C = x.shape
        # Generate Q, K, V
        qkv = self.qkv(x)  # (B, N, 3*C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each shape (B, num_heads, N, head_dim)

        # Compute attention weights
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention
        # (B, num_heads, N, head_dim) -> (B, N, C)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MultiHeadCrossAttention(nn.Module):
    """
    Multi‑Head Cross‑Attention module.
    """

    def __init__(self, in_channels, num_heads, dropout_rate=0.0):
        super().__init__()
        assert (
            in_channels % num_heads == 0
        ), "in_channels must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim**-0.5  # scaling factor

        self.q_proj = nn.Linear(in_channels, in_channels)
        self.kv_proj = nn.Linear(in_channels, in_channels * 2)

        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(in_channels, in_channels)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x, context):
        """
        Args:
            x: B, n_patches+1, embed_dim.    queries
            context: B, n_patches+1, embed_dim.    keys and values

        Returns:
            output: B, n_patches+1, embed_dim.
        """
        B, N_q, C = x.shape
        _, N_kv, _ = context.shape

        # Project queries, keys, values
        Q = self.q_proj(x)  # (B, N_q, C)
        KV = self.kv_proj(context)  # (B, N_kv, 2*C)
        K, V = torch.chunk(KV, 2, dim=-1)  # each (B, N_kv, C)

        # Reshape for multi-head attention
        Q = Q.view(B, N_q, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )  # (B, h, N_q, d)
        K = K.view(B, N_kv, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )  # (B, h, N_kv, d)
        V = V.view(B, N_kv, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )  # (B, h, N_kv, d)

        # Compute attention scores
        attn_scores = (Q @ K.transpose(-2, -1)) * self.scale  # (B, h, N_q, N_kv)
        attn_probs = attn_scores.softmax(dim=-1)
        attn_probs = self.attn_drop(attn_probs)

        # Apply attention to values
        # (B, h, N_q, d) -> (B, N_q, C)
        x = (attn_probs @ V).transpose(1, 2).contiguous().reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class FeedForward(nn.Module):
    def __init__(self, in_channels, hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, in_channels),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class MHSATransformerBlock(nn.Module):
    """
    Single Transformer encoder block.
    Consists of: -> LayerNorm -> Multi‑Head Self‑Attention (with residual)
                 -> LayerNorm -> FeedForward (with residual)
    """

    def __init__(self, in_channels, num_heads, mlp_ratio=4.0, dropout_rate=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_channels)
        self.attn = MultiHeadSelfAttention(in_channels, num_heads, dropout_rate)
        self.norm2 = nn.LayerNorm(in_channels)
        self.mlp = FeedForward(
            in_channels=in_channels,
            hidden_dim=int(in_channels * mlp_ratio),
            dropout_rate=dropout_rate,
        )

    def forward(self, x):
        # Self‑attention + residual
        x = x + self.attn(self.norm1(x))
        # MLP + residual
        x = x + self.mlp(self.norm2(x))
        return x


class MHSCATransformerBlock(nn.Module):
    """
    Single Transformer encoder block.
    Consists of: -> LayerNorm -> Multi‑Head Self‑Attention (with residual)
                 -> LayerNorm -> Multi‑Head Cross‑Attention (with residual)
                 -> LayerNorm -> FeedForward (with residual)
    """

    def __init__(self, in_channels, num_heads, mlp_ratio=4.0, dropout_rate=0.0):
        super().__init__()
        self.norm1_x = nn.LayerNorm(in_channels)
        self.norm1_context = nn.LayerNorm(in_channels)
        self.self_attn = MultiHeadSelfAttention(in_channels, num_heads, dropout_rate)
        self.norm2 = nn.LayerNorm(in_channels)
        self.cross_attn = MultiHeadCrossAttention(in_channels, num_heads, dropout_rate)
        self.norm3 = nn.LayerNorm(in_channels)
        self.mlp = FeedForward(
            in_channels=in_channels,
            hidden_dim=int(in_channels * mlp_ratio),
            dropout_rate=dropout_rate,
        )

    def forward(self, x, context):
        """
        x: main input (queries) of shape (B, n_patches+1, embed_dim)
        context: secondary input (keys and values) of shape (B, n_patches+1, embed_dim)
        """
        # Self‑attention + residual
        x = x + self.self_attn(self.norm1_x(x))
        # Cross‑attention + residual
        x = x + self.cross_attn(self.norm2(x), context=self.norm1_context(context))
        # MLP + residual
        x = x + self.mlp(self.norm3(x))
        return x
