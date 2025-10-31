from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import Mlp, Attention
from timm.models.registry import register_model
from itertools import accumulate


class PatchEmbed_3d(nn.Module):
    def __init__(self, img_size=9, patch_size=3, in_chans=1, embed_dim=768, num_frames=6000, tubelet_size=100):
        super(PatchEmbed_3d, self).__init__()
        if not isinstance(img_size, tuple):
            img_size = to_2tuple(img_size)
        if not isinstance(patch_size, tuple):
            patch_size = to_2tuple(patch_size)
        num_spatial_patches = (img_size[0] // patch_size[0]) * (
                img_size[1] // patch_size[1])
        num_patches = num_spatial_patches * (num_frames // tubelet_size)

        self.img_size = img_size
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.num_patches = num_patches
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(self.tubelet_size, patch_size[0], patch_size[1]),
            stride=(self.tubelet_size, patch_size[0], patch_size[1]))

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[
            1], "Input image size (%d * %d) doesn't match model (%d * %d)." % (H, W, self.img_size[0], self.img_size[1])
        # b, c, l -> b, l, c
        if T < self.tubelet_size:
            return torch.zeros(size=(B, 0, self.proj.out_channels)).cuda()
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbed_3d_seq(nn.Module):
    def __init__(self, img_size=9, patch_size=1, in_chans=1, embed_dim=768, num_frames=6000, tubelet_size=100):
        super(PatchEmbed_3d_seq, self).__init__()
        if not isinstance(img_size, tuple):
            self.img_size = to_2tuple(img_size)
        else:
            self.img_size = img_size
        if not isinstance(patch_size, tuple):
            self.patch_size = to_2tuple(patch_size)
        else:
            self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.in_channs = in_chans
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        num_spatial_patches = (self.img_size[0] // self.patch_size[0]) * (
                self.img_size[1] // self.patch_size[1])
        self.num_patches = num_spatial_patches * (num_frames // tubelet_size)
        self.linear = nn.Linear(patch_size * patch_size * in_chans * tubelet_size, embed_dim)

    def forward(self, x):
        B, C, T, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[
            1], "Input image size (%d * %d) doesn't match model (%d * %d)." % (H, W, self.img_size[0], self.img_size[1])
        # b, c, l -> b, l, c
        if T < self.tubelet_size:
            return torch.zeros(size=(B, 0, self.embed_dim)).cuda()

        x = x.reshape(shape=(
            B, C, T // self.tubelet_size, self.tubelet_size, H // self.patch_size[0], self.patch_size[0],
            W // self.patch_size[1], self.patch_size[1]))
        x = torch.einsum('ncfthpwq->nfhwctpq', x)
        x = x.reshape(B, (T * H * W) // (self.tubelet_size * self.patch_size[0] * self.patch_size[1]),
                      self.tubelet_size * self.patch_size[0] * self.patch_size[1] * self.in_channs)
        x = self.linear(x)
        return x


class PatchEmbed_1d_seq(nn.Module):
    def __init__(self, in_channels=1, tubelet_size=100, length=6000, embed_dim=768):
        super(PatchEmbed_1d_seq, self).__init__()
        self.modal_length = length
        self.tubelet_size = tubelet_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_patches = length // tubelet_size
        self.linear = nn.Linear(in_channels * tubelet_size, embed_dim)

    def forward(self, x):
        B, T, C = x.shape
        if T < self.tubelet_size:
            return torch.zeros(size=(B, 0, self.embed_dim)).cuda()
        assert C == 1
        x = x.reshape(shape=(B, T // self.tubelet_size, self.tubelet_size, C))
        x = x.reshape(shape=(B, -1, self.tubelet_size))
        x = self.linear(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                              proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(1)
            module.weight.data = 0.67 * 12 ** (-0.25) * module.weight.data

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim: int,
                 num_heads: int = 8,
                 qkv_bias: bool = False,
                 qk_norm: bool = False,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 norm_layer: nn.Module = nn.LayerNorm,
                 ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_q, x_kv):
        B, N_q, C = x_q.shape
        _, N_kv, _ = x_kv.shape

        q = self.q(x_q).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x_kv).reshape(B, N_kv, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q, k = self.q_norm(q), self.k_norm(k)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )
        x = x.transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SplitAttention(nn.Module):
    def __init__(self, dim: int,
                 num_heads: int = 8,
                 qkv_bias: bool = False,
                 qk_norm: bool = False,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 norm_layer: nn.Module = nn.LayerNorm,
                 num_splits: int = 6,
                 ) -> None:
        super(SplitAttention, self).__init__()
        self.num_splits = num_splits
        self.self_attentions = nn.ModuleList(
            [Attention(dim, num_heads, qkv_bias, qk_norm, attn_drop, proj_drop, norm_layer) for _ in
             range(self.num_splits)])
        self.norms_1 = nn.ModuleList([norm_layer(dim) for _ in range(self.num_splits)])
        self.cross_attention = CrossAttention(dim, num_heads, qkv_bias, qk_norm, attn_drop, proj_drop, norm_layer)
        self.norm_2 = norm_layer(dim)

    def forward(self, x, split_lengths):
        B, L, D = x.shape
        splits = torch.split(x, split_lengths, dim=1)
        self_attend_outputs = [self_norm(s + self_attn(s)) for self_norm, self_attn, s in
                               zip(self.norms_1, self.self_attentions, splits) if s.size(1) > 0]
        # concatenated_self_attend_output = torch.cat(self_attend_outputs, dim=1)
        cross_attend_outputs = []
        for i, s in enumerate(self_attend_outputs):
            if len(self_attend_outputs) > 1:
                other_indices = list(range(len(self_attend_outputs)))
                other_indices.pop(i)
                others_concatenated = torch.cat([self_attend_outputs[j] for j in other_indices], dim=1)
                cross_attend_output = self.cross_attention.forward(x_q=s, x_kv=others_concatenated)
                cross_attend_outputs.append(cross_attend_output)
            else:
                cross_attend_outputs.append(s)

        # 将交叉注意力的结果拼接起来
        output = torch.cat(cross_attend_outputs, dim=1)
        output = self.norm_2(x + output)

        return output


class Block_u(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(Block_u, self).__init__()
        self.attn = SplitAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm = norm_layer(dim)

    def forward(self, x, split_lengths):
        x = self.attn.forward(x, split_lengths)
        x = x + self.drop_path(self.mlp(self.norm(x)))
        return x


def patchify_3d(volumes, tubelet_size, patch_size):
    """
    :param volume: [N,channel_num,T,H,W]
    :param tubelet_size:
    :param patch_size:
    :return: [N,L,D]
    """
    N, channel_num, frame_length, height, width = volumes.size()
    x = volumes.reshape(shape=(
        N, channel_num, frame_length // tubelet_size, tubelet_size, height // patch_size, patch_size,
        width // patch_size, patch_size))
    x = torch.einsum('ncfthpwq->nfhwctpq', x)
    x = x.reshape(N, (frame_length * height * width) // (tubelet_size * patch_size * patch_size),
                  tubelet_size * patch_size * patch_size * channel_num)
    # x = nn.LayerNorm(tubelet_size * patch_size * patch_size * channel_num, eps=1e-3)(x)
    return x


def unpatchify_3d(seq, frame_length, height, width, tubelet_size, patch_size):
    """
    :param seq: [N,L,D]
    :param tubelet_size:
    :param patch_size:
    :return: volume [N,channel_num,T,H,W]
    """
    N, L, D = seq.size()
    channel_num = D // (tubelet_size * patch_size * patch_size)
    x = seq.reshape(N, frame_length // tubelet_size, height // patch_size, width // patch_size, channel_num,
                    tubelet_size, patch_size, patch_size)
    x = torch.einsum('nfhwctpq->ncfthpwq', x)
    x = x.reshape(N, channel_num, frame_length, height, width)
    return x


def patchify_1d(x, tubelet_size):
    """
    :param x: [N,seq_len,channel_num]
    :param tubelet_size:
    :return: [N,L,D]
    """
    N, seq_len, channel_num = x.size()
    x = x.reshape(N, seq_len // tubelet_size, tubelet_size, channel_num)
    x = x.reshape(N, seq_len // tubelet_size, tubelet_size * channel_num)
    # x = nn.LayerNorm(tubelet_size * channel_num, eps=1e-3)(x)
    return x


def unpatchify_1d(x, tubelet_size):
    """
    :param x: [N,L,D]
    :param tubelet_size:
    :return: [N,seq_len,channel_num]
    """
    N, L, D = x.size()
    channel_num = D // tubelet_size
    seq_len = L * tubelet_size
    x = x.reshape(N, L, tubelet_size, channel_num)
    x = x.reshape(N, seq_len, channel_num)
    return x
