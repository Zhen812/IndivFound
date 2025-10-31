import torch
import random


def random_tube_masking(x, mask_ratio, frames, height, width, patch_size, tubelet_size):
    '''
    0 means keep, 1 means remove
    x: (N,L,D)
    '''
    N, L, D = x.size()

    num_patches_per_frame = (height // patch_size) * (width // patch_size)
    num_masks_per_frame = int(num_patches_per_frame * mask_ratio)

    noise = torch.rand(N, num_patches_per_frame, device=x.device)
    noise = noise.repeat((1, frames // tubelet_size))
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :(frames // tubelet_size) * (num_patches_per_frame - num_masks_per_frame)]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    mask = torch.ones([N, L], device=x.device)
    mask[:, :(frames // tubelet_size) * (num_patches_per_frame - num_masks_per_frame)] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return x_masked, mask, ids_restore


def random_tube_masking_3d(x, mask_ratio, frames, height, width, patch_size, tubelet_size):
    '''
    0 means keep, 1 means remove
    x: (N,L,D)
    '''
    N, L, D = x.size()

    num_tubes_per_channel = frames // tubelet_size
    num_masks_per_channel = int(num_tubes_per_channel * mask_ratio)

    noise = torch.rand(N, num_tubes_per_channel, device=x.device).unsqueeze(2)
    noise = noise.repeat((1, 1, (height * width) // patch_size ** 2)).flatten(1)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :((height * width) // patch_size ** 2) * (num_tubes_per_channel - num_masks_per_channel)]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    mask = torch.ones([N, L], device=x.device)
    mask[:, :((height * width) // patch_size ** 2) * (num_tubes_per_channel - num_masks_per_channel)] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return x_masked, mask, ids_restore


def random_tube_masking_1d(x, mask_ratio, length, channel_num, tubelet_size):
    '''
        0 means keep, 1 means remove
        x: (N,channel_num * length,D)
    '''
    N, L, D = x.size()

    num_tubelet_per_channel = length // tubelet_size
    num_masks_per_channel = int(num_tubelet_per_channel * mask_ratio)

    noise = torch.rand(N, num_tubelet_per_channel, device=x.device)
    noise = noise.repeat(1, channel_num)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :channel_num * (num_tubelet_per_channel - num_masks_per_channel)]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    mask = torch.ones([N, L], device=x.device)
    mask[:, :channel_num * (num_tubelet_per_channel - num_masks_per_channel)] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return x_masked, mask, ids_restore


