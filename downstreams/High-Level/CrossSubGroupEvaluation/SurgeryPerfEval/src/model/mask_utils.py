import torch
import random


def random_masking_unstructured(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, embed_dim
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)  # noise in [0,1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small means keep, big means remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 means keep, 1 means remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return x_masked, mask, ids_restore


def random_masking_structured(x, mask_ratio, t=64, f=8, mode='time'):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.size()
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)
    assert L == f * t
    noise = noise.reshape(N, f, t)
    if mode == "time":
        for i in range(N):
            mask_t_list = random.sample(range(t), int(t * mask_ratio))
            for k in mask_t_list:
                noise[i, :, k] = 1.1
    elif mode == "freq":
        for i in range(N):
            mask_f_list = random.sample(range(f), int(f * mask_ratio))
            for k in mask_f_list:
                noise[i, k, :] = 1.1
    elif mode == "tf":
        for i in range(N):
            mask_t_list = random.sample(range(t), int(t * mask_ratio * 0.7))
            for k in mask_t_list:
                noise[i, :, k] = 1.1  # large value will be removed
        for i in range(N):
            mask_f_list = random.sample(range(f), int(f * mask_ratio * 0.7))
            for k in mask_f_list:
                noise[i, k, :] = 1.1  # large value will be removed
    noise = noise.reshape(N, L)
    # sort noise for each sample, only need to manuplate these two ids_shuffle, ids_restore
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore


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


def random_chunk_masking_3d(x, mask_ratio, frames, height, width, patch_size, tubelet_size):
    '''
        0 means keep, 1 means remove
        x: (N,L,D)
    '''
    N, L, D = x.size()

    num_chunks_per_tube = frames // tubelet_size
    num_masks_per_tube = int(num_chunks_per_tube * mask_ratio)

    noise = torch.rand(N, num_chunks_per_tube, device=x.device)
    noise = noise.unsqueeze(-1).repeat((1, 1, (height // patch_size) * (width // patch_size)))
    mask_start_indices = torch.randint(low=0, high=num_chunks_per_tube - num_masks_per_tube,
                                       size=(N, 1, (height // patch_size) * (width // patch_size))).repeat(1,
                                                                                                           num_chunks_per_tube,
                                                                                                           1)
    mask_end_indices = mask_start_indices + num_masks_per_tube
    indices = torch.arange(num_chunks_per_tube)[None, :, None].repeat(N, 1,
                                                                      (height // patch_size) * (width // patch_size))
    fill_mask = (indices >= mask_start_indices) & (indices < mask_end_indices)
    noise[fill_mask] = 1.

    noise = noise.flatten(1)
    ids_shffule = torch.argsort(noise, dim=1, stable=True)
    ids_restore = torch.argsort(ids_shffule, dim=1, stable=True)

    ids_keep = ids_shffule[:,
               :(num_chunks_per_tube - num_masks_per_tube) * (height // patch_size) * (width // patch_size)]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    mask = torch.ones([N, L], device=x.device)
    mask[:, :(num_chunks_per_tube - num_masks_per_tube) * (height // patch_size) * (width // patch_size)] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return x_masked, mask, ids_restore


def random_chunk_masking_1d(x, mask_ratio):
    '''
            0 means keep, 1 means remove
            x: (N,L,D)
        '''
    N, L, D = x.size()

    num_chunks_per_tube = L
    num_masks_per_tube = int(num_chunks_per_tube * mask_ratio)

    noise = torch.rand(N, num_chunks_per_tube, device=x.device)
    mask_start_indices = torch.randint(low=0, high=num_chunks_per_tube - num_masks_per_tube,
                                       size=(N, 1)).repeat(1, num_chunks_per_tube)
    mask_end_indices = mask_start_indices + num_masks_per_tube
    indices = torch.arange(num_chunks_per_tube)[None, :].repeat(N, 1)
    fill_mask = (indices >= mask_start_indices) & (indices < mask_end_indices)
    noise[fill_mask] = 1.

    ids_shffule = torch.argsort(noise, dim=1, stable=True)
    ids_restore = torch.argsort(ids_shffule, dim=1, stable=True)

    ids_keep = ids_shffule[:, :(num_chunks_per_tube - num_masks_per_tube)]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    mask = torch.ones([N, L], device=x.device)
    mask[:, :(num_chunks_per_tube - num_masks_per_tube)] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return x_masked, mask, ids_restore
