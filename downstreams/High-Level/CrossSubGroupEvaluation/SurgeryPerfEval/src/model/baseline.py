import torch
import torch.nn as nn
from .pos_embed import get_sinusoid_encoding_table, get_2d_sincos_pos_embed
from .mask_utils import random_tube_masking_1d, random_tube_masking, random_tube_masking_3d
from .model_utils import PatchEmbed_3d_seq, PatchEmbed_1d_seq, Block, PatchEmbed_3d, Block_u
from .model_utils import patchify_3d, patchify_1d, unpatchify_1d, unpatchify_3d
import itertools


class Baseline(nn.Module):
    """
        Individual Characterization Masked AutoEncoder
        input: modality of [Video, EEG, ECG, EOG, EMG, GSR]
    """

    def __init__(self, opt, norm_layer=nn.LayerNorm):
        super(Baseline, self).__init__()
        print('an Individual Characterization MAE with substitute missing modalities model')
        print('Learnable positional embedding:', opt['tr_pos'])
        self.opt = opt
        self.norm_pix_loss = opt['norm_pix_loss']
        self.num_negatives = opt['num_negatives']
        self.temp = opt['temp']

        # ----------------------------------------- encoder part --------------------------------------------
        print('defining encoder part...')
        self.patch_embed_eeg = PatchEmbed_3d_seq(img_size=(opt['eeg']['eeg_height'], opt['eeg']['eeg_width']),
                                                 patch_size=opt['eeg']['eeg_patch_size'],
                                                 in_chans=opt['eeg']['eeg_channel'],
                                                 embed_dim=opt['encoder']['embed_dim'],
                                                 num_frames=opt['eeg']['eeg_length'],
                                                 tubelet_size=opt['eeg']['eeg_tubelet_size'])
        self.patch_embed_eog = PatchEmbed_1d_seq(in_channels=1, tubelet_size=opt['eog']['eog_tubelet_size'],
                                                 length=opt['eog']['eog_length'] * opt['eog']['eog_channel'],
                                                 embed_dim=opt['encoder']['embed_dim'])

        print('[EEG, EOG]: [%d, %d]' % (self.patch_embed_eeg.num_patches,self.patch_embed_eog.num_patches))

        # encoder position embedding
        self.pos_embed_eeg_spatial = nn.Parameter(
            torch.zeros(1, (opt['eeg']['eeg_height'] * opt['eeg']['eeg_width']) // (opt['eeg']['eeg_patch_size'] ** 2),
                        opt['encoder']['embed_dim']),
            requires_grad=False)
        self.pos_embed_eeg_temporal = nn.Parameter(
            torch.zeros(1, opt['eeg']['eeg_length'] // opt['eeg']['eeg_tubelet_size'], opt['encoder']['embed_dim']),
            requires_grad=False)

        self.pos_embed_eog_temporal = nn.Parameter(
            torch.zeros(1, opt['eog']['eog_length'] // opt['eog']['eog_tubelet_size'],
                        opt['encoder']['embed_dim']), requires_grad=False)
        self.pos_embed_eog_chl = nn.Parameter(
            torch.zeros(1, opt['eog']['eog_channel'], opt['encoder']['embed_dim']), requires_grad=False)

        # modality embedding
        self.modality_eeg = nn.Parameter(torch.zeros(1, 1, opt['encoder']['embed_dim']))
        self.modality_eog = nn.Parameter(torch.zeros(1, 1, opt['encoder']['embed_dim']))

        # modality feature substitute embedding
        self.modality_eeg_substitute = nn.Parameter(
            torch.randn(1, opt['encoder']['embed_dim']))
        self.modality_eog_substitute = nn.Parameter(
            torch.randn(1, opt['encoder']['embed_dim']))

        # eeg branch
        self.blocks_eeg = nn.ModuleList(
            [Block(opt['encoder']['embed_dim'], opt['encoder']['num_heads'], opt['mlp_ratio'], qkv_bias=True,
                   norm_layer=norm_layer) for i in
             range(opt['encoder']['modality_specific_depth'])])
        # eog branch
        self.blocks_eog = nn.ModuleList(
            [Block(opt['encoder']['embed_dim'], opt['encoder']['num_heads'], opt['mlp_ratio'], qkv_bias=True,
                   norm_layer=norm_layer) for i in
             range(opt['encoder']['modality_specific_depth'])])

        # unified branch
        self.blocks_u = nn.ModuleList(
            [Block_u(dim=opt['encoder']['embed_dim'],
                     num_heads=opt['encoder']['num_heads'], mlp_ratio=opt['mlp_ratio'], qkv_bias=True,
                     norm_layer=norm_layer) for i in
             range(opt['encoder']['encoder_depth'] - opt['encoder']['modality_specific_depth'])])

        # --------------------------------------------------------------------------------------------------------------
        print('initializing weights...')
        self.initialize_weights()
        print('initializing finished!')

    def initialize_weights(self):
        # ---------------------------------- modality / mask embedding initialization ----------------------------
        print('initializing modalisty/mask embedding...')
        torch.nn.init.normal_(self.modality_eeg, std=.02)
        torch.nn.init.normal_(self.modality_eog, std=.02)

        # ------------------------------------ encoder pos embed initialization ---------------------------------------
        print('encoder positional embedding initialization ...')
        embed_dim = self.pos_embed_eeg_spatial.size()[-1]

        if self.patch_embed_eeg.num_patches > 0:
            pos_embed_eeg_temporal = get_sinusoid_encoding_table(
                self.opt['eeg']['eeg_length'] // self.opt['eeg']['eeg_tubelet_size'],
                embed_dim)
            self.pos_embed_eeg_temporal.data.copy_(pos_embed_eeg_temporal.float())
            pos_embed_eeg_spatial = get_2d_sincos_pos_embed(
                embed_dim=embed_dim, grid_h_size=self.opt['eeg']['eeg_height'] // self.opt['eeg']['eeg_patch_size'],
                grid_w_size=self.opt['eeg']['eeg_width'] // self.opt['eeg']['eeg_patch_size'])
            self.pos_embed_eeg_spatial.data.copy_(pos_embed_eeg_spatial.float())

        if self.patch_embed_eog.num_patches > 0:
            pos_embed_eog_temporal = get_sinusoid_encoding_table(
                self.opt['eog']['eog_length'] // self.opt['eog']['eog_tubelet_size'],
                embed_dim)
            self.pos_embed_eog_temporal.data.copy_(pos_embed_eog_temporal.float())
            pos_embed_eog_chl = get_sinusoid_encoding_table(self.opt['eog']['eog_channel'], embed_dim)
            self.pos_embed_eog_chl.data.copy_(pos_embed_eog_chl.float())

    def forward(self, video, eeg, ecg, eog, emg, gsr, modality_mask):
        # embed patches
        eeg = self.patch_embed_eeg(eeg)  # (b,c,t,h,w)-->(b,l,d)
        if self.patch_embed_eeg.num_patches > 0:
            pos_embed_eeg = self.pos_embed_eeg_spatial.unsqueeze(1).repeat(1, self.opt['eeg']['eeg_length'] //
                                                                           self.opt['eeg']['eeg_tubelet_size'], 1,
                                                                           1).reshape(1, eeg.size()[1],
                                                                                      -1) + self.pos_embed_eeg_temporal.unsqueeze(
                2).repeat(1, 1, (self.opt['eeg']['eeg_height'] * self.opt['eeg']['eeg_width']) // (
                    self.opt['eeg']['eeg_patch_size'] ** 2), 1).reshape(1, eeg.size()[1], -1)
            eeg = eeg + pos_embed_eeg + self.modality_eeg

        eog = torch.einsum('blc->bcl', eog).flatten(1).unsqueeze(2)  # (b,l,2)->(b,2l,1)
        eog = self.patch_embed_eog(eog)  # (b,l,c)-->(b,l,d)
        if self.patch_embed_eog.num_patches > 0:
            pos_embed_eog = self.pos_embed_eog_temporal.unsqueeze(1).repeat(1, self.opt['eog']['eog_channel'], 1,
                                                                            1).reshape(1, eog.size()[1], -1) + \
                            self.pos_embed_eog_chl.unsqueeze(2).repeat(1, 1,
                                                                       eog.size()[1] // self.opt['ecg']['ecg_channel'],
                                                                       1).reshape(1, eog.size()[1], -1)
            eog = eog + pos_embed_eog + self.modality_eog

        # modality-specific encoding
        for blk in self.blocks_eeg:
            eeg = blk(eeg)
        for blk in self.blocks_eog:
            eog = blk(eog)

        # joint stream
        lengths = [int(self.patch_embed_eeg.num_patches),
                   int(self.patch_embed_eog.num_patches)]
        x = torch.cat((eeg, eog), dim=1)
        for i, blk in enumerate(self.blocks_u):
            x = blk(x, split_lengths=lengths)

        valid_feat = torch.mean(x, dim=1)

        index_count = 0
        x_eeg = x[:, index_count:index_count + lengths[0], :]
        index_count += lengths[0]
        x_eog = x[:, index_count:, :]

        eeg_feat = modality_mask[:, 1].unsqueeze(1) * torch.mean(x_eeg, dim=1) + (
                1 - modality_mask[:, 1].unsqueeze(1)) * self.modality_eeg_substitute
        eog_feat = modality_mask[:, 3].unsqueeze(1) * torch.mean(x_eog, dim=1) + (
                1 - modality_mask[:, 3].unsqueeze(1)) * self.modality_eog_substitute

        inpainted_feat = (eeg_feat + eog_feat ) / 6.
        return valid_feat, inpainted_feat, eeg_feat,eeg_feat, eog_feat, eog_feat, eog_feat, eog_feat
