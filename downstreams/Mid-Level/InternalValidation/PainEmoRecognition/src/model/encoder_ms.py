import torch
import torch.nn as nn
from .pos_embed import get_sinusoid_encoding_table, get_2d_sincos_pos_embed
from .mask_utils import random_tube_masking_1d, random_tube_masking, random_tube_masking_3d
from .model_utils import PatchEmbed_3d_seq, PatchEmbed_1d_seq, Block, PatchEmbed_3d, Block_u
from .model_utils import patchify_3d, patchify_1d, unpatchify_1d, unpatchify_3d
import itertools


def _contrastive_loss_and_accuracy(feat1, feat2, modality_mask1, modality_mask2, bidirect_contrast=True):
    batch_size = feat2.size()[0]
    feat_matrix = torch.mm(feat1, feat2.transpose(0, 1))

    if not bidirect_contrast:
        nce_loss = -torch.sum(
            torch.diag(torch.log_softmax(feat_matrix, dim=0))
            * modality_mask1 * modality_mask2) / torch.clamp(
            (modality_mask1 * modality_mask2).sum(), 1e-6)
        c_acc = torch.sum(
            torch.eq(torch.argmax(torch.log_softmax(feat_matrix, dim=0), dim=0),
                     torch.arange(0, batch_size).cuda()) * modality_mask1 * modality_mask2) / torch.clamp(
            (modality_mask1 * modality_mask2).sum(), 1e-6)
    else:
        nce_loss = -torch.sum(
            torch.diag(torch.log_softmax(feat_matrix, dim=0))
            * modality_mask1 * modality_mask2) / torch.clamp(
            (modality_mask1 * modality_mask2).sum(), 1e-6) - \
                   torch.sum(
                       torch.diag(torch.log_softmax(feat_matrix.t(), dim=0))
                       * modality_mask1 * modality_mask2) / torch.clamp(
            (modality_mask1 * modality_mask2).sum(), 1e-6)
        nce_loss /= 2.
        c_acc = torch.sum(
            torch.eq(torch.argmax(torch.log_softmax(feat_matrix, dim=0), dim=0),
                     torch.arange(0, batch_size).cuda()) * modality_mask1 * modality_mask2) / torch.clamp(
            (modality_mask1 * modality_mask2).sum(), 1e-6) + torch.sum(
            torch.eq(torch.argmax(torch.log_softmax(feat_matrix.t(), dim=0), dim=0),
                     torch.arange(0, batch_size).cuda()) * modality_mask1 * modality_mask2) / torch.clamp(
            (modality_mask1 * modality_mask2).sum(), 1e-6)
        c_acc /= 2.
    return nce_loss, c_acc


def loo_contrastive_loss(video_rep, eeg_rep, ecg_rep, eog_rep, emg_rep, gsr_rep, modality_mask,
                         bidirect_contrast=False):
    video_rep = torch.nn.functional.normalize(video_rep, eps=1e-6, dim=-1)  # (b,embed_dim)
    eeg_rep = torch.nn.functional.normalize(eeg_rep, eps=1e-6, dim=-1)  # (b,embed_dim)
    ecg_rep = torch.nn.functional.normalize(ecg_rep, eps=1e-6, dim=-1)  # (b,embed_dim)
    eog_rep = torch.nn.functional.normalize(eog_rep, eps=1e-6, dim=-1)  # (b,embed_dim)
    emg_rep = torch.nn.functional.normalize(emg_rep, eps=1e-6, dim=-1)  # (b,embed_dim)
    gsr_rep = torch.nn.functional.normalize(gsr_rep, eps=1e-6, dim=-1)  # (b,embed_dim)

    wo_video_rep = modality_mask[:, 1].unsqueeze(1) * eeg_rep + modality_mask[:, 2].unsqueeze(
        1) * ecg_rep + modality_mask[:, 3].unsqueeze(1) * eog_rep + modality_mask[:, 4].unsqueeze(
        1) * emg_rep + modality_mask[:, 5].unsqueeze(1) * gsr_rep
    wo_video_rep = torch.nn.functional.normalize(wo_video_rep, eps=1e-6, dim=-1)

    wo_eeg_rep = modality_mask[:, 0].unsqueeze(1) * video_rep + modality_mask[:, 2].unsqueeze(
        1) * ecg_rep + modality_mask[:, 3].unsqueeze(1) * eog_rep + modality_mask[:, 4].unsqueeze(
        1) * emg_rep + modality_mask[:, 5].unsqueeze(1) * gsr_rep
    wo_eeg_rep = torch.nn.functional.normalize(wo_eeg_rep, eps=1e-6, dim=-1)

    wo_ecg_rep = modality_mask[:, 1].unsqueeze(1) * eeg_rep + modality_mask[:, 0].unsqueeze(
        1) * video_rep + modality_mask[:, 3].unsqueeze(1) * eog_rep + modality_mask[:, 4].unsqueeze(
        1) * emg_rep + modality_mask[:, 5].unsqueeze(1) * gsr_rep
    wo_ecg_rep = torch.nn.functional.normalize(wo_ecg_rep, eps=1e-6, dim=-1)

    wo_eog_rep = modality_mask[:, 1].unsqueeze(1) * eeg_rep + modality_mask[:, 2].unsqueeze(
        1) * ecg_rep + modality_mask[:, 0].unsqueeze(1) * video_rep + modality_mask[:, 4].unsqueeze(
        1) * emg_rep + modality_mask[:, 5].unsqueeze(1) * gsr_rep
    wo_eog_rep = torch.nn.functional.normalize(wo_eog_rep, eps=1e-6, dim=-1)

    wo_emg_rep = modality_mask[:, 1].unsqueeze(1) * eeg_rep + modality_mask[:, 2].unsqueeze(
        1) * ecg_rep + modality_mask[:, 3].unsqueeze(1) * eog_rep + modality_mask[:, 0].unsqueeze(
        1) * video_rep + modality_mask[:, 5].unsqueeze(1) * gsr_rep
    wo_emg_rep = torch.nn.functional.normalize(wo_emg_rep, eps=1e-6, dim=-1)

    wo_gsr_rep = modality_mask[:, 1].unsqueeze(1) * eeg_rep + modality_mask[:, 2].unsqueeze(
        1) * ecg_rep + modality_mask[:, 3].unsqueeze(1) * eog_rep + modality_mask[:, 4].unsqueeze(
        1) * emg_rep + modality_mask[:, 0].unsqueeze(1) * video_rep
    wo_gsr_rep = torch.nn.functional.normalize(wo_gsr_rep, eps=1e-6, dim=-1)

    nce_video_others, c_acc_video_others = _contrastive_loss_and_accuracy(feat1=video_rep, feat2=wo_video_rep,
                                                                          modality_mask1=modality_mask[:, 0],
                                                                          modality_mask2=torch.any(modality_mask[:, 1:],
                                                                                                   dim=1),
                                                                          bidirect_contrast=bidirect_contrast)
    nce_eeg_others, c_acc_eeg_others = _contrastive_loss_and_accuracy(feat1=eeg_rep, feat2=wo_eeg_rep,
                                                                      modality_mask1=modality_mask[:, 1],
                                                                      modality_mask2=torch.any(
                                                                          modality_mask[:, [0, 2, 3, 4, 5]], dim=1),
                                                                      bidirect_contrast=bidirect_contrast)
    nce_ecg_others, c_acc_ecg_others = _contrastive_loss_and_accuracy(feat1=ecg_rep, feat2=wo_ecg_rep,
                                                                      modality_mask1=modality_mask[:, 2],
                                                                      modality_mask2=torch.any(
                                                                          modality_mask[:, [0, 1, 3, 4, 5]], dim=1),
                                                                      bidirect_contrast=bidirect_contrast)
    nce_eog_others, c_acc_eog_others = _contrastive_loss_and_accuracy(feat1=eog_rep, feat2=wo_eog_rep,
                                                                      modality_mask1=modality_mask[:, 3],
                                                                      modality_mask2=torch.any(
                                                                          modality_mask[:, [0, 1, 2, 4, 5]], dim=1),
                                                                      bidirect_contrast=bidirect_contrast)
    nce_emg_others, c_acc_emg_others = _contrastive_loss_and_accuracy(feat1=emg_rep, feat2=wo_emg_rep,
                                                                      modality_mask1=modality_mask[:, 4],
                                                                      modality_mask2=torch.any(
                                                                          modality_mask[:, [0, 1, 2, 3, 5]], dim=1),
                                                                      bidirect_contrast=bidirect_contrast)
    nce_gsr_others, c_acc_gsr_others = _contrastive_loss_and_accuracy(feat1=gsr_rep, feat2=wo_gsr_rep,
                                                                      modality_mask1=modality_mask[:, 5],
                                                                      modality_mask2=torch.any(
                                                                          modality_mask[:, :-1], dim=1),
                                                                      bidirect_contrast=bidirect_contrast)
    nce_loss = (
                       nce_video_others + nce_eeg_others + nce_ecg_others + nce_eog_others + nce_emg_others + nce_gsr_others) / 6.
    c_acc = (
                    c_acc_video_others + c_acc_eeg_others + c_acc_ecg_others + c_acc_eog_others + c_acc_emg_others + c_acc_gsr_others) / 6.
    return nce_loss, c_acc, \
           [nce_video_others, nce_eeg_others, nce_ecg_others, nce_eog_others, nce_emg_others, nce_gsr_others], \
           [c_acc_video_others, c_acc_eeg_others, c_acc_ecg_others, c_acc_eog_others, c_acc_emg_others,
            c_acc_gsr_others]


class MultiMAE_FT(nn.Module):
    """
        Individual Characterization Masked AutoEncoder
        input: modality of [Video, EEG, ECG, EOG, EMG, GSR]
    """

    def __init__(self, opt, norm_layer=nn.LayerNorm):
        super(MultiMAE_FT, self).__init__()
        print('an Individual Characterization MAE with substitute missing modalities model')
        print('Learnable positional embedding:', opt['tr_pos'])
        self.opt = opt
        self.norm_pix_loss = opt['norm_pix_loss']
        self.num_negatives = opt['num_negatives']
        self.temp = opt['temp']

        # ----------------------------------------- encoder part --------------------------------------------
        print('defining encoder part...')
        self.patch_embed_video = PatchEmbed_3d(img_size=(opt['video']['height'], opt['video']['width']),
                                               patch_size=opt['video']['patch_size'], in_chans=opt['video']['channel'],
                                               embed_dim=opt['encoder']['embed_dim'], num_frames=opt['video']['length'],
                                               tubelet_size=opt['video']['tubelet_size'])
        self.patch_embed_eeg = PatchEmbed_3d_seq(img_size=(opt['eeg']['eeg_height'], opt['eeg']['eeg_width']),
                                                 patch_size=opt['eeg']['eeg_patch_size'],
                                                 in_chans=opt['eeg']['eeg_channel'],
                                                 embed_dim=opt['encoder']['embed_dim'],
                                                 num_frames=opt['eeg']['eeg_length'],
                                                 tubelet_size=opt['eeg']['eeg_tubelet_size'])
        self.patch_embed_ecg = PatchEmbed_1d_seq(in_channels=1, tubelet_size=opt['ecg']['ecg_tubelet_size'],
                                                 length=opt['ecg']['ecg_length'] * opt['ecg']['ecg_channel'],
                                                 embed_dim=opt['encoder']['embed_dim'])
        self.patch_embed_eog = PatchEmbed_1d_seq(in_channels=1, tubelet_size=opt['eog']['eog_tubelet_size'],
                                                 length=opt['eog']['eog_length'] * opt['eog']['eog_channel'],
                                                 embed_dim=opt['encoder']['embed_dim'])
        self.patch_embed_emg = PatchEmbed_1d_seq(in_channels=1, tubelet_size=opt['emg']['emg_tubelet_size'],
                                                 length=opt['emg']['emg_length'] * opt['emg']['emg_channel'],
                                                 embed_dim=opt['encoder']['embed_dim'])
        self.patch_embed_gsr = PatchEmbed_1d_seq(in_channels=1, tubelet_size=opt['gsr']['gsr_tubelet_size'],
                                                 length=opt['gsr']['gsr_length'] * opt['gsr']['gsr_channel'],
                                                 embed_dim=opt['encoder']['embed_dim'])

        print('[Video, EEG, ECG, EOG, EMG, GSR]: [%d, %d, %d, %d, %d, %d]' % (
            self.patch_embed_video.num_patches, self.patch_embed_eeg.num_patches,
            self.patch_embed_ecg.num_patches, self.patch_embed_eog.num_patches,
            self.patch_embed_emg.num_patches, self.patch_embed_gsr.num_patches))

        # encoder position embedding
        self.pos_embed_video_spatial = nn.Parameter(
            torch.zeros(1, (opt['video']['height'] * opt['video']['width']) // (opt['video']['patch_size'] ** 2),
                        opt['encoder']['embed_dim']), requires_grad=False)
        self.pos_embed_video_temporal = nn.Parameter(
            torch.zeros(1, opt['video']['length'] // opt['video']['tubelet_size'], opt['encoder']['embed_dim']),
            requires_grad=False)
        self.pos_embed_eeg_spatial = nn.Parameter(
            torch.zeros(1, (opt['eeg']['eeg_height'] * opt['eeg']['eeg_width']) // (opt['eeg']['eeg_patch_size'] ** 2),
                        opt['encoder']['embed_dim']),
            requires_grad=False)
        self.pos_embed_eeg_temporal = nn.Parameter(
            torch.zeros(1, opt['eeg']['eeg_length'] // opt['eeg']['eeg_tubelet_size'], opt['encoder']['embed_dim']),
            requires_grad=False)

        self.pos_embed_ecg_temporal = nn.Parameter(
            torch.zeros(1, opt['ecg']['ecg_length'] // opt['ecg']['ecg_tubelet_size'],
                        opt['encoder']['embed_dim']), requires_grad=False)
        self.pos_embed_ecg_chl = nn.Parameter(
            torch.zeros(1, opt['ecg']['ecg_channel'], opt['encoder']['embed_dim']), requires_grad=False)

        self.pos_embed_eog_temporal = nn.Parameter(
            torch.zeros(1, opt['eog']['eog_length'] // opt['eog']['eog_tubelet_size'],
                        opt['encoder']['embed_dim']), requires_grad=False)
        self.pos_embed_eog_chl = nn.Parameter(
            torch.zeros(1, opt['eog']['eog_channel'], opt['encoder']['embed_dim']), requires_grad=False)

        self.pos_embed_emg_temporal = nn.Parameter(
            torch.zeros(1, opt['emg']['emg_length'] // opt['emg']['emg_tubelet_size'], opt['encoder']['embed_dim']),
            requires_grad=False)
        self.pos_embed_emg_chl = nn.Parameter(torch.zeros(1, opt['emg']['emg_channel'], opt['encoder']['embed_dim']),
                                              requires_grad=False)

        self.pos_embed_gsr = nn.Parameter(torch.zeros(1, self.patch_embed_gsr.num_patches, opt['encoder']['embed_dim']),
                                          requires_grad=False)

        # modality embedding
        self.modality_video = nn.Parameter(torch.zeros(1, 1, opt['encoder']['embed_dim']))
        self.modality_eeg = nn.Parameter(torch.zeros(1, 1, opt['encoder']['embed_dim']))
        self.modality_ecg = nn.Parameter(torch.zeros(1, 1, opt['encoder']['embed_dim']))
        self.modality_eog = nn.Parameter(torch.zeros(1, 1, opt['encoder']['embed_dim']))
        self.modality_emg = nn.Parameter(torch.zeros(1, 1, opt['encoder']['embed_dim']))
        self.modality_gsr = nn.Parameter(torch.zeros(1, 1, opt['encoder']['embed_dim']))

        # modality feature substitute embedding
        self.modality_video_substitute = nn.Parameter(
            torch.randn(1, opt['encoder']['embed_dim']))
        self.modality_eeg_substitute = nn.Parameter(
            torch.randn(1, opt['encoder']['embed_dim']))
        self.modality_ecg_substitute = nn.Parameter(
            torch.randn(1, opt['encoder']['embed_dim']))
        self.modality_eog_substitute = nn.Parameter(
            torch.randn(1, opt['encoder']['embed_dim']))
        self.modality_emg_substitute = nn.Parameter(
            torch.randn(1, opt['encoder']['embed_dim']))
        self.modality_gsr_substitute = nn.Parameter(
            torch.randn(1, opt['encoder']['embed_dim']))

        # video branch
        self.blocks_video = nn.ModuleList(
            [Block(opt['encoder']['embed_dim'], opt['encoder']['num_heads'], opt['mlp_ratio'], qkv_bias=True,
                   norm_layer=norm_layer) for i in
             range(opt['encoder']['modality_specific_depth'])])
        # eeg branch
        self.blocks_eeg = nn.ModuleList(
            [Block(opt['encoder']['embed_dim'], opt['encoder']['num_heads'], opt['mlp_ratio'], qkv_bias=True,
                   norm_layer=norm_layer) for i in
             range(opt['encoder']['modality_specific_depth'])])
        # ecg branch
        self.blocks_ecg = nn.ModuleList(
            [Block(opt['encoder']['embed_dim'], opt['encoder']['num_heads'], opt['mlp_ratio'], qkv_bias=True,
                   norm_layer=norm_layer) for i in
             range(opt['encoder']['modality_specific_depth'])])
        # eog branch
        self.blocks_eog = nn.ModuleList(
            [Block(opt['encoder']['embed_dim'], opt['encoder']['num_heads'], opt['mlp_ratio'], qkv_bias=True,
                   norm_layer=norm_layer) for i in
             range(opt['encoder']['modality_specific_depth'])])
        # emg branch
        self.blocks_emg = nn.ModuleList(
            [Block(opt['encoder']['embed_dim'], opt['encoder']['num_heads'], opt['mlp_ratio'], qkv_bias=True,
                   norm_layer=norm_layer) for i in
             range(opt['encoder']['modality_specific_depth'])])
        # gsr branch
        self.blocks_gsr = nn.ModuleList(
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
        torch.nn.init.normal_(self.modality_video, std=.02)
        torch.nn.init.normal_(self.modality_eeg, std=.02)
        torch.nn.init.normal_(self.modality_ecg, std=.02)
        torch.nn.init.normal_(self.modality_eog, std=.02)
        torch.nn.init.normal_(self.modality_emg, std=.02)
        torch.nn.init.normal_(self.modality_gsr, std=.02)

        # ------------------------------------ encoder pos embed initialization ---------------------------------------
        print('encoder positional embedding initialization ...')
        embed_dim = self.pos_embed_video_spatial.size()[-1]

        if self.patch_embed_video.num_patches > 0:
            pos_embed_video_temporal = get_sinusoid_encoding_table(
                self.opt['video']['length'] // self.opt['video']['tubelet_size'],
                embed_dim)
            # print(pos_embed_video_temporal.size(),self.pos_embed_video_temporal.size())
            self.pos_embed_video_temporal.data.copy_(pos_embed_video_temporal.float())
            pos_embed_video_spatial = get_2d_sincos_pos_embed(
                embed_dim=embed_dim, grid_h_size=self.opt['video']['height'] // self.opt['video']['patch_size'],
                grid_w_size=self.opt['video']['width'] // self.opt['video']['patch_size'])
            self.pos_embed_video_spatial.data.copy_(pos_embed_video_spatial.float())

        if self.patch_embed_eeg.num_patches > 0:
            pos_embed_eeg_temporal = get_sinusoid_encoding_table(
                self.opt['eeg']['eeg_length'] // self.opt['eeg']['eeg_tubelet_size'],
                embed_dim)
            self.pos_embed_eeg_temporal.data.copy_(pos_embed_eeg_temporal.float())
            pos_embed_eeg_spatial = get_2d_sincos_pos_embed(
                embed_dim=embed_dim, grid_h_size=self.opt['eeg']['eeg_height'] // self.opt['eeg']['eeg_patch_size'],
                grid_w_size=self.opt['eeg']['eeg_width'] // self.opt['eeg']['eeg_patch_size'])
            self.pos_embed_eeg_spatial.data.copy_(pos_embed_eeg_spatial.float())

        if self.patch_embed_ecg.num_patches > 0:
            pos_embed_ecg_temporal = get_sinusoid_encoding_table(
                self.opt['ecg']['ecg_length'] // self.opt['ecg']['ecg_tubelet_size'],
                embed_dim)
            self.pos_embed_ecg_temporal.data.copy_(pos_embed_ecg_temporal.float())
            pos_embed_ecg_chl = get_sinusoid_encoding_table(self.opt['ecg']['ecg_channel'], embed_dim)
            self.pos_embed_ecg_chl.data.copy_(pos_embed_ecg_chl.float())

        if self.patch_embed_eog.num_patches > 0:
            pos_embed_eog_temporal = get_sinusoid_encoding_table(
                self.opt['eog']['eog_length'] // self.opt['eog']['eog_tubelet_size'],
                embed_dim)
            self.pos_embed_eog_temporal.data.copy_(pos_embed_eog_temporal.float())
            pos_embed_eog_chl = get_sinusoid_encoding_table(self.opt['eog']['eog_channel'], embed_dim)
            self.pos_embed_eog_chl.data.copy_(pos_embed_eog_chl.float())

        if self.patch_embed_emg.num_patches > 0:
            pos_embed_emg_temporal = get_sinusoid_encoding_table(
                self.opt['emg']['emg_length'] // self.opt['emg']['emg_tubelet_size'],
                embed_dim)
            self.pos_embed_emg_temporal.data.copy_(pos_embed_emg_temporal.float())
            pos_embed_emg_chl = get_sinusoid_encoding_table(self.opt['emg']['emg_channel'], embed_dim)
            self.pos_embed_emg_chl.data.copy_(pos_embed_emg_chl.float())

        if self.patch_embed_gsr.num_patches > 0:
            pos_embed_gsr = get_sinusoid_encoding_table(self.patch_embed_gsr.num_patches,
                                                        embed_dim)
            self.pos_embed_gsr.data.copy_(pos_embed_gsr.float())

    def forward(self, video, eeg, ecg, eog, emg, gsr, modality_mask):
        # embed patches

        video = self.patch_embed_video(video)  # (b,c,t,h,w)-->(b,l,d)
        if self.patch_embed_video.num_patches > 0:
            pos_embed_video = self.pos_embed_video_spatial.unsqueeze(1).repeat(1, self.opt['video']['length'] //
                                                                               self.opt['video']['tubelet_size'], 1,
                                                                               1).reshape(1, video.size()[1],
                                                                                          -1) + self.pos_embed_video_temporal.unsqueeze(
                2).repeat(1, 1, (self.opt['video']['height'] * self.opt['video']['width']) // (
                    self.opt['video']['patch_size'] ** 2), 1).reshape(1, video.size()[1], -1)
            video = video + pos_embed_video + self.modality_video

        eeg = self.patch_embed_eeg(eeg)  # (b,c,t,h,w)-->(b,l,d)
        if self.patch_embed_eeg.num_patches > 0:
            pos_embed_eeg = self.pos_embed_eeg_spatial.unsqueeze(1).repeat(1, self.opt['eeg']['eeg_length'] //
                                                                           self.opt['eeg']['eeg_tubelet_size'], 1,
                                                                           1).reshape(1, eeg.size()[1],
                                                                                      -1) + self.pos_embed_eeg_temporal.unsqueeze(
                2).repeat(1, 1, (self.opt['eeg']['eeg_height'] * self.opt['eeg']['eeg_width']) // (
                    self.opt['eeg']['eeg_patch_size'] ** 2), 1).reshape(1, eeg.size()[1], -1)
            eeg = eeg + pos_embed_eeg + self.modality_eeg

        ecg = torch.einsum('blc->bcl', ecg).flatten(1).unsqueeze(2)  # (b,l,2)-->(b,2l,1)
        ecg = self.patch_embed_ecg(ecg)  # (b,l,c)-->(b,l,d)
        if self.patch_embed_ecg.num_patches > 0:
            pos_embed_ecg = self.pos_embed_ecg_temporal.unsqueeze(1).repeat(1, self.opt['ecg']['ecg_channel'], 1,
                                                                            1).reshape(1, ecg.size()[1], -1) + \
                            self.pos_embed_ecg_chl.unsqueeze(2).repeat(1, 1,
                                                                       ecg.size()[1] // self.opt['ecg']['ecg_channel'],
                                                                       1).reshape(1, ecg.size()[1], -1)
            ecg = ecg + pos_embed_ecg + self.modality_ecg

        eog = torch.einsum('blc->bcl', eog).flatten(1).unsqueeze(2)  # (b,l,2)->(b,2l,1)
        eog = self.patch_embed_eog(eog)  # (b,l,c)-->(b,l,d)
        if self.patch_embed_eog.num_patches > 0:
            pos_embed_eog = self.pos_embed_eog_temporal.unsqueeze(1).repeat(1, self.opt['eog']['eog_channel'], 1,
                                                                            1).reshape(1, eog.size()[1], -1) + \
                            self.pos_embed_eog_chl.unsqueeze(2).repeat(1, 1,
                                                                       eog.size()[1] // self.opt['eog']['eog_channel'],
                                                                       1).reshape(1, eog.size()[1], -1)
            eog = eog + pos_embed_eog + self.modality_eog

        # print('origin emg:\n %s\n' % str(torch.isnan(emg.sum())))
        emg = torch.einsum('blc->bcl', emg).flatten(1).unsqueeze(2)  # (b,l,3)->(b,3l)->(b,3l,1)
        emg = self.patch_embed_emg(emg)  # (b,l,c)-->(b,l,d)
        if self.patch_embed_emg.num_patches > 0:
            pos_embed_emg = self.pos_embed_emg_temporal.unsqueeze(1).repeat(1, self.opt['emg']['emg_channel'], 1,
                                                                            1).reshape(1, emg.size()[1], -1) + \
                            self.pos_embed_emg_chl.unsqueeze(2).repeat(1, 1,
                                                                       emg.size()[1] // self.opt['emg']['emg_channel'],
                                                                       1).reshape(1, emg.size()[1], -1)
            emg = emg + pos_embed_emg + self.modality_emg

        gsr = self.patch_embed_gsr(gsr)  # (b,l,c)-->(b,l,d)
        if self.patch_embed_gsr.num_patches > 0:
            gsr = gsr + self.pos_embed_gsr + self.modality_gsr

        # modality-specific encoding
        for blk in self.blocks_video:
            video = blk(video)
        for blk in self.blocks_eeg:
            eeg = blk(eeg)
        for blk in self.blocks_ecg:
            ecg = blk(ecg)
        for blk in self.blocks_eog:
            eog = blk(eog)
        for blk in self.blocks_emg:
            emg = blk(emg)
        for blk in self.blocks_gsr:
            gsr = blk(gsr)

        # joint stream
        lengths = [int(self.patch_embed_video.num_patches),
                   int(self.patch_embed_eeg.num_patches),
                   int(self.patch_embed_ecg.num_patches),
                   int(self.patch_embed_eog.num_patches),
                   int(self.patch_embed_emg.num_patches),
                   int(self.patch_embed_gsr.num_patches)]
        x = torch.cat((video, eeg, ecg, eog, emg, gsr), dim=1)
        for i, blk in enumerate(self.blocks_u):
            x = blk(x, split_lengths=lengths)

        valid_feat = torch.mean(x, dim=1)

        index_count = 0
        x_video = x[:, index_count:index_count + lengths[0], :]
        index_count += lengths[0]
        x_eeg = x[:, index_count:index_count + lengths[1], :]
        index_count += lengths[1]
        x_ecg = x[:, index_count:index_count + lengths[2], :]
        index_count += lengths[2]
        x_eog = x[:, index_count:index_count + lengths[3], :]
        index_count += lengths[3]
        x_emg = x[:, index_count:index_count + lengths[4], :]
        index_count += lengths[4]
        x_gsr = x[:, index_count:, :]

        '''
        bs=x_video.size(0)
        inpainted_feat = (self.modality_video_substitute*(1-modality_mask[:,0].unsqueeze(1))+
            self.modality_eeg_substitute*(1-modality_mask[:,1].unsqueeze(1))+
            self.modality_ecg_substitute*(1-modality_mask[:,2].unsqueeze(1))+
            self.modality_eog_substitute*(1-modality_mask[:,3].unsqueeze(1))+
            self.modality_emg_substitute*(1-modality_mask[:,4].unsqueeze(1))+
            self.modality_gsr_substitute*(1-modality_mask[:,5].unsqueeze(1)))/torch.clamp(
                torch.sum(1-modality_mask,dim=1,keepdims=True),1e-5)
        inpainted_feat=valid_feat+inpainted_feat
        '''
        return valid_feat, torch.mean(x_video, dim=1), torch.mean(x_eeg, dim=1), torch.mean(x_ecg, dim=1), torch.mean(
            x_eog, dim=1), torch.mean(x_emg, dim=1), torch.mean(x_gsr, dim=1), \
               [self.modality_video_substitute, self.modality_eeg_substitute, self.modality_ecg_substitute,
                self.modality_eog_substitute, self.modality_emg_substitute, self.modality_gsr_substitute]


class Classifier(nn.Module):
    def __init__(self, num_classes, embed_dim):
        super(Classifier, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class Regressor(nn.Module):
    def __init__(self, embed_dim):
        super(Regressor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.mlp(x)
        return x
