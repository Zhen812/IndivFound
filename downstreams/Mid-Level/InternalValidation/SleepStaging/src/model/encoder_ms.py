import torch
import torch.nn as nn
from .pos_embed import get_sinusoid_encoding_table, get_2d_sincos_pos_embed
from .mask_utils import random_tube_masking_1d, random_tube_masking, random_tube_masking_3d
from .model_utils import PatchEmbed_3d_seq, PatchEmbed_1d_seq, Block, PatchEmbed_3d, Block_u
from .model_utils import patchify_3d, patchify_1d, unpatchify_1d, unpatchify_3d
import itertools


# torch.autograd.set_detect_anomaly(True)

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


def forward_contrastive_loss(video_rep, eeg_rep, ecg_rep, eog_rep, emg_rep, gsr_rep, modality_mask,
                             bidirect_contrast=True):
    video_rep = torch.nn.functional.normalize(video_rep, eps=1e-6, dim=-1)  # (b,embed_dim)
    eeg_rep = torch.nn.functional.normalize(eeg_rep, eps=1e-6, dim=-1)  # (b,embed_dim)
    ecg_rep = torch.nn.functional.normalize(ecg_rep, eps=1e-6, dim=-1)  # (b,embed_dim)
    eog_rep = torch.nn.functional.normalize(eog_rep, eps=1e-6, dim=-1)  # (b,embed_dim)
    emg_rep = torch.nn.functional.normalize(emg_rep, eps=1e-6, dim=-1)  # (b,embed_dim)
    gsr_rep = torch.nn.functional.normalize(gsr_rep, eps=1e-6, dim=-1)  # (b,embed_dim)

    nce_video_eeg, c_acc_video_eeg = _contrastive_loss_and_accuracy(feat1=video_rep, feat2=eeg_rep,
                                                                    modality_mask1=modality_mask[:, 0],
                                                                    modality_mask2=modality_mask[:, 1],
                                                                    bidirect_contrast=bidirect_contrast)
    nce_video_ecg, c_acc_video_ecg = _contrastive_loss_and_accuracy(feat1=video_rep, feat2=ecg_rep,
                                                                    modality_mask1=modality_mask[:, 0],
                                                                    modality_mask2=modality_mask[:, 2],
                                                                    bidirect_contrast=bidirect_contrast)
    nce_video_eog, c_acc_video_eog = _contrastive_loss_and_accuracy(feat1=video_rep, feat2=eog_rep,
                                                                    modality_mask1=modality_mask[:, 0],
                                                                    modality_mask2=modality_mask[:, 3],
                                                                    bidirect_contrast=bidirect_contrast)
    nce_video_emg, c_acc_video_emg = _contrastive_loss_and_accuracy(feat1=video_rep, feat2=emg_rep,
                                                                    modality_mask1=modality_mask[:, 0],
                                                                    modality_mask2=modality_mask[:, 4],
                                                                    bidirect_contrast=bidirect_contrast)
    nce_video_gsr, c_acc_video_gsr = _contrastive_loss_and_accuracy(feat1=video_rep, feat2=gsr_rep,
                                                                    modality_mask1=modality_mask[:, 0],
                                                                    modality_mask2=modality_mask[:, 5],
                                                                    bidirect_contrast=bidirect_contrast)

    nce_eeg_ecg, c_acc_eeg_ecg = _contrastive_loss_and_accuracy(feat1=eeg_rep, feat2=ecg_rep,
                                                                modality_mask1=modality_mask[:, 1],
                                                                modality_mask2=modality_mask[:, 2],
                                                                bidirect_contrast=bidirect_contrast)
    nce_eeg_eog, c_acc_eeg_eog = _contrastive_loss_and_accuracy(feat1=eeg_rep, feat2=eog_rep,
                                                                modality_mask1=modality_mask[:, 1],
                                                                modality_mask2=modality_mask[:, 3],
                                                                bidirect_contrast=bidirect_contrast)
    nce_eeg_emg, c_acc_eeg_emg = _contrastive_loss_and_accuracy(feat1=eeg_rep, feat2=emg_rep,
                                                                modality_mask1=modality_mask[:, 1],
                                                                modality_mask2=modality_mask[:, 4],
                                                                bidirect_contrast=bidirect_contrast)
    nce_eeg_gsr, c_acc_eeg_gsr = _contrastive_loss_and_accuracy(feat1=eeg_rep, feat2=gsr_rep,
                                                                modality_mask1=modality_mask[:, 1],
                                                                modality_mask2=modality_mask[:, 5],
                                                                bidirect_contrast=bidirect_contrast)

    nce_ecg_eog, c_acc_ecg_eog = _contrastive_loss_and_accuracy(feat1=ecg_rep, feat2=eog_rep,
                                                                modality_mask1=modality_mask[:, 2],
                                                                modality_mask2=modality_mask[:, 3],
                                                                bidirect_contrast=bidirect_contrast)
    nce_ecg_emg, c_acc_ecg_emg = _contrastive_loss_and_accuracy(feat1=ecg_rep, feat2=emg_rep,
                                                                modality_mask1=modality_mask[:, 2],
                                                                modality_mask2=modality_mask[:, 4],
                                                                bidirect_contrast=bidirect_contrast)
    nce_ecg_gsr, c_acc_ecg_gsr = _contrastive_loss_and_accuracy(feat1=ecg_rep, feat2=gsr_rep,
                                                                modality_mask1=modality_mask[:, 2],
                                                                modality_mask2=modality_mask[:, 5],
                                                                bidirect_contrast=bidirect_contrast)

    nce_eog_emg, c_acc_eog_emg = _contrastive_loss_and_accuracy(feat1=eog_rep, feat2=emg_rep,
                                                                modality_mask1=modality_mask[:, 3],
                                                                modality_mask2=modality_mask[:, 4],
                                                                bidirect_contrast=bidirect_contrast)
    nce_eog_gsr, c_acc_eog_gsr = _contrastive_loss_and_accuracy(feat1=eog_rep, feat2=gsr_rep,
                                                                modality_mask1=modality_mask[:, 3],
                                                                modality_mask2=modality_mask[:, 5],
                                                                bidirect_contrast=bidirect_contrast)

    nce_emg_gsr, c_acc_emg_gsr = _contrastive_loss_and_accuracy(feat1=emg_rep, feat2=gsr_rep,
                                                                modality_mask1=modality_mask[:, 4],
                                                                modality_mask2=modality_mask[:, 5],
                                                                bidirect_contrast=bidirect_contrast)
    nce_loss = (nce_video_eeg + nce_video_ecg + nce_video_eog + nce_video_emg + nce_video_gsr + nce_eeg_ecg +
                nce_ecg_eog + nce_eeg_emg + nce_eeg_gsr + nce_ecg_emg + nce_ecg_eog + nce_ecg_gsr +
                nce_eog_emg + nce_eog_gsr + nce_emg_gsr) / 15.
    c_acc = (c_acc_video_eeg + c_acc_video_ecg + c_acc_video_eog + c_acc_video_emg + c_acc_video_gsr +
             c_acc_eeg_ecg + c_acc_eeg_eog + c_acc_eeg_emg + c_acc_eeg_gsr + c_acc_ecg_emg + c_acc_ecg_eog +
             c_acc_ecg_gsr + c_acc_eog_emg + c_acc_eog_gsr + c_acc_emg_gsr) / 15.
    return nce_loss, c_acc, \
           [nce_video_eeg, nce_video_ecg, nce_video_eog, nce_video_emg, nce_video_gsr, nce_eeg_ecg,
            nce_eeg_ecg, nce_eeg_eog, nce_eeg_emg, nce_eeg_gsr, nce_ecg_eog, nce_ecg_emg, nce_ecg_gsr,
            nce_eog_emg, nce_eog_gsr, nce_emg_gsr], \
           [c_acc_video_eeg, c_acc_video_ecg, c_acc_video_eog, c_acc_video_emg, c_acc_video_gsr,
            c_acc_eeg_ecg,
            c_acc_eeg_eog, c_acc_eeg_emg, c_acc_eeg_gsr, c_acc_ecg_eog, c_acc_ecg_emg, c_acc_ecg_gsr,
            c_acc_eog_emg, c_acc_eog_gsr, c_acc_emg_gsr]


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


class MultiMAE(nn.Module):
    """
        Individual Characterization Masked AutoEncoder
        input: modality of [Video, EEG, ECG, EOG, EMG, GSR]
    """

    def __init__(self, opt, norm_layer=nn.LayerNorm):
        super(MultiMAE, self).__init__()
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

        # ----------------------------------------- decoder part --------------------------------------------
        print('defining decoder part...')
        # Project to lower dimension for the decoder
        self.decoder_embed = nn.Linear(opt['encoder']['embed_dim'], opt['decoder']['decoder_embed_dim'], bias=True)

        # token used for masking
        self.mask_video_token = nn.Parameter(torch.zeros(1, 1, opt['decoder']['decoder_embed_dim']))
        self.mask_eeg_token = nn.Parameter(torch.zeros(1, 1, opt['decoder']['decoder_embed_dim']))
        self.mask_ecg_token = nn.Parameter(torch.zeros(1, 1, opt['decoder']['decoder_embed_dim']))
        self.mask_eog_token = nn.Parameter(torch.zeros(1, 1, opt['decoder']['decoder_embed_dim']))
        self.mask_emg_token = nn.Parameter(torch.zeros(1, 1, opt['decoder']['decoder_embed_dim']))
        self.mask_gsr_token = nn.Parameter(torch.zeros(1, 1, opt['decoder']['decoder_embed_dim']))

        # decoder pos embed
        self.decoder_pos_embed_video_spatial = nn.Parameter(
            torch.zeros(1, (opt['video']['height'] * opt['video']['width']) // (opt['video']['patch_size'] ** 2),
                        opt['decoder']['decoder_embed_dim']), requires_grad=False)
        self.decoder_pos_embed_video_temporal = nn.Parameter(
            torch.zeros(1, opt['video']['length'] // opt['video']['tubelet_size'], opt['decoder']['decoder_embed_dim']),
            requires_grad=False)

        self.decoder_pos_embed_eeg_spatial = nn.Parameter(
            torch.zeros(1, (opt['eeg']['eeg_height'] * opt['eeg']['eeg_width']) // (opt['eeg']['eeg_patch_size'] ** 2),
                        opt['decoder']['decoder_embed_dim']),
            requires_grad=False)
        self.decoder_pos_embed_eeg_temporal = nn.Parameter(
            torch.zeros(1, opt['eeg']['eeg_length'] // opt['eeg']['eeg_tubelet_size'],
                        opt['decoder']['decoder_embed_dim']),
            requires_grad=False)

        self.decoder_pos_embed_ecg_temporal = nn.Parameter(
            torch.zeros(1, opt['ecg']['ecg_length'] // opt['ecg']['ecg_tubelet_size'],
                        opt['decoder']['decoder_embed_dim']), requires_grad=False)
        self.decoder_pos_embed_ecg_chl = nn.Parameter(
            torch.zeros(1, opt['ecg']['ecg_channel'], opt['decoder']['decoder_embed_dim']), requires_grad=False)

        self.decoder_pos_embed_eog_temporal = nn.Parameter(
            torch.zeros(1, opt['eog']['eog_length'] // opt['eog']['eog_tubelet_size'],
                        opt['decoder']['decoder_embed_dim']), requires_grad=False)
        self.decoder_pos_embed_eog_chl = nn.Parameter(
            torch.zeros(1, opt['eog']['eog_channel'], opt['decoder']['decoder_embed_dim']), requires_grad=False)

        self.decoder_pos_embed_emg_temporal = nn.Parameter(
            torch.zeros(1, opt['emg']['emg_length'] // opt['emg']['emg_tubelet_size'],
                        opt['decoder']['decoder_embed_dim']), requires_grad=False)
        self.decoder_pos_embed_emg_chl = nn.Parameter(
            torch.zeros(1, opt['emg']['emg_channel'], opt['decoder']['decoder_embed_dim']), requires_grad=False)
        self.decoder_pos_embed_gsr = nn.Parameter(
            torch.zeros(1, self.patch_embed_gsr.num_patches, opt['decoder']['decoder_embed_dim']), requires_grad=False)

        # decoder modality embedding
        self.decoder_modality_video = nn.Parameter(torch.zeros(1, 1, opt['decoder']['decoder_embed_dim']))
        self.decoder_modality_eeg = nn.Parameter(torch.zeros(1, 1, opt['decoder']['decoder_embed_dim']))
        self.decoder_modality_ecg = nn.Parameter(torch.zeros(1, 1, opt['decoder']['decoder_embed_dim']))
        self.decoder_modality_eog = nn.Parameter(torch.zeros(1, 1, opt['decoder']['decoder_embed_dim']))
        self.decoder_modality_emg = nn.Parameter(torch.zeros(1, 1, opt['decoder']['decoder_embed_dim']))
        self.decoder_modality_gsr = nn.Parameter(torch.zeros(1, 1, opt['decoder']['decoder_embed_dim']))

        # decoder block
        self.decoder_blocks = nn.ModuleList(
            [Block_u(dim=opt['decoder']['decoder_embed_dim'], num_heads=opt['decoder']['decoder_num_heads'],
                     mlp_ratio=opt['mlp_ratio'],
                     qkv_bias=True,
                     norm_layer=norm_layer)
             for i in range(opt['decoder']['decoder_depth'])])

        self.decoder_norm_video = norm_layer(opt['decoder']['decoder_embed_dim'])
        self.decoder_norm_eeg = norm_layer(opt['decoder']['decoder_embed_dim'])
        self.decoder_norm_ecg = norm_layer(opt['decoder']['decoder_embed_dim'])
        self.decoder_norm_eog = norm_layer(opt['decoder']['decoder_embed_dim'])
        self.decoder_norm_emg = norm_layer(opt['decoder']['decoder_embed_dim'])
        self.decoder_norm_gsr = norm_layer(opt['decoder']['decoder_embed_dim'])

        # project channel is different for different modality, use different projection heads
        self.decoder_pred_video = nn.Linear(opt['decoder']['decoder_embed_dim'],
                                            opt['video']['patch_size'] ** 2 * opt['video']['channel'] * opt['video'][
                                                'tubelet_size'])  # decoder to patch
        self.decoder_pred_eeg = nn.Linear(opt['decoder']['decoder_embed_dim'],
                                          opt['eeg']['eeg_patch_size'] ** 2 * opt['eeg']['eeg_channel'] * opt['eeg'][
                                              'eeg_tubelet_size'])
        self.decoder_pred_ecg = nn.Linear(opt['decoder']['decoder_embed_dim'],
                                          opt['ecg']['ecg_tubelet_size'])
        self.decoder_pred_eog = nn.Linear(opt['decoder']['decoder_embed_dim'],
                                          opt['eog']['eog_tubelet_size'])
        self.decoder_pred_emg = nn.Linear(opt['decoder']['decoder_embed_dim'],
                                          opt['emg']['emg_tubelet_size'])
        self.decoder_pred_gsr = nn.Linear(opt['decoder']['decoder_embed_dim'],
                                          opt['gsr']['gsr_tubelet_size'])
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

        torch.nn.init.normal_(self.decoder_modality_video, std=.02)
        torch.nn.init.normal_(self.decoder_modality_eeg, std=.02)
        torch.nn.init.normal_(self.decoder_modality_ecg, std=.02)
        torch.nn.init.normal_(self.decoder_modality_eog, std=.02)
        torch.nn.init.normal_(self.decoder_modality_emg, std=.02)
        torch.nn.init.normal_(self.decoder_modality_gsr, std=.02)

        torch.nn.init.normal_(self.mask_video_token, std=.02)
        torch.nn.init.normal_(self.mask_eeg_token, std=.02)
        torch.nn.init.normal_(self.mask_ecg_token, std=.02)
        torch.nn.init.normal_(self.mask_eog_token, std=.02)
        torch.nn.init.normal_(self.mask_emg_token, std=.02)
        torch.nn.init.normal_(self.mask_gsr_token, std=.02)

        # ------------------------------------ encoder pos embed initialization ---------------------------------------
        print('encoder positional embedding initialization ...')
        embed_dim = self.pos_embed_video_spatial.size()[-1]

        pos_embed_video_temporal = get_sinusoid_encoding_table(
            self.opt['video']['length'] // self.opt['video']['tubelet_size'],
            embed_dim)
        # print(pos_embed_video_temporal.size(),self.pos_embed_video_temporal.size())
        self.pos_embed_video_temporal.data.copy_(pos_embed_video_temporal.float())

        pos_embed_video_spatial = get_2d_sincos_pos_embed(
            embed_dim=embed_dim, grid_h_size=self.opt['video']['height'] // self.opt['video']['patch_size'],
            grid_w_size=self.opt['video']['width'] // self.opt['video']['patch_size'])
        self.pos_embed_video_spatial.data.copy_(pos_embed_video_spatial.float())

        pos_embed_eeg_temporal = get_sinusoid_encoding_table(
            self.opt['eeg']['eeg_length'] // self.opt['eeg']['eeg_tubelet_size'],
            embed_dim)
        self.pos_embed_eeg_temporal.data.copy_(pos_embed_eeg_temporal.float())

        pos_embed_eeg_spatial = get_2d_sincos_pos_embed(
            embed_dim=embed_dim, grid_h_size=self.opt['eeg']['eeg_height'] // self.opt['eeg']['eeg_patch_size'],
            grid_w_size=self.opt['eeg']['eeg_width'] // self.opt['eeg']['eeg_patch_size'])
        self.pos_embed_eeg_spatial.data.copy_(pos_embed_eeg_spatial.float())

        pos_embed_ecg_temporal = get_sinusoid_encoding_table(
            self.opt['ecg']['ecg_length'] // self.opt['ecg']['ecg_tubelet_size'],
            embed_dim)
        self.pos_embed_ecg_temporal.data.copy_(pos_embed_ecg_temporal.float())
        pos_embed_ecg_chl = get_sinusoid_encoding_table(self.opt['ecg']['ecg_channel'], embed_dim)
        self.pos_embed_ecg_chl.data.copy_(pos_embed_ecg_chl.float())

        pos_embed_eog_temporal = get_sinusoid_encoding_table(
            self.opt['eog']['eog_length'] // self.opt['eog']['eog_tubelet_size'],
            embed_dim)
        self.pos_embed_eog_temporal.data.copy_(pos_embed_eog_temporal.float())
        pos_embed_eog_chl = get_sinusoid_encoding_table(self.opt['eog']['eog_channel'], embed_dim)
        self.pos_embed_eog_chl.data.copy_(pos_embed_eog_chl.float())

        pos_embed_emg_temporal = get_sinusoid_encoding_table(
            self.opt['emg']['emg_length'] // self.opt['emg']['emg_tubelet_size'],
            embed_dim)
        self.pos_embed_emg_temporal.data.copy_(pos_embed_emg_temporal.float())
        pos_embed_emg_chl = get_sinusoid_encoding_table(self.opt['emg']['emg_channel'], embed_dim)
        self.pos_embed_emg_chl.data.copy_(pos_embed_emg_chl.float())

        pos_embed_gsr = get_sinusoid_encoding_table(self.patch_embed_gsr.num_patches,
                                                    embed_dim)
        self.pos_embed_gsr.data.copy_(pos_embed_gsr.float())

        # ------------------------------------ decoder pos embed initialization ---------------------------------------
        print('encoder positional embedding initialization ...')
        embed_dim = self.decoder_pos_embed_video_spatial.size()[-1]

        pos_embed_video_temporal = get_sinusoid_encoding_table(
            self.opt['video']['length'] // self.opt['video']['tubelet_size'],
            embed_dim)
        self.decoder_pos_embed_video_temporal.data.copy_(pos_embed_video_temporal.float())

        pos_embed_video_spatial = get_2d_sincos_pos_embed(
            embed_dim=embed_dim, grid_h_size=self.opt['video']['height'] // self.opt['video']['patch_size'],
            grid_w_size=self.opt['video']['width'] // self.opt['video']['patch_size'])
        self.decoder_pos_embed_video_spatial.data.copy_(pos_embed_video_spatial.float())

        pos_embed_eeg_temporal = get_sinusoid_encoding_table(
            self.opt['eeg']['eeg_length'] // self.opt['eeg']['eeg_tubelet_size'],
            embed_dim)
        self.decoder_pos_embed_eeg_temporal.data.copy_(pos_embed_eeg_temporal.float())

        pos_embed_eeg_spatial = get_2d_sincos_pos_embed(
            embed_dim=embed_dim, grid_h_size=self.opt['eeg']['eeg_height'] // self.opt['eeg']['eeg_patch_size'],
            grid_w_size=self.opt['eeg']['eeg_width'] // self.opt['eeg']['eeg_patch_size'])
        self.decoder_pos_embed_eeg_spatial.data.copy_(pos_embed_eeg_spatial.float())

        pos_embed_ecg_temporal = get_sinusoid_encoding_table(
            self.opt['ecg']['ecg_length'] // self.opt['ecg']['ecg_tubelet_size'],
            embed_dim)
        self.decoder_pos_embed_ecg_temporal.data.copy_(pos_embed_ecg_temporal.float())
        pos_embed_ecg_chl = get_sinusoid_encoding_table(self.opt['ecg']['ecg_channel'], embed_dim)
        self.decoder_pos_embed_ecg_chl.data.copy_(pos_embed_ecg_chl.float())

        pos_embed_eog_temporal = get_sinusoid_encoding_table(
            self.opt['eog']['eog_length'] // self.opt['eog']['eog_tubelet_size'],
            embed_dim)
        self.decoder_pos_embed_eog_temporal.data.copy_(pos_embed_eog_temporal.float())
        pos_embed_eog_chl = get_sinusoid_encoding_table(self.opt['eog']['eog_channel'], embed_dim)
        self.decoder_pos_embed_eog_chl.data.copy_(pos_embed_eog_chl.float())

        pos_embed_emg_temporal = get_sinusoid_encoding_table(
            self.opt['emg']['emg_length'] // self.opt['emg']['emg_tubelet_size'],
            embed_dim)
        self.decoder_pos_embed_emg_temporal.data.copy_(pos_embed_emg_temporal.float())

        pos_embed_emg_chl = get_sinusoid_encoding_table(self.opt['emg']['emg_channel'], embed_dim)
        self.decoder_pos_embed_emg_chl.data.copy_(pos_embed_emg_chl.float())

        pos_embed_gsr = get_sinusoid_encoding_table(self.patch_embed_gsr.num_patches,
                                                    embed_dim)
        self.decoder_pos_embed_gsr.data.copy_(pos_embed_gsr.float())

    def forward_encoder(self, video, eeg, ecg, eog, emg, gsr, modality_mask):
        # embed patches

        video = self.patch_embed_video(video)  # (b,c,t,h,w)-->(b,l,d)
        pos_embed_video = self.pos_embed_video_spatial.unsqueeze(1).repeat(1, self.opt['video']['length'] //
                                                                           self.opt['video']['tubelet_size'], 1,
                                                                           1).reshape(1, video.size()[1],
                                                                                      -1) + self.pos_embed_video_temporal.unsqueeze(
            2).repeat(1, 1, (self.opt['video']['height'] * self.opt['video']['width']) // (
                self.opt['video']['patch_size'] ** 2), 1).reshape(1, video.size()[1], -1)

        video = video + pos_embed_video + self.modality_video

        eeg = self.patch_embed_eeg(eeg)  # (b,c,t,h,w)-->(b,l,d)
        pos_embed_eeg = self.pos_embed_eeg_spatial.unsqueeze(1).repeat(1, self.opt['eeg']['eeg_length'] //
                                                                       self.opt['eeg']['eeg_tubelet_size'], 1,
                                                                       1).reshape(1, eeg.size()[1],
                                                                                  -1) + self.pos_embed_eeg_temporal.unsqueeze(
            2).repeat(1, 1, (self.opt['eeg']['eeg_height'] * self.opt['eeg']['eeg_width']) // (
                self.opt['eeg']['eeg_patch_size'] ** 2), 1).reshape(1, eeg.size()[1], -1)
        eeg = eeg + pos_embed_eeg + self.modality_eeg

        ecg = torch.einsum('blc->bcl', ecg).flatten(1).unsqueeze(2)  # (b,l,2)-->(b,2l,1)
        ecg = self.patch_embed_ecg(ecg)  # (b,l,c)-->(b,l,d)
        pos_embed_ecg = self.pos_embed_ecg_temporal.unsqueeze(1).repeat(1, self.opt['ecg']['ecg_channel'], 1,
                                                                        1).reshape(1, ecg.size()[1], -1) + \
                        self.pos_embed_ecg_chl.unsqueeze(2).repeat(1, 1,
                                                                   ecg.size()[1] // self.opt['ecg']['ecg_channel'],
                                                                   1).reshape(1, ecg.size()[1], -1)
        ecg = ecg + pos_embed_ecg + self.modality_ecg

        eog = torch.einsum('blc->bcl', eog).flatten(1).unsqueeze(2)  # (b,l,2)->(b,2l,1)
        eog = self.patch_embed_eog(eog)  # (b,l,c)-->(b,l,d)
        pos_embed_eog = self.pos_embed_eog_temporal.unsqueeze(1).repeat(1, self.opt['eog']['eog_channel'], 1,
                                                                        1).reshape(1, eog.size()[1], -1) + \
                        self.pos_embed_eog_chl.unsqueeze(2).repeat(1, 1,
                                                                   eog.size()[1] // self.opt['ecg']['ecg_channel'],
                                                                   1).reshape(1, eog.size()[1], -1)
        eog = eog + pos_embed_eog + self.modality_eog

        # print('origin emg:\n %s\n' % str(torch.isnan(emg.sum())))
        emg = torch.einsum('blc->bcl', emg).flatten(1).unsqueeze(2)  # (b,l,3)->(b,3l)->(b,3l,1)
        emg = self.patch_embed_emg(emg)  # (b,l,c)-->(b,l,d)
        pos_embed_emg = self.pos_embed_emg_temporal.unsqueeze(1).repeat(1, self.opt['emg']['emg_channel'], 1,
                                                                        1).reshape(1, emg.size()[1], -1) + \
                        self.pos_embed_emg_chl.unsqueeze(2).repeat(1, 1,
                                                                   emg.size()[1] // self.opt['emg']['emg_channel'],
                                                                   1).reshape(1, emg.size()[1], -1)
        emg = emg + pos_embed_emg + self.modality_emg

        gsr = self.patch_embed_gsr(gsr)  # (b,l,c)-->(b,l,d)
        gsr = gsr + self.pos_embed_gsr + self.modality_gsr

        # random mask tokens of each modality
        video, mask_video, ids_restore_video = random_tube_masking(video, 0.75,
                                                                   frames=self.patch_embed_video.num_frames,
                                                                   height=224, width=224, patch_size=16,
                                                                   tubelet_size=self.patch_embed_video.tubelet_size)
        eeg, mask_eeg, ids_restore_eeg = random_tube_masking_3d(eeg, 0.5, frames=self.opt['eeg']['eeg_length'],
                                                                height=self.opt['eeg']['eeg_height'],
                                                                width=self.opt['eeg']['eeg_width'],
                                                                tubelet_size=self.opt['eeg']['eeg_tubelet_size'],
                                                                patch_size=self.opt['eeg']['eeg_patch_size'])
        ecg, mask_ecg, ids_restore_ecg = random_tube_masking_1d(x=ecg, mask_ratio=0.5,
                                                                length=self.opt['ecg']['ecg_length'],
                                                                channel_num=self.opt['ecg']['ecg_channel'],
                                                                tubelet_size=self.opt['ecg']['ecg_tubelet_size'])
        eog, mask_eog, ids_restore_eog = random_tube_masking_1d(x=eog, mask_ratio=0.5,
                                                                length=self.opt['eog']['eog_length'],
                                                                channel_num=self.opt['eog']['eog_channel'],
                                                                tubelet_size=self.opt['eog']['eog_tubelet_size'])
        emg, mask_emg, ids_restore_emg = random_tube_masking_1d(x=emg, mask_ratio=0.5,
                                                                length=self.opt['emg']['emg_length'],
                                                                channel_num=self.opt['emg']['emg_channel'],
                                                                tubelet_size=self.opt['emg']['emg_tubelet_size'])
        gsr, mask_gsr, ids_restore_gsr = random_tube_masking_1d(x=gsr, mask_ratio=0.5,
                                                                length=self.opt['gsr']['gsr_length'],
                                                                channel_num=self.opt['gsr']['gsr_channel'],
                                                                tubelet_size=self.opt['gsr']['gsr_tubelet_size'])

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
        lengths = [int(self.patch_embed_video.num_patches * 0.25),
                   int(self.patch_embed_eeg.num_patches * 0.5),
                   int(self.patch_embed_ecg.num_patches * 0.5),
                   int(self.patch_embed_eog.num_patches * 0.5),
                   int(self.patch_embed_emg.num_patches * 0.5),
                   int(self.patch_embed_gsr.num_patches * 0.5)]
        x = torch.cat((video, eeg, ecg, eog, emg, gsr), dim=1)
        for i, blk in enumerate(self.blocks_u):
            x = blk(x, split_lengths=lengths)

        index_count = 0
        x_video = x[:, index_count:index_count + self.patch_embed_video.num_patches - int(mask_video[0].sum()), :]
        index_count += self.patch_embed_video.num_patches - int(mask_video[0].sum())
        x_eeg = x[:, index_count:index_count + self.patch_embed_eeg.num_patches - int(mask_eeg[0].sum()), :]
        index_count += self.patch_embed_eeg.num_patches - int(mask_eeg[0].sum())
        x_ecg = x[:, index_count:index_count + self.patch_embed_ecg.num_patches - int(mask_ecg[0].sum()), :]
        index_count += self.patch_embed_ecg.num_patches - int(mask_ecg[0].sum())
        x_eog = x[:, index_count:index_count + self.patch_embed_eog.num_patches - int(mask_eog[0].sum()), :]
        index_count += self.patch_embed_eog.num_patches - int(mask_eog[0].sum())
        x_emg = x[:, index_count:index_count + self.patch_embed_emg.num_patches - int(mask_emg[0].sum()), :]
        index_count += self.patch_embed_emg.num_patches - int(mask_emg[0].sum())
        x_gsr = x[:, index_count:, :]
        return x, x_video, x_eeg, x_ecg, x_eog, x_emg, x_gsr, mask_video, ids_restore_video, mask_eeg, ids_restore_eeg, mask_ecg, ids_restore_ecg, \
               mask_eog, ids_restore_eog, mask_emg, ids_restore_emg, mask_gsr, ids_restore_gsr

    def forward_decoder(self, x, mask_video, ids_restore_video,
                        mask_eeg, ids_restore_eeg, mask_ecg, ids_restore_ecg, mask_eog, ids_restore_eog,
                        mask_emg, ids_restore_emg, mask_gsr, ids_restore_gsr, modality_mask):
        x = self.decoder_embed(x)
        index_count = 0
        x_video = x[:, index_count:index_count + self.patch_embed_video.num_patches - int(mask_video[0].sum()), :]
        index_count += self.patch_embed_video.num_patches - int(mask_video[0].sum())
        x_eeg = x[:, index_count:index_count + self.patch_embed_eeg.num_patches - int(mask_eeg[0].sum()), :]
        index_count += self.patch_embed_eeg.num_patches - int(mask_eeg[0].sum())
        x_ecg = x[:, index_count:index_count + self.patch_embed_ecg.num_patches - int(mask_ecg[0].sum()), :]
        index_count += self.patch_embed_ecg.num_patches - int(mask_ecg[0].sum())
        x_eog = x[:, index_count:index_count + self.patch_embed_eog.num_patches - int(mask_eog[0].sum()), :]
        index_count += self.patch_embed_eog.num_patches - int(mask_eog[0].sum())
        x_emg = x[:, index_count:index_count + self.patch_embed_emg.num_patches - int(mask_emg[0].sum()), :]
        index_count += self.patch_embed_emg.num_patches - int(mask_emg[0].sum())
        x_gsr = x[:, index_count:, :]

        # ---------------------------------------- append mask tokens to sequence --------------------------------------
        mask_tokens_video = self.mask_video_token.repeat(x.size()[0], int(mask_video[0].sum()), 1)
        video_ = torch.cat([x_video, mask_tokens_video], dim=1)
        video_ = torch.gather(video_, dim=1, index=ids_restore_video.unsqueeze(-1).repeat(1, 1, x_video.size()[-1]))
        decoder_pos_embed_video = self.decoder_pos_embed_video_spatial.unsqueeze(1).repeat(1, self.opt['video'][
            'length'] // self.opt['video']['tubelet_size'], 1, 1).reshape(1, video_.size()[1],
                                                                          -1) + self.decoder_pos_embed_video_temporal.unsqueeze(
            2).repeat(1, 1, (self.opt['video']['height'] * self.opt['video']['width']) // (
                self.opt['video']['patch_size'] ** 2), 1).reshape(1, video_.size()[1], -1)
        video_ = video_ + self.decoder_modality_video + decoder_pos_embed_video

        mask_tokens_eeg = self.mask_eeg_token.repeat(x.size()[0], int(mask_eeg[0].sum()), 1)
        eeg_ = torch.cat([x_eeg, mask_tokens_eeg], dim=1)
        eeg_ = torch.gather(eeg_, dim=1, index=ids_restore_eeg.unsqueeze(-1).repeat(1, 1, x_eeg.size()[-1]))
        decoder_pos_embed_eeg = self.decoder_pos_embed_eeg_spatial.unsqueeze(1).repeat(1,
                                                                                       self.opt['eeg']['eeg_length'] //
                                                                                       self.opt['eeg'][
                                                                                           'eeg_tubelet_size'], 1,
                                                                                       1).reshape(1, eeg_.size()[1],
                                                                                                  -1) + \
                                self.decoder_pos_embed_eeg_temporal.unsqueeze(2).repeat(1, 1,
                                                                                        (self.opt['eeg']['eeg_height'] *
                                                                                         self.opt['eeg'][
                                                                                             'eeg_width']) // (
                                                                                                self.opt['eeg'][
                                                                                                    'eeg_patch_size'] ** 2),
                                                                                        1).reshape(1, eeg_.size()[1],
                                                                                                   -1)
        eeg_ = eeg_ + self.decoder_modality_eeg + decoder_pos_embed_eeg

        mask_tokens_ecg = self.mask_ecg_token.repeat(x.size()[0], int(mask_ecg[0].sum()), 1)
        ecg_ = torch.cat([x_ecg, mask_tokens_ecg], dim=1)
        ecg_ = torch.gather(ecg_, dim=1, index=ids_restore_ecg.unsqueeze(-1).repeat(1, 1, x_ecg.size()[-1]))
        decoder_pos_embed_ecg = self.decoder_pos_embed_ecg_temporal.unsqueeze(1).repeat(1,
                                                                                        self.opt['ecg']['ecg_channel'],
                                                                                        1, 1).reshape(1, ecg_.size()[1],
                                                                                                      -1) + \
                                self.decoder_pos_embed_ecg_chl.unsqueeze(2).repeat(1, 1,
                                                                                   ecg_.size()[1] // self.opt['ecg'][
                                                                                       'ecg_channel'],
                                                                                   1).reshape(1, ecg_.size()[1], -1)
        ecg_ = ecg_ + self.decoder_modality_ecg + decoder_pos_embed_ecg

        mask_tokens_eog = self.mask_eog_token.repeat(x.size()[0], int(mask_eog[0].sum()), 1)
        eog_ = torch.cat([x_eog, mask_tokens_eog], dim=1)
        eog_ = torch.gather(eog_, dim=1, index=ids_restore_eog.unsqueeze(-1).repeat(1, 1, x_eog.size()[-1]))
        decoder_pos_embed_eog = self.decoder_pos_embed_eog_temporal.unsqueeze(1).repeat(1,
                                                                                        self.opt['eog']['eog_channel'],
                                                                                        1, 1).reshape(1, eog_.size()[1],
                                                                                                      -1) + \
                                self.decoder_pos_embed_eog_chl.unsqueeze(2).repeat(1, 1,
                                                                                   eog_.size()[1] // self.opt['eog'][
                                                                                       'eog_channel'],
                                                                                   1).reshape(1, eog_.size()[1], -1)
        eog_ = eog_ + self.decoder_modality_eog + decoder_pos_embed_eog

        mask_tokens_emg = self.mask_emg_token.repeat(x.size()[0], int(mask_emg[0].sum()), 1)
        emg_ = torch.cat([x_emg, mask_tokens_emg], dim=1)
        emg_ = torch.gather(emg_, dim=1, index=ids_restore_emg.unsqueeze(-1).repeat(1, 1, x_emg.size()[-1]))
        decoder_pos_embed_emg = self.decoder_pos_embed_emg_temporal.unsqueeze(1).repeat(1,
                                                                                        self.opt['emg']['emg_channel'],
                                                                                        1, 1).reshape(1, emg_.size()[1],
                                                                                                      -1) + \
                                self.decoder_pos_embed_emg_chl.unsqueeze(2).repeat(1, 1,
                                                                                   emg_.size()[1] // self.opt['emg'][
                                                                                       'emg_channel'],
                                                                                   1).reshape(1, emg_.size()[1], -1)
        emg_ = emg_ + self.decoder_modality_emg + decoder_pos_embed_emg

        mask_tokens_gsr = self.mask_gsr_token.repeat(x.size()[0], int(mask_gsr[0].sum()), 1)
        gsr_ = torch.cat([x_gsr, mask_tokens_gsr], dim=1)
        gsr_ = torch.gather(gsr_, dim=1, index=ids_restore_gsr.unsqueeze(-1).repeat(1, 1, x_gsr.size()[-1]))
        gsr_ = gsr_ + self.decoder_modality_gsr + self.decoder_pos_embed_gsr

        x = torch.cat([video_, eeg_, ecg_, eog_, emg_, gsr_], dim=1)
        lengths = [self.patch_embed_video.num_patches,
                   self.patch_embed_eeg.num_patches,
                   self.patch_embed_ecg.num_patches,
                   self.patch_embed_eog.num_patches,
                   self.patch_embed_emg.num_patches,
                   self.patch_embed_gsr.num_patches]
        # transformer blocks forward
        for blk in self.decoder_blocks:
            x = blk(x, split_lengths=lengths)

        # predictor projection
        index_count = 0
        x_video = x[:, index_count:index_count + self.patch_embed_video.num_patches, :]
        index_count += self.patch_embed_video.num_patches
        x_eeg = x[:, index_count:index_count + self.patch_embed_eeg.num_patches, :]
        index_count += self.patch_embed_eeg.num_patches
        x_ecg = x[:, index_count:index_count + self.patch_embed_ecg.num_patches, :]
        index_count += self.patch_embed_ecg.num_patches
        x_eog = x[:, index_count:index_count + self.patch_embed_eog.num_patches, :]
        index_count += self.patch_embed_eog.num_patches
        x_emg = x[:, index_count:index_count + self.patch_embed_emg.num_patches, :]
        index_count += self.patch_embed_emg.num_patches
        x_gsr = x[:, index_count:index_count + self.patch_embed_gsr.num_patches, :]

        x_video = self.decoder_pred_video(self.decoder_norm_video(x_video))
        x_eeg = self.decoder_pred_eeg(self.decoder_norm_eeg(x_eeg))
        x_ecg = self.decoder_pred_ecg(self.decoder_norm_ecg(x_ecg))
        x_eog = self.decoder_pred_eog(self.decoder_norm_eog(x_eog))
        x_emg = self.decoder_pred_emg(self.decoder_norm_emg(x_emg))
        x_gsr = self.decoder_pred_gsr(self.decoder_norm_gsr(x_gsr))

        return x_video, x_eeg, x_ecg, x_eog, x_emg, x_gsr

    def forward_mae_loss(self, input, pred, mask, modality, chl_mask):
        if modality == "video":
            target = patchify_3d(input, self.patch_embed_video.tubelet_size, self.patch_embed_video.patch_size[0])
            chl_mask = chl_mask.repeat(1, self.opt['video']['length'] // self.opt['video']['tubelet_size'], 1,
                                       1).flatten(1)
        elif modality == "eeg":
            target = patchify_3d(input, self.patch_embed_eeg.tubelet_size, self.patch_embed_eeg.patch_size[0])
            chl_mask = chl_mask.repeat(1, self.opt['eeg']['eeg_length'] // self.opt['eeg']['eeg_tubelet_size'], 1,
                                       1).flatten(1)
        elif modality == "ecg":
            input = torch.einsum('blc->bcl', input).flatten(1).unsqueeze(2)
            target = patchify_1d(input, self.patch_embed_ecg.tubelet_size)
            chl_mask = chl_mask.repeat(1, self.opt['ecg']['ecg_length'] // self.opt['ecg']['ecg_tubelet_size'],
                                       1)
            chl_mask = torch.einsum('blc->bcl', chl_mask).flatten(1)
        elif modality == "eog":
            input = torch.einsum('blc->bcl', input).flatten(1).unsqueeze(2)
            target = patchify_1d(input, self.patch_embed_eog.tubelet_size)
            chl_mask = chl_mask.repeat(1, self.opt['eog']['eog_length'] // self.opt['eog']['eog_tubelet_size'],
                                       1)
            chl_mask = torch.einsum('blc->bcl', chl_mask).flatten(1)
        elif modality == "emg":
            input = torch.einsum('blc->bcl', input).flatten(1).unsqueeze(2)
            target = patchify_1d(input, self.patch_embed_emg.tubelet_size)
            chl_mask = chl_mask.repeat(1, self.opt['emg']['emg_length'] // self.opt['emg']['emg_tubelet_size'],
                                       1)
            chl_mask = torch.einsum('blc->bcl', chl_mask).flatten(1)

        elif modality == "gsr":
            target = patchify_1d(input, self.patch_embed_gsr.tubelet_size)
            chl_mask = chl_mask.repeat(1, self.opt['gsr']['gsr_length'] // self.opt['gsr']['gsr_tubelet_size'],
                                       1).flatten(1)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            std = target.std(dim=-1, keepdim=True)
            target = (target - mean) / torch.clamp(std, 1e-6)

        loss = torch.abs(pred - target)
        loss = loss.mean(dim=-1)  # [N,L,D]->[N,L
        mask = mask.squeeze(-1)
        loss = (loss * mask * chl_mask).sum(dim=-1) / torch.clamp((chl_mask * mask).sum(dim=-1),
                                                                  1e-6)  # loss on removed tokens
        return loss

    def forward_substitute_loss(self, video_rep, eeg_rep, ecg_rep, eog_rep, emg_rep, gsr_rep, modality_mask):
        loss_l1 = torch.sum(
            torch.abs(video_rep.detach() - self.modality_video_substitute) * modality_mask[:, 0].unsqueeze(
                1)) / torch.clamp(modality_mask[:, 0].sum(), 1e-6) + \
                  torch.sum(
                      torch.abs(eeg_rep.detach() - self.modality_eeg_substitute) * modality_mask[:, 1].unsqueeze(
                          1)) / torch.clamp(modality_mask[:, 1].sum(), 1e-6) + \
                  torch.sum(
                      torch.abs(ecg_rep.detach() - self.modality_ecg_substitute) * modality_mask[:, 2].unsqueeze(
                          1)) / torch.clamp(modality_mask[:, 2].sum(), 1e-6) + \
                  torch.sum(
                      torch.abs(eog_rep.detach() - self.modality_eog_substitute) * modality_mask[:, 3].unsqueeze(
                          1)) / torch.clamp(modality_mask[:, 3].sum(), 1e-6) + \
                  torch.sum(
                      torch.abs(emg_rep.detach() - self.modality_emg_substitute) * modality_mask[:, 4].unsqueeze(
                          1)) / torch.clamp(modality_mask[:, 4].sum(), 1e-6) + \
                  torch.sum(
                      torch.abs(gsr_rep.detach() - self.modality_gsr_substitute) * modality_mask[:, 5].unsqueeze(
                          1)) / torch.clamp(modality_mask[:, 5].sum(), 1e-6)

        loss = loss_l1 / 6
        embed_dim = video_rep.size()[-1]
        loss /= embed_dim
        return loss

    def forward(self, video, eeg, ecg, eog, emg, gsr,
                modality_mask, video_chl_mask, eeg_chl_mask, ecg_chl_mask, eog_chl_mask, emg_chl_mask,
                gsr_chl_mask,
                mae_loss_weight=1., contrast_loss_weight=0.01, substitute_loss_weight=1.,
                ):
        latent, cvideo, ceeg, cecg, ceog, cemg, cgsr, \
        mask_video, ids_restore_video, mask_eeg, ids_restore_eeg, mask_ecg, ids_restore_ecg, \
        mask_eog, ids_restore_eog, mask_emg, ids_restore_emg, mask_gsr, ids_restore_gsr = self.forward_encoder(
            video, eeg, ecg, eog, emg, gsr, modality_mask)
        x_video, x_eeg, x_ecg, x_eog, x_emg, x_gsr = self.forward_decoder(
            latent, mask_video, ids_restore_video, mask_eeg, ids_restore_eeg, mask_ecg, ids_restore_ecg,
            mask_eog, ids_restore_eog, mask_emg, ids_restore_emg, mask_gsr, ids_restore_gsr, modality_mask)
        # calculate loss
        loss_video = self.forward_mae_loss(input=video, pred=x_video, mask=mask_video, modality="video",
                                           chl_mask=video_chl_mask)
        loss_video = (loss_video * modality_mask[:, 0]).sum() / torch.clamp(modality_mask[:, 0].sum(), 1e-6)

        loss_mae_eeg = self.forward_mae_loss(input=eeg, pred=x_eeg, mask=mask_eeg, modality="eeg",
                                             chl_mask=eeg_chl_mask)
        loss_mae_eeg = (loss_mae_eeg * modality_mask[:, 1]).sum() / torch.clamp(modality_mask[:, 1].sum(), 1e-6)

        loss_mae_ecg = self.forward_mae_loss(input=ecg, pred=x_ecg, mask=mask_ecg, modality="ecg",
                                             chl_mask=ecg_chl_mask)
        loss_mae_ecg = (loss_mae_ecg * modality_mask[:, 2]).sum() / torch.clamp(modality_mask[:, 2].sum(), 1e-6)

        loss_mae_eog = self.forward_mae_loss(input=eog, pred=x_eog, mask=mask_eog, modality="eog",
                                             chl_mask=eog_chl_mask)
        loss_mae_eog = (loss_mae_eog * modality_mask[:, 3]).sum() / torch.clamp(modality_mask[:, 3].sum(), 1e-6)

        loss_mae_emg = self.forward_mae_loss(input=emg, pred=x_emg, mask=mask_emg, modality="emg",
                                             chl_mask=emg_chl_mask)
        loss_mae_emg = (loss_mae_emg * modality_mask[:, 4]).sum() / torch.clamp(modality_mask[:, 4].sum(), 1e-6)

        loss_mae_gsr = self.forward_mae_loss(input=gsr, pred=x_gsr, mask=mask_gsr, modality="gsr",
                                             chl_mask=gsr_chl_mask)
        loss_mae_gsr = (loss_mae_gsr * modality_mask[:, 5]).sum() / torch.clamp(modality_mask[:, 5].sum(), 1e-6)

        loss_mae = loss_video + loss_mae_eeg + loss_mae_ecg + loss_mae_eog + loss_mae_gsr + loss_mae_emg
        loss_mae = loss_mae * mae_loss_weight
        if contrast_loss_weight != 0:
            nce_loss, c_acc, nce_loss_list, c_acc_list = loo_contrastive_loss(cvideo.mean(dim=1),
                                                                              ceeg.mean(dim=1),
                                                                              cecg.mean(dim=1),
                                                                              ceog.mean(dim=1),
                                                                              cemg.mean(dim=1),
                                                                              cgsr.mean(dim=1),
                                                                              modality_mask)
            nce_loss = nce_loss * contrast_loss_weight
        else:
            nce_loss, c_acc = torch.tensor(0.0, device=eeg.device), torch.tensor(0.0, device=eeg.device)
            nce_loss_list = [torch.tensor(0.0, device=eeg.device) for i in range(6)]
            c_acc_list = [torch.tensor(0.0, device=eeg.device) for i in range(6)]

        if substitute_loss_weight != 0:
            subLoss = self.forward_substitute_loss(cvideo.mean(dim=1),
                                                   ceeg.mean(dim=1),
                                                   cecg.mean(dim=1),
                                                   ceog.mean(dim=1),
                                                   cemg.mean(dim=1),
                                                   cgsr.mean(dim=1),
                                                   modality_mask)
            subLoss = subLoss * substitute_loss_weight
        else:
            subLoss = torch.tensor(0.0, device=eeg.device)
        loss = loss_mae + nce_loss + subLoss
        return loss, nce_loss, c_acc, loss_video, loss_mae_eeg, loss_mae_ecg, loss_mae_eog, loss_mae_emg, loss_mae_gsr, \
               nce_loss_list, c_acc_list, x_video, x_eeg, x_ecg, x_eog, x_emg, x_gsr, subLoss


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
        # print(eog.size(),self.pos_embed_eog_temporal.size(),self.pos_embed_eog_chl.size())
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
        if self.patch_embed_video.num_patches>0:
            for blk in self.blocks_video:
                video = blk(video)
        if self.patch_embed_eeg.num_patches>0:
            for blk in self.blocks_eeg:
                eeg = blk(eeg)
        if self.patch_embed_ecg.num_patches>0:
            for blk in self.blocks_ecg:
                ecg = blk(ecg)
        if self.patch_embed_eog.num_patches>0:
            for blk in self.blocks_eog:
                eog = blk(eog)
        if self.patch_embed_emg.num_patches>0:
            for blk in self.blocks_emg:
                emg = blk(emg)
        if self.patch_embed_gsr.num_patches>0:
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
