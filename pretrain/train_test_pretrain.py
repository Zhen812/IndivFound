import os
import pickle
import time
import datetime
import torch
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from pretrain.src.utils.helper import AverageMeter
from tqdm import tqdm
from pretrain.src.utils.helper import makedir
import warnings


class OverflowWarningHandler:
    def __init__(self):
        self._overflow_warning_msg = "overflow encountered in multiply"

    def __call__(self, message, category, filename, lineno, line=None, source=None):
        # 检查警告消息和类别，如果匹配则抛出异常
        if (issubclass(category, RuntimeWarning) and
                self._overflow_warning_msg in str(message)):
            print('overflow!')
            return
            # 否则，使用默认的警告显示
        else:
            warnings.showwarning(message, category, filename, lineno, line=None)


def train(model, train_loader, args, opt, log):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.set_grad_enabled(True)

    batch_time, per_sample_time, data_time, per_sample_data_time, per_sample_dnn_time = \
        AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    loss_overall_meter, nce_loss_meter, c_acc_meter, loss_video_meter, loss_eeg_meter, loss_ecg_meter, loss_eog_meter, loss_emg_meter, loss_gsr_meter = \
        AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    progress = []

    best_epoch, best_loss = 0, np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = os.path.join(opt['model_save_root'], args.experiment_run, args.model, 'saved_models')
    makedir(exp_dir)
    makedir('%s/models/best' % exp_dir)
    makedir('%s/models/iteration' % exp_dir)

    def _save_progress():
        progress.append([epoch, global_step, best_epoch, best_loss, time.time() - start_time])
        with open("%s/progress.pkl" % exp_dir, 'wb') as f:
            pickle.dump(progress, f)

    model = model.to(device, dtype=torch.float32)

    trainables = [p for p in model.parameters() if p.requires_grad]
    log('Total parameter number is: {:.6f} million'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    log('Total trainable parameter number is: {:.6f} million'.format(sum(p.numel() for p in trainables) / 1e6))

    optimizer_specs = [
        {'params': trainables, 'weight_decay': 5e-7, 'betas': (0.95, 0.999), 'lr': opt['lr']},
    ]
    optimizer = torch.optim.Adam(optimizer_specs)
    ''' 
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=opt['lr'],
                                                    total_steps=opt['epochs'], pct_start=30. / opt['epochs'])
    '''
    '''
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=opt['epochs'],
                                                           eta_min=opt['lr'] * 0.1)
    '''

    log('==' * 15)
    epoch = 1

    scaler = GradScaler()
    log("current #steps=%s, #epochs=%s" % (global_step, epoch))
    log("start training...")
    result = np.zeros([opt['epochs'], 11])  # for each epoch, 21 metrics to be recorded
    model.train()
    while epoch <= opt['epochs']:
        begin_time = time.time()
        end_time = time.time()
        model.train()
        log('--' * 15)
        log(str(datetime.datetime.now()))
        log('current #epochs=%s, #steps=%s' % (epoch, global_step))
        log(
            'masking ratios for [Video, EEG, ECG, EOG, EMG, GSR] are [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]' % (
                opt['mask_ratio_video'], opt['mask_ratio_eeg'], opt['mask_ratio_ecg'], opt['mask_ratio_eog'],
                opt['mask_ratio_emg'],
                opt['mask_ratio_gsr']
            ))

        for i, (video, eeg, ecg, eog, emg, gsr, modality_mask, video_chl_mask, eeg_chl_mask, ecg_chl_mask,
                eog_chl_mask, emg_chl_mask, gsr_chl_mask) in enumerate(tqdm(train_loader)):
            B = eeg.size(0)
            video = video.to(device, non_blocking=True)
            eeg = eeg.to(device, non_blocking=True)
            ecg = ecg.to(device, non_blocking=True)
            eog = eog.to(device, non_blocking=True)
            emg = emg.to(device, non_blocking=True)
            gsr = gsr.to(device, non_blocking=True)
            modality_mask = modality_mask.to(device, non_blocking=True)
            video_chl_mask = video_chl_mask.to(device, non_blocking=True)
            eeg_chl_mask = eeg_chl_mask.to(device, non_blocking=True)
            ecg_chl_mask = ecg_chl_mask.to(device, non_blocking=True)
            eog_chl_mask = eog_chl_mask.to(device, non_blocking=True)
            emg_chl_mask = emg_chl_mask.to(device, non_blocking=True)
            gsr_chl_mask = gsr_chl_mask.to(device, non_blocking=True)

            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / B)
            dnn_start_time = time.time()

            with autocast():
                loss, nce_loss, c_acc, loss_video, loss_eeg, loss_ecg, loss_eog, loss_emg, loss_gsr, t1, _, \
                x_video, x_eeg, x_ecg, x_eog, x_emg, x_gsr, _ = model.forward(video, eeg, ecg, eog, emg, gsr,
                                                                                    modality_mask, video_chl_mask,
                                                                                    eeg_chl_mask, ecg_chl_mask,
                                                                                    eog_chl_mask, emg_chl_mask,
                                                                                    gsr_chl_mask)
            video_num = torch.sum(modality_mask[:, 0])
            eeg_num = torch.sum(modality_mask[:, 1])
            ecg_num = torch.sum(modality_mask[:, 2])
            eog_num = torch.sum(modality_mask[:, 3])
            emg_num = torch.sum(modality_mask[:, 4])
            gsr_num = torch.sum(modality_mask[:, 5])

            # this is due to for torch.nn.DataParallel, the output loss of 4 gpus won't be automatically averaged, need to be done manually
            loss, nce_loss, c_acc, loss_video, loss_eeg, loss_ecg, loss_eog, loss_emg, loss_gsr = \
                sum(loss) / len(loss), sum(nce_loss) / len(loss), sum(c_acc) / len(
                    loss), sum(loss_video) / len(loss), sum(loss_eeg) / len(loss), sum(loss_ecg) / len(
                    loss), sum(loss_eog) / len(loss), sum(loss_emg) / len(loss), sum(loss_gsr) / len(loss)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_overall_meter.update(loss.item(), B)
            nce_loss_meter.update(nce_loss.item(), B)
            c_acc_meter.update(c_acc.item(), B)
            loss_video_meter.update(loss_video.item(), max(1e-3, video_num.detach().item()))
            loss_eeg_meter.update(loss_eeg.item(), max(1e-3, eeg_num.detach().item()))
            loss_ecg_meter.update(loss_ecg.item(), max(1e-3, ecg_num.detach().item()))
            loss_eog_meter.update(loss_eog.item(), max(1e-3, eog_num.detach().item()))
            loss_emg_meter.update(loss_emg.item(), max(1e-3, emg_num.detach().item()))
            loss_gsr_meter.update(loss_gsr.item(), max(1e-3, gsr_num.detach().item()))

            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time) / B)
            per_sample_dnn_time.update((time.time() - dnn_start_time) / B)

            print_step = global_step % opt['n_print_steps'] == 0
            early_print_step = epoch == 0 and global_step % (opt['n_print_steps'] / 10) == 0
            print_step = print_step or early_print_step

            if print_step and global_step != 0:
                log('Epoch: [%06d][%06d/%06d]\t'
                    'Per Sample Total Time is %.6f\t'
                    'Per Sample Data Time %.6f\t'
                    'Per Sample DNN Time is %.6f\t'
                    'Train Total Loss is %.6f\t'
                    'Train MAE Loss for [Video, EEG, ECG, EOG, EMG, GSR] are [%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]\t'
                    'Train InfoNCE Loss is %.6f, c_acc is %.6f\t'

                    % (
                        epoch, i, len(train_loader), per_sample_time.avg, per_sample_data_time.avg,
                        per_sample_dnn_time.avg, loss_overall_meter.avg,
                        loss_video_meter.avg, loss_eeg_meter.avg, loss_ecg_meter.avg, loss_eog_meter.avg,
                        loss_emg_meter.avg, loss_gsr_meter.avg,
                        nce_loss_meter.avg, c_acc_meter.avg * 100
                    ))
                if np.isnan(loss_overall_meter.avg):
                    log("training diverged...")
                    return

            end_time = time.time()
            global_step += 1

        torch.save(model.state_dict(), '%s/models/iteration/model_%06d.pth' % (exp_dir, epoch))
        torch.save(optimizer.state_dict(), '%s/models/iteration/optim_state_%06d.pth' % (exp_dir, epoch))


        log('Epoch [%06d]: train total loss is %.6f' % (epoch, loss_overall_meter.avg))
        log('train MAE loss for [Video, EEG, ECG, EOG, EMG, GSR] are [%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]' % (
            loss_video_meter.avg, loss_eeg_meter.avg, loss_ecg_meter.avg, loss_eog_meter.avg,
            loss_emg_meter.avg, loss_gsr_meter.avg))
        log('train InfoNCE loss is %.6f, c_acc is %.6f' % (nce_loss_meter.avg, c_acc_meter.avg * 100))
        result[epoch - 1, :] = [loss_overall_meter.avg, nce_loss_meter.avg, c_acc_meter.avg, loss_video_meter.avg,
                                loss_eeg_meter.avg, loss_ecg_meter.avg, loss_eog_meter.avg, loss_emg_meter.avg,
                                loss_gsr_meter.avg,
                                loss_overall_meter.avg,
                                optimizer.param_groups[0]['lr']]
        np.savetxt(exp_dir + '/result.csv', result, delimiter=',')
        log('validation finished')
        log('Epoch [%06d]: lr = %.6f\n' % (
            epoch, optimizer.param_groups[0]['lr']))

        # scheduler.step()

        _save_progress()
        finish_time = time.time()
        log('Epoch [%06d]: training time = %.3f}' % (epoch, finish_time - begin_time))

        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        per_sample_dnn_time.reset()
        loss_overall_meter.reset()
        loss_video_meter.reset()
        loss_eeg_meter.reset()
        loss_ecg_meter.reset()
        loss_eog_meter.reset()
        loss_emg_meter.reset()
        loss_gsr_meter.reset()
        c_acc_meter.reset()
        nce_loss_meter.reset()

        epoch += 1
        global_step = 0

