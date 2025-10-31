import torch.nn.functional
import numpy as np
import pandas as pd
import torch
import os
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, recall_score, precision_score, \
    f1_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import argparse
import sys

import yaml
import torch.nn as nn
from torch.utils.data import Dataset
from src.model.encoder_ms import MultiMAE_FT
from src.utils.helper import set_seed

parser = argparse.ArgumentParser()
parser.add_argument('--gpuid', type=str, default="0,1")
parser.add_argument('--seed', type=int, default=21)
parser.add_argument('--num_classes', type=int, default=6)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
set_seed(args.seed)
print("CUDA_VISIBLE_DEVICES:", args.gpuid)
print("Random Seed: ", args.seed)


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


class IC_Dataset_SleepTelemetry(Dataset):
    """
    model: string; ['test'];
    sample_length: int; 30 s;
    """

    def __init__(self, opt):
        super(IC_Dataset_SleepTelemetry, self).__init__()
        self.opt = opt
        self.data_dir = opt['data_dir']
        self.file_paths, self.labels = self._obtain_data()

    def _contains_any(self, text, substrings):
        for i, substring in enumerate(substrings):
            if substring in text:
                return i
        return -1

    def _obtain_data(self):
        subjects = os.listdir(os.path.join(self.data_dir, 'eeg'))
        file_paths, labels = [], []
        for subj in subjects:
            pre_eeg_dir = os.path.join(self.data_dir, 'eeg', subj)
            pre_eog_dir = os.path.join(self.data_dir, 'eog', subj)
            pre_emg_dir = os.path.join(self.data_dir, 'emg', subj)
            nights = os.listdir(pre_eeg_dir)
            for night in nights:
                night_dir = os.path.join(pre_eeg_dir, night)
                clips = os.listdir(night_dir)
                for clip in clips:
                    file_paths.append(
                        [os.path.join(pre_dir, night, clip) for pre_dir in [pre_eeg_dir, pre_eog_dir, pre_emg_dir]])
                    label = self._contains_any(clip, ['SW', 'SR', 'S1', 'S2', 'S3', 'S4'])
                    labels.append(label)
        return file_paths, labels

    def get_eeg(self, eeg_name):
        eeg_width = self.opt['eeg']['eeg_width']
        eeg_height = self.opt['eeg']['eeg_height']
        eeg_length = self.opt['eeg']['eeg_length']
        eeg_channel = self.opt['eeg']['eeg_channel']
        eeg_tensor = torch.zeros((eeg_channel, eeg_length, eeg_height, eeg_width), dtype=torch.float32)
        eeg_indicator = 0
        # if not pd.isna(eeg_name):
        if os.path.exists(eeg_name):
            data = np.load(eeg_name)
            assert data.shape == (eeg_length, eeg_height, eeg_width)
            std = np.std(data, axis=0)
            std[std == 0] = np.nan
            minv = np.nanmin(std)
            if np.isnan(minv):
                print(eeg_name)
                return eeg_tensor, eeg_indicator
            exponent = int(np.floor(np.log10(minv)))
            # data = -np.log(np.maximum(data, 1e-6))
            data = (data - np.mean(data, axis=0)) / np.maximum(np.std(data, axis=0), 10 ** exponent)

            eeg_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
            eeg_indicator = 1
            eeg_chl_mask = torch.sum(eeg_tensor.reshape(eeg_channel * eeg_length, eeg_height, eeg_width), dim=0,
                                     keepdim=True) == 0
            eeg_chl_mask = 1 - eeg_chl_mask.float()
            eeg_chl_mask.requires_grad = False
            if np.mean(data) == 0 and np.std(data) == 0:
                eeg_indicator = 0
        return eeg_tensor, eeg_indicator  # channel * eeg_length * height * width

        # 读取处理EOG

    def get_eog(self, eog_name):
        eog_length = self.opt['eog']['eog_length']
        eog_channel = self.opt['eog']['eog_channel']
        eog_tensor = torch.zeros((eog_length, eog_channel), dtype=torch.float32)
        eog_chl_mask = torch.zeros((1, eog_channel), requires_grad=False, dtype=torch.float32)
        eog_indicator = 0
        # if not pd.isna(eog_name):
        if os.path.exists(eog_name):
            data = np.load(eog_name)
            # print(data.shape)
            assert data.shape == (eog_length, eog_channel)
            # data = -np.log(np.maximum(data, 1e-6))
            std = np.std(data, axis=0)
            std[std == 0] = np.nan
            minv = np.nanmin(std)
            if np.isnan(minv):
                return eog_tensor, eog_indicator
            exponent = int(np.floor(np.log10(minv)))
            data = (data - np.mean(data, axis=0)) / np.maximum(np.std(data, axis=0), 10 ** exponent)
            eog_tensor = torch.tensor(data, dtype=torch.float32)
            eog_indicator = 1
            eog_chl_mask = torch.ones((1, eog_channel), requires_grad=False, dtype=torch.float32)
            if np.mean(data) == 0 and np.std(data) == 0:
                eog_indicator = 0
        return eog_tensor, eog_indicator  # ecg_length*2

        # 读取处理EMG

    def get_emg(self, emg_name):
        emg_length = self.opt['emg']['emg_length']
        emg_channel = self.opt['emg']['emg_channel']
        emg_tensor = torch.zeros((emg_length, emg_channel), dtype=torch.float32)
        emg_indicator = 0
        emg_chl_mask = torch.zeros((1, emg_channel), requires_grad=False, dtype=torch.float32)
        # if not pd.isna(emg_name):
        if os.path.exists(emg_name):
            data = np.load(emg_name)
            assert data.shape == (emg_length, emg_channel)
            # data = -np.log(np.maximum(data, 1e-6))
            std = np.std(data, axis=0)
            std[std == 0] = np.nan
            minv = np.nanmin(std)
            if np.isnan(minv):
                return emg_tensor, emg_indicator
            exponent = int(np.floor(np.log10(minv)))
            data = (data - np.mean(data, axis=0)) / np.maximum(np.std(data, axis=0), 10 ** exponent)
            emg_tensor = torch.tensor(data, dtype=torch.float32)
            emg_indicator = 1
            emg_chl_mask = torch.sum(emg_tensor, dim=0, keepdim=True) == 0
            emg_chl_mask = 1 - emg_chl_mask.float()
            if np.mean(data) == 0 and np.std(data) == 0:
                emg_indicator = 0
        return emg_tensor, emg_indicator  # emg_length*3

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, item):
        eeg_path = self.file_paths[item][0]
        eog_path = self.file_paths[item][1]
        emg_path = self.file_paths[item][2]
        label = self.labels[item]

        # video_data, video_indicator = self.get_video(video_path)
        video_data = torch.zeros(size=(
            self.opt['video']['channel'], self.opt['video']['length'], self.opt['video']['height'],
            self.opt['video']['width']))
        video_indicator = 0

        eeg_data, eeg_indicator = self.get_eeg(eeg_path)
        '''
        eeg_data = torch.zeros(size=(
                        self.opt['eeg']['eeg_channel'], self.opt['eeg']['eeg_length'], self.opt['eeg']['eeg_height'],
                                    self.opt['eeg']['eeg_width'],))
        eeg_indicator = 0
        '''

        # ecg_data, ecg_indicator = self.get_ecg(ecg_path)

        ecg_data = torch.zeros(size=(
            self.opt['ecg']['ecg_length'], self.opt['ecg']['ecg_channel']))
        ecg_indicator = 0

        eog_data, eog_indicator = self.get_eog(eog_path)
        # eog_data = torch.zeros(size=(
        #     self.opt['eog']['eog_length'], self.opt['eog']['eog_channel']))
        # eog_indicator = 0

        emg_data, emg_indicator = self.get_emg(emg_path)

        # emg_data = torch.zeros(size=(
        #     self.opt['emg']['emg_length'], self.opt['emg']['emg_channel']))
        # emg_indicator = 0

        # gsr_data, gsr_indicator = self.get_gsr(gsr_path)

        gsr_data = torch.zeros(size=(
            self.opt['gsr']['gsr_length'], self.opt['gsr']['gsr_channel']))
        gsr_indicator = 0

        modality_mask = [video_indicator, eeg_indicator, ecg_indicator, eog_indicator, emg_indicator, gsr_indicator]
        modality_mask = torch.tensor(modality_mask, requires_grad=False)

        return video_data, eeg_data, ecg_data, eog_data, emg_data, gsr_data, modality_mask, label


def external_val_SleepCassette2SleepTelemetry():
    with open("./SleepTelemetry_config.yaml", encoding="UTF-8") as f:
        opt = yaml.safe_load(f)
    folds = list(range(5))
    found_model_ckpts = opt['found_model_ckpts']
    cls_model_ckpts = opt['cls_model_ckpts']
    fold_metrics = {k: [] for k in ['Acc', 'Precision', 'Recall', 'F1', 'AUC']}
    for foldID in folds:
        # load data
        test_dataset = IC_Dataset_SleepTelemetry(opt)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=opt['batch_size'],
                                                  shuffle=False,
                                                  num_workers=opt['num_workers'])
        print("length of test set is: %d\n" % len(test_dataset))

        # load model
        found_model = MultiMAE_FT(opt).cuda()
        classifier = Classifier(num_classes=args.num_classes, embed_dim=opt['encoder']['embed_dim']).cuda()
        if torch.cuda.device_count() > 1:
            print("Lets use {} GPUs.".format(torch.cuda.device_count()))
        found_model = nn.DataParallel(found_model, device_ids=[i for i in range(torch.cuda.device_count())])
        classifier = nn.DataParallel(classifier, device_ids=[i for i in range(torch.cuda.device_count())])

        # load weights
        found_model_ckpt = found_model_ckpts[foldID]
        found_model.load_state_dict(torch.load(found_model_ckpt),strict=False)
        cls_model_ckpt = cls_model_ckpts[foldID]
        classifier.load_state_dict(torch.load(cls_model_ckpt))

        # test
        found_model.eval()
        classifier.eval()

        targets_record, preds_record, pred_probs_record = [], [], []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            for i, (video, eeg, ecg, eog, emg, gsr, modality_mask, label) in enumerate(
                    tqdm(test_loader)):
                video = video.to(device, non_blocking=True)
                eeg = eeg.to(device, non_blocking=True)
                ecg = ecg.to(device, non_blocking=True)
                eog = eog.to(device, non_blocking=True)
                emg = emg.to(device, non_blocking=True)
                gsr = gsr.to(device, non_blocking=True)
                modality_mask = modality_mask.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)

                valid_feat, video_feat, _, ecg_feat, _, emg_feat, gsr_feat, subFeatures = found_model.forward(video,
                                                                                                              eeg, ecg,
                                                                                                              eog, emg,
                                                                                                              gsr,
                                                                                                              modality_mask)
                output = classifier(valid_feat)
                pred_probs_record.extend(torch.softmax(output, dim=-1).to('cpu').detach().tolist())
                _, prediction = torch.max(output, dim=-1)
                preds_record.extend(prediction.to('cpu').detach().tolist())
                targets_record.extend(label.to('cpu').detach().tolist())

        acc = accuracy_score(targets_record, preds_record)
        precision = precision_score(y_true=targets_record, y_pred=preds_record, average='macro')
        recall = recall_score(y_true=targets_record, y_pred=preds_record, average='macro')
        f1 = f1_score(y_true=targets_record, y_pred=preds_record, average='macro')
        y_true_oneHot = label_binarize(targets_record, classes=list(range(args.num_classes)))
        auc = roc_auc_score(y_true=y_true_oneHot, y_score=pred_probs_record, multi_class='ovr', average='macro')
        fold_metrics['Acc'].append(acc)
        fold_metrics['Precision'].append(precision)
        fold_metrics['Recall'].append(recall)
        fold_metrics['F1'].append(f1)
        fold_metrics['AUC'].append(auc)

        print("for Fold [%d]: [Acc, Precision, Recall, F1, AUC]=[%.3f, %.3f, %.3f, %.3f,%.3f]\n" % (
            foldID + 1, acc * 100, precision * 100, recall * 100, f1 * 100, auc * 100
        ))
    print('--' * 15)
    print("[Overall Metrics]:\n")
    for k in fold_metrics:
        print("[%s]: avg=%.3f, std=%.3f" % (k, np.mean(fold_metrics[k]) * 100, np.std(fold_metrics[k],ddof=1) * 100))


if __name__ == "__main__":
    print('[External Validation] on [Sleep Staging] Task, from [Sleep-Cassette Dataset] to [Sleep-Telemetry Dataset')
    print('--' * 15)
    external_val_SleepCassette2SleepTelemetry()
