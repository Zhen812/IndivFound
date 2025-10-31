import torch.nn.functional
import numpy as np
import pandas as pd
import torch
import torch.nn.functional
import cv2
from torchvision import transforms
import os
from torch.utils.data import Dataset
import pickle
import argparse
import yaml
import torch.nn as nn
import sys

cur_dir = os.getcwd()
par_dir = os.path.abspath(os.path.join(cur_dir, os.pardir))
grandpar_dir = os.path.abspath(os.path.join(par_dir, os.pardir))
great_gandpar_dir = os.path.abspath(os.path.join(grandpar_dir, os.pardir))
sys.path.append(great_gandpar_dir)
from src.utils.helper import set_seed
from src.model.encoder_ms import MultiMAE_FT, Classifier
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument('--label', type=str, default="Age_group")
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--gpuid', type=str, default="0,1,2,3")
parser.add_argument('--seed', type=int, default=21)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
set_seed(args.seed)
print("CUDA_VISIBLE_DEVICES:", args.gpuid)
print("Random Seed: ", args.seed)


class IC_Dataset_ECGID(Dataset):
    def __init__(self, opt, label):
        # label in ["Age","Sex"]
        super(IC_Dataset_ECGID, self).__init__()
        self.opt = opt
        self.label = label

        with open(opt['anno_path'], 'rb')as f:
            self.subj2anno = pickle.load(f)
        self.file_paths, self.ages, self.sexes = self._obtain_data()

    def _obtain_data(self):
        data_dir = self.opt['data_dir']
        subjects = list(self.subj2anno.keys())
        file_paths = []
        ages, sexes = [], []
        for subject in subjects:
            subject_dir = os.path.join(data_dir, subject)
            filenames = os.listdir(subject_dir)
            for filename in filenames:
                file_path = os.path.join(subject_dir, filename)
                key = subject
                age = self.subj2anno[key]['age']
                sex = self.subj2anno[key]['sex']
                file_paths.append(file_path)
                sexes.append(sex)
                ages.append(age)
        return file_paths, ages, sexes

    # 读取处理ECG
    def get_ecg(self, ecg_name):
        ecg_length = self.opt['ecg']['ecg_length']
        ecg_channel = self.opt['ecg']['ecg_channel']
        ecg_tensor = torch.zeros((ecg_length, ecg_channel), dtype=torch.float32)
        ecg_indicator = 0
        # if not pd.isna(ecg_name):
        if os.path.exists(ecg_name):
            data = np.load(ecg_name)
            # assert data.shape[0] <= ecg_length, data.shape[1] == ecg_channel
            # data = -np.log(np.maximum(data, 1e-6))
            std = np.std(data, axis=0)
            std[std == 0] = np.nan
            minv = np.nanmin(std)
            if np.isnan(minv):
                print(ecg_name)
                return ecg_tensor, ecg_indicator
            exponent = int(np.floor(np.log10(minv)))
            data = (data - np.mean(data, axis=0)) / np.maximum(np.std(data, axis=0), 10 ** exponent)
            ecg_tensor[:data.shape[0], :] = torch.tensor(data, dtype=torch.float32)
            ecg_indicator = 1
            if np.mean(data) == 0 and np.std(data) == 0:
                ecg_indicator = 0
        return ecg_tensor, ecg_indicator  # ecg_length*2

    def __getitem__(self, index):
        ecg_path = self.file_paths[index]
        age = self.ages[index]
        sex = self.sexes[index]
        label = age if 'Age' in self.label else sex

        # video_data, video_indicator = self.get_video(video_path)
        video_data = torch.zeros(size=(
            self.opt['video']['channel'], self.opt['video']['length'], self.opt['video']['height'],
            self.opt['video']['width']))
        video_indicator = 0

        eeg_data = torch.zeros(size=(
            self.opt['eeg']['eeg_channel'], self.opt['eeg']['eeg_length'], self.opt['eeg']['eeg_height'],
            self.opt['eeg']['eeg_width'],))
        eeg_indicator = 0

        ecg_data, ecg_indicator = self.get_ecg(ecg_path)

        eog_data = torch.zeros(size=(
            self.opt['eog']['eog_length'], self.opt['eog']['eog_channel']))
        eog_indicator = 0

        emg_data = torch.zeros(size=(
            self.opt['emg']['emg_length'], self.opt['emg']['emg_channel']))
        emg_indicator = 0

        gsr_data = torch.zeros(size=(
            self.opt['gsr']['gsr_length'], self.opt['gsr']['gsr_channel']))
        gsr_indicator = 0

        modality_mask = [video_indicator, eeg_indicator, ecg_indicator, eog_indicator, emg_indicator, gsr_indicator]
        modality_mask = torch.tensor(modality_mask, requires_grad=False)
        return video_data, eeg_data, ecg_data, eog_data, emg_data, gsr_data, modality_mask, label

    def __len__(self):
        return len(self.file_paths)


def external_val_AutonomicAging2ECGID():
    with open("./ECGID_config.yaml", encoding="UTF-8") as f:
        opt = yaml.safe_load(f)
    # load data
    test_dataset = IC_Dataset_ECGID(opt, label=args.label)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=opt['batch_size'],
                                              shuffle=False,
                                              num_workers=opt['num_workers'])
    print("length of test set is: %d\n" % len(test_dataset))

    folds = list(range(5))
    found_model_ckpts = opt['found_model_ckpts']['autonomicAging2ECGID']
    cls_model_ckpts = opt['cls_model_ckpts']['autonomicAging2ECGID']
    fold_metrics = {k: [] for k in ['Acc', 'Precision', 'Recall', 'F1', 'AUC']}
    for foldID in folds:
        # load model
        found_model = MultiMAE_FT(opt).cuda()
        classifier = Classifier(num_classes=args.num_classes, embed_dim=opt['encoder']['embed_dim']).cuda()
        if torch.cuda.device_count() > 1:
            print("Lets use {} GPUs.".format(torch.cuda.device_count()))
            found_model = nn.DataParallel(found_model, device_ids=[i for i in range(torch.cuda.device_count())])
            classifier = nn.DataParallel(classifier, device_ids=[i for i in range(torch.cuda.device_count())])

        # load weights
        found_model_ckpt = found_model_ckpts[foldID]
        found_model.load_state_dict(torch.load(found_model_ckpt), strict=False)
        cls_model_ckpt = cls_model_ckpts[foldID]
        classifier.load_state_dict(torch.load(cls_model_ckpt))

        # test
        found_model.eval()
        classifier.eval()

        targets_record, preds_record, pred_probs_record = [], [], []
        precisions = [0 for _ in range(args.num_classes)]
        recalls = [0 for _ in range(args.num_classes)]
        f1s = [0 for _ in range(args.num_classes)]
        TPs, pos_labels, pos_preds = [0] * args.num_classes, [0] * args.num_classes, [0] * args.num_classes

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

        for cls in range(args.num_classes):
            for idx in range(len(targets_record)):
                if preds_record[idx] == cls:
                    pos_preds[cls] += 1
                if targets_record[idx] == cls:
                    pos_labels[cls] += 1
                if targets_record[idx] == preds_record[idx] and targets_record[idx] == cls:
                    TPs[cls] += 1
            precisions[cls] = TPs[cls] * 1. / (pos_preds[cls] + 1e-3)
            recalls[cls] = TPs[cls] * 1. / (pos_labels[cls] + 1e-3)
            f1s[cls] = 2. * TPs[cls] / (pos_preds[cls] + pos_labels[cls] + 1e-3)

        acc = accuracy_score(targets_record, preds_record)
        precision = sum(precisions) / args.num_classes
        recall = sum(recalls) / args.num_classes
        f1 = sum(f1s) / args.num_classes
        auc = roc_auc_score(y_true=targets_record, y_score=np.array(pred_probs_record)[:, 1])
        print("for Fold [%d]: [Acc, Precision, Recall, F1, AUC]=[%.3f, %.3f, %.3f, %.3f,%.3f]\n" % (
            foldID + 1, acc * 100, precision * 100, recall * 100, f1 * 100, auc * 100
        ))
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
        print("[%s]: avg=%.3f, std=%.3f" % (k, np.mean(fold_metrics[k]) * 100, np.std(fold_metrics[k]) * 100))


if __name__ == '__main__':
    print('[External Validation] on [Age Grouping task], from [Autonomic-Aging] Datset to [ECGID] Datset')
    print('--' * 15)
    external_val_AutonomicAging2ECGID()
