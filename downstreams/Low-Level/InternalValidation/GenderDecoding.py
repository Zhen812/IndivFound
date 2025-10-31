import torch.nn.functional
import numpy as np
import pandas as pd
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import cv2
from torchvision import transforms
import os
import argparse
import yaml
import pickle
import torch.nn as nn
import sys
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize

cur_dir = os.getcwd()
par_dir = os.path.abspath(os.path.join(cur_dir, os.pardir))
grandpar_dir = os.path.abspath(os.path.join(par_dir, os.pardir))
sys.path.append(grandpar_dir)
from src.utils.helper import set_seed
from src.model.encoder_ms import MultiMAE_FT, Classifier

parser = argparse.ArgumentParser()
parser.add_argument('--label', type=str, default="Sex")
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--gpuid', type=str, default="0,1,2,3")
parser.add_argument('--seed', type=int, default=21)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
set_seed(args.seed)
print("CUDA_VISIBLE_DEVICES:", args.gpuid)
print("Random Seed: ", args.seed)


class IC_Dataset(Dataset):
    def __init__(self, opt, mode="train", label="Age_group", testFold=None):
        # label in ["Age_group","Sex","BMI"]
        super(IC_Dataset, self).__init__()
        self.opt = opt
        self.mode = mode
        self.label = label
        self.testFold = testFold
        self.file_paths, self.labels = self._obtain_data()

    def _get_bmi_label(self, num):
        if num < 30:
            return 0
        return 1

    def _obtain_data(self):
        subject2anno = {}
        subj_csv_path = self.opt['anno_path']
        df = pd.read_csv(subj_csv_path)
        for i in range(len(df)):
            subjID = "%04d" % int(df.at[i, 'ID'])
            if not pd.isna(df.at[i, self.label]):
                label = int(df.at[i, self.label])
                if self.label == "Age_group":
                    label -= 1
                if self.label == "BMI":
                    label = self._get_bmi_label(label)
                subject2anno[subjID] = label

        subjects = subject2anno.keys()
        final_subjects = []
        if 'train' in self.mode:
            for i, item in enumerate(subjects):
                if (i + 1) % 5 != self.testFold:
                    final_subjects.append(item)
        else:
            for i, item in enumerate(subjects):
                if (i + 1) % 5 == self.testFold:
                    final_subjects.append(item)

        data_dir = self.opt['data_dir']
        filenames = os.listdir(data_dir)
        file_paths, labels = [], []
        for filename in filenames:
            subjectID = filename.split('_')[0]
            if subjectID in final_subjects:
                labels.append(subject2anno[subjectID])
                file_paths.append(os.path.join(data_dir, filename))

        if os.path.join(data_dir,"0065_clip027.npy") in file_paths:
            idx = file_paths.index(os.path.join(data_dir,"0065_clip027.npy"))
            del file_paths[idx], labels[idx]
        return file_paths, labels

    # 读取处理ECG
    def get_ecg(self, ecg_name):
        ecg_length = self.opt['ecg']['ecg_length']
        ecg_channel = self.opt['ecg']['ecg_channel']
        ecg_tensor = torch.zeros((ecg_length, ecg_channel), dtype=torch.float32)
        ecg_indicator = 0
        # if not pd.isna(ecg_name):
        if os.path.exists(ecg_name):
            data = np.load(ecg_name)
            assert data.shape[0] <= ecg_length, data.shape[1] == ecg_channel
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
        label = self.labels[index]

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


def evaluate_ours():
    with open("./config.yaml", encoding="UTF-8") as f:
        opt = yaml.safe_load(f)

    found_model_ckpts = opt['found_model_ckpts']['genderDecoding']
    cls_model_ckpts = opt['cls_model_ckpts']['genderDecoding']
    folds = list(range(5))
    fold_metrics = {k: [] for k in ['Acc', 'Precision', 'Recall', 'F1', 'AUC']}
    for foldID in folds:
        # load data
        test_dataset = IC_Dataset(opt, mode='test', label=args.label, testFold=foldID)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=opt['batch_size'],
                                                  shuffle=False,
                                                  num_workers=opt['num_workers'])
        print("length of test set is: %d\n" % len(test_dataset))

        # load model
        found_model = MultiMAE_FT(opt).cuda()
        classifier = Classifier(num_classes=opt['num_classes'], embed_dim=opt['encoder']['embed_dim']).cuda()
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
        precisions = [0 for _ in range(opt['num_classes'])]
        recalls = [0 for _ in range(opt['num_classes'])]
        f1s = [0 for _ in range(opt['num_classes'])]
        TPs, pos_labels, pos_preds = [0] * opt['num_classes'], [0] * opt['num_classes'], [0] * opt['num_classes']

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
        print("[%s]: avg=%.3f, std=%.3f" % (k, np.mean(fold_metrics[k]) * 100, np.std(fold_metrics[k], ddof=1) * 100))


if __name__ == '__main__':
    print('[Internal Validation] on [Gender Decoding] Task with [Autonomic-Aging] Dataset')
    print('--' * 15)
    evaluate_ours()
