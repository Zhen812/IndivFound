import torch.nn.functional
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import os
import argparse
import yaml
import torch.nn as nn
import sys
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize
from src.utils.helper import set_seed
from src.model.encoder_ms import MultiMAE_FT

parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=4)
parser.add_argument('--gpuid', type=str, default="1,3,4,5,6,7")
parser.add_argument('--seed', type=int, default=21)
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


class IC_Dataset_HGD(Dataset):
    def __init__(self, opt, mode, testFold):
        super(IC_Dataset_HGD, self).__init__()
        self.opt = opt
        self.mode = mode
        self.data_dir = opt['data_dir']
        self.testFold = testFold
        self.num_classes = opt['num_classes']
        self.file_paths, self.labels = self._obtain_data()

    def _contains_any(self, text, substrings):
        for i, substring in enumerate(substrings):
            if substring in text:
                return i
        return -1

    def _obtain_data(self):
        subjects = ['s%02d' % i for i in range(1, 15)]
        final_subjects = []
        if 'train' in self.mode:
            for i, item in enumerate(subjects):
                if (i + 1) % 5 != self.testFold:
                    final_subjects.append(item)
        else:
            for i, item in enumerate(subjects):
                if (i + 1) % 5 == self.testFold:
                    final_subjects.append(item)

        file_paths, labels = [], []
        for subj in final_subjects:
            pre_eeg_dir = os.path.join(self.data_dir, 'eeg', subj)
            pre_eog_dir = os.path.join(self.data_dir, 'eog', subj)
            filenames = os.listdir(pre_eeg_dir)
            for filename in filenames:
                contain_idx = self._contains_any(filename,
                                                 ['left_hand', 'right_hand', 'rest', 'feet'])
                if contain_idx != -1:
                    file_paths.append([os.path.join(pre_eeg_dir, filename), os.path.join(pre_eog_dir, filename)])
                    labels.append(contain_idx)
        return file_paths, labels

    def get_eeg(self, eeg_name):
        eeg_width = self.opt['eeg']['eeg_width']
        eeg_height = self.opt['eeg']['eeg_height']
        eeg_length = self.opt['eeg']['eeg_length']
        eeg_channel = self.opt['eeg']['eeg_channel']
        eeg_tensor = torch.zeros((eeg_channel, eeg_length, eeg_height, eeg_width), dtype=torch.float32)
        if os.path.exists(eeg_name):
            data = np.load(eeg_name)
            length = min(eeg_length, data.shape[0])
            std = np.std(data, axis=0)
            std[std == 0] = np.nan
            minv = np.nanmin(std)
            exponent = int(np.floor(np.log10(minv)))
            data = (data - np.mean(data, axis=0)) / np.maximum(np.std(data, axis=0), 10 ** exponent)
            eeg_tensor = torch.zeros((eeg_channel, eeg_length, eeg_height, eeg_width), dtype=torch.float32)
            eeg_tensor[0, :length, :, :] = torch.tensor(data[:length, :, :], dtype=torch.float32)
        return eeg_tensor  # channel * eeg_length * height * width

    def get_eog(self, eog_name):
        eog_length = self.opt['eog']['eog_length']
        eog_channel = self.opt['eog']['eog_channel']
        eog_tensor = torch.zeros((eog_length, eog_channel), dtype=torch.float32)
        eog_chl_mask = torch.zeros((1, eog_channel), requires_grad=False, dtype=torch.float32)
        eog_indicator = 0
        # if not pd.isna(eog_name):
        if os.path.exists(eog_name):
            data = np.load(eog_name).transpose()
            length = data.shape[0]
            assert length <= eog_length, data.shape[1] == eog_channel
            # data = -np.log(np.maximum(data, 1e-6))
            std = np.std(data, axis=0)
            std[std == 0] = np.nan
            minv = np.nanmin(std)
            if np.isnan(minv):
                return eog_tensor, eog_indicator, eog_chl_mask
            exponent = int(np.floor(np.log10(minv)))
            data = (data - np.mean(data, axis=0)) / np.maximum(np.std(data, axis=0), 10 ** exponent)
            eog_tensor[:length, :] = torch.tensor(data, dtype=torch.float32)
            eog_indicator = 1
            eog_chl_mask = torch.ones((1, eog_channel), requires_grad=False, dtype=torch.float32)
            if np.mean(data) == 0 and np.std(data) == 0:
                eog_indicator = 0
        return eog_tensor, eog_indicator

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, item):
        eeg_path = self.file_paths[item][0]
        eog_path = self.file_paths[item][1]
        class_label = self.labels[item]

        video_data = torch.zeros(
            size=(self.opt['video']['channel'], self.opt['video']['length'], self.opt['video']['height'],
                  self.opt['video']['width']))
        video_indicator = 0

        eeg_data = self.get_eeg(eeg_path)
        eeg_indicator = 1

        ecg_data = torch.zeros(size=(self.opt['ecg']['ecg_length'], self.opt['ecg']['ecg_channel']))
        ecg_indicator = 0

        eog_data, eog_indicator = self.get_eog(eog_path)
        # eog_indicator = 1

        emg_data = torch.zeros(size=(self.opt['emg']['emg_length'], self.opt['emg']['emg_channel']))
        emg_indicator = 0

        gsr_data = torch.zeros(size=(self.opt['gsr']['gsr_length'], self.opt['gsr']['gsr_channel']))
        gsr_indicator = 0

        modality_mask = [video_indicator, eeg_indicator, ecg_indicator, eog_indicator, emg_indicator, gsr_indicator]
        modality_mask = torch.tensor(modality_mask, requires_grad=False)

        return video_data, eeg_data, ecg_data, eog_data, emg_data, gsr_data, modality_mask, class_label


def evaluate():
    with open("./HGD_config.yaml", encoding="UTF-8") as f:
        opt = yaml.safe_load(f)
    found_model_ckpts = opt['found_model_ckpts']
    cls_model_ckpts = opt['cls_model_ckpts']
    folds = list(range(5))
    fold_metrics = {k: [] for k in ['Acc', 'Precision', 'Recall', 'F1', 'AUC']}
    for foldID in folds:
        # load data
        test_dataset = IC_Dataset_HGD(opt, mode='test', testFold=foldID)
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
        found_model.load_state_dict(torch.load(found_model_ckpt), strict=False)
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
        print("for Fold [%d]: [Acc, Precision, Recall, F1, AUC]=[%.3f, %.3f, %.3f, %.3f,%.3f]\n" % (
            foldID + 1, acc * 100, precision * 100, recall * 100, f1 * 100, auc * 100
        ))
        fold_metrics['Acc'].append(acc)
        fold_metrics['Precision'].append(precision)
        fold_metrics['Recall'].append(recall)
        fold_metrics['F1'].append(f1)
        fold_metrics['AUC'].append(auc)
    print('--' * 15)
    print("[Overall Metrics]:\n")
    for k in fold_metrics:
        print("[%s]: avg=%.3f, std=%.3f" % (k, np.mean(fold_metrics[k]) * 100, np.std(fold_metrics[k], ddof=1) * 100))


if __name__ == '__main__':
    print('[Internal Validation] on [Motor Imagery Classification] Task with [High Gamma Dataset]')
    print('--' * 15)
    evaluate()
