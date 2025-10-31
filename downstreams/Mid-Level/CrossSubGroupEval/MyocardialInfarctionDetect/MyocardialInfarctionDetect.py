import torch.nn.functional
import numpy as np
import pandas as pd
import torch
import os
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, recall_score, precision_score, \
    f1_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
from dataset import IC_Dataset
import argparse
import sys
import yaml
import torch.nn as nn
from src.model.encoder_ms import MultiMAE_FT
from src.utils.helper import set_seed

parser = argparse.ArgumentParser()
parser.add_argument('--gpuid', type=str, default="1,3,4,5,6,7")
parser.add_argument('--seed', type=int, default=21)
parser.add_argument('--num_classes', type=int, default=2)
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


def cross_subgroup_val_M2F_mae():
    with open("./PTB_config.yaml", encoding="UTF-8") as f:
        opt = yaml.safe_load(f)
    folds = list(range(1))
    for foldID in folds:
        # load data
        test_dataset = IC_Dataset(opt, mode='test', filter_choice=('Sex', 0))
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
        found_model.load_state_dict(torch.load(opt['found_model_ckpts']['m2f']), strict=False)
        classifier.load_state_dict(torch.load(opt['cls_model_ckpts']['m2f']))

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
        precision = precision_score(y_true=targets_record, y_pred=preds_record)
        recall = recall_score(y_true=targets_record, y_pred=preds_record)
        f1 = f1_score(y_true=targets_record, y_pred=preds_record)
        print(np.array(pred_probs_record).shape)
        auc = roc_auc_score(y_true=targets_record, y_score=np.array(pred_probs_record)[:, 1])
        print("[IndivFound]: [Acc, Precision, Recall, F1, AUC]=[%.3f, %.3f, %.3f, %.3f,%.3f]\n" % (
            acc * 100, precision * 100, recall * 100, f1 * 100, auc * 100
        ))


def cross_subgroup_val_F2M_mae():
    with open("./PTB_config.yaml", encoding="UTF-8") as f:
        opt = yaml.safe_load(f)
    folds = list(range(1))
    for foldID in folds:
        # load data
        test_dataset = IC_Dataset(opt, mode='test', filter_choice=('Sex', 1))
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
        found_model.load_state_dict(torch.load(opt['found_model_ckpts']['f2m']), strict=False)
        classifier.load_state_dict(torch.load(opt['cls_model_ckpts']['f2m']))

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
        precision = precision_score(y_true=targets_record, y_pred=preds_record)
        recall = recall_score(y_true=targets_record, y_pred=preds_record)
        f1 = f1_score(y_true=targets_record, y_pred=preds_record)
        print(np.array(pred_probs_record).shape)
        auc = roc_auc_score(y_true=targets_record, y_score=np.array(pred_probs_record)[:, 1])
        print("[IndivFound]: [Acc, Precision, Recall, F1, AUC]=[%.3f, %.3f, %.3f, %.3f,%.3f]\n" % (
            acc * 100, precision * 100, recall * 100, f1 * 100, auc * 100
        ))


def cross_subgroup_val_Y2O_mae():
    with open("./PTB_config.yaml", encoding="UTF-8") as f:
        opt = yaml.safe_load(f)
    folds = list(range(1))
    for foldID in folds:
        # load data
        test_dataset = IC_Dataset(opt, mode='test', filter_choice=('Age_group', 0))
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
        found_model.load_state_dict(torch.load(opt['found_model_ckpts']['y2o']), strict=False)
        classifier.load_state_dict(torch.load(opt['cls_model_ckpts']['y2o']))

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
        precision = precision_score(y_true=targets_record, y_pred=preds_record)
        recall = recall_score(y_true=targets_record, y_pred=preds_record)
        f1 = f1_score(y_true=targets_record, y_pred=preds_record)
        print(np.array(pred_probs_record).shape)
        auc = roc_auc_score(y_true=targets_record, y_score=np.array(pred_probs_record)[:, 1])
        print("[IndivFound]: [Acc, Precision, Recall, F1, AUC]=[%.3f, %.3f, %.3f, %.3f,%.3f]\n" % (
            acc * 100, precision * 100, recall * 100, f1 * 100, auc * 100
        ))


def cross_subgroup_val_O2Y_mae():
    with open("./PTB_config.yaml", encoding="UTF-8") as f:
        opt = yaml.safe_load(f)
    folds = list(range(1))
    for foldID in folds:
        # load data
        test_dataset = IC_Dataset(opt, mode='test', filter_choice=('Age_group', 1))
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
        found_model.load_state_dict(torch.load(opt['found_model_ckpts']['o2y']), strict=False)
        classifier.load_state_dict(torch.load(opt['cls_model_ckpts']['o2y']))

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
        precision = precision_score(y_true=targets_record, y_pred=preds_record)
        recall = recall_score(y_true=targets_record, y_pred=preds_record)
        f1 = f1_score(y_true=targets_record, y_pred=preds_record)
        auc = roc_auc_score(y_true=targets_record, y_score=np.array(pred_probs_record)[:, 1])
        print("[IndivFound]: [Acc, Precision, Recall, F1, AUC]=[%.3f, %.3f, %.3f, %.3f,%.3f]\n" % (
            acc * 100, precision * 100, recall * 100, f1 * 100, auc * 100
        ))


if __name__ == "__main__":
    print('[Cross Sub-Group Validation] on [Myocardial Infarction Detection] Task with [PTB Dataset]')
    print('apply model trained on [Female] Group on [Male] Group')
    cross_subgroup_val_F2M_mae()
    print('apply model trained on [Male] Group on [Female] Group')
    cross_subgroup_val_M2F_mae()
    print('apply model trained on [Old] Group on [Young] Group')
    cross_subgroup_val_O2Y_mae()
    print('apply model trained on [Young] Group on [Old] Group')
    cross_subgroup_val_Y2O_mae()
