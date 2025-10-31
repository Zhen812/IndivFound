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
'''
cur_dir = os.getcwd()
par_dir = os.path.abspath(os.path.join(cur_dir, os.pardir))
grandpar_dir = os.path.abspath(os.path.join(par_dir, os.pardir))
sys.path.append(grandpar_dir)
'''
from src.model.encoder_ms import MultiMAE_FT
from src.utils.helper import set_seed

parser = argparse.ArgumentParser()
parser.add_argument('--gpuid', type=str, default="0,1,2,3,4,5,6,7")
parser.add_argument('--seed', type=int, default=21)
parser.add_argument('--num_classes', type=int, default=3)
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


def evaluate():
    with open("./WESAD_config.yaml", encoding="UTF-8") as f:
        opt = yaml.safe_load(f)
    folds = list(range(5))
    found_model_ckpts = opt['found_model_ckpts']
    cls_model_ckpts = opt['cls_model_ckpts']
    fold_metrics = {k: [] for k in ['Acc', 'Precision', 'Recall', 'F1', 'AUC']}
    for foldID in folds:
        # load data
        test_dataset = IC_Dataset(opt, mode='test', testFold=foldID)
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
        print("[%s]: avg=%.3f, std=%.3f" % (k, np.mean(fold_metrics[k]) * 100, np.std(fold_metrics[k],ddof=1) * 100))


if __name__ == "__main__":
    print('[Internal Validation] on [Stress Detection] Task with [WESAD Dataset]')
    print('--' * 15)
    evaluate()
