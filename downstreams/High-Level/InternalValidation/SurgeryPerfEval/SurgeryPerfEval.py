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
parser.add_argument('--gpuid', type=str, default="1,3,4,5,6,7")
parser.add_argument('--seed', type=int, default=21)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
set_seed(args.seed)
print("CUDA_VISIBLE_DEVICES:", args.gpuid)
print("Random Seed: ", args.seed)


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


class IC_Dataset_FLS(Dataset):
    """
    model: string; ['train','test'];
    testFold: int; 5-fold validation, so testFold ranges from 0 to 4;
    sample_length: int; 30 s;
    """

    def __init__(self, opt, mode, testFold):
        super(IC_Dataset_FLS, self).__init__()
        self.mode = mode
        self.opt = opt
        self.data_dir = opt['data_dir']
        self.testFold = testFold
        self.file_paths, self.labels = self._obtain_data()

    def _contains_any(self, text, substrings):
        for i, substring in enumerate(substrings):
            if substring in text:
                return i
        return -1

    def _obtain_data(self):
        subjects = os.listdir(self.data_dir)
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
            pre_eeg_dir = os.path.join(self.data_dir, subj)
            tasks = os.listdir(pre_eeg_dir)
            for task in tasks:
                task_dir = os.path.join(pre_eeg_dir, task)
                clips = os.listdir(task_dir)
                for clip in clips:
                    file_paths.append(os.path.join(task_dir, clip))
                    cur_label = int(clip.split('.')[0].split('_')[-2][5:])
                    labels.append((cur_label - 50.) / 50.)
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

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, item):
        eeg_path = self.file_paths[item]
        label = self.labels[item]

        # video_data, video_indicator = self.get_video(video_path)
        video_data = torch.zeros(size=(
            self.opt['video']['channel'], self.opt['video']['length'], self.opt['video']['height'],
            self.opt['video']['width']))
        video_indicator = 0

        eeg_data, eeg_indicator = self.get_eeg(eeg_path)

        # ecg_data, ecg_indicator = self.get_ecg(ecg_path)

        ecg_data = torch.zeros(size=(
            self.opt['ecg']['ecg_length'], self.opt['ecg']['ecg_channel']))
        ecg_indicator = 0

        # eog_data = torch.zeros(size=(self.opt['eog']['eog_length'], self.opt['eog']['eog_channel']))
        # eog_indicator = 0
        eog_data = torch.zeros(size=(
            self.opt['eog']['eog_length'], self.opt['eog']['eog_channel']))
        eog_indicator = 0

        # emg_data, emg_indicator = self.get_emg(emg_path)

        emg_data = torch.zeros(size=(
            self.opt['emg']['emg_length'], self.opt['emg']['emg_channel']))
        emg_indicator = 0

        # gsr_data, gsr_indicator = self.get_gsr(gsr_path)

        gsr_data = torch.zeros(size=(
            self.opt['gsr']['gsr_length'], self.opt['gsr']['gsr_channel']))
        gsr_indicator = 0

        modality_mask = [video_indicator, eeg_indicator, ecg_indicator, eog_indicator, emg_indicator, gsr_indicator]
        modality_mask = torch.tensor(modality_mask, requires_grad=False)

        return video_data, eeg_data, ecg_data, eog_data, emg_data, gsr_data, modality_mask, label


def evaluate():
    with open("./FLS_config.yaml", encoding="UTF-8") as f:
        opt = yaml.safe_load(f)

    found_model_ckpts = opt['found_model_ckpts']
    regress_model_ckpts = opt['regress_model_ckpts']
    folds = list(range(5))
    fold_metrics = {k: [] for k in ['MAE']}
    for foldID in folds:
        # load data
        test_dataset = IC_Dataset_FLS(opt, mode='test', testFold=foldID)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=opt['batch_size'],
                                                  shuffle=False,
                                                  num_workers=opt['num_workers'])
        print("length of test set is: %d\n" % len(test_dataset))

        # load model
        found_model = MultiMAE_FT(opt).cuda()
        regressor = Regressor(embed_dim=opt['encoder']['embed_dim']).cuda()

        if torch.cuda.device_count() > 1:
            print("Lets use {} GPUs.".format(torch.cuda.device_count()))
        found_model = nn.DataParallel(found_model, device_ids=[i for i in range(torch.cuda.device_count())])
        regressor = nn.DataParallel(regressor, device_ids=[i for i in range(torch.cuda.device_count())])

        # load weights
        found_model_ckpt = found_model_ckpts[foldID]
        found_model.load_state_dict(torch.load(found_model_ckpt), strict=False)
        regress_model_ckpt = regress_model_ckpts[foldID]
        regressor.load_state_dict(torch.load(regress_model_ckpt))

        # test
        found_model.eval()
        regressor.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = nn.L1Loss()
        A_loss_overall = []
        with torch.no_grad():
            for i, (video, eeg, ecg, eog, emg, gsr, modality_mask, label) in enumerate(tqdm(test_loader)):
                video = video.to(device, non_blocking=True)
                eeg = eeg.to(device, non_blocking=True)
                ecg = ecg.to(device, non_blocking=True)
                eog = eog.to(device, non_blocking=True)
                emg = emg.to(device, non_blocking=True)
                gsr = gsr.to(device, non_blocking=True)
                modality_mask = modality_mask.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)

                valid_feat, video_feat, _, ecg_feat, _, emg_feat, gsr_feat, subFeatures = found_model.forward(video,
                                                                                                              eeg,
                                                                                                              ecg, eog,
                                                                                                              emg,
                                                                                                              gsr,
                                                                                                              modality_mask)
                output = regressor(valid_feat).squeeze()
                loss = criterion(output, label)
                A_loss_overall.append(loss.to('cpu').detach())

        mae = np.mean(np.array(A_loss_overall))
        print("for Fold [%d]: MAE=%.3f" % (foldID + 1, mae * 50.))
        fold_metrics['MAE'].append(mae * 50.)
    print('--' * 15)
    print("[Overall Metrics]:\n")
    for k in fold_metrics:
        print("[%s]: avg=%.3f, std=%.3f" % (k, np.mean(fold_metrics[k]), np.std(fold_metrics[k], ddof=1)))


if __name__ == "__main__":
    print('[Internal Validation] on [Surgery Performance Evaluation] Task with [NIBIB-RPCCC-FLS Dataset]')
    print('--' * 15)
    evaluate()
