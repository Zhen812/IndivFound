import torch.nn.functional
import numpy as np
import pandas as pd
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import cv2
from torchvision import transforms
import os
import pickle

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
tfs = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)])


class IC_Dataset(Dataset):
    def __init__(self, opt, mode="train", testFold=None):
        super(IC_Dataset, self).__init__()
        self.opt = opt
        self.mode = mode
        self.testFold = testFold
        self.file_paths, self.labels = self._obtain_data()

    def _contains_any(self, text, substrings):
        for i, substring in enumerate(substrings):
            if substring in text:
                return i
        return -1

    def _obtain_data(self):
        data_dir = self.opt['data_dir']
        subjects = os.listdir(data_dir)
        subjects.sort()
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
        for subject in final_subjects:
            subject_dir = os.path.join(data_dir, subject)
            if not os.path.exists(subject_dir): pass
            filenames = os.listdir(subject_dir)
            for filename in filenames:
                label = self._contains_any(filename, ['Healthy', 'MyocardialInfarction'])
                if label != -1:
                    labels.append(label)
                    file_paths.append(os.path.join(subject_dir, filename))
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

