import torch.nn.functional
import numpy as np
import pandas as pd
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import cv2
import re
import os
from torchvision import transforms

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
tfs = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)])


class IC_Dataset(Dataset):
    """
    model: string; ['train','test'];
    testFold: int; 5-fold validation, so testFold ranges from 0 to 4;
    sample_length: int; 30 s;
    """

    def __init__(self, opt, mode, testFold):
        super(IC_Dataset, self).__init__()
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
        subjects = os.listdir(os.path.join(self.data_dir, 'eeg'))
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
