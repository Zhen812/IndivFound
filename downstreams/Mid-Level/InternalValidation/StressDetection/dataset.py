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

subj2anno = {
    "S2": {'age': 27, 'gender': 'm', 'height': 175, 'weight': 80},
    "S3": {'age': 27, 'gender': 'm', 'height': 173, 'weight': 69},
    "S4": {'age': 25, 'gender': 'm', 'height': 175, 'weight': 90},
    "S5": {'age': 35, 'gender': 'm', 'height': 189, 'weight': 80},
    "S6": {'age': 27, 'gender': 'm', 'height': 170, 'weight': 66},
    "S7": {'age': 28, 'gender': 'm', 'height': 184, 'weight': 74},
    "S8": {'age': 27, 'gender': 'f', 'height': 172, 'weight': 64},
    "S9": {'age': 26, 'gender': 'm', 'height': 181, 'weight': 75},
    "S10": {'age': 28, 'gender': 'm', 'height': 178, 'weight': 76},
    "S11": {'age': 26, 'gender': 'f', 'height': 171, 'weight': 54},
    "S13": {'age': 28, 'gender': 'm', 'height': 181, 'weight': 82},
    "S14": {'age': 27, 'gender': 'm', 'height': 180, 'weight': 80},
    "S15": {'age': 28, 'gender': 'm', 'height': 186, 'weight': 83},
    "S16": {'age': 24, 'gender': 'm', 'height': 184, 'weight': 69},
    "S17": {'age': 29, 'gender': 'f', 'height': 165, 'weight': 55},
}


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
        self.num_classes = opt['num_classes']
        self.file_paths, self.labels = self._obtain_data()

    def _contains_any(self, text, substrings):
        for i, substring in enumerate(substrings):
            if substring in text:
                return i
        return -1

    def _obtain_data(self):
        subjects = os.listdir(os.path.join(self.data_dir, 'ecg'))
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
        for subj in final_subjects:
            pre_ecg_dir = os.path.join(self.data_dir, 'ecg', subj)
            pre_emg_dir = os.path.join(self.data_dir, 'emg', subj)
            pre_gsr_dir = os.path.join(self.data_dir, 'gsr', subj)
            pre_dirs = [pre_ecg_dir, pre_emg_dir, pre_gsr_dir]
            filenames = os.listdir(pre_ecg_dir)
            for filename in filenames:
                if 'meditation' in filename:
                    continue
                file_dir = os.path.join(pre_ecg_dir, filename)
                clips = os.listdir(file_dir)
                for clip in clips:
                    contain_idx = self._contains_any(clip, ['baseline', 'stress', 'amusement'])
                    if contain_idx != -1:
                        file_paths.append([os.path.join(pre_dir, filename, clip) for pre_dir in pre_dirs])
                        labels.append(contain_idx)
        return file_paths, labels

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

    def get_emg(self, emg_name):
        emg_length = self.opt['emg']['emg_length']
        emg_channel = self.opt['emg']['emg_channel']
        emg_tensor = torch.zeros((emg_length, emg_channel), dtype=torch.float32)
        emg_indicator = 0
        # if not pd.isna(ecg_name):
        if os.path.exists(emg_name):
            data = np.load(emg_name)
            assert data.shape[0] <= emg_length, data.shape[1] == emg_channel
            # data = -np.log(np.maximum(data, 1e-6))
            std = np.std(data, axis=0)
            std[std == 0] = np.nan
            minv = np.nanmin(std)
            if np.isnan(minv):
                print(emg_name)
                return emg_tensor, emg_indicator
            exponent = int(np.floor(np.log10(minv)))
            data = (data - np.mean(data, axis=0)) / np.maximum(np.std(data, axis=0), 10 ** exponent)
            emg_tensor[:data.shape[0], :] = torch.tensor(data, dtype=torch.float32)
            emg_indicator = 1
            if np.mean(data) == 0 and np.std(data) == 0:
                emg_indicator = 0
        return emg_tensor, emg_indicator  # ecg_length*2

    def get_gsr(self, gsr_name):
        gsr_length = self.opt['gsr']['gsr_length']
        gsr_channel = self.opt['gsr']['gsr_channel']
        gsr_tensor = torch.zeros((gsr_length, gsr_channel), dtype=torch.float32)
        gsr_indicator = 0
        if os.path.exists(gsr_name):
            data = np.load(gsr_name)
            assert data.shape[0] <= gsr_length, data.shape[1] == gsr_channel
            std = np.std(data, axis=0)
            std[std == 0] = np.nan
            minv = np.nanmin(std)
            if np.isnan(minv):
                print(gsr_name)
                return gsr_tensor, gsr_indicator
            exponent = int(np.floor(np.log10(minv)))
            data = (data - np.mean(data, axis=0)) / np.maximum(np.std(data, axis=0), 10 ** exponent)
            gsr_tensor[:data.shape[0], :] = torch.tensor(data, dtype=torch.float32)
            gsr_indicator = 1
            if np.mean(data) == 0 and np.std(data) == 0:
                gsr_indicator = 0
        return gsr_tensor, gsr_indicator  # gsr_length*1

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, item):
        ecg_path = self.file_paths[item][0]
        emg_path = self.file_paths[item][1]
        gsr_path = self.file_paths[item][2]
        class_label = self.labels[item]

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
        '''
        ecg_data=torch.zeros(size=(
            self.opt['ecg']['ecg_length'],self.opt['ecg']['ecg_channel']))
        ecg_indicator=0
        '''
        # eog_data = torch.zeros(size=(self.opt['eog']['eog_length'], self.opt['eog']['eog_channel']))
        # eog_indicator = 0
        eog_data = torch.zeros(size=(
            self.opt['eog']['eog_length'], self.opt['eog']['eog_channel']))
        eog_indicator = 0

        emg_data, emg_indicator = self.get_emg(emg_path)

        # emg_data = torch.zeros(size=(
        #     self.opt['emg']['emg_length'], self.opt['emg']['emg_channel']))
        # emg_indicator = 0

        gsr_data, gsr_indicator = self.get_gsr(gsr_path)
        '''
        gsr_data=torch.zeros(size=(
            self.opt['gsr']['gsr_length'],self.opt['gsr']['gsr_channel']))
        gsr_indicator=0
        '''
        modality_mask = [video_indicator, eeg_indicator, ecg_indicator, eog_indicator, emg_indicator, gsr_indicator]
        modality_mask = torch.tensor(modality_mask, requires_grad=False)

        return video_data, eeg_data, ecg_data, eog_data, emg_data, gsr_data, modality_mask, class_label
