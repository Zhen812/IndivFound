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
    num_classes: 7 classes: [0-> anger, 1->disgust, 2->fear, 3->happy, 4->neutral, 5->pain, 6->sad]
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

    def _get_label(self,text):
        if "neutral" in text:
            return 0
        if "happy" in text:
            return 1
        if "pain" in text:
            return 2
        return 3


    def _obtain_data(self):
        subjects = os.listdir(os.path.join(self.data_dir, 'video'))
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
            pre_video_dir = os.path.join(self.data_dir, 'video', subj)
            pre_ecg_dir = os.path.join(self.data_dir, 'ecg',subj)
            pre_emg_dir = os.path.join(self.data_dir, 'emg',subj)
            pre_gsr_dir = os.path.join(self.data_dir, 'gsr',subj)
            pre_dirs = [pre_ecg_dir, pre_emg_dir, pre_gsr_dir]
            filenames = os.listdir(pre_video_dir)
            for filename in filenames:
                file_dir=os.path.join(pre_video_dir,filename)
                clips=os.listdir(file_dir)
                for clip in clips:
                    contain_idx = self._get_label(clip)
                    if contain_idx != -1:
                        tmp=[os.path.join(file_dir,clip)]
                        clip_name=clip.split('.')[0]
                        new_clip_name_list=clip_name.split('_')
                        tmp.extend([os.path.join(pre_dir,'%s%s.npy'%('_'.join(new_clip_name_list[:-1]),new_clip_name_list[-1]))for pre_dir in pre_dirs])
                        file_paths.append(tmp)
                        labels.append(contain_idx)
        return file_paths, labels

    def get_video(self, video_name=None):
        video_width = self.opt['video']['width']
        video_height = self.opt['video']['height']
        video_length = self.opt['video']['length']
        video_channel = self.opt['video']['channel']
        video_tensor = torch.zeros((video_channel, video_length, video_height, video_width), dtype=torch.float32)
        video_indicator = 0
        if os.path.exists(video_name):
            cap = cv2.VideoCapture(video_name)
            count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                # 检查帧的通道数
                channels = frame.shape[2] if len(frame.shape) == 3 else 1
                if channels != 3:  # 如果通道数不为3，则进行转换
                    if channels == 1:  # 如果是单通道灰度图像，则将其转换为三通道的灰度图像
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 将BGR颜色转换为RGB颜色
                frame = tfs(frame)
                if count < video_length:
                    video_tensor[:, count, :, :] = frame
                    count += 1
                video_indicator = 1
        return video_tensor, video_indicator  # channel * length * height * width

    def get_ecg(self, ecg_name):
        ecg_length = self.opt['ecg']['ecg_length']
        ecg_channel = self.opt['ecg']['ecg_channel']
        ecg_tensor = torch.zeros((ecg_length, ecg_channel), dtype=torch.float32)
        ecg_indicator = 0
        # if not pd.isna(ecg_name):
        if os.path.exists(ecg_name):
            data = np.load(ecg_name)
            assert data.shape == (ecg_length, ecg_channel)
            # data = -np.log(np.maximum(data, 1e-6))
            std = np.std(data, axis=0)
            std[std == 0] = np.nan
            minv = np.nanmin(std)
            if np.isnan(minv):
                return ecg_tensor, ecg_indicator
            exponent = int(np.floor(np.log10(minv)))
            data = (data - np.mean(data, axis=0)) / np.maximum(np.std(data, axis=0), 10 ** exponent)
            ecg_tensor = torch.tensor(data, dtype=torch.float32)
            ecg_indicator = 1
            if np.mean(data) == 0 and np.std(data) == 0:
                ecg_indicator = 0
        return ecg_tensor, ecg_indicator  # ecg_length*2

    def get_emg(self, emg_name):
        emg_length = self.opt['emg']['emg_length']
        emg_channel = self.opt['emg']['emg_channel']
        emg_tensor = torch.zeros((emg_length, emg_channel), dtype=torch.float32)
        emg_indicator = 0
        if os.path.exists(emg_name):
            data = np.load(emg_name)
            assert data.shape[0] == emg_length, data.shape[1]==3
            std = np.std(data, axis=0)
            std[std == 0] = np.nan
            minv = np.nanmin(std)
            if np.isnan(minv):
                return emg_tensor, emg_indicator
            exponent = int(np.floor(np.log10(minv)))
            data = (data - np.mean(data, axis=0)) / np.maximum(np.std(data, axis=0), 10 ** exponent)
            emg_tensor[:,0] = torch.tensor(data[:,-1], dtype=torch.float32)
            emg_indicator = 1
            if np.mean(data) == 0 and np.std(data) == 0:
                emg_indicator = 0
        return emg_tensor, emg_indicator  # emg_length*1

    def get_gsr(self, gsr_name):
        gsr_length = self.opt['gsr']['gsr_length']
        gsr_channel = self.opt['gsr']['gsr_channel']
        gsr_tensor = torch.zeros((gsr_length, gsr_channel), dtype=torch.float32)
        gsr_indicator = 0
        if os.path.exists(gsr_name):
            data = np.load(gsr_name)
            assert data.shape == (gsr_length, gsr_channel)
            std = np.std(data, axis=0)
            std[std == 0] = np.nan
            minv = np.nanmin(std)
            if np.isnan(minv):
                return gsr_tensor, gsr_indicator
            exponent = int(np.floor(np.log10(minv)))
            data = (data - np.mean(data, axis=0)) / np.maximum(np.std(data, axis=0), 10 ** exponent)
            gsr_tensor = torch.tensor(data, dtype=torch.float32)
            gsr_indicator = 1
            if np.mean(data) == 0 and np.std(data) == 0:
                gsr_indicator = 0
        return gsr_tensor, gsr_indicator  # gsr_length*1

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, item):
        video_path = self.file_paths[item][0]
        ecg_path = self.file_paths[item][1]
        emg_path = self.file_paths[item][2]
        gsr_path = self.file_paths[item][3]
        '''
        print(video_path)
        print(ecg_path)
        print(emg_path)
        print(gsr_path)
        '''
        class_label = self.labels[item]

        video_data, video_indicator = self.get_video(video_path)
        '''
        video_data=torch.zeros(size=(
            self.opt['video']['channel'],self.opt['video']['length'],self.opt['video']['height'],
            self.opt['video']['width']))
        video_indicator=0
        '''
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
        eog_data = torch.zeros(size=(self.opt['eog']['eog_length'], self.opt['eog']['eog_channel']))
        eog_indicator = 0

        emg_data, emg_indicator = self.get_emg(emg_path)
        '''
        emg_data=torch.zeros(size=(
            self.opt['emg']['emg_length'],self.opt['emg']['emg_channel']))
        emg_indicator=0
        '''
        gsr_data, gsr_indicator = self.get_gsr(gsr_path)
        '''
        gsr_data=torch.zeros(size=(
            self.opt['gsr']['gsr_length'],self.opt['gsr']['gsr_channel']))
        gsr_indicator=0
        '''
        modality_mask = [video_indicator, eeg_indicator, ecg_indicator, eog_indicator, emg_indicator, gsr_indicator]
        modality_mask = torch.tensor(modality_mask, requires_grad=False)

        return video_data, eeg_data, ecg_data, eog_data, emg_data, gsr_data, modality_mask, class_label
