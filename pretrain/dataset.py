import torch.nn.functional
import numpy as np
import pandas as pd
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import cv2
from torchvision import transforms
import os

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
tfs = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)])


class IC_Dataset(Dataset):
    def __init__(self, opt):
        super(IC_Dataset, self).__init__()
        self.opt = opt
        self.df = self._obtain_df()

    def _obtain_df(self):
        dataset_list = self.opt['dataset_list']
        dataset_csv_list = self.opt['dataset_csv_path']
        length = len(dataset_list)

        df = pd.DataFrame()
        valid_fields = ['Video', 'EEG', 'ECG', 'EOG', 'EMG', 'GSR']
        for i in range(length):
            dataset_name = dataset_list[i]
            dataset_csv_path = dataset_csv_list[i]
            subdf = pd.read_csv(dataset_csv_path)
            subdf = subdf[valid_fields]
            subdf['DatasetName'] = dataset_name
            df = pd.concat((df, subdf))
        df['Video'] = df['Video'].astype(str)
        df['EEG'] = df['EEG'].astype(str)
        df['ECG'] = df['ECG'].astype(str)
        df['EOG'] = df['EOG'].astype(str)
        df['EMG'] = df['EMG'].astype(str)
        df['GSR'] = df['GSR'].astype(str)
        return df

    # 读取处理video
    def get_video(self, video_name=None):
        video_width = self.opt['video']['width']
        video_height = self.opt['video']['height']
        video_length = self.opt['video']['length']
        video_channel = self.opt['video']['channel']
        video_tensor = torch.zeros((video_channel, video_length, video_height, video_width), dtype=torch.float32)
        video_chl_mask = torch.zeros(
            (1, video_height // self.opt['video']['patch_size'], video_width // self.opt['video']['patch_size']),
            requires_grad=False, dtype=torch.float32)
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
            video_chl_mask = torch.ones(
                (1, video_height // self.opt['video']['patch_size'], video_width // self.opt['video']['patch_size']),
                requires_grad=False)
        return video_tensor, video_indicator, video_chl_mask

    # 读取处理EEG
    def get_eeg(self, eeg_name):
        eeg_width = self.opt['eeg']['eeg_width']
        eeg_height = self.opt['eeg']['eeg_height']
        eeg_length = self.opt['eeg']['eeg_length']
        eeg_channel = self.opt['eeg']['eeg_channel']
        eeg_tensor = torch.zeros((eeg_channel, eeg_length, eeg_height, eeg_width), dtype=torch.float32)
        eeg_indicator = 0
        eeg_chl_mask = torch.zeros((1, eeg_height, eeg_width), requires_grad=False)
        if os.path.exists(eeg_name):
            data = np.load(eeg_name)
            assert data.shape == (eeg_length, eeg_height, eeg_width)
            std = np.std(data, axis=0)
            std[std == 0] = np.nan
            minv = np.nanmin(std)
            if np.isnan(minv):
                return eeg_tensor, eeg_indicator, eeg_chl_mask
            exponent = int(np.floor(np.log10(minv)))
            data = (data - np.mean(data, axis=0)) / np.maximum(np.std(data, axis=0), 10 ** exponent)

            eeg_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
            eeg_indicator = 1
            eeg_chl_mask = torch.sum(eeg_tensor.reshape(eeg_channel * eeg_length, eeg_height, eeg_width), dim=0,
                                     keepdim=True) == 0
            eeg_chl_mask = 1 - eeg_chl_mask.float()
            eeg_chl_mask.requires_grad = False
            if np.mean(data) == 0 and np.std(data) == 0:
                eeg_indicator = 0
        return eeg_tensor, eeg_indicator, eeg_chl_mask

    # 读取处理ECG
    def get_ecg(self, ecg_name):
        ecg_length = self.opt['ecg']['ecg_length']
        ecg_channel = self.opt['ecg']['ecg_channel']
        ecg_tensor = torch.zeros((ecg_length, ecg_channel), dtype=torch.float32)
        ecg_chl_mask = torch.zeros((1, ecg_channel), requires_grad=False)
        ecg_indicator = 0
        if os.path.exists(ecg_name):
            data = np.load(ecg_name)
            assert data.shape == (ecg_length, ecg_channel)
            # 计算数据的标准差，处理零值，并检查是否存在有效的标准差; 若输入信号完全相同，视为无效数据；
            std = np.std(data, axis=0)
            std[std == 0] = np.nan
            minv = np.nanmin(std)
            if np.isnan(minv):
                return ecg_tensor, ecg_indicator, ecg_chl_mask
            exponent = int(np.floor(np.log10(minv)))
            data = (data - np.mean(data, axis=0)) / np.maximum(np.std(data, axis=0), 10 ** exponent)
            ecg_tensor = torch.tensor(data, dtype=torch.float32)
            ecg_indicator = 1
            ecg_chl_mask = torch.ones((1, ecg_channel), requires_grad=False, dtype=torch.float32)
            if np.mean(data) == 0 and np.std(data) == 0:
                ecg_indicator = 0
        return ecg_tensor, ecg_indicator, ecg_chl_mask

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
            assert data.shape == (eog_length, eog_channel)
            # data = -np.log(np.maximum(data, 1e-6))
            std = np.std(data, axis=0)
            std[std == 0] = np.nan
            minv = np.nanmin(std)
            if np.isnan(minv):
                return eog_tensor, eog_indicator, eog_chl_mask
            exponent = int(np.floor(np.log10(minv)))
            data = (data - np.mean(data, axis=0)) / np.maximum(np.std(data, axis=0), 10 ** exponent)
            eog_tensor = torch.tensor(data, dtype=torch.float32)
            eog_indicator = 1
            eog_chl_mask = torch.ones((1, eog_channel), requires_grad=False, dtype=torch.float32)
            if np.mean(data) == 0 and np.std(data) == 0:
                eog_indicator = 0
        return eog_tensor, eog_indicator, eog_chl_mask

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
                return emg_tensor, emg_indicator, emg_chl_mask
            exponent = int(np.floor(np.log10(minv)))
            data = (data - np.mean(data, axis=0)) / np.maximum(np.std(data, axis=0), 10 ** exponent)
            emg_tensor = torch.tensor(data, dtype=torch.float32)
            emg_indicator = 1
            emg_chl_mask = torch.sum(emg_tensor, dim=0, keepdim=True) == 0
            emg_chl_mask = 1 - emg_chl_mask.float()
            if np.mean(data) == 0 and np.std(data) == 0:
                emg_indicator = 0
        return emg_tensor, emg_indicator, emg_chl_mask

    # 读取处理GSR
    def get_gsr(self, gsr_name):
        gsr_length = self.opt['gsr']['gsr_length']
        gsr_channel = self.opt['gsr']['gsr_channel']
        gsr_tensor = torch.zeros((gsr_length, gsr_channel), dtype=torch.float32)
        gsr_indicator = 0
        gsr_chl_mask = torch.zeros((1, gsr_channel), requires_grad=False, dtype=torch.float32)
        # if not pd.isna(gsr_name):
        if os.path.exists(gsr_name):
            data = np.load(gsr_name)
            assert data.shape == (gsr_length, gsr_channel)
            std = np.std(data, axis=0)
            std[std == 0] = np.nan
            minv = np.nanmin(std)
            if np.isnan(minv):
                return gsr_tensor, gsr_indicator, gsr_chl_mask
            exponent = int(np.floor(np.log10(minv)))
            data = (data - np.mean(data, axis=0)) / np.maximum(np.std(data, axis=0), 10 ** exponent)
            gsr_tensor = torch.tensor(data, dtype=torch.float32)
            gsr_indicator = 1
            gsr_chl_mask = torch.ones((1, 1), requires_grad=False, dtype=torch.float32)
            if np.mean(data) == 0 and np.std(data) == 0:
                gsr_indicator = 0
        return gsr_tensor, gsr_indicator, gsr_chl_mask

    def __getitem__(self, index):
        row = self.df.iloc[index]
        video_path = str(row['Video'])
        eeg_path = str(row['EEG'])
        ecg_path = str(row['ECG'])
        eog_path = str(row['EOG'])
        emg_path = str(row['EMG'])
        gsr_path = str(row['GSR'])

        video, video_indicator, video_chl_mask = self.get_video(video_path)
        EEG, EEG_indicator, EEG_chl_mask = self.get_eeg(eeg_path)
        ECG, ECG_indicator, ECG_chl_mask = self.get_ecg(ecg_path)
        EOG, EOG_indicator, EOG_chl_mask = self.get_eog(eog_path)
        EMG, EMG_indicator, EMG_chl_mask = self.get_emg(emg_path)
        GSR, GSR_indicator, GSR_chl_mask = self.get_gsr(gsr_path)

        modality_mask = [0] * 6
        modality_mask[0] = video_indicator
        modality_mask[1] = EEG_indicator
        modality_mask[2] = ECG_indicator
        modality_mask[3] = EOG_indicator
        modality_mask[4] = EMG_indicator
        modality_mask[5] = GSR_indicator
        modality_mask = torch.tensor(modality_mask, requires_grad=False, dtype=torch.bool)

        return (video, EEG, ECG, EOG, EMG, GSR, modality_mask, video_chl_mask, EEG_chl_mask, ECG_chl_mask,
                EOG_chl_mask, EMG_chl_mask, GSR_chl_mask)

    def __len__(self):
        return len(self.df)
