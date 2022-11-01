import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from utils.tools import StandardScaler
from utils.timefeatures import time_features
import torch
import numpy as np
import warnings

warnings.filterwarnings('ignore')


# SH 地铁数据集
class Dataset_SH_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                data_path='SHMetro.csv', split=[0.8, 0.1, 0.1]):
        if size == None:
            self.seq_len = 4
            self.label_len = 2
            self.pred_len = 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]  # 0

        self.train_split = split[0]
        self.test_split = split[1]
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path)).values
        number_of_day = 69
        C, N = df_raw.shape
        N = N - 1

        df_raw_sh_raw = df_raw[:, 1:].astype('float').reshape(-1, number_of_day, int(N / 2), 2)

        self.scaler.fit(df_raw_sh_raw.reshape(-1,2))
        df_raw_sh = self.scaler.transform(df_raw_sh_raw)
        df_raw_sh_raw = df_raw_sh_raw
        df_raw_sh = df_raw_sh.reshape(-1, number_of_day, int(N / 2), 2)
        df_raw_sh_stamp = df_raw.reshape(-1, number_of_day, N + 1)[:, :, 0]
        C, T, N, _ = df_raw_sh.shape

        self.seq_x_s = []
        self.seq_y_s = []
        self.seq_x_mark_s = []
        self.seq_y_mark_s = []
        self.raw_y_s = []
        for i in range(C):
            data = df_raw_sh[i, :, :]
            data_raw = df_raw_sh_raw[i, :, :]
            df_stamp = df_raw_sh_stamp[i, :]
            df_stamp = pd.DataFrame(df_stamp)
            df_stamp['date'] = pd.to_datetime(df_stamp[0])

            data_stamp = time_features(df_stamp)
            # print(data_stamp)

            data_len = len(data) - self.seq_len - self.pred_len + 1
            for s_begin in range(data_len):
                s_end = s_begin + self.seq_len
                r_begin = s_end - self.label_len
                r_end = r_begin + self.label_len + self.pred_len
                seq_x = data[s_begin:s_end]
                seq_y = data[r_begin:r_end]
                raw_y = data_raw[r_begin + self.label_len:r_end]
                seq_x_mark = data_stamp[s_begin:s_end]
                seq_y_mark = data_stamp[r_begin:r_end]
                self.seq_x_s.append(seq_x)
                self.seq_y_s.append(seq_y)
                self.seq_x_mark_s.append(seq_x_mark)
                self.seq_y_mark_s.append(seq_y_mark)
                self.raw_y_s.append(raw_y)
        self.seq_x_s = np.array(self.seq_x_s)
        self.seq_y_s = np.array(self.seq_y_s)
        self.seq_x_mark_s = np.array(self.seq_x_mark_s)
        self.seq_y_mark_s = np.array(self.seq_y_mark_s)
        self.raw_y_s = np.array(self.raw_y_s)
        sample_data = len(self.seq_x_s)
        # 训练集，验证集，测试集
        border1s = [0, int(sample_data * self.train_split) - self.seq_len,
                    int(sample_data * (self.train_split + self.test_split)) - self.seq_len]
        border2s = [int(sample_data * self.train_split), int(sample_data * (self.train_split + self.test_split)),
                    sample_data]
        self.border1 = border1s[self.set_type]
        self.border2 = border2s[self.set_type]

    def __getitem__(self, s_begin):
        seq_x = self.seq_x_s[self.border1 + s_begin]
        seq_y = self.seq_y_s[self.border1 + s_begin]
        seq_x_mark = self.seq_x_mark_s[self.border1 + s_begin]
        seq_y_mark = self.seq_y_mark_s[self.border1 + s_begin]
        raw_y = self.raw_y_s[self.border1 + s_begin]
        # print('seq_x', seq_x.shape)
        # print('seq_y', seq_y.shape)
        # print('seq_x_mark', seq_x_mark.shape)
        # print('seq_y_mark', seq_y_mark.shape)
        # print('raw_y', raw_y.shape)
        return seq_x, seq_y, seq_x_mark, seq_y_mark, raw_y

    def __len__(self):
        return self.border2 - self.border1

    def inverse_transform(self, data):
        # print('data.shape', data.shape)
        # print('self.scaler.std', self.scaler.std.shape)
        data = self.scaler.inverse_transform(data)
        return data


if __name__ == '__main__':
    train_data = Dataset_SH_minute(
        root_path='./SHMetro/',
        data_path='SHMetro.csv',
        flag='train',
        size=[4, 2, 4],
    )
    scaler = train_data.inverse_transform
    data_loader = DataLoader(
        train_data,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        drop_last=True)

    for i, (seq_x, seq_y, seq_x_mark, seq_y_mark, _) in enumerate(data_loader):
        print(seq_x.shape)
