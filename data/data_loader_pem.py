import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from utils.tools import StandardScaler
from utils.timefeatures import time_features
import torch
import warnings

warnings.filterwarnings('ignore')


# PEMS 地铁数据集
class Dataset_pems_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
               data_path='PEMS03.csv', split=[0.8,0.1, 0.1]):
        if size == None:
            self.seq_len = 12
            self.label_len = 6
            self.pred_len = 12
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
                                          self.data_path))
        sample_data = len(df_raw)
        # 训练集，验证集，测试集
        border1s = [0, int(sample_data * self.train_split) - self.seq_len,
                    int(sample_data * (self.train_split + self.test_split)) - self.seq_len]
        border2s = [int(sample_data * self.train_split), int(sample_data * (self.train_split + self.test_split)),
                    sample_data]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        cols_data = df_raw.columns[1:]
        nodes = len(cols_data)
        df_data = df_raw[cols_data].values  # (6348, 575)
        df_data = df_data.reshape(sample_data, nodes, -1)  # (6348, 288, 2)

        train_data = df_data[border1:border2, :, :]
        self.scaler.fit(train_data)  # g
        data = self.scaler.transform(df_data)

        raw_data = df_data
        '''
        data 归一化数据
        df_data 未归一化数据

        '''
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp)
        self.data_x = data[border1:border2]

        self.raw_data_y = raw_data[border1:border2]  # 原始标签

        self.data_y = data[border1:border2]  # 归一化decoder数据

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        raw_y = self.raw_data_y[r_begin + self.label_len:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, raw_y

    def __len__(self):
        # return len(self.data_x)
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


if __name__ == '__main__':
    train_data = Dataset_pems_minute(
        root_path='./PEMS03/',
        data_path='PEMS03.csv',
        flag='train',
        size=[12, 6, 12],
        scale=True,
        inverse=True
    )
    scaler = train_data.inverse_transform
    data_loader = DataLoader(
        train_data,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        drop_last=True)

    for i, (seq_x, seq_y, seq_x_mark, seq_y_mark) in enumerate(data_loader):
        print(seq_x.shape)
