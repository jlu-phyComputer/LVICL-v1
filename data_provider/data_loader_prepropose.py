import os
import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from data_provider.m4 import M4Dataset, M4Meta
from sklearn.preprocessing import StandardScaler
from utils.tools import convert_tsf_to_dataframe
import warnings
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv',
                 scale=True, seasonal_patterns=None, drop_short=False):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.token_len = self.seq_len - self.label_len
        self.token_num = self.seq_len // self.token_len
        self.flag = flag
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date).apply(str)
        self.data_stamp = df_stamp['date'].values
        self.data_stamp = [str(x) for x in self.data_stamp]

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        data_name = self.data_path.split('.')[0]
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        data_stamp_x = self.data_stamp[s_begin:s_end]
        data_stamp_y = self.data_stamp[r_begin:r_end]
        # seq_x_mark = self.data_stamp[s_begin:s_end:self.token_len]
        # seq_y_mark = self.data_stamp[s_end:r_end:self.token_len]

        return seq_x, seq_y, data_stamp_x, data_stamp_y

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


if __name__ == "__main__":
    data_set = Dataset_ETT_hour(
        'XXX',
        size=[672, 576, 96])
    data_loader = DataLoader(
        data_set,
        batch_size=1,
        shuffle=False,
        num_workers=10,
        drop_last=False)
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
        print("")

