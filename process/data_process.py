import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from sklearn.preprocessing import MinMaxScaler



class data_process:
    def __init__(self, path, window_length, num_features):
        super().__init__()
        self.x = []
        self.y = []
        self.num_features = num_features
        self.data = pd.read_excel(path)
        self.window_length = window_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def get(self, ):
        pass

    def normalize(self, x):
        return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-5)

    def do(self, ):
        for i in range(0, (len(self.data) // 96) - self.window_length):
            # 标准化
            x = self.normalize(self.data.iloc[i * 96:(i + self.window_length) * 96, 1:-1].values)
            # self.x.append(x.reshape(self.window_length, 96, self.num_features))
            self.x.append(x)
            y = self.data.iloc[(i+1) * 96:(i + self.window_length+1) * 96, -1:].values.reshape(self.window_length, 96)
            y = self.scaler.fit_transform(y.T)

            self.y.append(y.T[-1, :])

        return self.x, self.y


class dataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)


if __name__ == '__main__':
    
    import numpy as np
    scaler = MinMaxScaler(feature_range=(0, 1))
   
    d_process = data_process(path='./src/data/sub_df_train.xlsx', window_length=3, num_features=6)
    x, y = d_process.do()
    data_set = dataset(x=x, y=y)

    # print(x[-1].shape)
    print(y[-1], type(y[-1]))

    # DataLoader
    dataloader = DataLoader(dataset=data_set, num_workers=4, shuffle=True, batch_size=3)
    for index, batch in enumerate(dataloader):
        batch_x, batch_y = batch
        print('batch_x:{}, batch_y:{}'.format(batch_x.shape, batch_y.shape))


    
    



