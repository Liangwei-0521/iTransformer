import os
import sys
sys.path.append(os.getcwd())

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt

import torch
from src.model.ITransformer import Model
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error


def normalize(x):
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-5)


class test_data_process:
    def __init__(self, path, window_length, num_features):
        super().__init__()
        self.x = []
        self.y = []
        self.num_features = num_features
        self.data = pd.read_excel(path)
        self.window_length = window_length

    def get(self, ):
        pass

    def do(self, ):
        for i in range(0, (len(self.data) // 96) - self.window_length):
            # 标准化
            x = normalize(self.data.iloc[i * 96:(i + self.window_length) * 96, 1:-1].values)
            # self.x.append(x.reshape(self.window_length, 96, self.num_features))
            self.x.append(x)
            y = self.data.iloc[(i + 1) * 96:(i + self.window_length + 1) * 96, -1:].values.reshape(-1, 96)
            # 测试集的Y：不进行标准化
            self.y.append(y[-1, :])

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


class Model_test:
    def __init__(self, configs):
        self.test_model = Model(configs)
        self.test_model.load_state_dict(torch.load(configs.model_path,
                                                   map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
        self.test_model = self.test_model.eval()

    def predict(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        result = self.test_model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return result


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='ITransformer')

    parser.add_argument('--seq_len', type=int, required=False, default=288,
                        help='input the sequence of length')
    parser.add_argument('--pred_len', type=int, required=False, default=96,
                        help='output the sequence of length')
    parser.add_argument('--d_model', type=int, required=False, default=128, 
                        help='the dimension of model')
    parser.add_argument('--n_layers', type=int, required=False, default=4,
                        help='the number of layers')
    parser.add_argument('--factor', type=int, required=False, default=6,
                        help='the number of features')
    parser.add_argument('--n_heads', type=int, required=False, default=4,
                        help='the number of heads')
    parser.add_argument('--activation', type=str, default='gelu',
                        help='activation')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout')
    parser.add_argument('--d_ff', type=int, default=2048,
                        help='dimension of fcn')
    parser.add_argument('--output_attention', action='store_true',
                        help='whether to output attention in encoder')
    parser.add_argument('--use_norm', type=int, default=False,
                        help='use norm and denorm')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--lr', type=str, default=0.0001,
                        help='learning rate')
    parser.add_argument('--freq', type=str, default='t',
                        help='time frequency')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--model_path', type=str, 
                        default='./result/iTransformer_model_version_l1.pth')

    args = parser.parse_args()

    # 实例化测试模型
    iTransformer_ = Model_test(configs=args)

    # 最大-最小值
    history_train = pd.read_excel('./src/data/sub_df_train.xlsx')
    predict_max = history_train.iloc[-96:, -1].max()
    predict_min = history_train.iloc[-96:, -1].min()
    print('最大值：', predict_max, '最小值：', predict_min)

    # 测试数据
    d_process = test_data_process(path='./src/data/sub_df_test.xlsx', window_length=3, num_features=6)
    x, y = d_process.do()
    data_set = dataset(x=x, y=y)
    test_loader = DataLoader(dataset=data_set, num_workers=4, shuffle=False, batch_size=1)

    # 测试时间
    df = d_process.data['Date']
    df['timestamp'] = pd.to_datetime(d_process.data['Date'])
    # 提取日期部分，并获取唯一值
    unique_dates = df['timestamp'].dt.date.unique()
    # 格式化日期为 'YYYY-MM-DD'
    test_time = [pd.Timestamp(date).strftime('%Y-%m-%d') for date in unique_dates][2:]

    window_len = 3
  
    for idx, (x, y) in enumerate(test_loader):

        predict_max = history_train.iloc[96*(idx+window_len-1):96*(idx+window_len), -1].mean()
        predict_min = history_train.iloc[96*(idx+window_len-1):96*(idx+window_len), -1].std()

        
        predict_result = iTransformer_.predict(x_enc=x, x_mark_enc=None, x_dec=None, x_mark_dec=None)

        print(predict_result)

        
        # predict_result = predict_result * (predict_max - predict_min) + predict_min
        predict_result = predict_result * predict_min + predict_max

        predict_mae = mean_absolute_error(predict_result[0].tolist(), y[0].tolist())
        print('MAE:', predict_mae)
        # 可视化预测和真实结果
        plt.figure(figsize=(8, 5))
        plt.title('Date:' +  test_time[idx] +' \n MAE:'+str(predict_mae))
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.plot(range(0, 96), predict_result[0].tolist(), label='Predict')
        plt.plot(range(0, 96), y[0].tolist(), label='Real')
        plt.legend()
        plt.show()




