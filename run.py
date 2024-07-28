import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'test'))

import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn
from src.model.ITransformer import Model
from iTransformer_test import Model_test
from sklearn.metrics import mean_absolute_error
from src.data.api_get import make



def normalize(x):
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-5)


def main():

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
    sub_data = make()
    x = torch.from_numpy(normalize(x = sub_data.iloc[:, 1:-1]).to_numpy(dtype='float32')).unsqueeze(dim=0)
    iTransformer_ = Model_test(configs=args)
    predict_result = iTransformer_.predict(x_enc=x, x_mark_enc=None, x_dec=None, x_mark_dec=None)
                                            

    predict_max = sub_data['clean_price'][-96*2:-96].max()
    predict_min = sub_data['clean_price'][-96*2:-96].min()
    predict_result = predict_result * (predict_max - predict_min) + predict_min
    
    # 创建DataFrame
    predict_result = predict_result.detach().numpy()
    predict_result = pd.DataFrame(predict_result[0])
    print(predict_result)

    df = pd.DataFrame(columns=['Time', 'Value', 'Data_Type', 'Algorithm'])
    # 时间
    tomorrow = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
    time_series = pd.date_range(start=tomorrow, periods=96, freq='15T')
    time_series = pd.DataFrame(time_series, columns=['Datetime'])

    df['Time'] = time_series
    # 预测值
    df['Value'] = predict_result
    # 数据类型
    df['Data_Type'] = 'day_ahead_price' 
    # Algorithm
    df['Algorithm'] = 'iTransformer'
    print(df)

    import datetime
    from database.result_upload import Prediction_Upload

    upload_time_Str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    upload_time = pd.Timestamp(upload_time_Str)
    # result_smooth.df_smooth: 对应整理好的预测结果

    Prediction_Upload(df = df,
                    Province ='Shanxi',
                    Data_Type ='day_ahead_price',
                    Algorithm = 'iTransformer',
                    Data_Source= None,  # 未使用到气象数据
                    upload_time = upload_time)

    #%%触发api，数据同步至influxdb
    from database.result_upload import Data_sync
    Data_sync(database ='Shanxi',
            upload_time = upload_time_Str,
            bucket ='Shanxi_Price_Forecast',
            model_version='v1.0.0')

    return predict_result


if __name__ == '__main__':
    main()