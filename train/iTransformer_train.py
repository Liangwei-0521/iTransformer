import os
import sys
import time
sys.path.append(os.getcwd())

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from src.model.ITransformer import Model
from process.data_process import data_process, dataset

import matplotlib.pyplot as plt

# 数据
d_process = data_process(path='./src/data/sub_df_train.xlsx', window_length=3, num_features=6)
x, y = d_process.do()

data_set = dataset(x=x, y=y)
dataloader = DataLoader(dataset=data_set, num_workers=4, shuffle=True, batch_size=3)


class Model_train:
    def __init__(self, configs, n_epochs):
        super().__init__()
        self.itransformer_model = Model(configs).to(device=configs.device)
        self.train_data = dataloader
        self.lr = configs.lr
        self.optimizer = optim.Adam(lr=self.lr,
                                     params=self.itransformer_model.parameters())
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                              'min', patience=10, 
                                                              verbose=True)
        self.n_epochs = n_epochs
        self.device = configs.device

    def do(self, ):
        all_loss_value = []
        for epoch in tqdm(range(self.n_epochs)):
            epoch_loss = 0
            for idx, (x_batch, y_batch) in enumerate(self.train_data):

                self.optimizer.zero_grad()
                predict = self.itransformer_model(x_enc=x_batch.to(self.device), x_mark_enc=None, x_dec=None, x_mark_dec=None)
                # L2 Loss 均方误差损失函数MSE
                # L1 Loss 绝对误差损失函数MAE
                loss_value = F.l1_loss(predict, y_batch.to(self.device))
                epoch_loss = epoch_loss + loss_value.item()
                print(f'------ >>> Epochs: {epoch} >>>  ------ idx: {idx}, 损失值：{loss_value.item():.4f}')

                loss_value.backward()
                self.optimizer.step()

                if epoch % 5 == 0 and epoch >= 5:
                    print('预测出清价格 --- 日前：', predict, '\n', '真实出清出清价格 --- 日前：', y_batch)
                    break
                # with SummaryWriter() as writer:
                #     writer.add_graph()
                #     writer.add_scalar("loss", loss_value)

            epoch_loss = epoch_loss / len(self.train_data)
            all_loss_value.append(epoch_loss)
            # 自适应调整 learning rate
            self.scheduler.step(epoch_loss)

        print('\n ---------- 模型训练完毕---------- \n ')
        torch.save(self.itransformer_model.state_dict(),
                   './result/iTransformer_model.pth',
                    _use_new_zipfile_serialization=False)

        # 损失
        plt.figure(figsize=(8, 5))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid()
        plt.plot(range(0, len(all_loss_value)), all_loss_value, linestyle='-', label='Loss')
        plt.legend(loc='upper right')
        plt.savefig('./result/model_loss.png')
        plt.show()


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
    parser.add_argument('--lr', type=str, default=0.0005,
                        help='learning rate')
    parser.add_argument('--freq', type=str, default='t',
                        help='time frequency')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    train_model = Model_train(configs=args, n_epochs=100)
    train_model.do()
