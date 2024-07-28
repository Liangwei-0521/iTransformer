import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layer.encoder import EncoderLayer, Encoder
from src.layer.full_attention import FullAttention, AttentionLayer
from src.layer.embed import DataEmbedding_inverted
import argparse


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # 输入序列长度
        self.seq_len = configs.seq_len
        # 输出预测长度
        self.pred_len = configs.pred_len

        self.use_norm = configs.use_norm

        self.enc_embedding = DataEmbedding_inverted(configs.seq_len,
                                                    configs.d_model,
                                                    configs.embed,
                                                    configs.freq,
                                                    configs.dropout)
        # 初始化模型
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(mask_flag=False,
                                      factor=configs.factor,
                                      attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model,
                        configs.n_heads
                    ),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for layer in range(configs.n_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        self.number_features = configs.factor
        self.net = nn.Sequential(
            nn.Linear(self.number_features, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.ReLU()

        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        stdev = None
        means = None
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # Input: [B, L, N]
        # B: Batch_size
        # L: Length
        # N: The number of variate (变量个数）

        B, L, N = x_enc.shape
        # First step: Embedding (转置输入矩阵，对整个序列进行编码）
        # 编码输出的是转至后的编码矩阵：B L N -> B N E
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # Second step: 进行基于特征维度的注意力编码
        # B N E -> B N E
        enc_out, attns = self.encoder(enc_out, x_mark_enc)

        # Third step: 最后的模块, 进行线性编码，随后转置
        # B N E -> B N S -> B S N
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return self.net(dec_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :].squeeze(-1).contiguous()  # [B, L, D]



