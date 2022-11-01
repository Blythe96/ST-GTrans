import torch
import torch.nn as nn
import numpy as np
from model.layers import *


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, E):
        '''
        Q: [B, T,h, N, d_k]
        K: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        V: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k][B,H,T,N,D]
        E: [B,T,H,N,N]
        '''
        B, T, H, N, d_k = Q.shape
        s_ = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # [B,T,h,N,N]
        e_ = torch.mul(s_, E)  # [B,T,h,N,N]

        s = nn.Softmax(dim=-1)(e_)  # [12, 8, 4, 288, 288] * [12, 8,4,288, 288]
        context = torch.matmul(s, V)  # [B,T,h,,N,D]
        return context, e_


class SMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads, seq_length, dropout=0.0):
        super(SMultiHeadAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.seq_len = seq_length
        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        # 用Linear来做投影矩阵
        # 但这里如果是多头的话，是不是需要声明多个矩阵？？？

        self.W_V = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_Q = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_E = nn.Linear(self.embed_size, self.seq_len * self.heads, bias=False)

        self.norm_o = nn.LayerNorm(self.embed_size)
        self.norm_e = nn.LayerNorm(self.embed_size)

        self.fc_out_o = nn.Linear(heads * self.head_dim, embed_size)
        self.fc_out_e = nn.Linear(heads * self.seq_len, embed_size)

        self.dropout = dropout

    def forward(self, input_Q, input_K, input_V, input_E):
        '''
        input_Q: [batch_size, N, T, C]
        input_K: [batch_size, N, T, C]
        input_V: [batch_size, N, T, C]
        input_E: [batch_size, N, N, C]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        # 输入B,D,N,T
        input_Q = input_Q.transpose(1, 3)  # (B,T,N,D)
        input_K = input_K.transpose(1, 3)
        input_V = input_V.transpose(1, 3)

        B, T, N, C = input_Q.shape

        # (B,T,N,D) --> [B,T,N, h * d_k] --> [B,T,N, h * d_k] --> [B, h, T, N, d_k]
        Q = self.W_Q(input_Q).view(B, T, N, self.heads, self.head_dim).transpose(2, 3)  # Q: [B, T,h, N, d_k]
        K = self.W_K(input_K).view(B, T, N, self.heads, self.head_dim).transpose(2, 3)  # K: [B,  T,h, N, d_k]
        V = self.W_V(input_V).view(B, T, N, self.heads, self.head_dim).transpose(2, 3)  # V: [B,  T,h, N, d_k]
        E = self.W_E(input_E).view(B, N, N, self.heads, T).permute(0, 4, 3, 1, 2)  # E: [B,N,N,H,T] ---> [B,T,H,N,N]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, E = ScaledDotProductAttention()(Q, K, V, E)  # [B, T,H, N, D_k]
        context = context.transpose(2, 3).contiguous().view(B, T, N, self.heads * self.head_dim)  # [B, T, N,H, D_k]
        E = E.permute(0, 4, 3, 1, 2).reshape(B, N, N, -1)  # [B,N,N,h*T]
        context, E = self.fc_out_o(context), self.fc_out_e(E)
        context, E = self.norm_o(context + input_Q), self.norm_e(E + input_E)
        context, E = F.dropout(context, self.dropout, training=self.training), F.dropout(E, self.dropout,
                                                                                         training=self.training)
        context = context.permute(0, 3, 2, 1) # (B,D,N,T)
        return context, E


'''
spatial  
输入 ： B,D,N,T
输出 :  (B,N,T,D)
'''
if __name__ == '__main__':
    model = SMultiHeadAttention(embed_size=64, seq_length=4, heads=8)
    data = torch.randn(12, 64, 288, 4)
    e = torch.randn(12, 288, 288, 64)
    res, E = model(data, data, data, e)
    print('res shape is {},E shape is {}'.format(res.shape, E.shape))
