import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import math

import numpy as np


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.3, max_len=5000, lookup_index=None):
        super(TemporalPositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.lookup_index = lookup_index
        self.max_len = max_len
        # computing the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0).unsqueeze(0)  # (1, 1, T_max, d_model)
        self.register_buffer('pe', pe)
        # register_buffer:
        # Adds a persistent buffer to the module.
        # This is typically used to register a buffer that should not to be considered a model parameter.

    def forward(self, x):
        '''
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, N, T, F_out)
        '''
        if self.lookup_index is not None:
            x = self.pe[:, :, self.lookup_index, :]  # (batch_size, N, T, F_in) + (1,1,T,d_model)
        else:
            x = self.pe[:, :, :x.size(2), :]
        res = self.dropout(x.detach())  # [1, 1, 4, 64]
        return res  #


class startConv(nn.Module):
    def __init__(self, in_dim=2, residual_channels=64):
        '''
        :param in_dim: 2
        :param residual_channels: 32
        '''

        super(startConv, self).__init__()
        self.tokenConv = nn.Conv2d(in_channels=in_dim,
                                   out_channels=residual_channels,
                                   kernel_size=(1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        '''
        :param x: (B,N,T,D)
        :return:
        '''
        x = x.transpose(1, 3)  # [B,D,N,T]
        x = self.tokenConv(x)
        x = x.transpose(1, 3)
        return x  # (B,N,T,D)


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        '''
        d_model 默认 512
        c_in 默认 7
        '''
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        y = self.emb(x).detach()
        return y


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()
        minute_size = 4;  # 15*4 = 60
        hour_size = 24  # 24*1 = 24
        weekday_size = 7;  # 7*1 = 7
        day_size = 32;
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])
        y = hour_x + weekday_x + day_x + month_x + minute_x  # (B,T,D)
        y = y.unsqueeze(1)
        return y


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='t'):
        super(TimeFeatureEmbedding, self).__init__()
        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)

    def forward(self, x):
        y = self.embed(x)
        return y


class DataEmbedding(nn.Module):
    def __init__(self, in_dim, d_model, embed_type='fixed', freq='t', dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = startConv(in_dim, d_model)
        self.position_embedding = TemporalPositionalEncoding(d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark, TE=True, PE=True):
        '''
        :param x: (B,N,T,D)
        :param x_mark:
        :return:
        '''
        a = self.value_embedding(x)  # (B,N,T,D)
        if PE:
            b = self.position_embedding(x)  # [1,1,T,D ] #
        else:
            b = torch.zeros_like(a)
        if TE:
            c = self.temporal_embedding(x_mark)
        else:
            c = torch.zeros(a)
        x = a + b + c
        return self.dropout(x)  # x s shape is (B,N,D,T)


# mixprop
class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncwl,vw->ncvl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias)

    def forward(self, x):
        return self.mlp(x)


class mixprop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        adj = adj.to(torch.float32)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)
        return ho


class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask
        return adj

    def fullA(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        return adj


class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input):
        if self.elementwise_affine:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight[:, :, :], self.bias[:, :, :], self.eps)
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
               'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


# ------------------------
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        '''
        Q: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        K: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        V: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        attn_mask: [batch_size, n_heads, seq_len, seq_len] 可能没有
        '''
        B, n_heads, len1, len2, d_k = Q.shape
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        # scores : [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), N(Spatial) or T(Temporal)]
        # scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn,
                               V)  # [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]]
        return context


# SMHA 空间Transformer
class SMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SMultiHeadAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        # 用Linear来做投影矩阵
        # 但这里如果是多头的话，是不是需要声明多个矩阵？？？

        self.W_V = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_Q = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, N, T, C]
        input_K: [batch_size, N, T, C]
        input_V: [batch_size, N, T, C]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        # (B,D,N,T) 输出也是B,D,N,T
        input_Q = input_Q.permute(0, 2, 3, 1)  # (B,N,T,D)
        input_K = input_K.permute(0, 2, 3, 1)
        input_V = input_V.permute(0, 2, 3, 1)

        B, N, T, C = input_Q.shape
        # [B, N, T, C] --> [B, N, T, h * d_k] --> [B, N, T, h, d_k] --> [B, h, T, N, d_k]
        Q = self.W_Q(input_Q).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)  # Q: [B, h, T, N, d_k]
        K = self.W_K(input_K).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)  # K: [B, h, T, N, d_k]
        V = self.W_V(input_V).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)  # V: [B, h, T, N, d_k]

        # attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context = ScaledDotProductAttention()(Q, K, V)  # [B, h, T, N, d_k]
        context = context.permute(0, 3, 2, 1, 4)  # [B, N, T, h, d_k]
        context = context.reshape(B, N, T, self.heads * self.head_dim)  # [B, N, T, C]
        # context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        # output = self.fc_out(context)  # [batch_size, len_q, d_model]
        output = context.permute(0, 3, 1, 2)  # B,D,N,T
        return output


# temporal transformer

def attention(query, key, value, mask=None, dropout=None):
    '''

    :param query:  (batch, h, d_k, N, T)
    :param key: (batch, h, d_k, N, T)
    :param value: (batch, h, d_k, N, T)
    :param mask: (batch, 1, 1, T2, T2)
    :param dropout:
    '''
    d_k = query.size(2)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # scores: (batch, h, d_k, N, N)
    if mask is not None:
        scores = scores.masked_fill_(mask == 0, -1e9)  # -1e9 means attention scores=0
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn  # (batch, h, d_k, N, T), (batch, h, d_k, N, N)


class TemporalHeadAttention(nn.Module):
    def __init__(self, d_model, nb_head, kernel_size, T_dim, dropout):
        '''
        :param d_model:  64
        :param nb_head: 8
        (B,D,N,T)
        '''
        super(TemporalHeadAttention, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head  # d_k = 8
        self.h = nb_head  # 8
        self.padding = (kernel_size - 1) // 2
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)

        self.Conv1D_Q = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(1, kernel_size),
                                  padding=(0, self.padding))
        self.Conv1D_K = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(1, kernel_size),
                                  padding=(0, self.padding))
        self.linear_V = nn.Linear(T_dim, T_dim)
        self.linear_end = nn.Linear(T_dim, T_dim)

    def forward(self, input_q, input_k, input_v, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, T, T), same mask applied to all h heads.
        nbatches = input_q.shape[0]
        N = input_q.shape[2]
        query = self.Conv1D_Q(input_q).view(nbatches, self.h, self.d_k, N, -1)
        key = self.Conv1D_K(input_k).view(nbatches, self.h, self.d_k, N, -1)
        value = self.linear_V(input_v).view(nbatches, self.h, self.d_k, N, -1)
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.view(nbatches, self.h * self.d_k, N, -1)
        x = self.linear_end(x)
        return x
