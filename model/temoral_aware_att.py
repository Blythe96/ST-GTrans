import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import copy


def clones(module, N):
    '''
    Produce N identical layers.
    :param module: nn.Module
    :param N: int
    :return: torch.nn.ModuleList
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    '''

    :param query:  (batch, N, h, T1, d_k)
    :param key: (batch, N, h, T2, d_k)
    :param value: (batch, N, h, T2, d_k)
    :param mask: (batch, 1, 1, T2, T2)
    :param dropout:
    :return: (batch, N, h, T1, d_k), (batch, N, h, T1, T2)
    '''
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # scores: (batch, N, h, T1, T2)

    if mask is not None:
        scores = scores.masked_fill_(mask == 0, -1e9)  # -1e9 means attention scores=0
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # p_attn: (batch, N, h, T1, T2)

    return torch.matmul(p_attn, value), p_attn  # (batch, N, h, T1, d_k), (batch, N, h, T1, T2)


class MultiHeadAttention(nn.Module):
    def __init__(self, nb_head, d_model, dropout=.0):
        super(MultiHeadAttention, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        '''
        :param query: (batch, N, T, d_model)
        :param key: (batch, N, T, d_model)
        :param value: (batch, N, T, d_model)
        :param mask: (batch, T, T)
        :return: x: (batch, N, T, d_model)
        '''
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, T, T), same mask applied to all h heads.

        nbatches = query.size(0)

        N = query.size(1)

        # (batch, N, T, d_model) -linear-> (batch, N, T, d_model) -view-> (batch, N, T, h, d_k) -permute(2,3)-> (batch, N, h, T, d_k)
        query, key, value = [l(x).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3) for l, x in
                             zip(self.linears, (query, key, value))]

        # apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # x:(batch, N, h, T1, d_k)
        # attn:(batch, N, h, T1, T2)

        x = x.transpose(2, 3).contiguous()  # (batch, N, T1, h, d_k)
        x = x.view(nbatches, N, -1, self.h * self.d_k)  # (batch, N, T1, d_model)
        return self.linears[-1](x)


class MultiHeadAttentionAwareTemporalContex_q1d_k1d(nn.Module):  # 1d conv on query, 1d conv on key
    def __init__(self, nb_head, d_model, kernel_size=3, dropout=.0):
        super(MultiHeadAttentionAwareTemporalContex_q1d_k1d, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        self.linears = clones(nn.Linear(d_model, d_model), 2)  # 2 linear layers: 1  for W^V, 1 for W^O
        self.padding = (kernel_size - 1) // 2

        self.conv1Ds_aware_temporal_context = clones(
            nn.Conv2d(d_model, d_model, (1, kernel_size), padding=(0, self.padding)),
            2)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        '''
        :param query: (batch, N, T, d_model)
        :param key: (batch, N, T, d_model)
        :param value: (batch, N, T, d_model)
        :param mask:  (batch, T, T)
        if query/key has multiple time segments, causal convolution should be applied separately for each time segment.
        :return: (batch, N, T, d_model)
        '''

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, T, T), same mask applied to all h heads.

        nbatches = query.size(0)

        N = query.size(1)

        # deal with key and query: temporal conv
        # (batch, N, T, d_model)->permute(0, 3, 1, 2)->(batch, d_model, N, T) -conv->(batch, d_model, N, T)-view->(batch, h, d_k, N, T)-permute(0,3,1,4,2)->(batch, N, h, T, d_k)
        query, key = [
            l(x.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2) for
            l, x in zip(self.conv1Ds_aware_temporal_context, (query, key))]
        # deal with value:
        # (batch, N, T, d_model) -linear-> (batch, N, T, d_model) -view-> (batch, N, T, h, d_k) -permute(2,3)-> (batch, N, h, T, d_k)
        value = self.linears[0](value).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3)
        # apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(2, 3).contiguous()  # (batch, N, T1, h, d_k)
        x = x.view(nbatches, N, -1, self.h * self.d_k)  # (batch, N, T1, d_model)
        return self.linears[-1](x)


class MultiHeadAttentionAwareTemporalContex_qc_k1d(nn.Module):  # query: causal conv; key 1d conv
    def __init__(self, nb_head, d_model, kernel_size=3, dropout=.0):
        super(MultiHeadAttentionAwareTemporalContex_qc_k1d, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        self.linears = clones(nn.Linear(d_model, d_model), 2)  # 2 linear layers: 1  for W^V, 1 for W^O
        self.causal_padding = kernel_size - 1
        self.padding_1D = (kernel_size - 1) // 2
        self.query_conv1Ds_aware_temporal_context = nn.Conv2d(d_model, d_model, (1, kernel_size),
                                                              padding=(0, self.causal_padding))
        self.key_conv1Ds_aware_temporal_context = nn.Conv2d(d_model, d_model, (1, kernel_size),
                                                            padding=(0, self.padding_1D))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        '''
        :param query: (batch, N, T, d_model)
        :param key: (batch, N, T, d_model)
        :param value: (batch, N, T, d_model)
        :param mask:  (batch, T, T)
        :return: (batch, N, T, d_model)
        '''
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, T, T), same mask applied to all h heads.
        nbatches = query.size(0)

        N = query.size(1)

        query = self.query_conv1Ds_aware_temporal_context(query.permute(0, 3, 1, 2))[:, :, :,
                :-self.causal_padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
        key = self.key_conv1Ds_aware_temporal_context(key.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h,
                                                                                                 self.d_k, N,
                                                                                                 -1).permute(0, 3, 1, 4,
                                                                                                             2)

        value = self.linears[0](value).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3)
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(2, 3).contiguous()  # (batch, N, T1, h, d_k)
        x = x.view(nbatches, N, -1, self.h * self.d_k)  # (batch, N, T1, d_model)
        return self.linears[-1](x)


class multi_local_temporal_att(nn.Module):
    def __init__(self, kernel_size, num_heads, n_dim, dropout):
        super(multi_local_temporal_att, self).__init__()
        self.length = len(kernel_size)
        self.temporal_conv = nn.ModuleList()
        for i in range(self.length):
            self.temporal_conv.append(MultiHeadAttentionAwareTemporalContex_q1d_k1d(nb_head=num_heads, d_model=n_dim,
                                                                                    kernel_size=kernel_size[i],
                                                                                    dropout=dropout))

    def forward(self, k, q, v):
        '''
        :param k:
        :param q: dec 的值
        :param v:
        :return:
        '''
        # filter = self.temporal_conv[0](k, q, v)
        # gate = self.temporal_conv[1](k, q, v)
        # res = self.tanh(filter) * self.sigmoid(gate)
        res = None
        for i in range(self.length):
            if res is None:
                res = self.temporal_conv[i](k, q, v)
            else:
                res += self.temporal_conv[i](k, q, v)
        return res


class multi_local_temporal_causal_att(nn.Module):
    def __init__(self, kernel_size, num_heads, n_dim, dropout):
        super(multi_local_temporal_causal_att, self).__init__()
        self.length = len(kernel_size)
        self.temporal_conv = nn.ModuleList()
        for i in range(self.length):
            self.temporal_conv.append(MultiHeadAttentionAwareTemporalContex_qc_k1d(nb_head=num_heads, d_model=n_dim,
                                                                                   kernel_size=kernel_size[i],
                                                                                   dropout=dropout))

    def forward(self, k, q, v):

        res = None
        for i in range(self.length):
            if res is None:
                res = self.temporal_conv[i](k, q, v)
            else:
                res += self.temporal_conv[i](k, q, v)
        return res
