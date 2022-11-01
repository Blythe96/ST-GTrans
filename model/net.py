import torch.nn as nn
from model.layers import DataEmbedding, mixprop, graph_constructor, LayerNorm
from model.temoral_aware_att import multi_local_temporal_att
import torch.nn.functional as F
import torch
from model.spatial import SMultiHeadAttention


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, generator, DEVICE):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.prediction_generator = generator
        self.to(DEVICE)

    def forward(self, x, x_dec):
        encoder_output = self.encode(x)
        return self.decode(x_dec, encoder_output)

    def encode(self, x):
        en_h = self.encoder(x)
        return en_h

    def decode(self, x_dec, memory, ):
        de_h = self.decoder(x_dec, memory)
        y = self.prediction_generator(de_h)
        return y


class Encoderlayer(nn.Module):
    def __init__(self, gcn_true, st, gcn_depth, num_nodes, device, predefined_A=None, buildA_true=None,
                 static_feat=None, dropout=0.3, node_dim=40, num_heads=4, subgraph_size=20, gcn_dim=32, d_model=64,
                 propalpha=0.05,
                 tanhalpha=3, seq_length=4, layer_norm_affline=True):
        super(Encoderlayer, self).__init__()
        self.buildA_true = buildA_true
        self.gcn_true = gcn_true
        self.predefined_A = predefined_A
        self.num_nodes = num_nodes
        self.seq_length = seq_length
        self.dropout = dropout
        self.st = st
        self.idx = torch.arange(self.num_nodes).to(device)
        self.mlt = multi_local_temporal_att(kernel_size=[1,3], num_heads=num_heads, n_dim=d_model, dropout=dropout)

        self.norm1 = LayerNorm((self.num_nodes, self.seq_length, d_model), elementwise_affine=layer_norm_affline)
        self.norm2 = LayerNorm((d_model, self.num_nodes, self.seq_length), elementwise_affine=layer_norm_affline)

        if self.gcn_true:
            self.gconv1 = mixprop(d_model, gcn_dim, gcn_depth, dropout, propalpha)  # 出战
            self.gconv2 = mixprop(d_model, gcn_dim, gcn_depth, dropout, propalpha)  # 入站

        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha,
                                    static_feat=static_feat)  # 构图
        if self.st:
            self.s_transformer = SMultiHeadAttention(embed_size=d_model, seq_length=self.seq_length, heads=num_heads)

    def forward(self, x, e):
        '''
        :param x:  (B,N,T,D)
        :return:
        '''

        residual1 = x  # (12, 288, 4, 64)
        x = self.mlt(x, x, x)  # (12,288,4,2) 对时间维度进行操作

        x = self.norm1(x + residual1)  # [B,N,T,D]
        x = F.dropout(x, self.dropout, training=self.training)

        # spatial 建模
        x = x.permute(0, 3, 1, 2)  # (B,D,N,T)

        residual1 = x
        if self.gcn_true:
            if self.buildA_true:
                adp = self.gc(self.idx)
            else:
                adp = self.predefined_A

        if self.st and self.gcn_true:
            st_x, e = self.s_transformer(x, x, x, e)  # (B,D,N,T) 输出也是B,D,N,T
            gcn_x = self.gconv1(x, adp) + self.gconv2(x, adp.transpose(1, 0))
            x = gcn_x + st_x
        elif self.st and not (self.gcn_true):
            st_x, e = self.s_transformer(x, x, x, e)
            x = st_x
        else:
            gcn_x = self.gconv1(x, adp) + self.gconv2(x, adp.transpose(1, 0))
            x = gcn_x
        # if self.gcn_true:
        #     gcn_x = self.gconv1(x, adp) + self.gconv2(x, adp.transpose(1, 0))  # [12, 64, 288, 4]
        # if self.st:
        #     st_x, e = self.s_transformer(x, x, x, e)  # (B,D,N,T) 输出也是B,D,N,T
        #
        #     x = gcn_x + st_x
        # else:
        #     x = gcn_x

        x = residual1 + x
        x = self.norm2(x)  # (B,D,N,T)
        x = x.permute(0, 2, 3, 1)  # # (B,N,T,D)

        return x, e


class encoder(nn.Module):
    def __init__(self, layers, gcn_true, st, gcn_depth, num_nodes, device, predefined_A=None, buildA_true=None,
                 static_feat=None,
                 dropout=0.3, node_dim=40, num_heads=4, subgraph_size=20, gcn_dim=32, d_model=64, propalpha=0.05,
                 tanhalpha=3, seq_length=4, layer_norm_affline=True):
        super(encoder, self).__init__()
        self.layers = nn.ModuleList(
            [Encoderlayer(gcn_true, st, gcn_depth, num_nodes, device, predefined_A, buildA_true, static_feat, dropout,
                          node_dim, num_heads, subgraph_size, gcn_dim, d_model, propalpha, tanhalpha, seq_length,
                          layer_norm_affline) for _ in range(layers)])

    def forward(self, x, e):
        for layer in self.layers:
            x, e = layer(x, e)
        return x, e


class Decoderlayer(nn.Module):
    def __init__(self, gcn_true, st, gcn_depth, num_nodes, device, predefined_A=None, buildA_true=None, label=2,
                 static_feat=None, dropout=0.3, node_dim=40, num_heads=4, subgraph_size=20, gcn_dim=32, d_model=64,
                 propalpha=0.05,
                 tanhalpha=3, seq_length=4, out_length=4, layer_norm_affline=True):
        super(Decoderlayer, self).__init__()
        self.buildA_true = buildA_true
        self.gcn_true = gcn_true
        self.predefined_A = predefined_A
        self.num_nodes = num_nodes
        self.seq_length = seq_length
        self.out_length = 4
        self.dropout = dropout
        self.st = st
        self.label_len = label
        self.idx = torch.arange(self.num_nodes).to(device)
        self.mlt1 = multi_local_temporal_att(kernel_size=[1,3], num_heads=num_heads, n_dim=d_model, dropout=dropout)
        self.mlt2 = multi_local_temporal_att(kernel_size=[1,3], num_heads=num_heads, n_dim=d_model,
                                             dropout=dropout)
        self.norm1 = LayerNorm((self.num_nodes, self.out_length + self.label_len, d_model),
                               elementwise_affine=layer_norm_affline)

        self.norm2 = LayerNorm((self.num_nodes, self.seq_length, d_model), elementwise_affine=layer_norm_affline)
        self.norm3 = LayerNorm((d_model, self.num_nodes, self.seq_length), elementwise_affine=layer_norm_affline)
        if self.gcn_true:
            self.gconv1 = mixprop(d_model, gcn_dim, gcn_depth, dropout, propalpha)  # 出战
            self.gconv2 = mixprop(d_model, gcn_dim, gcn_depth, dropout, propalpha)  # 入站

        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha,
                                    static_feat=static_feat)  # 构图
        if self.st:
            self.s_transformer = SMultiHeadAttention(embed_size=d_model, seq_length=self.seq_length, heads=num_heads)
        self.conv_e = nn.Conv2d(in_channels=self.num_nodes, out_channels=self.num_nodes, kernel_size=(3, 1))

    def forward(self, x_dec_input, x_enc, e):
        '''
        :param x:  (B,N,T,D)
        :return:
        '''
        residual1 = x_dec_input  # (12, 288, 6, 64)
        x = self.mlt1(x_dec_input, x_dec_input, x_dec_input)  # (12, 288, 6, 64)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.norm1(x + residual1)  # [B,N,T,D]

        x = self.conv_e(x)
        # x = x[:, :, -self.out_length:, :]

        # ---------
        # ED
        residual2 = x
        x = self.mlt2(x_enc, x, x_enc)  # k, q, v

        x = F.dropout(x, self.dropout, training=self.training)

        x = self.norm2(x + residual2)

        # spatial 建模
        x = x.permute(0, 3, 1, 2)  # (B,D,N,T)

        residual3 = x
        if self.gcn_true:
            if self.buildA_true:
                adp = self.gc(self.idx)
            else:
                adp = self.predefined_A
        # if self.gcn_true:
        #     gcn_x = self.gconv1(x, adp) + self.gconv2(x, adp.transpose(1, 0))  # [12, 64, 288, 4]
        if self.st and self.gcn_true:
            st_x, e = self.s_transformer(x, x, x, e)  # (B,D,N,T) 输出也是B,D,N,T
            gcn_x = self.gconv1(x, adp) + self.gconv2(x, adp.transpose(1, 0))
            x = gcn_x + st_x
        elif self.st and not (self.gcn_true):
            st_x, e = self.s_transformer(x, x, x, e)
            x = st_x
        else:
            gcn_x = self.gconv1(x, adp) + self.gconv2(x, adp.transpose(1, 0))
            x = gcn_x

        x = residual3 + x
        x = self.norm3(x)  # (B,D,N,T)
        x = x.permute(0, 2, 3, 1)  # (B,N,T,D)

        return x, e


class decoder(nn.Module):
    def __init__(self, layers, gcn_true, st, gcn_depth, num_nodes, device, predefined_A=None, buildA_true=None, label=1,
                 static_feat=None,
                 dropout=0.3, node_dim=40, num_heads=4, subgraph_size=20, gcn_dim=32, d_model=64, propalpha=0.05,
                 tanhalpha=3, seq_length=4, layer_norm_affline=True):
        super(decoder, self).__init__()
        self.layers = nn.ModuleList(
            [Decoderlayer(gcn_true, st, gcn_depth, num_nodes, device, predefined_A, buildA_true, label, static_feat,
                          dropout,
                          node_dim, num_heads, subgraph_size, gcn_dim, d_model, propalpha, tanhalpha, seq_length,
                          layer_norm_affline) for _ in range(layers)])

    def forward(self, x_dec_input, x_enc, e):
        for layer in self.layers:
            x, e = layer(x_dec_input, x_enc, e)
        return x, e


class predictor(nn.Module):
    def __init__(self, d_model, out_dim, dropout):
        super(predictor, self).__init__()
        # self.predict_1 = nn.Linear(d_model, d_model / 2)  # (B,N,D,D_)
        # self.predict_2 = nn.Linear(d_model / 2, out_dim)
        self.predict = nn.Linear(d_model, out_dim)
        self.relu = nn.ReLU()
        self.dropout = dropout

    def forward(self, x):
        '''
        :param x: (B,N,T,D)
        '''
        # x = self.relu(self.predict_1(x))  # [12, 4,288, 2] # (B,N,T,D)
        # x = self.predict_2(x)
        x = self.predict(x)
        x = x.transpose(1, 2)
        return x


class stgt(nn.Module):
    def __init__(self, gcn_true, st, gcn_depth, num_nodes, device, predefined_A, buildA_true, static_feat,
                 dropout, node_dim, subgraph_size, num_heads, seq_length, in_dim, out_length,
                 out_dim, label, emb_dim, gcn_dim, d_model, layers, propalpha=0.05, tanhalpha=3,
                 layer_norm_affline=True):
        '''
        :param in_dim: 输入维度
        :param d_model: 输入Encoder的特征维度
        '''
        super(stgt, self).__init__()
        self.layers = layers
        self.enc_embed = DataEmbedding(in_dim, emb_dim)
        self.dec_embed = DataEmbedding(in_dim, emb_dim)
        self.edge_embed = nn.Linear(1, emb_dim)
        self.encoder = encoder(layers, gcn_true, st, gcn_depth, num_nodes, device, predefined_A,
                               buildA_true=buildA_true,
                               static_feat=static_feat, dropout=dropout, node_dim=node_dim, num_heads=num_heads,
                               subgraph_size=subgraph_size, gcn_dim=gcn_dim, d_model=d_model, propalpha=propalpha,
                               tanhalpha=tanhalpha, seq_length=seq_length, layer_norm_affline=layer_norm_affline)
        self.decoder = decoder(layers, gcn_true, st, gcn_depth, num_nodes, device, predefined_A,
                               buildA_true=buildA_true, label=label,
                               static_feat=static_feat, dropout=dropout, node_dim=node_dim, num_heads=num_heads,
                               subgraph_size=subgraph_size, gcn_dim=gcn_dim, d_model=d_model, propalpha=propalpha,
                               tanhalpha=tanhalpha, seq_length=seq_length, layer_norm_affline=layer_norm_affline)
        self.predictor = predictor(d_model, out_dim, dropout)

    def forward(self, e, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc_input = self.enc_embed(x_enc, x_mark_enc)  # (B,N,T,D) (64,288,4,64)
        x_dec_input = self.dec_embed(x_dec, x_mark_dec)
        e = self.edge_embed(e)
        x_enc, e = self.encoder(x_enc_input, e)
        x, e = self.decoder(x_dec_input, x_enc, e)
        x = self.predictor(x)  # (B,T,N,D)
        return x
