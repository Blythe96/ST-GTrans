import argparse
import time
from utils.tools import *
from utils.metric_function import *
from engine import Trainer
from data.data_loader_pem import Dataset_pems_minute
from torch.utils.data import Dataset, DataLoader
from model.net_new import stgt
import torch


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


parser = argparse.ArgumentParser()


# Namespace(adj_data='data/PEMS03/graph_PEMS03_conn.pkl', batch_size=8, buildA_true=False, cl=True, clip=5, d_model=64, data='PEMS03.csv', device='cuda:1', dilation_exponential=1, dropout=0.3, dtw_data='data/PEMS03/graph_PEMS03_cor.pkl',
# emb_dim=64, epochs=100, expid=1, gcn_depth=2, gcn_dim=64, gcn_true=True,
# in_dim=1, label_len=6, layers=3, learning_rate=0.001,
# load_static_feature=False, node_dim=40, num_heads=8, num_nodes=358,
# num_split=1, out_dim=1, print_every=50, propalpha=0.05,
# root_path='data/PEMS03/', runs=1, save='./save/pems/926',
# seed=101, seq_in_len=12, seq_out_len=12, step_size1=2500, step_size2=100, subgraph_size=20, tanhalpha=3, weight_decay=0.0001)

parser.add_argument('--device', type=str, default='cpu', help='')
parser.add_argument('--root_path', type=str, default='data/PEMS03/', help='data root path')
parser.add_argument('--data', type=str, default='PEMS03.csv', help='data path')

parser.add_argument('--adj_data', type=str, default='data/PEMS03/graph_PEMS03_conn.pkl', help='adj data path')
parser.add_argument('--dtw_data', type=str, default='data/PEMS03/graph_PEMS03_cor.pkl', help='dtw data path')

parser.add_argument('--gcn_true', type=str_to_bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--buildA_true', type=str_to_bool, default=False,
                    help='whether to construct adaptive adjacency matrix')
parser.add_argument('--load_static_feature', type=str_to_bool, default=False, help='whether to load static feature')
parser.add_argument('--cl', type=str_to_bool, default=True, help='whether to do curriculum learning')

parser.add_argument('--gcn_depth', type=int, default=4, help='graph convolution depth')
parser.add_argument('--num_nodes', type=int, default=358, help='number of nodes/variables')
parser.add_argument('--num_heads', type=int, default=8, help='number of heads')

parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--subgraph_size', type=int, default=20, help='k')
parser.add_argument('--node_dim', type=int, default=40, help='dim of nodes')
parser.add_argument('--dilation_exponential', type=int, default=1, help='dilation exponential')

parser.add_argument('--d_model', type=int, default=64, help='d_model')

parser.add_argument('--gcn_dim', type=int, default=64, help='gcn_dim')

parser.add_argument('--emb_dim', type=int, default=64, help='emb_dim')

parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
parser.add_argument('--out_dim', type=int, default=1, help='outputs dimension')

parser.add_argument('--seq_in_len', type=int, default=12, help='input sequence length')
parser.add_argument('--label_len', type=int, default=6, help='label length of decoder')
parser.add_argument('--seq_out_len', type=int, default=12, help='output sequence length')

parser.add_argument('--layers', type=int, default=3, help='number of layers')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')

parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--clip', type=int, default=5, help='clip')
parser.add_argument('--step_size1', type=int, default=2500, help='step_size')
parser.add_argument('--step_size2', type=int, default=100, help='step_size')

parser.add_argument('--epochs', type=int, default=50, help='')
parser.add_argument('--print_every', type=int, default=50, help='')
parser.add_argument('--seed', type=int, default=101, help='random seed')
parser.add_argument('--save', type=str, default='./save/pems/1020_2', help='save path')
parser.add_argument('--expid', type=int, default=1, help='experiment id')

parser.add_argument('--propalpha', type=float, default=0.05, help='prop alpha')
parser.add_argument('--tanhalpha', type=float, default=3, help='adj alpha')

parser.add_argument('--num_split', type=int, default=1, help='number of splits for graphs')

parser.add_argument('--runs', type=int, default=1, help='number of runs')

args = parser.parse_args()
torch.set_num_threads(3)


def generate_loader(flag, shuffle):
    size = [args.seq_in_len, args.label_len, args.seq_out_len]
    _data = Dataset_pems_minute(
        root_path=args.root_path,
        data_path=args.data,
        flag=flag,
        size=size,
    )

    scaler = _data.inverse_transform

    data_loader = DataLoader(
        _data,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=True)
    print('{} dataset is {}'.format(flag, len(_data)))
    return data_loader, scaler


def main(runid):
    # torch.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(args.seed)
    # load data
    device = torch.device(args.device)
    train_loader, train_scaler = generate_loader('train', True)
    vali_loader, vali_scaler = generate_loader('val', False)
    test_loader, test_scaler = generate_loader('test', False)

    predefined_A = load_adj(args.adj_data)
    predefined_A = torch.tensor(predefined_A) - torch.eye(args.num_nodes)
    predefined_A = predefined_A.to(device)
    predefined_e = load_dtw(args.dtw_data)
    predefined_e = torch.tensor(predefined_e).float().to(device)
    predefined_e = predefined_e.expand(args.batch_size, args.num_nodes, args.num_nodes).unsqueeze(3)
    print(predefined_e.shape)

    model = stgt(gcn_true=args.gcn_true, st=True, gcn_depth=args.gcn_depth, num_nodes=args.num_nodes,
                 device=args.device,
                 predefined_A=predefined_A, buildA_true=args.buildA_true, static_feat=None, dropout=args.dropout,
                 node_dim=args.node_dim,
                 subgraph_size=args.subgraph_size, num_heads=args.num_heads, seq_length=args.seq_in_len,
                 in_dim=args.in_dim,
                 out_length=args.seq_out_len, out_dim=args.out_dim, label=args.label_len, emb_dim=args.emb_dim,
                 gcn_dim=args.gcn_dim, d_model=args.d_model, layers=args.layers, propalpha=args.propalpha,
                 tanhalpha=args.tanhalpha,
                 layer_norm_affline=True)

    # model = stgt_new(in_dim=2, out_dim=2, kernel=[1, 3, 5], t_num_heads=4, gcn_true=False, st=True, buildA_true=False,
    #                  predefined_A=predefined_A, g_num_heads=4,
    #                  num_nodes=288,
    #                  subgraph_size=20, node_dim=40, d_model=64, gcn_dim=62, seq_len=4, gcn_depth=3, dropout=0.3,
    #                  layer_norm_affline=True,
    #                  propalpha=0.05, tanhalpha=3, static_feat=None, device=args.device)

    print(args)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams)

    engine = Trainer(model, args.learning_rate, args.weight_decay, args.clip, args.step_size1, args.seq_out_len,
                     device, args.cl)

    print("start training...", flush=True)
    his_loss = []
    vali_time = []
    train_time = []
    minl = 1e5
    print('arg.epoches', args.epochs)
    for i in range(1, args.epochs + 1):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        print('epoch is', i)
        for iter, (seq_x, seq_y, seq_x_mark, seq_y_mark, label_y) in enumerate(train_loader):
            # seq_x B,T,N,D
            train_x = seq_x.float().to(device)
            train_y = seq_y.float().to(device)
            train_label_y = label_y.float().to(device)
            train_x_mark = seq_x_mark.float().to(device)
            train_dec_mark = seq_y_mark.float().to(device)  # [B, T', 5]
            # decoder input
            dec_inp = torch.zeros_like(seq_y[:, -args.seq_out_len:, :]).float().to(device)
            dec_inp = torch.cat([train_y[:, :args.label_len, :], dec_inp], dim=1).float()

            # Temporal 维度transpose
            train_x = train_x.transpose(1, 2)  # (B,N,T,D)
            dec_inp = dec_inp.transpose(1, 2)  # (B,N,T,D)

            # train_y = train_y[:, -args.seq_out_len:, :, :]  # [B, T, N, D] 标签

            # output = train_scaler(train_y)

            # start training 模型输出维度需要是 [B, T, N, D]
            metrics = engine.train(predefined_e, train_x, train_x_mark, dec_inp, train_dec_mark, train_label_y,
                                   train_scaler)

            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            #
            # if iter % args.print_every == 0:
            #     log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
            #     print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)

        t2 = time.time()
        train_time.append(t2 - t1)

        # validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []
        s1 = time.time()
        for iter, (seq_x, seq_y, seq_x_mark, seq_y_mark, label_y) in enumerate(vali_loader):
            vali_x = seq_x.float().to(device)
            vali_y = seq_y.float()
            vali_label_y = label_y.float().to(device)

            vali_x_mark = seq_x_mark.float().to(device)
            vali_dec_mark = seq_y_mark.float().to(device)  # [B, T', 5]
            # decoder input
            dec_inp = torch.zeros_like(seq_y[:, -args.seq_out_len:, :]).float()
            dec_inp = torch.cat([vali_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

            # Temporal 维度transpose
            vali_x = vali_x.transpose(1, 2)
            dec_inp = dec_inp.transpose(1, 2)
            # vali_y = vali_y[:, -args.seq_out_len:, :].to(device)
            metrics = engine.eval(predefined_e, vali_x, vali_x_mark, dec_inp, vali_dec_mark, vali_label_y, vali_scaler)
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])

        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2 - s1)))
        vali_time.append(s2 - s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),
              flush=True)

        if mvalid_loss < minl:
            # ./save/exp1_1_
            torch.save(engine.model.state_dict(),
                       args.save + "exp" + str(args.expid) + "_" + str(runid) + '_' + str(args.layers) + '_' + str(
                           args.epochs) + '_' + str(args.d_model) + '_' + str(args.label_len) + ".pth")
            minl = mvalid_loss
        if i % 10 == 0:
            torch.save(engine.model.state_dict(),
                       args.save + "exp" + str(args.expid) + "_" + str(runid) + '_' + str(args.layers) + '_' + str(
                           args.epochs) + '_' + str(args.d_model) + '_' + str(args.label_len) + '_' + 'epoch' + '_' + str(
                           i) + ".pth")

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(vali_time)))

    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(
        args.save + "exp" + str(args.expid) + "_" + str(runid) + '_' + str(args.layers) + '_' + str(
            args.epochs) + '_' + str(args.d_model) + '_' + str(args.label_len) + ".pth"))

    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid], 4)))

    # valid data
    outputs = []
    all_vali_y = torch.Tensor().to(device)

    for iter, (seq_x, seq_y, seq_x_mark, seq_y_mark, label_y) in enumerate(vali_loader):
        vali_x = seq_x.float().to(device)
        vali_y = seq_y.float()

        vali_label_y = label_y.float().to(device)

        vali_x_mark = seq_x_mark.float().to(device)
        vali_dec_mark = seq_y_mark.float().to(device)  # [B, T', 5]
        # decoder input
        dec_inp = torch.zeros_like(seq_y[:, -args.seq_out_len:, :]).float()
        dec_inp = torch.cat([vali_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

        # Temporal 维度transpose
        vali_x = vali_x.transpose(1, 2)
        dec_inp = dec_inp.transpose(1, 2)
        # vali_y = vali_y[:, -args.seq_out_len:, :].to(device)

        if all_vali_y is None:
            all_vali_y = vali_label_y
        else:
            all_vali_y = torch.cat((all_vali_y, vali_label_y), dim=0)

        with torch.no_grad():
            preds = engine.model(predefined_e, vali_x, vali_x_mark, dec_inp, vali_dec_mark)
        outputs.append(preds)

    yhat = torch.cat(outputs, dim=0).to(device)

    # yhat = yhat[:realy.size(0), ...]
    pred = vali_scaler(yhat)

    vmae, vmape, vrmse = metric(pred, all_vali_y)

    # test data
    outputs = []
    all_test_y = torch.Tensor().to(device)
    for iter, (seq_x, seq_y, seq_x_mark, seq_y_mark, label_y) in enumerate(test_loader):
        test_x = seq_x.float().to(device)
        test_y = seq_y.float()
        test_label_y = label_y.float().to(device)
        test_x_mark = seq_x_mark.float().to(device)
        test_dec_mark = seq_y_mark.float().to(device)  # [B, T', 5]
        # decoder input
        dec_inp = torch.zeros_like(seq_y[:, -args.seq_out_len:, :]).float()
        dec_inp = torch.cat([test_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

        # Temporal 维度transpose
        test_x = test_x.transpose(1, 2)  # ([64, 2, 288, 4]) B,D,N,T
        dec_inp = dec_inp.transpose(1, 2)  # ([64, 2, 288, 4])
        # test_y = test_y[:, -args.seq_out_len:, :].to(device)  # [B, 4, N, D]标签

        with torch.no_grad():
            preds = engine.model(predefined_e, test_x, test_x_mark, dec_inp, test_dec_mark)

        outputs.append(preds)
        yhat = torch.cat(outputs, dim=0).to(device)
        if all_test_y is None:
            all_test_y = test_label_y
        else:
            all_test_y = torch.cat((all_test_y, test_label_y), dim=0)
    mae = []
    mape = []
    rmse = []
    all_pred = test_scaler(yhat)
    all_real = all_test_y

    for i in range(args.seq_out_len):
        pred = all_pred[:, i, :, :]
        real = all_real[:, i, :, :]
        metrics = metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        mae.append(metrics[0])
        mape.append(metrics[1])
        rmse.append(metrics[2])
    return vmae, vmape, vrmse, mae, mape, rmse


if __name__ == "__main__":

    vmae = []
    vmape = []
    vrmse = []
    mae = []
    mape = []
    rmse = []
    for i in range(args.runs):
        vm1, vm2, vm3, m1, m2, m3 = main(i)
        vmae.append(vm1)
        vmape.append(vm2)
        vrmse.append(vm3)
        mae.append(m1)
        mape.append(m2)
        rmse.append(m3)

    mae = np.array(mae)
    mape = np.array(mape)
    rmse = np.array(rmse)

    amae = np.mean(mae, 0)
    amape = np.mean(mape, 0)
    armse = np.mean(rmse, 0)

    smae = np.std(mae, 0)
    smape = np.std(mape, 0)
    srmse = np.std(rmse, 0)

    print('\n\nResults for 10 runs\n\n')
    # valid data
    print('valid\tMAE\tRMSE\tMAPE')
    log = 'mean:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.mean(vmae), np.mean(vrmse), np.mean(vmape)))
    log = 'std:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.std(vmae), np.std(vrmse), np.std(vmape)))
    print('\n\n')

    # test data
    print('test|horizon\tMAE-mean\tRMSE-mean\tMAPE-mean\tMAE-std\tRMSE-std\tMAPE-std')
    for i in range(args.seq_out_len):
        log = '{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
        print(log.format(i + 1, amae[i], armse[i], amape[i], smae[i], srmse[i], smape[i]))
