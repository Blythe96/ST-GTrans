import pickle
import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from torch.autograd import Variable


class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(
            data) else self.mean  # mean shape is (288,2)
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(
            data) else self.std  # std shape(288,2) 分别对每个站点的进出口进行归一化处理
        data = (data - mean) / (std)
        data = data * 2. - 1.
        return data

    def inverse_transform(self, data):

        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        data = (data + 1.) / 2.
        data = (data * (std)) + mean

        return data


# class StandardScaler():
#     def __init__(self):
#         self.mean = 0.
#         self.std = 1.
#
#     def fit(self, data):
#         self.mean = data.mean(0)
#         self.std = data.std(0)
#
#     def transform(self, data):
#         mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(
#             data) else self.mean  # mean shape is (288,2)
#         std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(
#             data) else self.std  # std shape(288,2) 分别对每个站点的进出口进行归一化处理
#         data = (data - mean) / (std)
#         return data * 2. - 1.
#
#     def inverse_transform(self, data):
#         data = (data + 1.)/ 2.
#         mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
#         std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
#         data = (data * (std)) + mean
#         return data


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def asym_adj(adj):
    """Asymmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_adj(pkl_filename):
    # sensor_ids, sensor_id_to_ind, adj = load_pickle(pkl_filename)
    adj = load_pickle(pkl_filename)
    return adj


def load_dtw(pkl_filename):
    dtw = load_pickle(pkl_filename)
    return dtw


def load_node_feature(path):
    fi = open(path)
    x = []
    for li in fi:
        li = li.strip()
        li = li.split(",")
        e = [float(t) for t in li[1:]]
        x.append(e)
    x = np.array(x)
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    z = torch.tensor((x - mean) / std, dtype=torch.float)
    return z
