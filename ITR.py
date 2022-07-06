import torch.utils.data
from torch import nn
import numpy as np
import torch
import torch.nn.functional as F


class encoder(nn.Module):
    def __init__(self, n_fts, n_hid1, n_hid2, dropout, args):
        super(encoder, self).__init__()
        self.GCN3 = GCNLayer(n_fts, n_hid1, dropout=dropout, args=args)
        self.GCN4 = GCNLayer(n_hid1, n_hid2, dropout=dropout, args=args)
        self.dropout = dropout

    def forward(self, X_o, A_o):
        Z_a = self.GCN3(X_o, A_o, is_sparse_input=True)
        Z_a = F.dropout(Z_a, self.dropout, training=self.training)
        Z_a = self.GCN4(Z_a, A_o)
        return Z_a


class Model(nn.Module):
    def __init__(self, n_nodes, n_fts, n_hid1, n_hid2, dropout, args):
        super(Model, self).__init__()
        self.dropout = dropout
        self.args = args
        self.GCN1 = GCNLayer(n_nodes, n_hid1, dropout=dropout, args=args)
        self.GCN2 = GCNLayer(n_hid1, n_hid2, dropout=dropout, args=args)
        self.encoder = encoder(
            n_fts=n_fts,
            n_hid1=n_hid1,
            n_hid2=n_hid2,
            dropout=dropout,
            args=args)
        self.shared_d1 = GCNLayer(n_hid2, n_hid1, dropout=dropout, args=args)
        self.shared_d2 = GCNLayer(n_hid1, n_fts, dropout=dropout, args=args)

    def forward(self, X_o, A_o, D, A, H, train_fts_idx, vali_test_fts_idx):
        X_o = F.dropout(X_o, self.dropout, training=self.training)
        index = torch.cat((train_fts_idx, vali_test_fts_idx), 0).argsort()
        Z_a = self.encoder(X_o, A_o)
        Z_s = self.GCN1(D, A, is_sparse_input=True)
        Z_s = F.dropout(Z_s, self.dropout, training=self.training)
        Z_s = self.GCN2(Z_s, A)
        Z_i = torch.cat((Z_a, Z_s[vali_test_fts_idx]), 0)
        Z = torch.mm(H, Z_i[index])
        Z_tilde = torch.cat((Z_a, Z[vali_test_fts_idx]), 0)
        Z_tilde = Z_tilde[index]
        A_hat = torch.mm(Z_tilde, torch.transpose(Z_s, 0, 1))
        Z_d = F.relu(self.shared_d1(Z_tilde, A, is_sparse_input=True))
        Z_d = F.dropout(Z_d, self.dropout, training=self.training)
        X_hat = self.shared_d2(Z_d, A)
        return X_hat, A_hat, F.relu(Z_tilde)


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, args):
        super(GCNLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.args = args
        self.W = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(in_features, out_features).type(
            torch.cuda.FloatTensor if args.cuda else torch.FloatTensor), gain=np.sqrt(2.0)),
            requires_grad=True)

    def forward(self, x, sp_adj, is_sparse_input=False):
        if is_sparse_input:
            h = torch.spmm(x, self.W)
        else:
            h = torch.mm(x, self.W)
        h_prime = torch.spmm(sp_adj, h)
        return F.elu(h_prime)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'