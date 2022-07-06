import scipy.sparse as sp
import torch
import torch.nn as nn
import sys
import pickle as pkl
import torch.nn.functional as F
import pickle
import os
from sklearn.svm import SVC
import numpy as np
from sklearn.utils import shuffle
import networkx as nx


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_npz_to_sparse_graph(file_name):
    with np.load(file_name) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']), shape=loader['adj_shape'])

        if 'attr_data' in loader:
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']), shape=loader['attr_shape'])
        elif 'attr_matrix' in loader:
            attr_matrix = loader['attr_matrix']
        else:
            attr_matrix = None

        if 'labels_data' in loader:
            labels = sp.csr_matrix((loader['labels_data'], loader['labels_indices'], loader['labels_indptr']), shape=loader['labels_shape'])
        elif 'labels' in loader:
            labels = loader['labels']
        else:
            labels = None

        node_names = loader.get('node_names')
        attr_names = loader.get('attr_names')
        class_names = loader.get('class_names')
        metadata = loader.get('metadata')

    return SparseGraph(adj_matrix, attr_matrix, labels, node_names, attr_names, class_names, metadata)


class SparseGraph:

    def __init__(self, adj_matrix, attr_matrix=None, labels=None,
                 node_names=None, attr_names=None, class_names=None, metadata=None):

        if sp.isspmatrix(adj_matrix):
            adj_matrix = adj_matrix.tocsr().astype(np.float32)
        else:
            raise ValueError("Adjacency matrix must be in sparse format (got {0} instead)".format(type(adj_matrix)))

        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError("Dimensions of the adjacency matrix don't agree")

        if attr_matrix is not None:
            if sp.isspmatrix(attr_matrix):
                attr_matrix = attr_matrix.tocsr().astype(np.float32)
            elif isinstance(attr_matrix, np.ndarray):
                attr_matrix = attr_matrix.astype(np.float32)
            else:
                raise ValueError("Attribute matrix must be a sp.spmatrix or a np.ndarray (got {0} instead)".format(type(attr_matrix)))

            if attr_matrix.shape[0] != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency and attribute matrices don't agree")

        if labels is not None:
            if labels.shape[0] != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency matrix and the label vector don't agree")

        if node_names is not None:
            if len(node_names) != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency matrix and the node names don't agree")

        if attr_names is not None:
            if len(attr_names) != attr_matrix.shape[1]:
                raise ValueError("Dimensions of the attribute matrix and the attribute names don't agree")

        self.adj_matrix = adj_matrix
        self.attr_matrix = attr_matrix
        self.labels = labels
        self.node_names = node_names
        self.attr_names = attr_names
        self.class_names = class_names
        self.metadata = metadata

    def num_nodes(self):
        return self.adj_matrix.shape[0]

    def num_edges(self):
        if self.is_directed():
            return int(self.adj_matrix.nnz)
        else:
            return int(self.adj_matrix.nnz / 2)

    def get_neighbors(self, idx):
        return self.adj_matrix[idx].indices

    def is_directed(self):
        return (self.adj_matrix != self.adj_matrix.T).sum() != 0

    def to_undirected(self):
        if self.is_weighted():
            raise ValueError("Convert to unweighted graph first.")
        else:
            self.adj_matrix = self.adj_matrix + self.adj_matrix.T
            self.adj_matrix[self.adj_matrix != 0] = 1
        return self

    def is_weighted(self):
        return np.any(np.unique(self.adj_matrix[self.adj_matrix != 0].A1) != 1)

    def to_unweighted(self):
        self.adj_matrix.data = np.ones_like(self.adj_matrix.data)
        return self

    def unpack(self):
        return self.adj_matrix, self.attr_matrix, self.labels


def load_data(args):
    print('loading dataset: {}'.format(args.dataset))
    if args.dataset in ['cora', 'citeseer']:
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("./data/{}/ind.{}.{}".format(args.dataset, args.dataset, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))
        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file("./data/{}/ind.{}.test.index".format(args.dataset, args.dataset))
        test_idx_range = np.sort(test_idx_reorder)
        if args.dataset == 'citeseer':
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range - min(test_idx_range), :] = ty
            ty = ty_extended
        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
        labels = np.argmax(labels, 1)
        labels = torch.from_numpy(labels).long()
        if not args.generative_flag:
            features = normalize_features(features)
        adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
        indices_norm = torch.from_numpy(np.stack([adj_norm.tocoo().row, adj_norm.tocoo().col], axis=0).astype(float)).long()
        values_norm = torch.from_numpy(adj_norm.tocoo().data.astype(float)).float()
        adj_norm = torch.sparse.FloatTensor(indices_norm, values_norm, torch.Size(adj_norm.shape))
        indices = torch.from_numpy(np.stack([adj.tocoo().row, adj.tocoo().col], axis=0).astype(float)).long()
        values = torch.from_numpy(adj.tocoo().data.astype(float)).float()
        adj = torch.sparse.FloatTensor(indices, values, torch.Size(adj.shape))
        features = torch.from_numpy(np.array(features.todense())).float()
    elif args.dataset in ['amac']:
        data = load_npz_to_sparse_graph(os.path.join(os.getcwd(), 'data', 'amac', 'amazon_electronics_computers.npz'))
        features = data.attr_matrix.todense()
        if not args.generative_flag:
            features = normalize_features(features)
        features = torch.from_numpy(features).float()
        adj = data.adj_matrix
        adj = adj + adj.T
        adj.data = np.ones_like(adj.data)
        adj = adj.tocoo()
        adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
        indices_norm = torch.from_numpy(np.stack([adj_norm.tocoo().row, adj_norm.tocoo().col], axis=0).astype(float)).long()
        values_norm = torch.from_numpy(adj_norm.tocoo().data.astype(float)).float()
        adj_norm = torch.sparse.FloatTensor(indices_norm, values_norm, torch.Size(adj_norm.shape))
        indices = torch.from_numpy(np.stack([adj.tocoo().row, adj.tocoo().col], axis=0).astype(float)).long()
        values = torch.from_numpy(adj.data).float()
        adj = torch.sparse.FloatTensor(indices, values, torch.Size(adj.shape))
        labels = torch.from_numpy(data.labels).long()
    elif args.dataset in ['amap']:
        data = load_npz_to_sparse_graph(os.path.join(os.getcwd(), 'data', 'amap', 'amazon_electronics_photo.npz'))
        features = data.attr_matrix.todense()
        if not args.generative_flag:
            features = normalize_features(features)
        features = torch.from_numpy(features).float()
        adj = data.adj_matrix
        adj = adj + adj.T
        adj.data = np.ones_like(adj.data)
        adj = adj.tocoo()
        adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
        indices_norm = torch.from_numpy(np.stack([adj_norm.tocoo().row, adj_norm.tocoo().col], axis=0).astype(float)).long()
        values_norm = torch.from_numpy(adj_norm.tocoo().data.astype(float)).float()
        adj_norm = torch.sparse.FloatTensor(indices_norm, values_norm, torch.Size(adj_norm.shape))
        indices = torch.from_numpy(np.stack([adj.tocoo().row, adj.tocoo().col], axis=0).astype(float)).long()
        values = torch.from_numpy(adj.data).float()
        adj = torch.sparse.FloatTensor(indices, values, torch.Size(adj.shape))
        labels = torch.from_numpy(data.labels).long()
    else:
        print('Cannot process this dataset!')
        raise Exception

    pickle.dump(adj.to_dense().numpy(), open(os.path.join(os.getcwd(), 'features', '{}_sp_adj.pkl'.format(args.dataset)), 'wb'))
    if args.dataset in ['cora', 'citeseer', 'amap', 'amac']:
        pickle.dump(labels.numpy(), open(os.path.join(os.getcwd(), 'data', args.dataset, '{}_labels.pkl'.format(args.dataset)), 'wb'))

    return adj, adj_norm, features, labels


def load_generated_features(path):
    fts = pkl.load(open(path, 'rb'))
    norm_fts = normalize_features(fts)
    norm_fts = torch.from_numpy(norm_fts).float()
    return norm_fts


def normalize_adj(mx):
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def cal_accuracy(train_fts, train_lbls, test_fts, test_lbls):
    clf = SVC(gamma='auto')
    clf.fit(train_fts, train_lbls)

    preds_lbls = clf.predict(test_fts)
    acc = accuracy(preds_lbls, test_lbls)
    return acc


def RECALL_NDCG(estimated_fts, true_fts, topN=10):
    preds = np.argsort(-estimated_fts, axis=1)
    preds = preds[:, :topN]

    gt = [np.where(true_fts[i, :] != 0)[0] for i in range(true_fts.shape[0])]
    recall_list = []
    ndcg_list = []
    for i in range(preds.shape[0]):
        if len(gt[i]) != 0:
            if np.sum(estimated_fts[i, :]) != 0:
                recall = len(set(preds[i, :]) & set(gt[i])) * 1.0 / len(set(gt[i]))
                recall_list.append(recall)

                intersec = np.array(list(set(preds[i, :]) & set(gt[i])))
                if len(intersec) > 0:
                    dcg = [np.where(preds[i, :] == ele)[0] for ele in intersec]
                    dcg = np.sum([1.0 / (np.log2(x + 1 + 1)) for x in dcg])
                    idcg = np.sum([1.0 / (np.log2(x + 1 + 1)) for x in range(len(gt[i]))])
                    ndcg = dcg * 1.0 / idcg
                else:
                    ndcg = 0.0
                ndcg_list.append(ndcg)
            else:
                temp_preds = shuffle(np.arange(estimated_fts.shape[1]))[:topN]

                recall = len(set(temp_preds) & set(gt[i])) * 1.0 / len(set(gt[i]))
                recall_list.append(recall)

                intersec = np.array(list(set(temp_preds) & set(gt[i])))
                if len(intersec) > 0:
                    dcg = [np.where(temp_preds == ele)[0] for ele in intersec]
                    dcg = np.sum([1.0 / (np.log2(x + 1 + 1)) for x in dcg])
                    idcg = np.sum([1.0 / (np.log2(x + 1 + 1)) for x in range(len(gt[i]))])
                    ndcg = dcg * 1.0 / idcg
                else:
                    ndcg = 0.0
                ndcg_list.append(ndcg)

    avg_recall = np.mean(recall_list)
    avg_ndcg = np.mean(ndcg_list)

    return avg_recall, avg_ndcg


class MLP(nn.Module):
    def __init__(self, fts_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(fts_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_fts):
        h1 = F.relu(self.fc1(input_fts))
        h2 = self.fc2(h1)
        return F.log_softmax(h2, dim=1)


def class_eva(train_fts, train_lbls, test_fts, test_lbls):
    test_featured_idx = np.where(test_fts.sum(1) != 0)[0]
    test_non_featured_idx = np.where(test_fts.sum(1) == 0)[0]

    featured_test_fts = test_fts[test_featured_idx]
    featured_test_lbls = test_lbls[test_featured_idx]
    non_featured_test_lbls = test_lbls[test_non_featured_idx]

    fts_dim = train_fts.shape[1]
    hid_dim = 64
    n_class = int(max(max(train_lbls), max(test_lbls)) + 1)
    is_cuda = torch.cuda.is_available()

    model = MLP(fts_dim, hid_dim, n_class)
    if is_cuda:
        model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    featured_test_lbls_arr = featured_test_lbls.copy()

    train_fts = torch.from_numpy(train_fts).float()
    train_lbls = torch.from_numpy(train_lbls).long()
    featured_test_fts = torch.from_numpy(featured_test_fts).float()
    featured_test_lbls = torch.from_numpy(featured_test_lbls).long()
    if is_cuda:
        train_fts = train_fts.cuda()
        train_lbls = train_lbls.cuda()
        featured_test_fts = featured_test_fts.cuda()
        featured_test_lbls = featured_test_lbls.cuda()

    acc_list = []
    for i in range(1000):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_fts)

        loss = F.nll_loss(outputs, train_lbls)
        loss.backward()
        optimizer.step()

        model.eval()
        featured_test_outputs = model(featured_test_fts)
        test_loss = F.nll_loss(featured_test_outputs, featured_test_lbls)
        if is_cuda:
            featured_test_outputs = featured_test_outputs.data.cpu().numpy()
        else:
            featured_test_outputs = featured_test_outputs.data.numpy()
        featured_preds = np.argmax(featured_test_outputs, axis=1)

        random_preds = np.random.choice(n_class, len(test_non_featured_idx))

        preds = np.concatenate((featured_preds, random_preds))
        lbls = np.concatenate((featured_test_lbls_arr, non_featured_test_lbls))

        acc = np.sum(preds == lbls) * 1.0 / len(lbls)
        acc_list.append(acc)
        print('Epoch: {}, train loss: {:.4f}, test loss: {:.4f}, test acc: {:.4f}'.format(i, loss.item(), test_loss.item(), acc))

    print('Best epoch:{}, best acc: {:.4f}'.format(np.argmax(acc_list), np.max(acc_list)))
    return np.max(acc_list)


def observed_data_process(args, adj, train_fts_idx, true_features):
    adj_train = sp.csr_matrix(adj.to_dense()[train_fts_idx, :][:, train_fts_idx])
    adj_train = normalize_adj(adj_train + sp.eye(adj_train.shape[0]))
    indices = torch.from_numpy(np.stack([adj_train.tocoo().row, adj_train.tocoo().col], axis=0).astype(float)).long()
    values = torch.from_numpy(adj_train.tocoo().data.astype(float)).float()
    adj_train = torch.sparse.FloatTensor(indices, values, torch.Size(adj_train.shape))

    X_o = true_features[train_fts_idx]

    if args.cuda:
        X_o = X_o.cuda()
        adj_train = adj_train.cuda()
    else:
        X_o = X_o
        adj_train = adj_train
    return X_o, adj_train


def adj_loss_process(args, norm_adj):
    n_pos = len(norm_adj._values())
    norm_adj_arr = norm_adj.to_dense().numpy()
    if args.cuda:
        pos_indices = norm_adj._indices().cpu().numpy()
    else:
        pos_indices = norm_adj._indices().numpy()
    pos_indices = list(zip(pos_indices[0, :], pos_indices[1, :]))

    if not os.path.exists(os.path.join(os.getcwd(), 'data', args.dataset, '{}_{}_neg_indices.pkl'.format(args.dataset, args.train_fts_ratio))):
        zero_indices = np.where(norm_adj_arr == 0)
        neg_indices = list(zip(zero_indices[0], zero_indices[1]))
        neg_indices = shuffle(neg_indices, random_state=args.seed)[:args.neg_times * n_pos]
        pickle.dump(neg_indices, open(os.path.join(os.getcwd(), 'data', args.dataset, '{}_{}_neg_indices.pkl'.format(args.dataset, args.train_fts_ratio)), 'wb'))
    else:
        neg_indices = pickle.load(open(os.path.join(os.getcwd(), 'data', args.dataset, '{}_{}_neg_indices.pkl'.format(args.dataset, args.train_fts_ratio)), 'rb'))

    if args.cuda:
        neg_indices = torch.LongTensor(neg_indices).cuda()
        neg_values = torch.zeros(size=[len(neg_indices)]).cuda()
        pos_values = torch.ones(size=[len(pos_indices)]).cuda()
        pos_indices = torch.LongTensor(pos_indices).cuda()
    else:
        neg_indices = torch.LongTensor(neg_indices)
        neg_values = torch.zeros(size=[len(neg_indices)])
        pos_values = torch.ones(size=[len(pos_indices)])
        pos_indices = torch.LongTensor(pos_indices)
    return neg_indices, neg_values, pos_values, pos_indices


def data_split(args, adj):
    shuffled_nodes = shuffle(np.arange(adj.shape[0]), random_state=args.seed)
    train_fts_idx = torch.from_numpy(shuffled_nodes[:int(args.train_fts_ratio * adj.shape[0])]).long()
    vali_fts_idx = torch.from_numpy(
        shuffled_nodes[
        int(args.train_fts_ratio * adj.shape[0]):int((args.train_fts_ratio + 0.1) * adj.shape[0])]).long()
    test_fts_idx = torch.from_numpy(shuffled_nodes[int((args.train_fts_ratio + 0.1) * adj.shape[0]):]).long()
    vali_test_fts_idx = torch.from_numpy(shuffled_nodes[int(args.train_fts_ratio * adj.shape[0]):]).long()

    print("Dataset loading done!")
    pickle.dump(test_fts_idx,
                open(os.path.join(os.getcwd(), 'features', '{}_{}_test_fts_idx.pkl'.format(
                    args.dataset, args.train_fts_ratio)), 'wb'))
    return train_fts_idx, vali_fts_idx, test_fts_idx, vali_test_fts_idx


def loss_weight(args, true_features, train_fts_idx):
    if args.dataset in ['cora', 'citeseer', 'amac', 'amap']:
        fts_loss_func = fts_loss_discrete
        pos_weight = torch.sum(true_features[train_fts_idx] == 0.0).item() / (
            torch.sum(true_features[train_fts_idx] != 0.0).item())
    else:
        fts_loss_func = None
        pos_weight = None
        print("Error!")
    if args.cuda:
        pos_weight_tensor = torch.from_numpy(np.array([pos_weight])).float().cuda()
        neg_weight_tensor = torch.from_numpy(np.array([1.0])).float().cuda()
    else:
        pos_weight_tensor = torch.from_numpy(np.array([pos_weight])).float()
        neg_weight_tensor = torch.from_numpy(np.array([1.0])).float()
    return fts_loss_func, pos_weight_tensor, neg_weight_tensor


def input_matrix(args, adj, norm_adj, true_features):
    indices = torch.from_numpy(np.stack([np.arange(adj.shape[0]), np.arange(adj.shape[0])], axis=0)).long()
    values = torch.from_numpy(np.ones(indices.shape[1])).float()
    diag_fts = torch.sparse.FloatTensor(indices, values, torch.Size([adj.shape[0], adj.shape[0]]))
    if args.cuda:
        A = norm_adj.cuda()
        D = diag_fts.to_dense().cuda()
        true_features = true_features.cuda()
    else:
        A = norm_adj
        D = diag_fts.to_dense()
        true_features = true_features
    A_temp = A
    return A, D, true_features, A_temp


def model_optimizer(args, Model, optim):
    model = Model(n_nodes=args.n_nodes,
                  n_fts=args.feat,
                  n_hid1=args.hidden1,
                  n_hid2=args.hidden2,
                  dropout=args.dropout,
                  args=args)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                           weight_decay=args.weight_decay)
    if args.cuda:
        model = model.cuda()
    else:
        model = model
    return model, optimizer


def graph_loss_func(graph_recon=None, pos_indices=None, neg_indices=None, pos_values=None, neg_values=None):
    BCE = torch.nn.BCEWithLogitsLoss(reduction='none')
    loss_indices = torch.cat([pos_indices, neg_indices], dim=0)
    preds_logits = graph_recon[loss_indices[:, 0], loss_indices[:, 1]]
    labels = torch.cat([pos_values, neg_values])
    loss_bce = torch.mean(BCE(preds_logits, labels))
    return loss_bce


def fts_loss_discrete(recon_x=None, x=None, p_weight=None, n_weight=None):
    BCE = torch.nn.BCEWithLogitsLoss(reduction='none')
    output_fts_reshape = torch.reshape(recon_x, shape=[-1])
    out_fts_lbls_reshape = torch.reshape(x, shape=[-1])
    weight_mask = torch.where(out_fts_lbls_reshape != 0.0, p_weight, n_weight)
    loss_bce = torch.mean(BCE(output_fts_reshape, out_fts_lbls_reshape) * weight_mask)
    return loss_bce


def train_loss(args, X_hat, A_hat, T, fts_loss_func, train_fts_idx,
               pos_weight_tensor, neg_weight_tensor, pos_indices, neg_indices, pos_values, neg_values):
    L_x = args.lambda_xr * fts_loss_func(X_hat[train_fts_idx], T[train_fts_idx],
                                         pos_weight_tensor, neg_weight_tensor)
    L_a = args.lambda_ar * graph_loss_func(graph_recon=A_hat, pos_indices=pos_indices, neg_indices=neg_indices,
                                           pos_values=pos_values, neg_values=neg_values)
    return L_x+L_a


def adj_update(args, A, Z_f, cosine_similarity):
    A_f = normalize_adj(
        sp.csr_matrix(cosine_similarity(Z_f.data.cpu().numpy(), Z_f.data.cpu().numpy())).tocoo())
    indices = torch.from_numpy(np.stack([A_f.tocoo().row, A_f.tocoo().col], axis=0).astype(float)).long()
    values = torch.from_numpy(A_f.tocoo().data.astype(float)).float()
    A_f = torch.sparse.FloatTensor(indices, values, torch.Size(A_f.shape))
    A_temp = (0.5 * A.cpu() + 0.5 * A_f).to_dense()
    if args.cuda:
        A_temp = A_temp.cuda()
    else:
        A_temp = A_temp
    return A_temp


def save_generative_fts(args, gene_X, T, train_fts_idx, vali_fts_idx, test_fts_idx):
    if args.dataset in ['cora', 'citeseer', 'amap', 'amac']:
        output_fts = 1.0 / (1.0 + np.exp(-gene_X))
    else:
        output_fts = None
        print("Error!")
    if args.cuda:
        train_fts = T[train_fts_idx].data.cpu().numpy()
        vali_fts = T[vali_fts_idx].data.cpu().numpy()
        train_fts_idx_arr = train_fts_idx.cpu().numpy()
        vali_fts_idx_arr = vali_fts_idx.cpu().numpy()
        test_fts_idx_arr = test_fts_idx.cpu().numpy()
    else:
        train_fts = T[train_fts_idx].data.numpy()
        vali_fts = T[vali_fts_idx].data.numpy()
        train_fts_idx_arr = train_fts_idx.numpy()
        vali_fts_idx_arr = vali_fts_idx.numpy()
        test_fts_idx_arr = test_fts_idx.numpy()
    save_fts = np.zeros(shape=T.shape)
    save_fts[train_fts_idx_arr] = train_fts
    save_fts[vali_fts_idx_arr] = vali_fts
    save_fts[test_fts_idx_arr] = output_fts
    pickle.dump(save_fts, open(os.path.join(os.getcwd(), 'features', 'final_gene_fts_train_ratio_{}_{}.pkl'.format(args.dataset, args.train_fts_ratio, )), 'wb'))


def test_model(args, model, T, X_o, A_o, D, A, A_temp, train_id, vali_id, vali_test_id, test_id):
    print('Loading well-trained model'.format(args.epoch))
    model.load_state_dict(
        torch.load(os.path.join(os.getcwd(), 'model', 'final_model_{}_{}.pkl'.format(args.dataset, args.train_fts_ratio))))
    model.eval()
    X_hat, _, _ = model(X_o, A_o, D, A, A_temp, train_id, vali_test_id)
    gene_fts = X_hat[test_id]
    print('Profiling performance on {}:'.format(args.dataset))
    if args.cuda:
        gene_fts = gene_fts.data.cpu().numpy()
        gt_fts = T[test_id].cpu().numpy()
    else:
        gene_fts = gene_fts.data.numpy()
        gt_fts = T[test_id].numpy()
    for topK in args.topK_list:
        avg_recall, avg_ndcg = RECALL_NDCG(gene_fts, gt_fts, topN=topK)
        print('tpoK: {}, recall: {}, ndcg: {}'.format(topK, avg_recall, avg_ndcg))
    save_generative_fts(args, gene_fts, T, train_id, vali_id, test_id)
    if args.cuda:
        T = T.cpu().data.numpy()
    else:
        T = T.data.numpy()
    pickle.dump(T, open(os.path.join(os.getcwd(), 'features', '{}_true_features.pkl'.format(args.dataset)), 'wb'))