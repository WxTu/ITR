import argparse
from torch import optim
from ITR import Model
from utils import *
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import random
from tqdm import tqdm
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--method_name', type=str, default='Model')
parser.add_argument('--topK_list', type=list, default=[10, 20, 50])
parser.add_argument('--update', type=int, default=30)
parser.add_argument('--seed', type=int, default=72)
parser.add_argument('--hidden2', type=int, default=64)
parser.add_argument('--patience', type=int, default=30)
parser.add_argument('--neg_times', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--lambda_xr', type=float, default=10)
parser.add_argument('--lambda_ar', type=float, default=0.5)
parser.add_argument('--train_fts_ratio', type=float, default=0.4)
parser.add_argument('--generative_flag', type=bool, default=True)
parser.add_argument('--cuda', action='store_true', default=torch.cuda.is_available())
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.dataset == "cora":
    args.n_nodes = 2708
    args.feat = 1433
    args.hidden1 = 200
    args.epoch = 250
    args.n_cluster = 7
elif args.dataset == "citeseer":
    args.n_nodes = 3327
    args.feat = 3703
    args.hidden1 = 512
    args.epoch = 400
    args.n_cluster = 6
elif args.dataset == "amap":
    args.n_nodes = 7650
    args.feat = 745
    args.hidden1 = 512
    args.epoch = 8000
    args.n_cluster = 8
elif args.dataset == "amac":
    args.n_nodes = 13752
    args.feat = 767
    args.hidden1 = 512
    args.epoch = 8000
else:
    print("Error!")

if __name__ == "__main__":
    adj, norm_adj, true_features, node_labels = load_data(args)
    A, D, T, A_temp = input_matrix(args, adj, norm_adj, true_features)
    train_id, vali_id, test_id, vali_test_id = data_split(args, adj)
    X_o, A_o = observed_data_process(args, adj, train_id, true_features)
    fts_loss_func, pos_weight_tensor, neg_weight_tensor = loss_weight(args, true_features, train_id)
    neg_indices, neg_values, pos_values, pos_indices = adj_loss_process(args, norm_adj)
    model, optimizer = model_optimizer(args, Model, optim)

    best = 0.0
    best_mse = 10000.0
    bad_counter = 0
    best_epoch = 0
    L_list = []
    eva_values_list = []

    for epoch in tqdm(range(1, args.epoch + 1)):
        model.train()
        optimizer.zero_grad()
        X_hat, A_hat, Z_f = model(X_o, A_o, D, A, A_temp, train_id, vali_test_id)
        if (epoch + 1) % args.update == 0:
            A_temp = adj_update(args, A, Z_f, cosine_similarity)
        L = train_loss(args, X_hat, A_hat, T, fts_loss_func, train_id, pos_weight_tensor, neg_weight_tensor, pos_indices, neg_indices, pos_values, neg_values)
        L.backward()
        optimizer.step()

        if epoch == args.epoch:
            torch.save(model.state_dict(), os.path.join(os.getcwd(), 'model', 'final_model_{}_{}.pkl'.format(args.dataset, args.train_fts_ratio)))

    test_model(args, model, T, X_o, A_o, D, A, A_temp, train_id, vali_id, vali_test_id, test_id)