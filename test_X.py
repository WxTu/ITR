from sklearn.model_selection import KFold
from utils import *
dataset = 'cora'
train_fts_ratio = 0.4 * 1.0
print('begining......')

gene_fts = pickle.load(open(os.path.join(os.getcwd(), 'features', 'final_gene_fts_train_ratio_{}_{}.pkl'.format(dataset, train_fts_ratio)), 'rb'))
gene_fts_idx = pickle.load(open(os.path.join(os.getcwd(), 'features', '{}_{}_test_fts_idx.pkl'.format(dataset, train_fts_ratio)), 'rb'))
true_features = pickle.load(open(os.path.join(os.getcwd(), 'features', '{}_true_features.pkl'.format(dataset)), 'rb'))
all_labels = pickle.load(open(os.path.join(os.getcwd(), 'data', dataset, '{}_labels.pkl'.format(dataset)), 'rb'))
gene_fts = gene_fts[gene_fts_idx]
labels_of_gene = all_labels[gene_fts_idx]
gene_data = np.concatenate((gene_fts, np.reshape(labels_of_gene, newshape=[-1, 1])), axis=1)
final_list = []
for i in range(10):
    gene_data = shuffle(gene_data, random_state=72)
    KF = KFold(n_splits=5)
    split_data = KF.split(gene_data)
    acc_list = []
    for train_idx, test_idx in split_data:
        train_data = gene_data[train_idx]
        train_featured_idx = np.where(train_data.sum(1) != 0)[0]
        train_data = train_data[train_featured_idx]
        test_data = gene_data[test_idx]
        acc = class_eva(train_fts=train_data[:, :-1], train_lbls=train_data[:, -1],
                        test_fts=test_data[:, :-1], test_lbls=test_data[:, -1])
        acc_list.append(acc)
    avg_acc = np.mean(acc_list)
    final_list.append(avg_acc)

print('classification performance: {}'.format(np.mean(final_list)))
