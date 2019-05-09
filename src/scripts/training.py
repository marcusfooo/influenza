from src.models import models, train_model
from src.data import make_dataset
from src.features import build_features
from src.utils import utils
import torch
import numpy as np
import operator

torch.manual_seed(1)
np.random.seed(1)

data_set = './data/processed/triplet'
train_trigram_vecs, train_labels = utils.read_dataset(data_set + '_train.csv', concat=False)
test_trigram_vecs, test_labels = utils.read_dataset(data_set + '_test.csv', concat=False)

#train_trigram_vecs = build_features.get_diff_vecs(train_trigram_vecs)
#test_trigram_vecs = build_features.get_diff_vecs(test_trigram_vecs)

samples = 8192
X_train = torch.tensor(train_trigram_vecs[:, :samples], dtype=torch.float32)
Y_train = torch.tensor(train_labels[:samples], dtype=torch.int64)
X_test = torch.tensor(test_trigram_vecs, dtype=torch.float32)
Y_test = torch.tensor(test_labels, dtype=torch.int64)

_, counts = np.unique(Y_train, return_counts=True)
imbalance = max(counts) / Y_train.shape[0]
print('Training class imbalance: %.3f' % imbalance)
_, counts = np.unique(Y_test, return_counts=True)
imbalance = max(counts) / Y_test.shape[0]
print('Testing class imbalance:  %.3f' % imbalance)
with open(data_set + '_test_baseline.txt', 'r') as f:
    print('Test baselines:')
    print(f.read())

input_dim = X_train.shape[2]
seq_length = X_train.shape[0]
output_dim = 2
hidden_size = 100
#net = models.LstmModel(input_dim, hidden_size, output_dim)
#net = models.AttentionModel(input_dim, seq_length, hidden_size, output_dim)
net = models.DaRnnModel(input_dim, seq_length, hidden_size, output_dim)

num_of_epochs = 100
learning_rate = 0.01
batch_size = 512
train_model.train_rnn(net, True, num_of_epochs, learning_rate, batch_size, X_train, Y_train, X_test, Y_test)