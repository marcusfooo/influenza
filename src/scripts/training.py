#%% Creating training data
from src.models import models, train_model
from src.data import make_dataset
from src.features import build_features
from src.features import original
from src.utils import utils
import torch
import numpy as np
import operator

torch.manual_seed(1)
np.random.seed(1)

train_trigram_vecs, train_labels = utils.read_dataset('./data/processed/triplet_train_data.csv')
test_trigram_vecs, test_labels = utils.read_dataset('./data/processed/triplet_test_data.csv')

X_train = torch.FloatTensor(train_trigram_vecs[:, :8192])
Y_train = torch.LongTensor(train_labels[:8192])
X_test = torch.FloatTensor(test_trigram_vecs)
Y_test = torch.LongTensor(test_labels)

#%% Training model
_, counts = np.unique(Y_train, return_counts=True)
imbalance = max(counts) / Y_train.shape[0]
print('Training class imbalance: %.3f' % imbalance)
_, counts = np.unique(Y_test, return_counts=True)
imbalance = max(counts) / Y_test.shape[0]
print('Testing class imbalance:  %.3f' % imbalance)

input_dim = X_train.shape[2]
seq_length = X_train.shape[0]
output_dim = 2
hidden_size = 5
num_of_layers = 1
net = models.LSTMModel(input_dim, hidden_size, num_of_layers, output_dim)
#net = models.AttentionModel(input_dim, seq_length, hidden_size, num_of_layers, output_dim)

num_of_epochs = 100
learning_rate = 0.01
batch_size = 256
train_model.train_rnn(net, num_of_epochs, learning_rate, batch_size, X_train, Y_train, X_test, Y_test)