
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

data_path = './data/raw/'
train_files = ['2011.csv', '2012.csv', '2013.csv', '2014.csv', '2015.csv', '2016.csv']
train_trigram_vecs, test_trigram_vecs, train_trigram_idxs, test_trigram_idxs = utils.read_and_process_to_trigram_vecs(train_files, data_path, sample_size=625, test_split=0.2)

train_trigram_vecs = np.array(train_trigram_vecs)
test_trigram_vecs = np.array(test_trigram_vecs)
train_labels = build_features.indexes_to_mutations(train_trigram_idxs[-2], train_trigram_idxs[-1])
test_labels = build_features.indexes_to_mutations(test_trigram_idxs[-2], test_trigram_idxs[-1])

X_train = torch.FloatTensor(train_trigram_vecs)
Y_train = torch.LongTensor(train_labels)
X_test = torch.FloatTensor(test_trigram_vecs)
Y_test = torch.LongTensor(test_labels)

#%% Training model
_, counts = np.unique(Y_test, return_counts=True)
majority_acc = max(counts) / Y_test.shape[0]
print('Accuracy for majority vote: %.4f' % majority_acc)

input_dim = X_train.shape[2]
output_dim = 2
hidden_size = 10
num_of_layers = 1
net = models.LSTMModel(input_dim, hidden_size, num_of_layers, output_dim)

num_of_epochs = 500
learning_rate = 0.1
batch_size = 256
train_model.train_rnn(net, num_of_epochs, learning_rate, batch_size, X_train, Y_train, X_test, Y_test)

#%%
