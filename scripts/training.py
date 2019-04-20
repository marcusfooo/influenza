
#%% Creating training data
from src.models import models, train_model
from src.data import make_dataset
from src.features import build_features
from src.features import original
from src.utils import utils
import torch
import numpy as np
import operator

torch.manual_seed(0)
np.random.seed(0)

data_files = ['2011.csv', '2012.csv', '2013.csv', '2014.csv', '2015.csv', '2016.csv']
data_path = './data/raw/'
trigram_vecs, trigram_idxs = utils.read_and_process_to_trigram_vecs(data_files, data_path)
labels = build_features.indexes_to_mutations(trigram_idxs[-2], trigram_idxs[-1])

trigram_vecs = np.array(trigram_vecs)
labels = np.array(labels)

num_of_training_examples = int(len(trigram_vecs[0]) // (5 / 4))

train_vecs = trigram_vecs[:, :training_examples]
train_labels = labels[:training_examples]
test_vecs = trigram_vecs[:, training_examples:]
test_labels = labels[training_examples:]

X_train = torch.FloatTensor(train_vecs)
Y_train = torch.LongTensor(train_labels)
X_test = torch.FloatTensor(test_vecs)
Y_test = torch.LongTensor(test_labels)

#%% Training model
_, counts = np.unique(Y_test, return_counts=True)
majority_acc = max(counts) / Y_test.shape[0]
print('Accuracy for majority vote: %.4f' % majority_acc)

input_dim = X_train.shape[2]
output_dim = 2
hidden_size = 400
num_of_layers = 1
net = models.LSTMModel(input_dim, hidden_size, num_of_layers, output_dim)

num_of_epochs = 3
learning_rate = 0.1
batch_size = 256
train_model.train_rnn(net, num_of_epochs, learning_rate, batch_size, X_train, Y_train, X_test, Y_test)

#%%
