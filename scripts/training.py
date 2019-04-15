
#%%
from src.models import models, train_model
from src.data import make_dataset
from src.features import build_features
from src.features import original
from src.utils import utils
import torch
import numpy as np

data_files = ['2011.csv', '2012.csv', '2013.csv', '2014.csv', '2015.csv', '2016.csv']
data_path = './data/raw/'
trigram_vecs, trigram_idxs = utils.read_and_process_to_trigram_vecs(data_files, data_path)

labels = build_features.indexes_to_mutations(trigram_idxs[-2], trigram_idxs[-1])
training_input = torch.FloatTensor(trigram_vecs[:-1])
training_output = torch.LongTensor(labels)

#%%
torch.manual_seed(0)
np.random.seed(0)

input_dim = trigram_vecs_data.shape[1]
output_dim = 2
hidden_size = 400
num_of_layers = 1

net = models.LSTMModel(input_dim, hidden_size, num_of_layers, output_dim)
train_model.train_rnn(net, 5, 0.1, 256, training_input, training_output)


#%%
