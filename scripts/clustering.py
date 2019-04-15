from src.data import clustering
from src.utils import utils
import numpy as np

data_files = ['2011.csv', '2012.csv', '2013.csv', '2014.csv', '2015.csv', '2016.csv']
data_path = './data/raw/'

trigram_vecs = utils.read_and_process_to_trigram_vecs(data_files, data_path, sample_size=10, concat=False)
print('Summed up to one vector per strain')
print(f'Shape: {np.array(trigram_vecs).shape}')

numpi = np.array(trigram_vecs)
prot_vecs = numpi.sum(axis=3)
print('Summed up to one vector per strain')
print(f'Shape: {prot_vecs.shape}')
