#%%
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

#%%
import src.utils.utils as utils
import numpy as np

data_files = ['2011.csv', '2012.csv', '2013.csv', '2014.csv', '2015.csv', '2016.csv']
data_path = './data/raw/'

trigram_vecs, trigram_idxs = utils.read_and_process_to_trigram_vecs(data_files)
print(type(trigram_vecs[0][0]))
print('Summed up to one vector per strain')
print(f'Shape: {np.array(trigram_vecs).shape}')
print(f'Shape: {np.array(trigram_idxs).shape}')

numpi = np.array(trigram_vecs)
prot_vecs = numpi.sum(axis=3)
print('Summed up to one vector per strain')
print(f'Shape: {prot_vecs.shape}')
