#%%
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

#%%
import src.utils.utils as utils
from src.data import cluster
import numpy as np

data_files = ['2011.csv', '2012.csv', '2013.csv', '2014.csv', '2015.csv', '2016.csv']
data_path = './data/raw/'

trigram_vecs = utils.read_and_process_to_trigram_vecs(data_files, data_path, sample_size=0, concat=False)
print(f'Shape: {len(trigram_vecs)}x{len(trigram_vecs[0])}x{len(trigram_vecs[0][0])}')

year_trigram_vecs = trigram_vecs[0]
cluster.cluster_year(year_trigram_vecs)

