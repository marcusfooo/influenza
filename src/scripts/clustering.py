#%%
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

#%%
from src.utils import utils
from src.data import cluster
from src.visualization import visualize

data_files = ['2011.csv', '2012.csv', '2013.csv', '2014.csv', '2015.csv', '2016.csv']
data_files = ['2015.csv', '2016.csv']
data_files = ['2015.csv']
data_path = './data/raw/'

trigram_vecs, _ = utils.read_and_process_to_trigram_vecs(data_files, data_path, sample_size=0, squeeze=False)
print(f'Shape: {len(trigram_vecs)}x{len(trigram_vecs[0])}x{len(trigram_vecs[0][0])}')

clusters = cluster.cluster_years(trigram_vecs, method='KMeans')
average = cluster.evaluate_clusters(clusters)

clusters = cluster.cluster_years(trigram_vecs, method='DBSCAN')
cluster.evaluate_clusters(clusters)
# visualize.show_clusters(clusters, method='PCA')
