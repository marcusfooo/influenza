from src.utils import utils
from src.data import cluster
from src.visualization import visualize

# data_files = ['2011.csv', '2012.csv', '2013.csv', '2014.csv', '2015.csv', '2016.csv']
data_files = ['2014.csv', '2015.csv', '2016.csv']
data_path = './data/raw/'

methods = ['DBSCAN', 'KMeans', 'MeanShift']
methods = ['DBSCAN']

trigram_vecs, _ = utils.read_and_process_to_trigram_vecs(data_files, data_path, sample_size=0, concat=False)

# concated_years = [[]]
# for year_trigram_vecs in trigram_vecs:
#     concated_years[0] += year_trigram_vecs
# trigram_vecs = concated_years

print(f'Shape: {len(trigram_vecs)}x{len(trigram_vecs[0])}x{len(trigram_vecs[0][0])}')

for method in methods:
    clusters = cluster.cluster_years(trigram_vecs, method=method)
    print(f'Number of clusters in first year: {len(clusters[0][2])}')
    average = cluster.evaluate_clusters(clusters)
    print(f'Average score of {method}: {average}')

    linked_clusters = cluster.link_clusters(clusters)

    # visualize.show_clusters(clusters, method='TSNE')
