import pandas as pd
import math
import numpy as np
from src.data import make_dataset
from src.features import build_features
from src.utils import utils
from src.data import cluster
from src.visualization import visualize

def main():
  data_path = './data/raw/'
  years = list(range(1975, 2017))
  data_files = list(map(lambda x: str(x) + '.csv', years))
  training_samples = 400
  test_samples = 100
  test_split = test_samples / (training_samples + test_samples)
  clustering_method = 'random' # options: 'dbscan' 'hierarchy' 'random'

  trigram_to_idx, _ = make_dataset.read_trigram_vecs(data_path)
  epitope_a = [122, 124, 126, 130, 131, 132, 133, 135, 137, 138, 140, 142, 143, 144, 145, 146, 150, 152, 168]
  epitope_b = [128, 129, 155, 156, 157, 158, 159, 160, 163, 165, 186, 187, 188, 189, 190, 192, 193, 194, 196, 197, 198]
  epitope_c = [44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 273, 275, 276, 278, 279, 280, 294, 297, 299, 300, 304, 305, 307, 308, 309, 310, 311, 312]
  epitope_d = [96, 102, 103, 117, 121, 167, 170, 171, 172, 173, 174, 175, 176, 177, 179, 182, 201, 203, 207, 208, 209, 212, 213, 214, 215, 216, 217, 218, 219, 226, 227, 228, 229, 230, 238, 240, 242, 244, 246, 247, 248]
  epitope_e = [57, 59, 62, 63, 67, 75, 78, 80, 81, 82, 83, 86, 87, 88, 91, 92, 94, 109, 260, 261, 262, 265]
  epitope_positions = epitope_a + epitope_b + epitope_c + epitope_d + epitope_e
  epitope_positions.sort()

  strains_by_year = make_dataset.read_strains_from(data_files, data_path)
  strains_by_year = list(map(make_dataset.replace_uncertain_AAs,strains_by_year))
  train_strains_by_year, test_strains_by_year = make_dataset.train_test_split_strains(strains_by_year, test_split)

  if (clustering_method != 'random'):
    train_strains_by_year, train_clusters_by_year  = utils.cluster_years(train_strains_by_year, data_path, clustering_method)
    test_strains_by_year, test_clusters_by_year = utils.cluster_years(test_strains_by_year, data_path, clustering_method)
    print('Train clusters over the years: ')
    for i, year_clusters in enumerate(train_clusters_by_year):
        print('Year: {}\n{}'.format(i, year_clusters['population']))
    visualize.show_clusters(train_clusters_by_year, data_files, method='PCA', dims=3)
    print('Test clusters over the years: ')
    for i, year_clusters in enumerate(test_clusters_by_year):
        print('Year: {}\n{}'.format(i, year_clusters['population']))

    train_strains_by_year = cluster.sample_from_clusters(train_strains_by_year, train_clusters_by_year, training_samples)
    test_strains_by_year = cluster.sample_from_clusters(test_strains_by_year, test_clusters_by_year, test_samples)

  else:
    train_strains_by_year = build_features.sample_strains(train_strains_by_year, training_samples)
    test_strains_by_year = build_features.sample_strains(test_strains_by_year, test_samples)

  create_triplet_trigram_dataset(train_strains_by_year, trigram_to_idx, epitope_positions, file_name=(f'./data/processed/triplet_{clustering_method}_train'))
  create_triplet_trigram_dataset(test_strains_by_year, trigram_to_idx, epitope_positions, file_name=(f'./data/processed/triplet_{clustering_method}_test'))

def create_triplet_trigram_dataset(strains_by_year, trigram_to_idx, epitope_positions, file_name):
  """Creates a dataset in csv format.
  X: Time series of three overlapping trigram vectors, one example for each epitope.
  Y: 0 if epitope does not mutate, 1 if it does.
  """
  triplet_strains_by_year = build_features.make_triplet_strains(strains_by_year, epitope_positions)
  trigrams_by_year = build_features.split_to_trigrams(triplet_strains_by_year)
  trigram_idxs = build_features.map_trigrams_to_idxs(trigrams_by_year, trigram_to_idx)
  labels = build_features.make_triplet_labels(triplet_strains_by_year)

  acc, p, r, f1, mcc = build_features.get_majority_baselines(triplet_strains_by_year, labels)
  with open(file_name + '_baseline.txt', 'w') as f:
    f.write(' Accuracy:\t%.3f\n' % acc)
    f.write(' Precision:\t%.3f\n' % p)
    f.write(' Recall:\t%.3f\n' % r)
    f.write(' F1-score:\t%.3f\n' % f1)
    f.write(' Matthews CC:\t%.3f' % mcc)

  data_dict = {'y': labels}
  for year in range(len(triplet_strains_by_year) - 1):
    data_dict[year] = trigram_idxs[year]

  pd.DataFrame(data_dict).to_csv(file_name + '.csv', index=False)


if __name__ == '__main__':
  main()