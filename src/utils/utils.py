import math
import ast
import pandas as pd
import numpy as np
from src.data import make_dataset
from src.features import build_features

def read_and_process_to_trigram_vecs(data_files, data_path='../data/raw/', sample_size=100, test_split=0.0, squeeze=True, extract_epitopes=False):
  trigram_to_idx, trigram_vecs_data = make_dataset.read_trigram_vecs(data_path)

  strains_by_year = make_dataset.read_strains_from(data_files, data_path)

  train_strains_by_year, test_strains_by_year = make_dataset.train_test_split_strains(strains_by_year, test_split)
  training_samples = int(math.floor(sample_size * (1 - test_split)))
  test_samples = sample_size - training_samples

  if training_samples > 0:
    train_strains_by_year = build_features.sample_strains(train_strains_by_year, training_samples)

  if test_samples > 0:
    test_strains_by_year = build_features.sample_strains(test_strains_by_year, test_samples)
  
  train_trigram_vecs, train_trigram_idxs = process_years(train_strains_by_year, squeeze, extract_epitopes, trigram_to_idx, trigram_vecs_data)
  test_trigram_vecs, test_trigram_idxs = process_years(test_strains_by_year, squeeze, extract_epitopes, trigram_to_idx, trigram_vecs_data)

  return train_trigram_vecs, test_trigram_vecs, train_trigram_idxs, test_trigram_idxs


def process_years(strains_by_year, squeeze, extract_epitopes, trigram_to_idx, trigram_vecs_data):
  if(len(strains_by_year[0]) == 0): return [],[]
  trigrams_by_year = build_features.split_to_trigrams(strains_by_year)

  if extract_epitopes:
    epitope_a = [122, 124, 126, 130, 131, 132, 133, 135, 137, 138, 140,142, 143, 144, 145, 146, 150, 152, 168]
    epitope_b = [128, 129, 155, 156, 157, 158, 159, 160, 163, 165, 186, 187, 188, 189, 190, 192, 193, 194, 196, 197, 198]
    epitope_c = [44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 273, 275, 276, 278, 279, 280, 294, 297, 299, 300, 304, 305, 307, 308, 309, 310, 311, 312]
    epitope_d = [96, 102, 103, 117, 121, 167, 170, 171, 172, 173, 174, 175, 176, 177, 179, 182, 201, 203, 207, 208, 209, 212, 213, 214, 215, 216, 217, 218, 219, 226, 227, 228, 229, 230, 238, 240, 242, 244, 246, 247, 248]
    epitope_e = [57, 59, 62, 63, 67, 75, 78, 80, 81, 82, 83, 86, 87, 88, 91, 92, 94, 109, 260, 261, 262, 265]
    epitope_positions = epitope_a + epitope_b + epitope_c + epitope_d + epitope_e
    epitope_positions.sort()

    trigrams_by_year = build_features.extract_positions_by_year(epitope_positions, trigrams_by_year)

  if squeeze:
    trigrams_by_year = build_features.squeeze_trigrams(trigrams_by_year)

  trigram_idxs = build_features.map_trigrams_to_idxs(trigrams_by_year, trigram_to_idx)
  trigram_vecs = build_features.map_idxs_to_vecs(trigram_idxs, trigram_vecs_data)

  return trigram_vecs, trigram_idxs


def read_dataset(path):
  _, trigram_vecs_data = make_dataset.read_trigram_vecs('./data/raw/')

  df = pd.read_csv(path)
  labels = df['y'].values
  trigram_idx_strings = df.loc[:, df.columns != 'y'].values
  parsed_trigram_idxs = [list(map(lambda x: ast.literal_eval(x), example)) for example in trigram_idx_strings]
  trigram_vecs = np.array(build_features.map_idxs_to_vecs(parsed_trigram_idxs, trigram_vecs_data))
  trigram_vecs_concatenated = np.reshape(trigram_vecs, [len(df.columns) - 1, len(df.index), -1])

  return trigram_vecs_concatenated, labels