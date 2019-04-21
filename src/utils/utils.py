from src.data import make_dataset
from src.features import build_features
import math

def read_and_process_to_trigram_vecs(data_files, data_path='../data/raw/', sample_size=100, test_split=0.0, squeeze=True):
  trigram_to_idx, trigram_vecs_data = make_dataset.read_trigram_vecs(data_path)

  strains_by_year = make_dataset.read_strains_from(data_files, data_path)

  train_strains_by_year, test_strains_by_year = make_dataset.train_test_split_strains(strains_by_year, test_split)
  training_samples = int(math.floor(sample_size * (1 - test_split)))
  test_samples = sample_size - training_samples

  if training_samples > 0:
    train_strains_by_year = build_features.sample_strains(train_strains_by_year, training_samples)

  if test_samples > 0:
    test_strains_by_year = build_features.sample_strains(test_strains_by_year, test_samples)
  
  train_trigram_vecs, train_trigram_idxs = process_years(train_strains_by_year, squeeze, trigram_to_idx, trigram_vecs_data)
  test_trigram_vecs, test_trigram_idxs = process_years(test_strains_by_year, squeeze, trigram_to_idx, trigram_vecs_data)

  return train_trigram_vecs, test_trigram_vecs, train_trigram_idxs, test_trigram_idxs


def process_years(strains_by_year, squeeze, trigram_to_idx, trigram_vecs_data):
  if(len(strains_by_year[0]) == 0): return [],[]
  trigrams_by_year = build_features.split_to_trigrams(strains_by_year)
  trigram_idxs = build_features.trigrams_to_indexes(trigrams_by_year, trigram_to_idx)

  if squeeze:
    trigram_idxs = build_features.squeeze_trigrams(trigram_idxs)

  trigram_vecs = build_features.indexes_to_trigram_vecs(trigram_idxs, trigram_vecs_data)


  return trigram_vecs, trigram_idxs