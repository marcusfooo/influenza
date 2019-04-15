from src.data import make_dataset
from src.features import build_features

def read_and_process_to_trigram_vecs(data_files, data_path='../data/raw/', sample_size=100, concat=True):
  trigram_to_idx, trigram_vecs_data = make_dataset.read_trigram_vecs(data_path)

  strains_by_year = make_dataset.read_strains_from(data_files, data_path)

  if sample_size > 0:
    strains_by_year = build_features.sample_strains(strains_by_year, sample_size)

  trigrams_by_year = build_features.split_to_trigrams(strains_by_year)

  trigram_idxs = build_features.trigrams_to_indexes(trigrams_by_year, trigram_to_idx)

  if concat:
    trigram_idxs = build_features.concat_trigrams(trigram_idxs)
    trigram_vecs = build_features.indexes_to_trigram_vecs(trigram_idxs, trigram_vecs_data)
  else:
    trigram_vecs = build_features.indexes_by_year_to_trigram_vecs(trigram_idxs, trigram_vecs_data)

  return trigram_vecs, trigram_idxs


