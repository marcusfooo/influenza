import pandas as pd
import random
import numpy as np

def sample_strains(strains_by_year, num_of_samples):
  """TODO: DOCSTRING"""
  sampled_strains_by_year = []

  for year_strains in strains_by_year:
    sampled_strains_by_year.append(random.choices(year_strains, k=num_of_samples))

  return sampled_strains_by_year

def split_to_trigrams(strains_by_year, overlapping=True):
  """TODO: DOCSTRING"""
  if overlapping:
    step_size = 1
    num_of_trigrams = len(strains_by_year[0][0]) - 2
  else:
    step_size = 3
    num_of_trigrams = len(strains_by_year[0][0]) // step_size

  trigrams_by_year = []
  for year_strains in strains_by_year:
    year_trigrams = []

    for strain in year_strains:
      strain_trigrams = []

      for pos in range(num_of_trigrams):
        strain_trigrams.append(strain[pos*step_size : pos*step_size + 3])

      year_trigrams.append(strain_trigrams)
    
    trigrams_by_year.append(year_trigrams)

  return trigrams_by_year

def to_time_series(trigrams_by_year):
  """TODO: DOCSTRING"""
  num_of_samples = len(trigrams_by_year[0][0])
  time_series = []

  for _ in trigrams_by_year:
    year_series = [[]] * len(trigrams_by_year[0][0])
    for year_trigrams in trigrams_by_year:
      year_series = [[]] * len(year_trigrams[0])

      for trigrams in year_trigrams:
        for pos, trigram in enumerate(trigrams):
            year_series[pos] = year_series[pos] + [trigram]
    
        year_series[pos] = year_series[pos] + [trigrams[pos]]

    print(year_series)
    time_series.append(year_series)
      

  

  return time_series

def trigrams_to_indexes(trigrams_series, trigram_to_idx):
  """TODO: DOCSTRING"""
  dummy_idx = len(trigram_to_idx)
  
  def mapping(trigram):
    if '-' not in trigram:
      return trigram_to_idx[trigram]
    else:
      return dummy_idx
   
  training_indexes = []
  for example in trigrams_series:
    training_indexes.append(list(map(mapping, example)))
    
  # Comment on transpose
  return np.transpose(np.array(training_indexes))

def indexes_to_trigram_vecs(training_indexes, trigram_vecs):
  """TODO: DOCSTRING"""
  dummy_vec = np.array([0] * trigram_vecs.shape[1])
  
  def mapping(idx):
    if idx < trigram_vecs.shape[0]:
      return trigram_vecs[idx]
    else:
      return dummy_vec
  
  training_vecs = []
  for example in training_indexes:
    training_vecs.append(list(map(mapping, example)))
    
  return training_vecs