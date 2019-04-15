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

def trigrams_to_indexes(trigrams_by_year, trigram_to_idx):
  """TODO: DOCSTRING"""
  dummy_idx = len(trigram_to_idx)
  
  def mapping(trigram):
    if '-' not in trigram:
      return trigram_to_idx[trigram]
    else:
      return dummy_idx
   
  trigrams_idxs_by_year = []
  for year_trigrams in trigrams_by_year:
    year_trigrams_idxs = []

    for trigrams in year_trigrams:
        year_trigrams_idxs.append(list(map(mapping, trigrams)))
    
    trigrams_idxs_by_year.append(year_trigrams_idxs)
    
  return trigrams_idxs_by_year

def concat_trigrams(trigrams_by_year):
  """Takes all strains (represented by trigrams) from each year 
  and concatenates them into a single array"""
  concated_trigrams_by_year = []

  for year_trigrams in trigrams_by_year:
      concated_trigrams = []

      for trigrams in year_trigrams:
          concated_trigrams += trigrams

      concated_trigrams_by_year.append(concated_trigrams)

  return concated_trigrams_by_year

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

def indexes_by_year_to_trigram_vecs(training_indexes_by_year, trigram_vecs_source):
  """TODO: DOCSTRING"""
  dummy_vec = np.array([0] * trigram_vecs_source.shape[1])
  
  def mapping(idx):
    if idx < trigram_vecs_source.shape[0]:
      return trigram_vecs_source[idx]
    else:
      return dummy_vec
  
  trigram_vecs = []
  for year_trigrams in training_indexes_by_year:
    year_trigram_vecs = []

    for trigrams in year_trigrams:
        year_trigram_vecs.append(list(map(mapping, trigrams)))

    trigram_vecs.append(year_trigram_vecs)

  return trigram_vecs