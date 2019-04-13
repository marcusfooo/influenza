import pandas as pd
import random
import numpy as np

def construct_training_data(raw_amino_sequences, num_of_samples, overlapping=True):
  """TODO: DOCSTRING"""
  num_of_years = len(raw_amino_sequences)
  if overlapping:
    step_size = 1
    num_of_trigrams = len(raw_amino_sequences[0][0]) - 2
  else:
    step_size = 3
    num_of_trigrams = len(raw_amino_sequences[0][0]) // step_size
  
  training_data = []
  for i in range(num_of_samples):
    trigrams = [[]] * num_of_trigrams
    
    for year_idx in range(num_of_years):
      sample_sequence = random.choice(raw_amino_sequences[year_idx])
      
      for pos in range(num_of_trigrams):
        trigrams[pos] = trigrams[pos] + [sample_sequence[pos*step_size : pos*step_size + 3]]
        
    training_data += trigrams
  
  return training_data


def convert_to_indexes(training_data, trigram_to_idx):
  """TODO: DOCSTRING"""
  dummy_idx = len(trigram_to_idx)
  
  def mapping(trigram):
    if '-' not in trigram:
      return trigram_to_idx[trigram]
    else:
      return dummy_idx
   
  training_indexes = []
  for example in training_data:
    training_indexes.append(list(map(mapping, example)))
    
  # Comment on transpose
  return np.transpose(np.array(training_indexes))


def convert_to_prot_vecs(training_indexes, prot_vecs):
  """TODO: DOCSTRING"""
  dummy_vec = np.array([0] * prot_vecs.shape[1])
  
  def mapping(idx):
    if idx < prot_vecs.shape[0]:
      return prot_vecs[idx]
    else:
      return dummy_vec
  
  training_vecs = []
  for example in training_indexes:
    training_vecs.append(list(map(mapping, example)))
    
  return training_vecs
  