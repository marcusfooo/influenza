import pandas as pd
import random
import numpy as np
from src.features.trigram import Trigram

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

      for i in range(num_of_trigrams):
        pos = i * step_size
        trigram = Trigram(strain[pos:pos + 3], pos)
        strain_trigrams.append(trigram)

      remainder = len(strain) % step_size
      if remainder > 0:
        padding = '-' * (3 - remainder)
        amino_acids = strain[-remainder:] + padding
        trigram = Trigram(amino_acids, len(strain) - remainder)
        strain_trigrams.append(trigram)

      year_trigrams.append(strain_trigrams)
    
    trigrams_by_year.append(year_trigrams)

  return trigrams_by_year


def make_triplet_strains(strains_by_year, positions):
  triplet_strains_by_year = []
  triplet_strain_margin = 2

  for strains_in_year in strains_by_year:
    triplet_strains_in_year = []
    for strain in strains_in_year:
      for p in positions:
        if p < triplet_strain_margin:
          padding_size = triplet_strain_margin - p
          triplet_strain = '-' * padding_size + strain[:p + triplet_strain_margin + 1]
        elif p > len(strain) - 1 - triplet_strain_margin:
          padding_size = p - (len(strain) - 1 - triplet_strain_margin)
          triplet_strain = strain[p - triplet_strain_margin:] + '-' * padding_size
        else:
          triplet_strain = strain[p - triplet_strain_margin:p + triplet_strain_margin + 1]
        triplet_strains_in_year.append(triplet_strain)
    triplet_strains_by_year.append(triplet_strains_in_year)

  return triplet_strains_by_year


def make_triplet_labels(triplet_strains_by_year):
  num_of_triplets = len(triplet_strains_by_year[0])
  epitope_position = 2

  labels = []
  for i in range(num_of_triplets):
    if triplet_strains_by_year[-1][i][epitope_position] == triplet_strains_by_year[-2][i][epitope_position]:
      labels.append(0)
    else:
      labels.append(1)

  return labels


def get_majority_baseline(triplet_strains_by_year, labels):
  epitope_position = 2

  correct = 0
  for i in range(len(labels)):
    epitopes = []
    for year in range(len(triplet_strains_by_year) - 1):
      epitopes.append(triplet_strains_by_year[year][i][epitope_position])
    majority_epitope = max(set(epitopes), key = epitopes.count)

    if triplet_strains_by_year[-2][i][epitope_position] == majority_epitope:
      if not labels[i]:
        correct += 1
    else:
      if labels[i]:
        correct += 1

  return correct / len(labels)


def extract_positions_by_year(positions, trigrams_by_year):
  strain = trigrams_by_year[0][0]
  strain_idxs_to_extract = []
  idx = 0

  for pos in positions:
    pos_found = False
    while not pos_found:
      trigram = strain[idx]
      if trigram.contains_position(pos):
        pos_found = True
      else:
        idx += 1

    pos_extracted = False
    while not pos_extracted:
      trigram = strain[idx]
      if trigram.contains_position(pos):
        strain_idxs_to_extract.append(idx)
        idx += 1
      else:
        pos_extracted = True

  def extract_idxs(strain_trigrams):
    return [strain_trigrams[i] for i in strain_idxs_to_extract]

  extracted_by_year = []
  for year_trigrams in trigrams_by_year:
    extracted_by_year.append(list(map(extract_idxs, year_trigrams)))

  return extracted_by_year


def squeeze_trigrams(trigrams_by_year):
  """Takes all strains (represented by trigrams) from each year 
  and squeezes them into a single array"""
  squeezed_trigrams_by_year = []

  for year_trigrams in trigrams_by_year:
      squeezed_trigrams = []

      for trigrams in year_trigrams:
          squeezed_trigrams += trigrams

      squeezed_trigrams_by_year.append(squeezed_trigrams)

  return squeezed_trigrams_by_year


def replace_uncertain_amino_acids(amino_acids):
  replacements = {'B': 'DN',
                  'J': 'IL',
                  'Z': 'EQ',
                  'X': 'ACDEFGHIKLMNPQRSTVWY'}

  for uncertain in replacements.keys():
    amino_acids = amino_acids.replace(uncertain, random.choice(replacements[uncertain]))

  return amino_acids


def map_trigrams_to_idxs(nested_trigram_list, trigram_to_idx):
  """TODO: DOCSTRING"""
  dummy_idx = len(trigram_to_idx)
  
  def mapping(trigram):
    if isinstance(trigram, Trigram):
      trigram.amino_acids = replace_uncertain_amino_acids(trigram.amino_acids)

      if '-' not in trigram.amino_acids:
        return trigram_to_idx[trigram.amino_acids]
      else:
        return dummy_idx

    elif isinstance(trigram, list):
      return list(map(mapping, trigram))
      
    else:
      raise TypeError('Expected nested list of Trigrams, but encountered {} in recursion.'.format(type(trigram)))
   
  return list(map(mapping, nested_trigram_list))


def map_idxs_to_vecs(nested_idx_list, idx_to_vec):
  """TODO: DOCSTRING"""
  dummy_vec = np.array([0] * idx_to_vec.shape[1])
  
  def mapping(idx):
    if isinstance(idx, int):
      if idx < idx_to_vec.shape[0]:
        return idx_to_vec[idx]
      else:
        return dummy_vec

    elif isinstance(idx, list):
      return list(map(mapping, idx))
      
    else:
      raise TypeError('Expected nested list of ints, but encountered {} in recursion.'.format(type(idx)))

  return list(map(mapping, nested_idx_list))


def indexes_to_mutations(trigram_indexes_x, trigram_indexes_y):
  """
  Creates an numpy array containing 1's in positions where trigram_indexes_x and
  trigram_indexes_y differ, corresponding to mutated sites and zeros elsewhere.
  """
  assert(len(trigram_indexes_x) == len(trigram_indexes_y))

  mutations = np.zeros(len(trigram_indexes_x))
  for i in range(len(trigram_indexes_x)):
    if trigram_indexes_x[i] != trigram_indexes_y[i]:
        mutations[i] = 1
  
  return mutations