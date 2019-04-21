import pandas as pd
import random
import math

def read_trigram_vecs(data_path='../data/raw/'):
  """TODO: DOCSTRING"""
  prot_vec_file = 'protVec_100d_3grams.csv'
    
  df = pd.read_csv(data_path + prot_vec_file, delimiter = '\t')
  trigrams = list(df['words'])
  trigram_to_idx = {trigram: i for i, trigram in enumerate(trigrams)}
  trigram_vecs = df.loc[:, df.columns != 'words'].values  
  
  return trigram_to_idx, trigram_vecs

def read_strains_from(data_files, data_path='../data/raw/'):
  """TODO: DOCSTRING"""
  raw_strains = []
  for file_name in data_files:
    df = pd.read_csv(data_path + file_name)
    uncertain_strains = df['seq']
    sequences = replace_uncertain_AAs(uncertain_strains)
    raw_strains.append(sequences)
    
  return raw_strains

def replace_uncertain_AAs(uncertain_df):
  """TODO: DOCSTRING"""
  replacements = {'B': 'DN',
                  'J': 'IL',
                  'Z': 'EQ',
                  'X': 'ACDEFGHIKLMNPQRSTVWY'}

  certain_df = uncertain_df
  for AA in replacements.keys():
    certain_df = certain_df.replace(AA, random.choice(replacements[AA]), regex=True)
    
  return certain_df

def train_test_split_strains(strains_by_year, test_split):
  train_strains, test_strains = [], []
  for strains in strains_by_year:
    num_of_training_examples = int(math.floor(strains.count() * (1 - test_split)))
    shuffled_strains = strains.sample(frac=1).reset_index(drop=True)
    train = shuffled_strains.iloc[:num_of_training_examples].reset_index(drop=True)
    test = shuffled_strains.iloc[num_of_training_examples:].reset_index(drop=True)
    train_strains.append(train)
    test_strains.append(test)
  
  return train_strains, test_strains