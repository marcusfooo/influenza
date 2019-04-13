import pandas as pd
import random

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