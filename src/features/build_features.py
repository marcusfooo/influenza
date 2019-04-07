import pandas as pd
import random

def read_prot_vecs(data_path):
  """TODO: DOCSTRING"""
  prot_vec_file = 'protVec_100d_3grams.csv'
    
  df = pd.read_csv(data_path + prot_vec_file, delimiter = '\t')
  prot_trigrams = list(df['words'])
  trigram_to_idx = {trigram: i for i, trigram in enumerate(prot_trigrams)}
  prot_vecs = df.loc[:, df.columns != 'words'].values  
  
  return trigram_to_idx, prot_vecs

def read_amino_sequences(data_path, data_files):
  """TODO: DOCSTRING"""
  raw_amino_sequences = []
  for file_name in data_files:
    df = pd.read_csv(data_path + file_name)
    uncertain_sequences = df['seq']
    sequences = replace_uncertain_AAs(uncertain_sequences)
    raw_amino_sequences.append(sequences)
    
  return raw_amino_sequences

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