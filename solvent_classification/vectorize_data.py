import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--datafile', type=str, default='heteroaryl_suzuki.csv')
parser.add_argument('--mol2vec_model', default='model_300dim.pkl')
args= parser.parse_args()

import numpy as np
from mol2vec.features import mol2alt_sentence, sentences2vec
from gensim.models import word2vec
from my_neural_fgp import *
from sklearn import preprocessing
from rdkit.Chem import Descriptors, AllChem
from rdkit import Chem
import gzip
import pickle
import time
import pandas as pd
import tqdm
import logging
from tqdm.contrib.concurrent import process_map
import multiprocessing
N_CPUS = multiprocessing.cpu_count()

#1. COMMON FILES
m2v_model = word2vec.Word2Vec.load(args.mol2vec_model)
literature_csv_file = args.datafile 

#aga_solvents = pd.read_csv('../other/aga_smiles_solv.csv', sep=';')#DEL?
#aga_bases = pd.read_csv('../other/aga_smiles_base.csv', sep=';')#DEL?
#ligands = pd.read_csv('../other/suzuki_ligandy.csv', sep=';')#DEL?

#2. VECTORIZATIONS
#2a. Mol2Vec
def processor_mol2vec(line):
   #s = '.'.join(line)#list(s)[0]
   mol=Chem.MolFromSmiles(line)
   return mol2alt_sentence(mol,1)

def embedd_m2v(smiles_list):
   sentences = [processor_mol2vec(line) for line in smiles_list]
   table = sentences2vec(sentences, m2v_model, unseen='UNK').astype(np.float32)
   return table

#2b. RdKIT
rdkit_desc = dict(Descriptors.descList)
rdkit_keys = list(rdkit_desc.keys())
rdkit_keys.sort()

N_WORKERS = min(10, N_CPUS)

def describe(smiles_list):
   L = len(smiles_list)
   result = np.zeros((L,len(rdkit_keys)))
   
   with tqdm.tqdm(desc='rdkit', total=L) as pgbar:
      for i,s in enumerate(smiles_list):
         #if len(s)>1: raise ValueError('WTF: %i %s'%(i,str(s)))
         if s=='':
            pgbar.update()
            continue

         try:
            mol=Chem.MolFromSmiles(s)
            for j,k in enumerate(rdkit_keys):
               d=rdkit_desc[k](mol)
               if not np.isfinite(d): d=0
               result[i,j]=d
         except:
            print(s, k)
            raise
         pgbar.update()
   
   return result

#2b. Morgan Fingerprints
      
def morganize(smiles_list, rad=3, lenght=512, counts=True, clean_iso=True):
   L = len(smiles_list)
   result = np.zeros((L,lenght))
   
   with tqdm.tqdm(desc='ecfp6', total=L) as pgbar:
      for i,s in enumerate(smiles_list):
         if s=='':
            pgbar.update()
            continue
         try:
            mol=Chem.MolFromSmiles(s)
            if clean_iso:
               for atom in mol.GetAtoms(): atom.SetIsotope(0)
            #if len(s)>1:raise
            fgp = AllChem.GetMorganFingerprint(mol,rad,useCounts=counts)
         except:
            print(i,s)
            raise
         details= fgp.GetNonzeroElements()
         for d in details:
            result[i,d%lenght]+=details[d]
         pgbar.update()
   
   return result

#2c. One-hot encodings
def str_one_hot(labels, v=False):
   label_encoder = preprocessing.LabelEncoder()
   onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
   nums = label_encoder.fit_transform(labels).reshape(-1, 1)
   if v:
      msg = f'Classes: {label_encoder.classes_}'
      logging.info(msg)
   return onehot_encoder.fit_transform(nums), label_encoder.classes_

#2d. Graphs
def make_graphs(all_smiles, filter_name='first_order'):
   graphs = smiles_data_processor(all_smiles)
   logging.info('SMILES processed for %s'%k)
   graphs, input_shapes = align_and_make_filters(graphs, filter_name)
   logging.info('Graphs done for %s'%k)
   return (graphs, input_shapes)

#3. Some utils
def is_low(dc):
   sml = list(dc['halogen'])[0]
   n = Chem.MolFromSmiles(sml).GetNumHeavyAtoms()
   return n<=100

def select_smallest(smiles_iter):
   result, min_n='', 1000
   do_wtf_check = 'O=Cc1ccc(-c2ccc(Br)c3nc4c5ccccc5c5ccccc5c4nc23)s1' in smiles_iter
   for x in smiles_iter:
      N = Chem.MolFromSmiles(x).GetNumHeavyAtoms()
      if N<min_n:
         min_n = N
         result= x
   if do_wtf_check:
      logging.info('WTF check: %s'%str(smiles_iter))
      logging.info('WTF check: %s'%result)
   return [result]

def check_missing_data(df, null='', columns=['solvent_class', 'base_class', 'yield', 'bromide', 'boronate']):
   missing_cols, missing_vals = [], []
   for x in columns:
      if x not in df.columns:
         missing_cols.append(x)
      else:
         full = (df[x]!=null).all()
         if not full:
            missing_vals.append(x)
   assert missing_cols==[], f'missing columns: {missing_cols}'
   assert missing_vals==[], f'missing values in columns: {missing_vals}'

FILTER_NAME = 'first_order'

#===================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')

logging.info('START')

data = pd.read_csv(literature_csv_file, sep=';').fillna('')
check_missing_data(data)
weird_bases = [',,' in x for x in data.base_class]
logging.info(f'Converting more than one base class into "other" - {sum(weird_bases)}/{len(data)} ({np.mean(weird_bases):.1f})')
data['base_class'] = data.base_class.apply(lambda x: 'other' if ',,' in x else x)

output_vectors = {'yield':data['yield'].values}
for k in ['solvent_class','base_class']:
   enc, labels = str_one_hot(data[k], True)
   output_vectors[k+'_enc'] = enc
   output_vectors[k+'_labels'] = labels
   logging.info('%s converted to one-hot'%k)

input_vectors = {}
mismatch = []
func_names, funcs = ['ecfp6', 'rdkit', 'm2v', 'graph'], [morganize, describe, embedd_m2v, make_graphs]
for k in ['boronate', 'bromide']:
   for n,func in zip(func_names, funcs):
      input_vectors[f'{k}_{n}'] = func(data[k])
      logging.info('%s converted to %s'%(k,n))
      this_len = len(input_vectors[f'{k}_{n}']) if n!='graph' else len(input_vectors[f'{k}_{n}'][0]['X'])
      if this_len!=len(data):
         mismatch.append((f'{k}_{n}', this_len))
status = mismatch==[]
logging.info(f'All tables with len {len(data)}: {status}')
if not status: logging.info(f'incorrect lens: {mismatch}')

logging.info('Saving')
with gzip.open('vectorized_data.pkz', 'wb') as f:
   pickle.dump((input_vectors, output_vectors),f)
logging.info('EOT, NCR')   

