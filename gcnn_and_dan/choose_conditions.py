import gzip
import pickle
import pandas as pd
datafile = 'preprocessed_data_sml_all_morgan_rdkit.pkz'
from rdkit import Chem
from rdkit.Chem.Descriptors import MolLogP
import numpy as np

def smiles_to_logP(smiles):
   if smiles=='': return 0
   m=Chem.MolFromSmiles(smiles)
   return MolLogP(m)

solvent_names = pd.read_csv('data/smiles_solv.csv', sep=';').set_index('solv_smiles')

def get_name(smiles):
   name = solvent_names['solv_name'][smiles]
   if not isinstance(name, str):
      name = name.values[0]
   return name

with gzip.open(datafile, 'rb') as f:
   data = pickle.load(f)

def make_counts(data):
   base_mols={}
   solv_mols={}
   all_conditions={}
   more_than_two=set()
   
   for i, base_sml in enumerate(data['base_sml']):
      base_sml = '.'.join(base_sml)
      solv_sml = '.'.join(data['solvent_sml'][i])
      base_class = data['base_enc'][i].argmax()
      solv_class = data['solvent_enc'][i].argmax()
      #print(solv_sml, base_sml, base_class, solv_class)
   
      if base_class not in base_mols: base_mols[base_class]=set()
      base_mols[base_class].add(base_sml)
      if solv_class not in solv_mols: solv_mols[solv_class]=set()
      solv_mols[solv_class].add(solv_sml)
      
      condition = base_sml + ';'+solv_sml
      if condition not in all_conditions:
         all_conditions[condition]=0
      all_conditions[condition]+=1
      if solv_sml.count('.')>1:
         more_than_two.add(solv_sml)
   return {'base':base_mols, 'solv':solv_mols, 'cond':all_conditions, 'more_than_two':more_than_two, 'N':len(data['base_sml'])}


def calc_av_logP(solv, water_content=0.3):
   smls = solv.split('.')
   organic = [x for x in smls if x!='O']
   water = 'O' if 'O' in smls else ''

   #case 1: no organic
   if organic==[]:
      result = smiles_to_logP(water)
   else:
      logPs = [smiles_to_logP(x) for x in organic]
      max_logP = round(max(logPs),1)
      #case 2: only polar
      if max_logP<1.0:
         result = sum(logPs)/len(logPs)
         if water == '':
            result = water_content*smiles_to_logP(water) + (1-water_content)*result
      else:
         #case 3: two phase system
         #assumption: uniform distribution of organics
         av_logP, denominator = 0, 0
         for x in logPs:
            #P = co/cw, then just solved
            alpha = 1/((10**(-x))+1)
            av_logP += alpha*x
            denominator += alpha
         result = av_logP/denominator
   return result


def choose_solvent_and_base(halogen_smiles, boronic_smiles, solvent_class, base_class, count_dict, th_freq=20):
   base_mols = count_dict['base'][base_class]
   solv_mols = count_dict['solv'][solvent_class]
   
   halogen_logP = smiles_to_logP(halogen_smiles)
   boronic_logP = smiles_to_logP(boronic_smiles)

   #prepare combinations
   combinations=[]
   for base in base_mols:
      if base=='':continue
      for solv in solv_mols:
         if solv=='':continue
         c = base+';'+solv
         f = count_dict['cond'].get(c,0)
         av_logP = calc_av_logP(solv)
         dev = abs(halogen_logP-av_logP) + abs(boronic_logP-av_logP)
         combinations.append((c,1000*dev -f))#-10*f+dev))
         #if f >=th_freq:
         #   combinations.append((c,dev))
   combinations.sort(key=lambda x:x[1])

   return combinations[0][0]

def test_on_random(n=10):
   count_dict = make_counts(data)
   indices = np.random.choice(np.arange(len(data['boronic'])), n, replace=False)
   for i in indices:
      boronic = list(data['boronic'][i])[0]
      halogen = list(data['halogen'][i])[0]
      solvent_sml = '.'.join(data['solvent_sml'][i])
      base_sml = '.'.join(data['base_sml'][i])
      solvent_class = data['solvent_enc'][i].argmax()
      base_class = data['base_enc'][i].argmax()
      condition = choose_solvent_and_base(halogen, boronic, solvent_class, base_class, count_dict)
      print('.'.join([boronic,halogen]), '\t', ';'.join([base_sml,solvent_sml]), '\t', condition)

if __name__=='__main__':
   import argparse
   parser = argparse.ArgumentParser()
   parser.add_argument('--boronic', type=str, default='c1ccccc1B(OC(=O)C)OC(=O)C')
   parser.add_argument('--halogen', type=str, default='c1ccccc1Br')
   parser.add_argument('--solvent_class', type=int, default=1)
   parser.add_argument('--base_class', type=int, default=1)
   parser.add_argument('--test', type=int, default=0)
   args = parser.parse_args()
   if args.test==0:
      count_dict = make_counts(data)
      print(choose_solvent_and_base(args.halogen, args.boronic, args.solvent_class, args.base_class, count_dict))
   else:
      test_on_random(args.test)

#mtt=[]
#for x in more_than_two:
#   x=x.split('.')
#   logP = [smiles_to_logP(xx) for xx in x]
#   names = [get_name(xx) for xx in x]
#   d = list(zip(x,names,logP))
#   d.sort(key=lambda x:-x[-1])
#   rng = max(logP)-min(logP)
#   x , names, logP = list(zip(*d))
#   x='.'.join(x)
#   name = ';'.join(names)
#   mtt.append((x,name,logP,rng))
#
#mtt.sort(key=lambda x:-x[-1])
##for x in mtt:print(x)
#
