from makeDataForKeras import parse, getBaseClass, getSolventClass, getSolventClassAP, Chem, AllChem, Descriptors, preprocessing, numpy as np, makeOutput
from mol2vec.features import mol2alt_sentence, sentences2vec
from gensim.models import word2vec
from my_neural_fgp import *
import gzip
import time
import pandas as pd

aga_solvents = pd.read_csv('data/smiles_solv.csv', sep=';')
aga_bases = pd.read_csv('data/smiles_base.csv', sep=';')
ligands = pd.read_csv('data/suzuki_ligandy.csv', sep=';')


def parse2(lines):
    data=[]
    #["ONERX", ["benzoin6m1N"], [], ["benzoin6m1N:::Brc1cccc2ncccc12"], [], 
    #5 ["ethanol", "toluene", "water"], ["sodium carbonate"], [80], [], ["Pd[P(Ph)3]4"], ["Inert atmosphere"], 
    #11 [13.0], ["ARTICLE", "Article; De Vreese, Rob; Muylaas; Tetrahedron Letters; vol. 58; 40; (2017); p. 3803 - 3807;"], 
    #13: {"Cc1ccc(S(=O)(=O)Oc2ccc(-c3ccc(-c4ccc(OS(=O)(=O)c5ccc(C)cc5)c5ncccc45)s3)c3cccnc23)cc1": [["CC1=CC=C(C=C1)S(=O)(=O)OC1=CC=C(Br)C2=CC=CN=C12", "Cc1ccc(S(=O)(=O)Oc2ccc(-c3ccc(B(O)O)s3)c3cccnc23)cc1"]]}, "CC1=CC=C(C=C1)S(=O)(=O)OC1=CC=C(Br)C2=CC=CN=C12.OB(O)C1=CC=C(S1)B(O)O.CC1=CC=C(C=C1)S(=O)(=O)OC1=CC=C(Br)C2=CC=CN=C12>>CC1=CC=C(C=C1)S(=O)(=O)OC1=C2N=CC=CC2=C(C=C1)C1=CC=C(S1)C1=C2C=CC=NC2=C(OS(=O)(=O)C2=CC=C(C)C=C2)C=C1"]
    for line in lines:
        ev=eval(line)
        boro = set()
        halo = set()
        for prod in ev[13]:
            for sbsList in ev[13][prod]:
                halo.add(sbsList[0])
                boro.add(sbsList[1])
        if len(boro) >1:
            boro2=boro.difference(halo)
            if len(boro2) != len(boro) and boro2:
                boro=boro2
        if len(halo) >1:
            halo2= halo.difference(boro)
            if halo2 and len(halo2) != len(halo):
                halo = halo2
        data.append( {'solvent':ev[5], 'base':ev[6], 'temp':ev[7], 'yield':ev[11], 'boronic':boro, 'halogen':halo, 'ligand':ev[8]} )
    return data


def convert_name_list_to_smiles_list(name_list, case='solv'):
   assert case in ['solv','base','lig']
   if case=='solv':
      source = aga_solvents
   elif case=='base':
      source=aga_bases
   else:
      source=ligands
   name_c, sml_c = '%s_name'%case, '%s_smiles'%case
   result = []
   for name in name_list:
       mask = np.where(source[name_c]==name)[0]
       val = source[sml_c][mask].values
       assert len(val) <=1, 'Found too many matches: %s for %s'%(str(val),name)
       if len(val)!=0:
          val = val[0]
          if val not in result: result.append(val)
   #result = '.'.join(result)
   result = [x for x in result if x.strip()!='']
   return result

model = word2vec.Word2Vec.load('/home/wbeker/projects/lingwistyka/siec/models/model_300dim.pkl')

def processor_mol2vec(line):
   s = '.'.join(line)#list(s)[0]
   mol=Chem.MolFromSmiles(s)
   return mol2alt_sentence(mol,1)


def embedd_m2v(smiles_list):
   sentences=[processor_mol2vec(line) for line in smiles_list]
   tablica=sentences2vec(sentences, model, unseen='UNK').astype(np.float32)
   return tablica

rdkit_desc = dict(Descriptors.descList)
rdkit_keys = list(rdkit_desc.keys())
rdkit_keys.sort()

#datafile='minJedenHeteroNoPatenWithYieldNewFormat.onerx'
datafile = '/chematica/madness/suzukiWarunki/withRxid.onerx'

def getNewSolventClass(solvent_list):
   return getSolventClass(  frozenset([ getSolventClassAP(s) for s in solvent_list ]) )


def str_one_hot(labels, v=False):
   label_encoder = preprocessing.LabelEncoder()
   onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
   nums = label_encoder.fit_transform(labels).reshape(-1, 1)
   if v:
      print('Classes: ',len(label_encoder.classes_))
   return onehot_encoder.fit_transform(nums)
   

def is_low(dc):
   sml = list(dc['halogen'])[0]
   n = Chem.MolFromSmiles(sml).GetNumHeavyAtoms()
   return n<=100

def load_rr(name=datafile, keys=['solvent', 'base', 'boronic', 'halogen','ligand', 'yield']):
   with open(name, 'r') as f:
      raw =f.readlines()
      print('Raw data: ',len(raw))
      dict_list = parse2(raw)
      dict_list = [x for x in dict_list if len(x['boronic'])==1 and len(x['halogen'])==1]
      print('single: ',len(dict_list))
      dict_list = [x for x in dict_list if is_low(x)]
      #for i,x in enumerate(dict_list):
      #   dict_list[i]['halogen'] = select_smallest(x['halogen'])
      print('included: ', len(dict_list))
      #solvent base temp yield boronic halogen
   result={}
   sample=True
   for key in keys:
      d=[]
      d2=[]
      for x in dict_list:
         y = x[key]
         if sample:
            print(x)
            sample=False
         if key=='solvent':
            solv_smiles = convert_name_list_to_smiles_list(y, case='solv')
            y=getNewSolventClass(y)
            d2.append(solv_smiles)
         elif key=='base':
            base_smiles = convert_name_list_to_smiles_list(y, case='base')
            solv =getSolventClass(  frozenset([ getSolventClassAP(s) for s in x['solvent'] ]) )
            y=frozenset([getBaseClass(b) for b in y ])
            y, _ = makeOutput(y ,solv)
            d2.append(base_smiles)
         elif key=='ligand':
            lig_smiles = convert_name_list_to_smiles_list(y, case='lig')
            d2.append(lig_smiles)
         elif key=='yield':
            y = y[0] if y!=[] else -1
         d.append(y)
      result[key]=d
      if d2!=[]:
         result[key + '_sml']=d2
   return result
     
      
def morganize(smiles_list, rad=3, lenght=512, counts=True, clean_iso=True):
   L = len(smiles_list)
   result = np.zeros((L,lenght))
   
   for i,s in enumerate(smiles_list):
      s = '.'.join(s).strip('.').strip()#list(s)[0]
      if s=='':continue
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

   return result


def describe(smiles_list):
   L = len(smiles_list)
   result = np.zeros((L,len(rdkit_keys)))
    
   for i,s in enumerate(smiles_list):
      #if len(s)>1: raise ValueError('WTF: %i %s'%(i,str(s)))
      s = '.'.join(s).strip('.').strip()#list(s)[0]
      if s=='':continue

      try:
         mol=Chem.MolFromSmiles(s)
         for j,k in enumerate(rdkit_keys):
            d=rdkit_desc[k](mol)
            if not np.isfinite(d): d=0
            result[i,j]=d
      except:
         print(s, k)
         raise
   
   return result

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


if __name__=='__main__':
   import argparse
   parser = argparse.ArgumentParser()
   parser.add_argument('--use_previous', type=str, default=None)
   args= parser.parse_args()
   import pickle
   import gzip
   import logging
   logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')

   filter_name='first_order'

   logging.info('START')
   if args.use_previous == None:
      d=load_rr()
      logging.info('Raw RR data loaded')
   else:
      with gzip.open(args.use_previous, 'rb') as f:
         d = pickle.load(f)
      logging.info('Previous vectors loaded')

   for k in ['solvent','base']:
      enc = str_one_hot(d[k], True)
      d[k+'_enc']=enc
      logging.info('%s converted to one-hot'%k)

   for k in ['boronic', 'halogen', 'solvent_sml', 'base_sml', 'ligand_sml']:
      for n,func in zip(['ecfp6', 'rdkit', 'm2v'], [morganize, describe, embedd_m2v]):
         if n=='m2v':continue
         key = '%s_%s'%(k.split('_')[0],n)
         if key in d: continue
         enc = func(d[k])
         logging.info('%s converted to %s'%(k,n))
         d[key] = enc
      if False:#'%s_graph'%k not in d:
         #all_smiles = ['.'.join(s) for s in d[k]]
         all_smiles = [list(s)[0] for s in d[k]]
         graphs = smiles_data_processor(all_smiles)
         logging.info('SMILES processed for %s'%k)
         graphs, input_shapes = align_and_make_filters(graphs, filter_name)
         logging.info('Graphs done for %s'%k)
         d['%s_graphs'%k] = (graphs, input_shapes)

   logging.info('Saving')
   with gzip.open('preprocessed_data_sml_all_morgan_rdkit.pkz', 'wb') as f:
      pickle.dump(d,f)
   logging.info('EOT, NCR')   
