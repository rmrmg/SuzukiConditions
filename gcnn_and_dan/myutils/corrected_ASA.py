from rdkit import Chem 
#from rdkit.Chem.PeriodicTable import numTable 
#from rdkit.Chem import Crippen 
from rdkit.Chem import rdPartialCharges,rdMolDescriptors 
import numpy 
import bisect 
radCol = 5
vdwCol = -1 
bondScaleFacts = [ .1,0,.2,.3] # aromatic,single,double,triple 
import math 

from os.path import dirname, join
import pickle
dir_ = dirname(__file__)
table_path = join(dir_, 'numTable.pkl')
with open(table_path, 'rb') as f:
   numTable = pickle.load(f)


def myLabuteContribs(mol,includeHs=1,force=0, correction_factor=1): #0.79 
  """ 
    My corrected code, beceause RdKit's was fucked up. 
  """ 
  if not force: 
    try: 
      res = mol._labuteContribs 
    except AttributeError: 
      pass 
    else: 
      if res.all(): 
        return res 
     
  nAts = mol.GetNumAtoms() 
  Vi = numpy.zeros(nAts+1,'d') 
  rads = numpy.zeros(nAts+1,'d') 
  VDWrads = numpy.zeros(nAts+1,'d')
 
  # 0 contains the H information 
  rads[0] = numTable[1][radCol]
  VDWrads[0] = numTable[1][vdwCol]*correction_factor
  #print(numTable[1])
  #print(VDWrads[0], rads[0])
  for i in range(nAts): 
    rads[i+1] = numTable[mol.GetAtomWithIdx(i).GetAtomicNum()][radCol]
    VDWrads[i+1] = numTable[mol.GetAtomWithIdx(i).GetAtomicNum()][vdwCol]*correction_factor
  #print( rads, VDWrads)
  # start with explicit bonds 
  for bond in mol.GetBonds(): 
    idx1 = bond.GetBeginAtomIdx()+1 
    idx2 = bond.GetEndAtomIdx()+1 
    ri = rads[idx1] 
    rj = rads[idx2] 
    Ri = VDWrads[idx1] 
    Rj = VDWrads[idx2] 
     
    if not bond.GetIsAromatic(): 
      bij = ri+rj - bondScaleFacts[bond.GetBondType()] 
    else: 
      bij = ri+rj - bondScaleFacts[0] 
    dij = min( max( abs(Ri-Rj), bij), Ri+Rj) 
    #print(bij, dij)
    Vi[idx1] += (Rj*Rj - (Ri-dij)**2 )/ dij 
    Vi[idx2] += (Ri*Ri - (Rj-dij)**2 )/ dij 
   
  # add in hydrogens 
  if includeHs: 
    j = 0 
    numAllHs=0
    Rj = VDWrads[j]
    rj = rads[j] 
    for i in range(1,nAts+1): 
      NHs = mol.GetAtomWithIdx(i-1).GetTotalNumHs()
      numAllHs += NHs
      ri = rads[i]
      Ri = VDWrads[i] 
      bij = ri+rj 
      dij = min( max( abs(Ri-Rj), bij), Ri+Rj) 
      Vi[i] += NHs*(Rj*Rj - (Ri-dij)**2 )/ dij 
      Vi[j] += NHs*(Ri*Ri - (Rj-dij)**2 )/ dij 
  
    Vi[j] = numAllHs*4*math.pi * Rj**2 - math.pi * Rj * Vi[j]
    
  for i in range(1, nAts+1): 
    Ri = VDWrads[i] 
    Vi[i] = 4*math.pi * Ri**2 - math.pi * Ri * Vi[i] 
 
  mol._labuteContribs=Vi 
  return Vi 


def  myLabuteASA(mol,includeHs=1,force=0, correction_factor=1): 
    contribs = myLabuteContribs(mol, includeHs,force,correction_factor)
    return sum(contribs)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--correction_factor', type=float, default=1)
    ns = parser.parse_args()
    
    mols = ['C', 'CC', 'CCO', 'CCCO', 'CC(C)CO', 'CC(C)(C)CO']
    for mol in mols:
        print('\nMOL: ', mol)
        mol=Chem.MolFromSmiles(mol)
        print('  LContribs, H=0 ',['%8.3f'%x for x in myLabuteContribs(mol, includeHs=0, force=1, correction_factor=ns.correction_factor)], 
                                   '%8.3f'%myLabuteASA(mol, includeHs=0, correction_factor=ns.correction_factor))
        print('  LContribs, H=1 ', ['%8.3f'%x for x in myLabuteContribs(mol, includeHs=1, force=1, correction_factor=ns.correction_factor)], 
                                   '%8.3f'%myLabuteASA(mol,includeHs=1, force=1,correction_factor=ns.correction_factor))
        print('Adding Hs')
        mol2=Chem.AddHs(mol)
        print('  LContribs, H=0 ',['%8.3f'%x for x in myLabuteContribs(mol2, includeHs=0, force=1, correction_factor=ns.correction_factor)],
                                   '%8.3f'%myLabuteASA(mol2, includeHs=0, correction_factor=ns.correction_factor))
