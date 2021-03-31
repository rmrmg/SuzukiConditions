from rdkit import Chem
from rdkit.Chem import AllChem
from .corrected_ASA import myLabuteContribs
from random import random
from numpy import zeros, array, identity, diag
from multiprocessing import Pool

p_table = Chem.GetPeriodicTable()

mi = {
     'Te.0,Te.0': 17.016219,
     'P.0,P.0': 12.718968,
     'Ag.1,Mg.1': 12.630596,
     'Te.1,Te.1': 11.649775,
     'Mg.1,Mn.1': 11.630596,
     'Au.2,P.2': 11.254668,
     'Cu.1,Li.1': 10.996320,
     'B.3,N.3': 10.715739,
     'N.3,S.3': 10.715739,
     'C.3,O.3':  9.640335,
     'Se.0,Se.0':  9.578583,
     'C.3,N.3':  9.575869,
     'P.0,Se.0':  9.524811,
     'C.3,P.3':  9.318407,
     'P.2,Se.2':  8.948859,
     'I.1,Mn.1':  8.873403,
     'N.2,Pb.2':  8.729469,
     'Cl.1,Pt.1':  8.606612,
     'Br.1,Mg.1':  8.586384,
     'I.1,Zn.1':  8.519766,
     'N.3,P.3':  8.393811,
     'N.2,Si.2':  8.314431,
     'Ce.1,Cl.1':  8.244042,
     'Br.1,Ca.1':  8.235042,
     'I.1,Pd.1':  8.195331,
     'B.2,S.2':  8.102166,
     'F.1,O.1': -8.030081,
     'P.1,Pd.1':  7.838800,
     'Cl.1,Fe.1':  7.829004,
     'Ca.1,Cl.1':  7.829004,
     'Au.1,Cl.1':  7.829004,
     'C.3,C.3':  7.769159,
     'Bi.1,Se.1':  7.656592,
     'Se.1,Se.1':  7.564531,
     'O.0,P.0':  7.562475,
     'Br.1,Zn.1':  7.296442,
     'I.1,Mg.1':  7.283639,
     'Ni.1,P.1':  7.253837,
     'As.1,Se.1':  7.066048,
     'B.0,N.0':  6.787200,
     'Al.1,Se.1':  6.690539,
     'Mn.1,Si.1':  6.539801,
     'O.0,Te.0':  6.527709,
     'Cl.1,Mn.1':  6.507076,
     'Cl.1,Ti.1':  6.507076,
     'Br.1,Hg.1':  6.427687,
     'Al.1,Cl.1':  6.343578,
     'Cl.1,Tl.1':  6.244042,
     'Br.1,Pd.1':  6.235042,
     'Cl.1,Hg.1':  6.214295,
     'Bi.2,O.2':  6.160745,
     'O.2,Re.2':  6.160745,
     'O.2,Sb.2':  6.160745,
     'O.2,Sn.2':  6.160745,
     'Fe.2,O.2':  6.160745,
     'I.2,N.2':  6.144506,
     'Cl.1,Sb.1':  6.128565,
     'Cl.1,Ge.1':  6.092039,
     'Cl.1,Zn.1':  5.983514,
     'O.2,S.2':  5.933290,
     'As.2,O.2':  5.912817,
     'O.2,P.2':  5.849947,
     'I.2,O.2':  5.745707,
     'Cl.1,Mg.1':  5.724668,
     'N.2,N.2':  5.719360,
     'Au.1,S.1':  5.692375,
     'C.2,Ta.2':  5.635330,
     'C.2,Te.2':  5.635330,
     'P.2,S.2':  5.601143,
     'Hg.1,I.1':  5.580621,
     'Cl.1,Cl.1': -5.528135,
     'Al.1,Br.1':  5.427687,
     'C.2,O.2':  5.335283,
     'P.0,S.0':  5.303071,
     'F.1,Sb.1':  5.230263,
     'N.3,N.3':  5.176072,
     'B.2,O.2':  5.160745,
     'O.1,O.1': -4.962284,
     'Cl.1,Pb.1':  4.954535,
     'O.2,Se.2':  4.814295,
     'Fe.1,N.1':  4.683180,
     'As.1,Sn.1':  4.607093,
     'N.2,P.2':  4.559544,
     'Ge.1,Si.1':  4.539801,
     'N.2,O.2':  4.466659,
     'As.1,Cl.1':  4.456052,
     'Cl.1,O.1': -4.440617,
     'As.1,B.1':  4.435473,
     'I.1,Te.1':  4.427147,
     'C.2,N.2':  4.422275,
     'Co.1,N.1':  4.361252,
     'Cl.1,P.1':  4.329239,
     'N.2,Se.2':  4.253735,
     'C.2,Se.2':  4.113793,
     'Sn.1,Sn.1':  4.061659,
     'B.1,I.1':  4.046431,
     'Co.1,O.1':  3.962326,
     'Br.1,Se.1':  3.939109,
     'B.1,Se.1':  3.933956,
     'S.1,Sb.1':  3.898825,
     'B.1,F.1':  3.897279,
     'O.1,P.1':  3.892269,
     'Cl.1,Cu.1':  3.829004,
     'N.0,P.0':  3.821965,
     'N.0,Se.0':  3.805347,
     'B.1,O.1':  3.742191,
     'B.1,Cl.1':  3.738998,
     'I.1,N.1': -3.722670,
     'Ge.1,S.1':  3.692375,
     'C.2,Si.2':  3.635330,
     'P.1,Se.1':  3.609981,
     'I.1,I.1':  3.526446,
     'O.1,Ti.1':  3.447753,
     'N.0,O.0':  3.426255,
     'Bi.1,Cl.1':  3.402740,
     'S.2,S.2': -3.396849,
     'Al.1,O.1':  3.377364,
     'O.1,Tl.1':  3.377364,
     'C.2,C.2':  3.268151,
     'O.1,Sb.1':  3.261886,
     'As.1,I.1':  3.178523,
     'O.1,Pt.1':  3.154971,
     'Ga.1,N.1':  3.098218,
     'I.1,Se.1':  3.092043,
     'Cu.1,S.1':  3.014303,
     'F.1,N.1': -2.987299,
     'Cl.1,Se.1':  2.973644,
     'As.1,O.1':  2.967886,
     'O.0,O.0': -2.960800,
     'B.1,Br.1':  2.960610,
     'Br.1,N.1': -2.875605,
     'As.2,C.2':  2.846834,
     'As.1,Br.1':  2.803196,
     'C.2,S.2':  2.739941,
     'N.0,N.0':  2.731997,
     'Br.1,Br.1': -2.716061,
     'I.1,Sn.1':  2.633089,
     'S.2,Se.2':  2.626433,
     'S.0,S.0':  2.609034,
     'Mg.1,N.1': -2.550439,
     'C.0,S.0':  2.491985,
     'As.2,N.2':  2.481541,
     'Br.1,Te.1':  2.466857,
     'As.1,S.1':  2.456926,
     'B.1,S.1':  2.450365,
     'I.1,O.1': -2.443524,
     'O.1,Si.1':  2.436653,
     'B.1,P.1':  2.426865,
     'C.0,O.0':  2.410483,
     'Cl.1,N.1': -2.370179,
     'C.0,N.0':  2.369457,
     'Bi.1,O.1':  2.343416,
     'O.1,Pb.1':  2.310249,
     'C.0,Se.0':  2.292723,
     'N.1,S.1':  2.270811,
     'Cl.1,Si.1':  2.246056,
     'C.1,Cs.1':  2.206506,
     'C.1,W.1':  2.206506,
     'C.1,Mo.1':  2.206506,
     'C.1,Ta.1':  2.206506,
     'C.1,K.1':  2.206506,
     'C.1,In.1':  2.206506,
     'C.1,Re.1':  2.206506,
     'C.1,Zr.1':  2.206506,
     'C.1,Cd.1':  2.206506,
     'C.1,Li.1':  2.196591,
     'C.1,F.1':  2.181145,
     'N.1,P.1':  2.172938,
     'Br.1,C.1':  2.159097,
     'B.1,N.1':  2.148990,
     'C.1,I.1':  2.144211,
     'N.0,S.0':  2.132044,
     'P.2,P.2':  2.130547,
     'C.1,Sn.1':  2.130093,
     'N.1,Se.1': -2.098179,
     'C.1,Cl.1':  2.089263,
     'C.1,Na.1':  2.086212,
     'C.1,O.1':  2.081748,
     'C.1,Cu.1':  2.064487,
     'C.2,I.2':  2.050367,
     'C.1,N.1':  2.043036,
     'C.1,Se.1':  1.978200,
     'N.2,S.2':  1.967419,
     'O.1,Pd.1':  1.962326,
     'P.1,S.1':  1.952360,
     'Bi.1,C.1':  1.950166,
     'C.1,Ga.1':  1.943472,
     'Br.1,P.1':  1.917444,
     'C.1,Te.1':  1.914055,
     'C.1,S.1':  1.910990,
     'Ag.1,C.1':  1.884578,
     'O.1,Zn.1': -1.883164,
     'C.1,Si.1':  1.849497,
     'C.1,Pb.1':  1.834537,
     'C.1,Ni.1':  1.791469,
     'C.0,Te.0':  1.774502,
     'Cl.1,S.1':  1.749920,
     'C.1,Hg.1':  1.721079,
     'B.1,B.1':  1.718419,
     'S.1,Se.1':  1.718370,
     'C.1,Ge.1':  1.691933,
     'Al.1,S.1':  1.621985,
     'S.1,Si.1': -1.609673,
     'Cl.1,Sn.1':  1.588690,
     'C.1,Zn.1':  1.570469,
     'Al.1,C.1':  1.542109,
     'Cl.1,I.1':  1.538631,
     'As.1,C.1':  1.529548,
     'Si.1,Te.1':  1.508582,
     'P.1,Te.1':  1.485653,
     'S.1,S.1':  1.482706,
     'C.0,C.0':  1.482042,
     'B.1,Si.1': -1.457096,
     'C.2,P.2':  1.411213,
     'Hg.1,S.1':  1.399593,
     'I.1,S.1': -1.391548,
     'Br.1,S.1': -1.351837,
     'C.1,Sb.1':  1.313421,
     'C.0,P.0':  1.303082,
     'Si.1,Sn.1': -1.285476,
     'C.1,Mg.1':  1.267507,
     'F.1,P.1':  1.261430,
     'S.1,Sn.1':  1.259415,
     'S.1,Te.1':  1.246118,
     'C.1,Tl.1':  1.206506,
     'O.1,Se.1': -1.140962,
     'N.1,Sn.1': -1.071707,
     'Cl.1,Te.1':  1.060820,
     'C.1,Ti.1':  1.054503,
     'N.1,Na.1':  1.039324,
     'Si.1,Si.1':  0.989441,
     'Se.1,Si.1':  0.980834,
     'O.1,S.1':  0.976599,
     'F.1,Sn.1': -0.953468,
     'Hg.1,N.1': -0.931530,
     'O.1,Sn.1': -0.899477,
     'I.1,P.1':  0.847986,
     'As.1,Si.1':  0.844921,
     'N.1,O.1':  0.798516,
     'C.1,Pd.1':  0.791469,
     'B.1,C.1':  0.790056,
     'C.1,C.1':  0.773433,
     'C.1,P.1':  0.772550,
     'I.1,Si.1': -0.714046,
     'Al.1,N.1': -0.709137,
     'Br.1,Sn.1':  0.672799,
     'Hg.1,O.1':  0.669544,
     'C.1,Ce.1':  0.621544,
     'P.1,P.1':  0.606932,
     'Br.1,Si.1': -0.586873,
     'Au.1,C.1':  0.469540,
     'C.1,Mn.1':  0.469540,
     'C.1,Co.1':  0.469540,
     'As.1,F.1': -0.408034,
     'F.1,S.1':  0.367670,
     'Ge.1,O.1': -0.359602,
     'P.1,Si.1': -0.333612,
     'Na.1,O.1':  0.318470,
     'N.1,Si.1': -0.283683,
     'Br.1,I.1':  0.244228,
     'C.1,Ca.1':  0.206506,
     'N.1,N.1':  0.133322,
     'As.1,N.1':  0.125804,
     'F.1,Si.1': -0.110779}

electronegativity = {
   'H':  2.20 ,   
   'Li':    0.98 ,   
   'Be':    1.57 ,   
   'B':  2.04 ,   
   'C':  2.55 ,   
   'N':  3.04 ,   
   'O':  3.44 ,   
   'F':  3.98 ,   
   'Na':    0.93 ,   
   'Mg':    1.31 ,   
   'Al':    1.61 ,   
   'Si':    1.90 ,   
   'P':  2.19 ,   
   'S':  2.59 ,   
   'Cl':    3.16 ,   
   'K':  0.82 ,   
   'Ca':    1.00 ,   
   'Sc':    1.36 ,   
   'Ti':    1.54 ,   
   'V':  1.63 ,   
   'Cr':    1.66 ,   
   'Mn':    1.55 ,   
   'Fe':    1.83 ,   
   'Co':    1.88 ,   
   'Ni':    1.91 ,   
   'Cu':    1.90 ,   
   'Zn':    1.65 ,   
   'Ga':    1.81 ,   
   'Ge':    2.01 ,   
   'As':    2.18 ,   
   'Se':    2.55 ,   
   'Br':    2.96 ,   
   'Kr':    3.00 ,   
   'Rb':    0.82 ,   
   'Sr':    0.95 ,   
   'Y':  1.22 ,   
   'Zr':    1.33 ,   
   'Nb':    1.6  ,   
   'Mo':    2.16 ,   
   'Tc':    1.9  ,   
   'Ru':    2.2  ,   
   'Rh':    2.28 ,   
   'Pd':    2.20 ,   
   'Ag':    1.93 ,   
   'Cd':    1.69 ,   
   'In':    1.78 ,   
   'Sn':    1.96 ,   
   'Sb':    2.05 ,   
   'Te':    2.1  ,   
   'I':  2.66 ,   
   'Xe':    2.6  ,   
   'Cs':    0.79 ,   
   'Ba':    0.89 ,   
   'La':    1.10 ,   
   'Ce':    1.12 ,   
   'Pr':    1.13 ,   
   'Nd':    1.14 ,   
   'Sm':    1.17 ,   
   'Gd':    1.20 ,   
   'Dy':    1.22 ,   
   'Ho':    1.23 ,   
   'Er':    1.24 ,   
   'Tm':    1.25 ,   
   'Lu':    1.27 ,   
   'Hf':    1.3  ,   
   'Ta':    1.5  ,   
   'W':  2.36 ,   
   'Re':    1.9  ,   
   'Os':    2.2  ,   
   'Ir':    2.20 ,   
   'Pt':    2.28 ,   
   'Au':    2.54 ,   
   'Hg':    2.00 ,   
   'Tl':    1.62 ,   
   'Pb':    2.33 ,   
   'Bi':    2.02 ,   
   'Po':    2.0  ,   
   'At':    2.2  ,   
   'Ra':    0.9  ,   
   'Ac':    1.1  ,   
   'Th':    1.3  ,   
   'Pa':    1.5  ,   
   'U':  1.38 ,   
   'Np':    1.36 ,   
   'Pu':    1.28 ,   
   'Am':    1.3  ,   
   'Cm':    1.3  ,   
   'Bk':    1.3  ,   
   'Cf':    1.3  ,   
   'Es':    1.3  ,   
   'Fm':    1.3  ,   
   'Md':    1.3  }

def to_one_hot(k, tot):
   result = [0 for _ in range(tot)]
   result[k]=1
   return result


class RandomDescriptors():
   def __init__(self, key='Lv',target_element='O', use_elements=True, size=8, keepHs=False ):
      self.key = key
      self.target_element = target_element
      self.size = size
      self.use_elements = use_elements
      self.keepHs=keepHs
      self.memory={}


   def _get_random_descriptor_of_element(self, element):
      if element in self.memory:
         return self.memory[element]
      else:
         vector = [random() for _ in range(self.size)]
         self.memory[element] = vector
         return vector


   def make_random_descriptors(self, smiles):
      mol = Chem.MolFromSmiles(smiles.replace('['+self.key, '['+self.target_element))
      result = []
      for Atom in mol.GetAtoms():
         idx=Atom.GetIdx()
         S = Atom.GetSymbol()
         NH = Atom.GetTotalNumHs()
         if self.use_elements:
            vector=self._get_random_descriptor_of_element(S)[:]
         else:
            vector=[random() for _ in range(self.size)]
         if self.keepHs:
            vector+=[float(NH)]
         result.append(vector)
      return result

   def __call__(self, smiles):
      return self.make_random_descriptors(smiles)


def get_bond_mi(bond, dc=mi, absolute=True):
    deg=bond.GetBondTypeAsDouble()
    if deg==1.5:
        deg=1
    else:
        deg-=1
    deg=int(deg)
    a,b=bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()
    key='{at1:s}.{deg:d},{at2:s}.{deg:d}'.format(at1=a, at2=b, deg=deg)
    alter_key='{at1:s}.{deg:d},{at2:s}.{deg:d}'.format(at1=b, at2=a, deg=deg)
    if key in dc:
        result = dc[key]
    elif alter_key in dc:
        result = dc[alter_key]
    else:
        result=0.0
    if absolute:
        result=abs(result)
    return result


def _check_non_diagonal_zeros(matrix):
 N,M = matrix.shape
 assert N==M
 for i in range(N):
   for j in range(i+1,N):
      if matrix[i,j]==0:
         return True
 return False


def adjacency_to_distance(adj_matrix, min_dist_to_include=1, factor=1.0, max_dist_to_include=10):
   N,M = adj_matrix.shape
   assert N==M
   dist = zeros((N,N))
   b = adj_matrix[:,:]
   I=1.0
   
   while(_check_non_diagonal_zeros(dist) and I<=max_dist_to_include):
     for i in range(len(b)):
       for j in range(i+1,len(b)):
          if dist[i,j]==0 and b[i,j]>=min_dist_to_include:
            if factor==1.0:
               to_set = I
            else:
               to_set = I**factor
            dist[i,j] = to_set
            dist[j,i] = to_set
     I+=1.0
     b=b.dot(adj_matrix)
   return dist


def featurize_bond(bond, use_mi=False, use_polarization=False):
    aromaticity = float(bond.GetIsAromatic())
    order = bond.GetBondTypeAsDouble()
    data= [order, aromaticity]
    begin_atom, end_atom = bond.GetBeginAtom(), bond.GetEndAtom()
    if use_mi:
       data.append(get_bond_mi(bond))
    if use_polarization:
       atom1, atom2 = begin_atom.GetSymbol(), end_atom.GetSymbol()
       el1, el2 = [electronegativity[x] for x in [atom1,atom2]]
       data.append(abs(el1-el2))
    return data, begin_atom.GetIdx(), end_atom.GetIdx()
      

def get_bond_space_matrices(mol, use_mi=False, use_polarization=False):
   '''Returns: bond features, bond_adj_mtx, bond2atom_mtx ( X: A = X.dot(B) )'''
   Nbonds = mol.GetNumBonds()
   Natoms = mol.GetNumAtoms()
   bond_features = []
   bond2atom = zeros((Natoms, Nbonds))
   for bi, bond in enumerate(mol.GetBonds()):
      F, beg_i, end_i = featurize_bond(bond, use_mi, use_polarization)
      bond_features.append(F)
      bond2atom[beg_i,bi]=1
      bond2atom[end_i,bi]=1
   bond_adj = bond2atom.T.dot(bond2atom)
   bond_adj = bond_adj - diag(diag(bond_adj))
   return array(bond_features), bond_adj, bond2atom
   

def compute_Gasteiger_charges(smiles, idx=0, Gasteiger_iterations=200):
   mol = Chem.MolFromSmiles(smiles)
   mol_ionized = Chem.MolFromSmiles(smiles)
   atom_ionized = mol_ionized.GetAtomWithIdx(idx)
   atom = mol.GetAtomWithIdx(idx)
   numHs = atom.GetTotalNumHs()
   if numHs>0:
      atom_ionized.SetNoImplicit(1)
      atom_ionized.SetFormalCharge(-1)
      atom_ionized.SetNumExplicitHs(numHs-1)
      Chem.rdmolops.SanitizeMol(mol_ionized)
   try:
      Chem.rdPartialCharges.ComputeGasteigerCharges(mol, Gasteiger_iterations, True)
      Chem.rdPartialCharges.ComputeGasteigerCharges(mol_ionized, Gasteiger_iterations, True)
      q_in_neu = atom.GetDoubleProp('_GasteigerHCharge') + atom.GetDoubleProp('_GasteigerCharge')
      q_in_ion = atom_ionized.GetDoubleProp('_GasteigerHCharge') + atom_ionized.GetDoubleProp('_GasteigerCharge')
   except ValueError:
      q_in_neu, q_in_ion = 0.0, 0.0
   is_ion_aromatic = atom_ionized.GetIsAromatic()
   return {'q_neu': q_in_neu, 'q_ion':q_in_ion, 'is_aromatic':is_ion_aromatic}


known_elements=['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
                'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
                'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',    # H?
                'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
                                    'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']
UNKN = len(known_elements)-1

def describe_atom(atom_object, use_formal_charge=False, use_Gasteiger=False, use_mi=True):
   mol = atom_object.GetOwningMol()
   contribs = myLabuteContribs(mol, includeHs=0)
   idx = atom_object.GetIdx()
   code = {'SP':1, 'SP2':2, 'SP3':3,'UNSPECIFIED':-1, 'UNKNOWN':-1, 'S':0, 'SP3D':4, 'SP3D2':5}
   result = []
   symbol = atom_object.GetSymbol()
   result.append(atom_object.GetAtomicNum())
   symbol_idx = UNKN if symbol not in known_elements else known_elements.index(symbol)
   result.extend(to_one_hot(symbol_idx, len(known_elements)))
   try:
      one_hot = [0.0 for _ in range(7)]
      hib = code[atom_object.GetHybridization().name]
      one_hot[hib+1]=1.0
      result+=one_hot
      #result.append(hib)
      result.extend(to_one_hot(atom_object.GetTotalValence(),8))
   except:
      print(Chem.MolToSmiles(mol, canonical=0),idx)
      raise
   result.extend(to_one_hot(len(atom_object.GetNeighbors()),7))
   result.extend(to_one_hot(max(atom_object.GetNumImplicitHs(), atom_object.GetNumExplicitHs()),5))
   result.append(p_table.GetNOuterElecs(symbol))
   result.append(electronegativity.get(symbol,0))
   result.append(float(atom_object.GetIsAromatic()))
   if use_formal_charge:
      result.append(atom_object.GetFormalCharge())
   if use_Gasteiger:
      q_in_neu = atom_object.GetDoubleProp('_GasteigerHCharge') + atom_object.GetDoubleProp('_GasteigerCharge')
      result.append(q_in_neu)
   result.append(contribs[idx+1])
   if use_mi:
      bond_mis = [get_bond_mi(bond) for bond in atom_object.GetBonds()]
      Nb=len(bond_mis)
      if bond_mis!=[]:
         mi_desc = [max(bond_mis), min(bond_mis), sum(bond_mis), sum(bond_mis)/Nb]
      else:
         mi_desc = [0, 0, 0, 0]
      result.extend(mi_desc)
   return result


def process_smiles(smiles, use_bond_orders=False, use_formal_charge=False, add_connections_to_aromatic_rings=False, use_Gasteiger=True, use_mi=True):
   if type(smiles).__name__=='str':
      mol = Chem.MolFromSmiles(smiles)
   elif type(smiles).__name__=='Mol':
      mol = smiles
   else:
      raise TypeError('Unknown type')
   A  = Chem.rdmolops.GetAdjacencyMatrix(mol).astype(float)
   if use_bond_orders:
      for bond in mol.GetBonds():
         order = bond.GetBondTypeAsDouble()
         if bond.GetIsAromatic():
            order=1.5
         idx_beg = bond.GetBeginAtomIdx()
         idx_end = bond.GetEndAtomIdx()
         A[idx_beg, idx_end]=order
         A[idx_end, idx_beg]=order
   if add_connections_to_aromatic_rings:
      rings = mol.GetRingInfo().AtomRings()
      for R in rings:
         if not all([mol.GetAtomWithIdx(xx).GetIsAromatic() for xx in R]):continue
         for xx, idx1 in enumerate(R):
            for idx2 in R[xx+1:]:
               order =0.5 if use_bond_orders else 1.0
               if A[idx1, idx2] == 0:
                  A[idx1, idx2] = order
                  A[idx2, idx1] = order
   if use_Gasteiger:
      try:
         Chem.rdPartialCharges.ComputeGasteigerCharges(mol, 200, True)
      except ValueError:
         for atom in mol.GetAtoms():
            atom.SetProp('_GasteigerCharge','0.0')
            atom.SetProp('_GasteigerHCharge','0.0')
      
   desc = [describe_atom(x, use_formal_charge=use_formal_charge, use_Gasteiger=use_Gasteiger, use_mi=use_mi) for x in mol.GetAtoms()]
   return array(desc), A


def process_smiles_compressed(smiles, use_bond_orders=False, use_formal_charge=False, add_connections_to_aromatic_rings=False, use_Gasteiger=True, use_mi=True):
   if type(smiles).__name__=='str':
      mol = Chem.MolFromSmiles(smiles)
   elif type(smiles).__name__=='Mol':
      mol = smiles
   else:
      raise TypeError('Unknown type')
   A  = array([[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in mol.GetBonds()])
   if use_Gasteiger:
      try:
         Chem.rdPartialCharges.ComputeGasteigerCharges(mol, 200, True)
      except ValueError:
         for atom in mol.GetAtoms():
            atom.SetProp('_GasteigerCharge','0.0')
            atom.SetProp('_GasteigerHCharge','0.0')
      
   desc = [describe_atom(x, use_formal_charge=use_formal_charge, use_Gasteiger=use_Gasteiger, use_mi=use_mi) for x in mol.GetAtoms()]
   return array(desc), A



def _get_max_size(matrices):
   shapes = array([x.shape for x in matrices])
   return tuple(shapes.max(axis=0))


def zero_pad(source_matrix, target_shape):
   source_shape = source_matrix.shape
   D = len(target_shape)

   assert D<=2 and D>0
   assert len(source_shape)==D
   assert all([source_shape[x]<=target_shape[x] for x in range(D)])

   if source_shape==target_shape:
      result = source_matrix
   else:
      result = zeros(target_shape)
      if D==2:
         N, M = source_shape
         result[:N,:M] = source_matrix[:N,:M]
      elif D==1:
         N, = source_shape 
         result[:N] = source_matrix[:N]

   return result


def process_smiles_set(smiles_set, smiles_config, threads=0):
   if threads==0:
      result = [process_smiles(x, **smiles_config) for x in smiles_set]
   else:
      p=Pool(threads)
      result = p.map( lambda x: process_smiles(x, **smiles_config), smiles_set)
   X, A =  list(zip(*result))
   L = array([x.shape[0] for x in A])
   X_target_shape = _get_max_size(X)
   A_target_shape = _get_max_size(A)
   X=[zero_pad(x, X_target_shape) for x in X]
   A=[zero_pad(x, A_target_shape) for x in A]
   return {'X':array(X), 'A':array(A), 'L':L}



def test():
   mol = 'c1ccccc1CC(=O)O'
   X, A = process_smiles(mol)
   D = adjacency_to_distance(A)
   for x in X:print( x)
   print( A)
   print('\nWith bond orders')
   X, A = process_smiles(mol, use_bond_orders=1)
   print( A)
   print('\nDistance Matrix')
   print( D)
   mol=Chem.MolFromSmiles(mol)
   bf, bad, b2a = get_bond_space_matrices(mol, 1, 1)
   print('\nBond features\n',bf)
   print('\nBond adjacency\n',bad)
   print('\nBond to atom transform\n',b2a)
   print('\nAtom adj reconstructed\n',b2a.dot(b2a.T))

if __name__=='__main__':
   test()
