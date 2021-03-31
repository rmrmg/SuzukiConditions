from keras import backend as K
import tensorflow as tf
import numpy as np

def _assign_indices_from_adj_to_bnd(adj2d, bnd, mol_idx):
   n=adj2d.shape[0]
   bond_idx=0
   for i in range(n):
      for j in range(i+1, n):
         if adj2d[i, j]==0:
               continue
         bnd[mol_idx, bond_idx, 0]= i
         bnd[mol_idx, bond_idx, 1]= j
         bond_idx+=1


def np_adj_to_bnd_3d(adjacency_matrix):
   '''assertion: zero-one 3d matrix'''
   D = adjacency_matrix.sum(axis=-1)
   D = D.sum(axis=-1)/2
   max_bnd = int(D.max())
   N = adjacency_matrix.shape[0]
   bond_indices = -np.ones((N, max_bnd, 2)).astype(int)
    
   for mol_idx in range(N):
      _assign_indices_from_adj_to_bnd(adjacency_matrix[mol_idx], bond_indices, mol_idx)
   
   return bond_indices


def tf_bnd_to_adj(bond_indices, max_atom):
   N=bond_indices.shape[0]
   prev = K.one_hot(bond_indices[:,:,0], max_atom)
   nxt = K.one_hot(bond_indices[:,:,1], max_atom)
   incidence = nxt-prev
      
   incidence_t = K.permute_dimensions(incidence, (0, 2, 1))
   lap = K.batch_dot(incidence_t, incidence)
   
   dg = K.ones((N,max_atom))
   mask = K.ones_like(lap) - tf.linalg.diag(dg)
   adj = -lap*mask
   
   return adj
   

def flatten_graph_data(features, filters, lens, identity, adjacency):
   '''Returns: concatenated data, Nfeatures, Natoms, Nbonds '''
   Nmol, Nat, Nfeat = features.shape
   #assuming first-irder unnormalized filters
    
   new_feat = features.reshape((Nmol, -1))
   new_lens = lens.reshape(-1,1)
   bonds = np_adj_to_bnd_3d(adjacency)
   max_bonds = bonds.shape[1]
   bonds = bonds.reshape(Nmol, -1)
    
   data = np.concatenate([new_feat, bonds, new_lens], axis=1)
    
   return data, Nfeat, Nat, max_bonds


def almost_equal(ar1, ar2, epsilon=1e-5):
   return (abs(ar1-ar2)<=epsilon).all()


def tf_reconstruct_graph_matrices(data, Nfeatures, Natoms, Nbonds):
   pos_x = Nfeatures*Natoms
   pos_b = Nbonds*2 + pos_x
   nmol = data.shape[0]

   x = K.reshape(data[:,:pos_x], (nmol, Natoms, Nfeatures))
   b = K.reshape(data[:,pos_x:pos_b], (nmol, Nbonds, 2))
   b = K.cast(b, 'int32')
   lens = K.reshape(data[:,pos_b:], (nmol,))
   
   adj = tf_bnd_to_adj(b, Natoms)
   identity = tf.linalg.diag(K.ones((nmol, Natoms)))
   filters = K.concatenate((identity, adj), axis=1)
   
   return x, filters, lens, identity, adj


if __name__=='__main__':
   from system import path
   path.append('../')
   from data_preprocess import gz_unpickle, gz_pickle, np
   from myutils.graph_utils import human_readable_size
   
   def get_size_str(array):
      S = array.size*array.itemsize
      return human_readable_size(S)
      
   from os.path import isfile
   chk_file = '../experiments/compression_tests.pkz'
   source = '../experiments/gcnn_testZ_repaired_data.pkz'
   if isfile(chk_file):
      test_features, test_filters, test_lens, test_I, test_A = gz_unpickle(chk_file)
   else:
      x,y,w = gz_unpickle(source)
      features, filters, lens, identity, adjacency = x
      lens = np.array(lens)
      short = np.random.choice(np.arange(lens.shape[0]), 10, replace=False)#np.where(lens==4)[0][:3]
      test_A = adjacency[short,:,:]
      test_I = identity[short,:,:]
      test_lens = lens[short]
      test_filters = filters[short,:,:]
      test_features = features[short,:,:]
      gz_pickle(chk_file, [test_features, test_filters, test_lens, test_I, test_A])
          
   bonds = np_adj_to_bnd_3d(test_A)
   max_atom = test_A.shape[-1]
      
   stuff = test_features, test_filters, test_lens, test_I, test_A
      
   #============================================== 
   print('Test adjacency reconstruction from bond data')
      
   new_A = tf_bnd_to_adj(bonds, max_atom)
   print(type(new_A))
   new_A = K.eval(new_A)
   print((new_A==test_A).all())
      
   #==============================
   print('Test compression')
   data, Nfeatures, Natoms, Nbonds = flatten_graph_data(*stuff)
   old_size = sum([x.size*x.itemsize for x in stuff])
   print('Original data: ', human_readable_size(old_size))
   print('Compressed: ', get_size_str(data))
      
   #===========================================
   print('Test reconstruction')
   tf_data = K.variable(data)
   new_stuff = list(tf_reconstruct_graph_matrices(tf_data, Nfeatures, Natoms, Nbonds))
   print([type(x) for x in new_stuff])
   names=['new_features', 'new_filters', 'new_lens', 'new_I', 'new_A']
      
   for i in range(len(stuff)):
      if type(new_stuff[i]).__name__=='Tensor':
         new_stuff[i] = K.eval(new_stuff[i])
      print('  %s'%names[i], almost_equal(stuff[i], new_stuff[i], 1e-5))


