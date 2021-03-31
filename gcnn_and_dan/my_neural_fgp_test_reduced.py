from keras.models import Model
from keras.layers import Input, Dense, Dropout, Lambda, BatchNormalization, Concatenate, TimeDistributed, Add, Activation
from keras.regularizers import l2
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import Callback
from os.path import isfile 
import logging
from rdkit import Chem
from rdkit.Chem import AllChem
import gzip
import pickle
import numpy as np
import yaml

def gz_unpickle(name):
   with gzip.open(name, 'rb') as f:
      return pickle.load(f)

def temporal_zero_padding(x):
   input_shape=x.shape
   pad_shape = (input_shape[0],
                1,
                input_shape[2])
   output = K.zeros_like(x)
   output = K.sum(output, axis=1, keepdims=True)
   output = K.concatenate([output, x], axis=1)
   return output


def edges_to_degree(edges, reverse=False):
   atom_degrees = K.sum(K.cast(K.not_equal(edges, -1), 'float32'), axis=-1, keepdims=True)
   if reverse:
      atom_degrees=1/K.maximum(atom_degrees,1)
   return atom_degrees


def lookup(atoms, neighbour_list):
   nei_in_lookup = K.cast(neighbour_list+1, dtype='int32')
   atoms_to_lookup = temporal_zero_padding(atoms)
   #
   atoms_shape = K.shape(atoms_to_lookup)
   batch_n = atoms_shape[0]
   lookup_size = atoms_shape[1]
   num_atom_features = atoms_shape[2]
   #
   nei_shape = K.shape(nei_in_lookup)
   max_atoms = nei_shape[1]
   max_degree = nei_shape[2]
   #
   offset_shape = (batch_n, 1, 1)
   offset = K.reshape(K.arange(batch_n, dtype=K.dtype(nei_in_lookup)), offset_shape)
   offset *= lookup_size
   #
   flattened_atoms = K.reshape(atoms_to_lookup, (-1, num_atom_features))
   flattened_edges = K.reshape(nei_in_lookup + offset, (batch_n, -1))
   flattened_result = K.gather(flattened_atoms, flattened_edges)
   #
   output_shape = (batch_n, max_atoms, max_degree, num_atom_features)
   output = K.reshape(flattened_result, output_shape) 
   #print('LOOKUP Target SHAPE: ',output_shape)
   #print('LOOKUP SHAPE: ',output.shape)
   #
   return output


def adjacencies_to_nei_list(adjs):
   Dmax = int(adjs.sum(axis=1).max())
   N,M = adjs.shape[:2]
   result = -np.ones((N,M,Dmax)).astype(int)
   for i,A in enumerate(adjs):
      for j, v in enumerate(A):
         idx = np.where(v>0)[0]
         result[i,j,:len(idx)] = idx
   return result


default_smiles_config = dict(add_connections_to_aromatic_rings=False,
                             use_Gasteiger=False,
                             use_bond_orders=False,
                             use_formal_charge=False,
                             use_mi=False)
#
#def smiles_data_processor(datalines, process_smiles_config=default_smiles_config, use_semi_colon=False):
#   '''makes X, A and L arrays'''
#   total_X, total_A, total_lens = [], [] , []
#   for line in datalines:
#         if use_semi_colon:
#            line = line.split(';')[0]
#         try:
#            X, A = process_smiles(line, **process_smiles_config)
#         except:
#            logging.info('Error with %s'%str(line))
#            raise
#         n,f = X.shape
#         total_lens.append(n)
#         total_X.append(X)
#         total_A.append(A)
#   return {'X':total_X, 'A':total_A, 'L':total_lens, 'F':f}
#
#
#def align_and_make_filters(data_dict, filter_name='first_order'):
#   '''makes processed data_dict and input_shapes '''
#   max_size = max(data_dict['L'])
#   f = data_dict['F']
#   print('MAX SIZE:',max_size)
#   data_dict['X'] = np.array([zero_pad(x, (max_size, f)) for x in data_dict['X']])
#   data_dict['A'] = np.array([zero_pad(x, (max_size, max_size)) for x in data_dict['A']])
#   data_dict['I'] = np.array([np.identity(max_size) for _ in data_dict['A']])
#   data_dict['L'] = np.array(data_dict['L'])
#   data_dict['filters'] = FILTERS[filter_name](data_dict['A'])
#   input_shapes = [data_dict[x].shape for x in ['X', 'filters', 'L', 'I', 'A']] 
#   return data_dict, input_shapes
#
#
def make_morgan(smiles_list):
   result = []
   for sml in smiles_list:
      mol=Chem.MolFromSmiles(sml)
      fgp = AllChem.GetMorganFingerprintAsBitVect(mol, 4, nBits=4096).ToBitString()
      result.append(list(fgp))
   return np.array(result).astype(float)


def make_mlp_regressor(input_shape, model_config):
   I = Input(shape=(input_shape[1],))
   L=model_config.get('n_layers', 1)
   H=model_config.get('n_hidden',100)
   l2_val = model_config.get('l2', 0.13)
   H=Dense(H, kernel_regularizer=l2(l2_val), activation='relu')(I)
   for _ in range(L-1):
      H=Dense(H, kernel_regularizer=l2(l2_val))(I)
   output = Dense(1, kernel_regularizer=l2(l2_val))(I)
   model = Model(inputs=[I], outputs=output)
   model.compile(loss='mse', optimizer=Adam(lr=1e-3), metrics=['mse'])
   return model, 'mse'


def make_reduced_gcnn_module(prev_layer, reduced_graph_inputs, gcnn_config):
   #X_input, filters_input, nums_input, identity_input, adjacency_input 
   features, nei = reduced_graph_inputs
   max_atoms = gcnn_config['shapes']['max_atoms']
   
   features_shape = gcnn_config['shapes']['features']
   n_features = features_shape[-1]

   #control parameters
   N_H = gcnn_config.get('hidden_units', 128)
   dropout_prob = gcnn_config.get('dropout', 0.031849402173891934)
   lr = gcnn_config.get('lr', 1e-3)
   l2_val = gcnn_config.get('l2', 1e-3)
   N_it = gcnn_config.get('num_layers', 8)
   activation = gcnn_config.get('activation', 'relu')
   drp_flag = gcnn_config.get('dropout_flag', False)

   concat = gcnn_config.get('concat', False)
   make_fgp = gcnn_config.get('fgp',False)
   fgp_size = gcnn_config.get('fgp_size',50)
  
   #control
   logging.info('Echo of GCNN module config:')
   #logging.info('num_filters : %i'%num_filters)
   logging.info('N_H         : %i'%N_H)
   logging.info('max_atoms   : %i'%max_atoms)
   logging.info('n_features  : %i'%n_features)
   logging.info('dropout_prob: %5.2f'%dropout_prob)
   logging.info('lr          : %f'%lr)
   logging.info('l2_val      : %f'%l2_val)
   logging.info('N_it        : %i'%N_it)
   logging.info('activation  : %s'%activation)
   logging.info('drp_flag    : %s'%str(drp_flag))
   logging.info('concat      : %s'%str(concat))
   logging.info('make_fgp    : %s'%str(make_fgp))
   logging.info('fgp_size    : %i'%fgp_size)
   #=============================

   mask = Lambda(lambda x: K.cast(K.any(K.not_equal(x,-1), axis=-1,keepdims=True), dtype='float32'))(nei)
   lens = Lambda(lambda x: K.sum(K.reshape(x, K.shape(x)[:-1]), axis=-1, keepdims=True))(mask)

   if make_fgp:
      Z = K.zeros((features_shape[-1],fgp_size))
      fgp = Lambda(lambda x: K.sum(K.dot(x,Z), axis=1))(prev_layer)
   normalization_factor = Lambda(lambda x: edges_to_degree(x, reverse=True))(nei)
   #initial convolution
   H = TimeDistributed(Dense(N_H,  activation=activation, kernel_regularizer=l2(l2_val)))(prev_layer)
   H = Concatenate()([H, features])
   H = BatchNormalization()(H)
   H=Dropout(dropout_prob)(H, training=drp_flag)
   for it in range(N_it):
      #print('WTF H-shape: ',H.shape)
      curr_nei = Lambda(lambda x: K.sum(lookup(x[0], x[1]), axis=2)*x[2])([H, nei, normalization_factor])
      curr_nei = Lambda(lambda x: K.reshape(x, (-1,max_atoms,N_H+n_features)))(curr_nei)
      #print('WTF shape: ',curr_nei.shape)
      self_ = TimeDistributed(Dense(N_H,  activation=activation, kernel_regularizer=l2(l2_val)))(H)
      curr_nei = TimeDistributed(Dense(N_H, activation=activation, kernel_regularizer=l2(l2_val)), input_shape=(None,max_atoms,N_H+n_features))(curr_nei)
      H = Add()([self_, curr_nei])
      if make_fgp:
         sparse = TimeDistributed(Dense(fgp_size,  activation='linear', kernel_regularizer=l2(l2_val)))(H)
         sparse = Lambda(lambda x: K.softmax(x, axis=-1))(sparse)
         sparse = Lambda(lambda x: x[0]*x[1])([mask, sparse])
         fgp = Lambda(lambda x: K.sum(x[0], axis=1)+x[1])([sparse, fgp])
      if concat and it<(N_it-1):
         H = Concatenate()([H, features])
      H = BatchNormalization()(H)
      H = Dropout(dropout_prob)(H, training=drp_flag)

   #Pooling
   if make_fgp:
      output = fgp
   else:
      output = Lambda(lambda X: K.sum(X[0], axis=1)/X[1])([H, lens])  # adding a node invariant layer to make sure output does not depends upon the node order in a graph.
      output = Dropout(dropout_prob)(output, training=drp_flag)
   return output


def make_reduced_gcnn_regressor(input_shapes, output_shape, model_config):
   ''' order: X_input, filters_input, nums_input, identity_input, adjacency_input 
   '''
   #training_data = ([X, graph_conv_filters, lens], Y)
   #X_input, filters_input, nums_input, identity_input, adjacency_input 
   features_shape, nei_shape = input_shapes
   
   X_input = Input(shape=features_shape[1:])
   nei_input = Input(shape=nei_shape[1:])

   #control parameters
   N_H = model_config.get('hidden_units', 128)
   N_H_mlp = model_config.get('hidden_units_mlp', 100)
   fgp_size = model_config.get('fgp_size', 50)
   dropout_prob = model_config.get('dropout', 0.031849402173891934)
   lr = model_config.get('lr', 1e-3)
   l2_val = model_config.get('l2', 1e-3)
   N_it = model_config.get('num_layers', 4)
   activation = model_config.get('activation', 'relu')
   drp_flag = model_config.get('dropout_flag', False)
 

   logging.info('Internal check')
   logging.info('   N_H:          %i'%N_H)
   logging.info('   N_H_mlp:      %i'%N_H_mlp)
   logging.info('   fgp_size:     %i'%fgp_size)
   logging.info('   dropout_prob: %f'%dropout_prob)
   logging.info('   lr:           %f'%lr)
   logging.info('   l2:           %f'%l2_val)
   logging.info('   N_it:         %i'%N_it)
   logging.info('   activation:   %s'%activation)
   logging.info('   drp_flag:     %s'%drp_flag)

   graph = [X_input, nei_input] 
   gcnn_config = {'hidden_units':N_H, 'dropout':dropout_prob, 'lr':lr, 'l2':l2_val, 'N_it':'num_layers', 'activation':activation, 'dropout_flag':drp_flag}
   gcnn_config['shapes']={'filters':input_shapes[1], 'max_atoms':nei_shape[1], 'features':features_shape}
   gcnn_config['concat'] = True
   gcnn_config['fgp'] = True
   gcnn_config['fgp_size'] = fgp_size

   output = make_reduced_gcnn_module(X_input, graph, gcnn_config)

   mlp_hidden = Dense(N_H_mlp, activation='relu', kernel_regularizer=l2(l2_val))(output)
   if len(output_shape)==2:
      N_output=output_shape[1]
   else:
      N_output=1
   output_activation='linear'
   metric='mse'
   loss_f='mse'
   output = Dropout(dropout_prob)(mlp_hidden, training=drp_flag)
   output = Dense(N_output, activation=output_activation)(mlp_hidden)#output)
   
   model = Model(inputs=graph, outputs=output)
   model.compile(loss=loss_f, optimizer=Adam(lr=lr), metrics=[metric])

   return model, metric




if __name__=='__main__':
   import pandas as pd
   import argparse
   parser=argparse.ArgumentParser()
   parser.add_argument('--version',type=int, choices=[0,1,2,3], default=0)
   args=parser.parse_args()

   logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')

   task_params = {'target_name' : 'measured log solubility in mols per litre',
                  'data_file'   : 'neural-fingerprint/examples/delaney.csv'}
   checkpoint_name = 'first_order_test.pkz' #'cheb_test.pkz'#'first_order_test.pkz'
   filter_name = 'first_order'##'cheb'#'first_order'

   logging.info('START')

   graphs, input_shapes, Y, morgan = gz_unpickle(checkpoint_name)
   logging.info('Read from pickle')

   #from Douvenaud:
   model_params = dict(fp_length=50,    # Usually neural fps need far fewer dimensions than morgan.
                    fp_depth=4,      # The depth of the network equals the fingerprint radius.
                    conv_width=20,   # Only the neural fps need this parameter.
                    h1_size=100,     # Size of hidden layer of network on top of fps.
                    L2_reg=np.exp(-2))
   train_params = dict(num_iters=100,
                    batch_size=100,
                    init_scale=np.exp(-4),
                    step_size=np.exp(-6))
   N_train = 800
   N_val   = 20
   N_test  = 20
   
   seps = np.cumsum([0, N_train,N_val,N_test])

   #my turn

   model_config = dict(max_atoms = int(max(graphs['L'])),
                       name='gcnn',
                       num_filters = 2 if filter_name=='first_order' else 3,
                       hidden_units = model_params['conv_width'],
                       dropout = 0.2,
                       lr = 1e-4,
                       l2 = 1e-3,
                       num_layers = model_params['fp_depth'],
                       hidden_units_mlp = model_params['h1_size'],
                       fgp_size = model_params['fp_length'],
                       activation = 'relu',
                       drp_flag = False)
   order_reduced = ['X', 'nei']
   graphs['nei'] = adjacencies_to_nei_list(graphs['A'])
   input_shapes_reduced = [graphs[x].shape for x in order_reduced]
   uY, sY = np.mean(Y), np.std(Y)
   Y = (Y-uY)/sY
   logging.info('Y_std : %8.3f'%sY)
   d = [], []
   sets_reduced=[]
   for i in range(3):
      j,k = seps[i], seps[i+1]
      sets_reduced.append(([graphs[x][j:k] for x in order_reduced], Y[j:k]))

   train_r, val_r, test_r = sets_reduced
   logging.info('Data divided')
   logging.info('Model parameters:\n\n'+yaml.dump(model_config))
   logging.info('smiles parameters:\n\n'+yaml.dump(default_smiles_config))

   def test_provider(gcnn_provider, input_shapes=input_shapes_reduced, train=train_r, test=test_r, val=val_r):
      model, _ = gcnn_provider(input_shapes, Y.shape, model_config)
      model.fit(train[0], train[1], validation_data=val, 
                epochs=train_params['num_iters'], 
                batch_size=train_params['batch_size'], 
                verbose=True)
      evaluation = model.evaluate(test[0], test[1])
      evaluation[1] = sY*(evaluation[1]**0.5)
      logging.info('model evaluation: '+str(evaluation))

   logging.info('Test reduced')
   test_provider(make_reduced_gcnn_regressor, input_shapes=input_shapes_reduced, train=train_r, val=val_r, test=test_r)

   logging.info('Test morgan')
   if True:#test_morgan:
      logging.info('Morgan_baseline')
      train_x, val_x, test_x = [morgan[seps[i]:seps[i+1]] for i in range(3)]
      morgan_model, _ = make_mlp_regressor(train_x.shape,{})
      morgan_model.fit(train_x, train_r[1], validation_data=(val_x, val[1]),
                epochs=train_params['num_iters'], 
                batch_size=train_params['batch_size'], 
                verbose=True)
      evaluation_morgan = morgan_model.evaluate(test_x, test[1])
      evaluation_morgan[1] = sY*(evaluation_morgan[1]**0.5)
      logging.info('Morgan: '+str(evaluation_morgan))
   
