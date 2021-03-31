from os.path import isfile
from utils import *

def eliminate_constant_columns(X):
   std = X.std(axis=0)
   idx = np.where(std!=0)[0]
   return X[:,idx], idx


def data_provider(loader_config):
   name = loader_config.get('checkpoint','checkpoint.pkz')
   logging.info('using checkpoint: %s'%name)
   if isfile(name):
      X, Y, splits = gz_unpickle(name)
      logging.info('checkpoint loaded')
   else:
      subset = loader_config.get('subset',1.0)
      logging.info('using %.1f%% of data'%(100*subset))
      data_dict = gz_unpickle(loader_config['file'])
      Y = np.array(data_dict[loader_config['output']])
      N = len(Y)
      indices = np.arange(N)

      if subset<1.0:
         indices = np.random.choice(indices, int(N*subset), replace=False)
         Y=Y[indices]

      X = []
      for input_name in loader_config['inputs']: #X_input, filters_input, nums_input, identity_input, adjacency_input 
         for desc_name in loader_config['descriptors']:
            key='%s_%s'%(input_name,desc_name)
            X.append(np.array(data_dict[key])[indices])
      folds = KFold(n_splits=5)
      splits = list(folds.split(Y))
      gz_pickle(name, (X, Y, splits))
      logging.info('checkpoint saved')
   return X, Y, splits


def define_dense_block(input_layer, hidden, layers, activation, dropout=0.2, l2val=1e-3, batchnorm=False):
   hidden_type  = type(hidden).__name__
   assert hidden_type in ['int', 'list'], 'bad type for hidden: %s'%hidden_type
   assert type(layers).__name__=='int', 'bad type for layers: %s'%str(layers)
   #no block 
   if hidden==0 or layers==0: return input_layer
   
   if hidden_type=='int': 
      units=[hidden]*layers
   else: 
      units = hidden
      if len(hidden)!=layers:
         msg = ("Mismatch between hidden list (len %i) and n_layers (len %i),"
                " using hidden list")%(len(units), layers)
         logging.warning(msg)

   current = input_layer
   for N_H in units:
      current = Dense(N_H, activation=activation, kernel_regularizer=l2(l2val))(current)
      if dropout!=0:
         current = Dropout(dropout)(current)
      if batchnorm:
         current = BatchNormalization()(current)

   return current


def build_model(model_config, X_shapes, Y_shape):
   #common regularization
   regularization_keys = ['dropout', 'l2val', 'batchnorm']
   regularization_cfg = {x:model_config[x] for x in regularization_keys}
   #loss
   loss_f = model_config['loss']
   #pre-merge
   hidden1 = model_config['hidden1']
   n_layers1 = model_config['n_layers1']
   activation1 = model_config['act1']
   #intemediate_merge
   intermerge_idx = model_config['inter_idx'] #list of index lists
   intermerge_hidden = model_config['inter_hidden']
   intermerge_activation = model_config['inter_act']
   intermerge_layers = model_config['inter_layers']
   do_intermerge = (intermerge_idx not in [None, []])\
                   and (intermerge_hidden not in [0, None, []])\
                   and (intermerge_layers!=0)
   #merge
   merge = model_config['merge']
   assert merge in ['mult','add','concat'], 'Bad merge type:%s'%str(merge)
   #post-merge
   hidden2 = model_config['hidden2']
   n_layers2 = model_config['n_layers2']
   activation2 = model_config['act2']
   out_act = model_config['out_act']
   ######

   inputs = [Input(shape=x[1:]) for x in X_shapes]
   to_merge_layers = [define_dense_block(XX, hidden1, n_layers1, activation1,
                      **regularization_cfg) for XX in inputs]
   if do_intermerge:
      all_indices = set(range(len(to_merge_layers)))
      used_indices = set(sum(intermerge_idx, []))
      intermerged = [ Concatenate()([to_merge_layers[x] for x in y]) for y in intermerge_idx]
      intermerged = [define_dense_block(XX, intermerge_hidden, intermerge_layers, intermerge_activation,
                            **regularization_cfg) for XX in intermerged]
      not_used = all_indices-used_indices
      for i in not_used:
         intermerged.append(to_merge_layers[i])
      level_1 = intermerged
   else:
      level_1 = to_merge_layers
   if merge=='mult':
      merged = Multiply()(level_1)
   elif merge=='concat':
      merged = Concatenate()(level_1)
   elif merge=='add':
      merged = Add()(level_1)
    
   output = define_dense_block(merged, hidden2, n_layers2, activation2, **regularization_cfg)
   out_shape = 1 if len(Y_shape)==1 else Y_shape[1]
   last_layer = Dense(out_shape, activation=out_act, kernel_regularizer=l2(model_config['l2val']))(output) 
   model = Model(inputs=inputs, output=last_layer)
   model.compile(optimizer='adam', loss=loss_f, metrics=['mae'])
   str_list = ['Model info:']
   model.summary(print_fn = lambda x: str_list.append(x))
   logging.info('\n'.join(str_list))
   return model

#update config
config={
      'model_config':{\
         'inter_idx': None,
         'inter_hidden': 0,
         'inter_act': 'relu',
         'inter_layers':0,
         'dropout'  :0.2,
         'l2val'    :1e-3,
         'batchnorm':True,
         'loss'     :'mse',
         'hidden1'  :0,
         'n_layers1':0,
         'act1'     :'relu',
         'merge'    :'concat',
         'hidden2'  :128,
         'n_layers2':1,
         'act2'     :'relu',
         'out_act'  :'linear',
         'standarization': 'normal'},
      'loader_config':{\
         'checkpoint':'checkpoint.pkz',
         'file': 'preprocessed_data_sml_all_morgan_rdkit.pkz',
         'inputs': ['halogen', 'boronic', 'solvent', 'base', 'ligand'],
         'descriptors':['ecfp6'],
         'output':'yield'},
      'train_config':{\
         'epochs':30,
         'verbose':True,
         'batch_size':100}
      }


def standardize(arr, mode='normal', axis=0):
   assert mode in ['normal','min-max'], 'Unknown mode: %s'%str(mode)
   if mode=='normal':
      u, s = arr.mean(axis=axis), arr.std(axis=axis)
   elif mode=='min-max':
      mn, mx = arr.min(axis=axis), arr.max(axis=axis)
      u=mn
      s=mx-mn
   if isinstance(s, float):
      s=1 if s==0 else s
   else:
      assert isinstance(s, np.array)
      s=np.where(s!=0, s,1)
   return (arr-u)/s, u, s


def make_permutation_test(model, Xtest, Ytest, Ys, permutation_test):
   result = [0,0,0]
   if permutation_test==None:
      return result
   try:
      assert isinstance(permutation_test, int)
      assert permutation_test<len(Xtest)
   except AssertionError as e:
         logging.info(str(e))
         logging.info('Permutation test ommitted')
         return result

   normal_value = model.evaluate(Xtest,Ytest)[-1]*Ys
   np.random.shuffle(Xtest[permutation_test])
   perturbed_value=model.evaluate(Xtest,Ytest)[-1]*Ys
   logging.info("'RR's permutation test (input %i): Before: %8.3f  After:%8.3f"%(permutation_test, normal_value, perturbed_value))
   return [normal_value, perturbed_value, perturbed_value-normal_value]


def evaluate_regressor(model_config, X, Y, splits, one_fold=False, 
         train_config={'epochs':100, 'batch_size':100}, permutation_test=None):
         
   X = [eliminate_constant_columns(XX)[0] for XX in X]
   logging.info('Constant cols eliminated')
   Xshapes = [XX.shape for XX in X]
   Yshape = Y.shape
   
   logging.info('data shapes: %s'%str(Xshapes))
   
   standarization_mode = model_config.get('standarization','normal')
   logging.info('%s standarization of Y'%(standarization_mode.capitalize()))
   Y, Yu, Ys = standardize(Y, mode=standarization_mode)
   
   all_folds = []
   all_folds_mae=[]
   permutation_tests=[]
   Fid = 0
   for train_idx, test_idx in splits:
      logging.info('Fold %i'%Fid)
      Fid+=1
      Xtrain = [XX[train_idx] for XX in X]
      Xtest = [XX[test_idx] for XX in X]
      Ytrain = Y[train_idx]
      Ytest = Y[test_idx]
      model = build_model(model_config, Xshapes, Yshape)
      result = model.fit(Xtrain, Ytrain, validation_data=(Xtest, Ytest),
                         **train_config)
      all_folds.append(result.history['val_loss'])
      all_folds_mae.append(result.history['val_mean_absolute_error'])
      permut_result = make_permutation_test(model, Xtest, Ytest, Ys, permutation_test)
      permutation_tests.append(permut_result)
      if one_fold: break
   

   logging.info('Transfroming mse to rmse, back transformation to Y space')
   all_folds = np.sqrt(all_folds)*Ys
   all_folds_mae = np.array(all_folds_mae)*Ys
   cv_mean = np.mean(all_folds, axis=0)
   cv_std = np.std(all_folds, axis=0)
   cv_mean_mae = np.mean(all_folds_mae, axis=0)
   cv_std_mae = np.std(all_folds_mae, axis=0)
   
   permut_m = np.mean(permutation_tests, axis=0)
   permut_s = np.std(permutation_tests, axis=0)
   print_permut = any(permut_m!=0)
   if print_permut:
      logging.info('Permutation test average:')
      logging.info('  Before: %8.3f (%.3f)  After %8.3f (%.3f)'%(\
         permut_m[0], permut_s[0], permut_m[1], permut_s[1]))
      logging.info('  Diff: %8.3f (%.3f)'%(permut_m[2], permut_s[2]))
   idx = np.argmin(cv_mean)
   logging.info('MAE: %8.3f (%8.3f)'%(cv_mean_mae[idx],cv_std_mae[idx]))
   return idx, cv_mean[idx], cv_std[idx]


if __name__=='__main__':
   import argparse
   parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
   parser.add_argument('--log', type=str, default='regression.log')
   parser.add_argument('--chk', type=str, default=None)
   parser.add_argument('--permute', type=int, default=None)
   parser.add_argument('--epochs', type=int, default=None)
   parser.add_argument('--one_fold', action='store_true')
   parser.add_argument('--config', type=str, default=None)
   args = parser.parse_args()
   
   logger = make_logger(args.log)
   logging.info('START')
   logging.info('command line arguments:\n'+yaml.dump(args.__dict__))
   
   #['yield', 'solvent_enc', 'base_enc', 'boronic_ecfp6', 'boronic_rdkit', 'halogen_ecfp6', 'halogen_rdkit', 'solvent_ecfp6', 'solvent_rdkit', 'base_ecfp6', 'base_rdkit', 'ligand_ecfp6', 'ligand_rdkit']
   
   #default config
   
   config.update(load_config(args.config))
   if args.chk!=None:
      config['loader_config']['checkpoint'] = args.chk
   if args.epochs!=None:
      config['train_config']['epochs'] = args.epochs
   
   logging.info('Config:\n'+yaml.dump(config))
   
   model_config = config['model_config']
   loader_config = config['loader_config']
   train_config = config['train_config']
   
   all_ks = []
   X, Y, splits = data_provider(loader_config) 
   
   idx, best_cv_mean, best_cv_std = evaluate_regressor(model_config, X, Y, splits, args.one_fold, train_config=train_config, permutation_test=args.permute)

   logging.info('Best epoch: %i'%idx)
   logging.info('CV AV: %8.3f'%best_cv_mean)
   logging.info('CV std %8.3f'%best_cv_std)
