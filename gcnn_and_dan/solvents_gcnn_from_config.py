from os.path import isfile
from solvents_modules import *
import argparse
parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--log', type=str, default='gcnn.log')
parser.add_argument('--chk', type=str, default='gcnn_chk.pkz')
parser.add_argument('--one_fold', action='store_true')
parser.add_argument('--config', type=str, default='gcnn_config.yaml')
args = parser.parse_args()

def data_provider():
   name = args.chk
   if isfile(name):
      graphs, Y, splits = gz_unpickle(name)
   else:
      X, Y = prepare_data(desc=['graphs'])
      graphs = []
      for XX in X: #X_input, filters_input, nums_input, identity_input, adjacency_input 
         nodes = XX['X']
         nei = adjacencies_to_nei_list(XX['A'])
         graphs.extend([nodes,nei])
      folds = KFold(n_splits=5)
      splits = list(folds.split(Y))
      gz_pickle(name, (graphs, Y, splits))
   return graphs, Y, splits


def make_model_and_evaluate(Xtrain, Xtest, Ytrain, Ytest, model_config, train_config):
#   model_config = dict(\
#      hidden_units =  {{choice([10,20,64,100])}},
#      hidden_units_mlp = {{choice([50,100,200])}},
#      fgp_size = 50,
#      dropout =  {{uniform(0.1,0.7)}},
#      lr = {{choice([1e-3,1e-2,1e-4])}},
#      l2 = {{uniform(0,0.5)}},
#      num_layers = 3,
#      activation = 'relu',
#      dropout_flag = False,
#      concat = True,
#      fgp = False
#      )
#
#   train_config=dict(\
#      epochs = {{choice([50,100,150])}},
#      batch = {{choice([32,64,128])}}
#      )
         
   N=Ytrain.shape[1]
   Ytrain_ = [Ytrain[:,i] for i in range(N)]
   Ytest_ = [Ytest[:,i] for i in range(N)]
   baseline_ps = []
   models = []
   for i,t in enumerate(Ytrain_):
     logging.info('OUT %i'%i)
     p, model=baseline(model_config, Xtrain, t, Xtest, Ytest_[i], graph=True, train_config=train_config)
     baseline_ps.append(p)
     models.append(model)
   Yref = np.hstack([x.reshape(-1,1) for x in Ytest_])
   p = np.hstack([x.reshape(-1,1) for x in baseline_ps])
   ks = categorical_stats(Yref,p)

   #logging.info('Model parameters:\n'+yaml.dump(model_config)) 
   #logging.info('Training parameters:\n'+yaml.dump(train_config)) 
   logging.info('Top 1,2,3: ' + ' '.join(['%5.3f'%x for x in ks]))
    
   return ks


logger = make_logger(args.log)
logging.info('START')
logging.info('command line arguments:\n'+yaml.dump(args.__dict__))

config = load_config(args.config)
logging.info('Config:\n'+yaml.dump(config))

model_config = config['model_config']
train_config = config['train_config']

all_ks = []
X, Y, splits = data_provider() 
logging.info('data loaded')

Fid = 0
for train_idx, test_idx in splits:
   logging.info('Fold %i'%Fid)
   Fid+=1
   Xtrain = [XX[train_idx] for XX in X]
   Xtest = [XX[test_idx] for XX in X]
   Ytrain = Y[train_idx]
   Ytest = Y[test_idx]
   fold_ks = make_model_and_evaluate(Xtrain, Xtest, Ytrain, Ytest, model_config, train_config)
   all_ks.append(fold_ks)
   if args.one_fold: break

k_m = np.mean(all_ks, axis=0)
k_s = np.std(all_ks, axis=0)

logging.info('Top-1,2,3 mean: '+' '.join(['%.3f'%x for x in k_m]))
logging.info('Top-1,2,3 std : '+' '.join(['%.3f'%x for x in k_s]))
