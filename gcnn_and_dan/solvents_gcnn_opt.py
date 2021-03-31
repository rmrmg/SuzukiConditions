from os.path import isfile
from solvents_modules import *
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
import argparse
parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--log', type=str, default='nfp_opt.log')
parser.add_argument('--trials', type=str, default=None)
parser.add_argument('--n_trials', type=int, default=10)

args = parser.parse_args()

def data_provider():
   name = 'data_for_opt.pkz'
   if isfile(name):
      Xtrain, Xtest, Ytrain, Ytest = gz_unpickle(name)
   else:
      X, Y = prepare_data(desc=['graphs'])
      Y = Y[:,1:] #in the first class, there's to little positive examples
      graphs = []
      for XX in X: #X_input, filters_input, nums_input, identity_input, adjacency_input 
         nodes = XX['X']
         nei = adjacencies_to_nei_list(XX['A'])
         graphs.extend([nodes,nei])
      all_idx = np.arange(len(Y))
      reduction_factor = 5
      selection = np.random.choice(all_idx, int((all_idx[-1]+1)/reduction_factor), replace=False)
      train_idx, test_idx = train_test_split(selection, test_size=0.2)
      out_size = 1 if  len(Y.shape)==1 else Y.shape[1]
      
      Xtrain = [xx[train_idx] for xx in graphs]
      Xtest = [xx[test_idx] for xx in graphs]
      Ytrain, Ytest = Y[train_idx,:], Y[test_idx,:]
      gz_pickle(name, (Xtrain, Xtest, Ytrain, Ytest))
   return Xtrain, Xtest, Ytrain, Ytest


def make_model_and_evaluate(Xtrain, Xtest, Ytrain, Ytest):
   model_config = dict(\
      hidden_units =  {{choice([10,20,64,100])}},
      hidden_units_mlp = {{choice([50,100,200])}},
      fgp_size = 50,
      dropout =  {{uniform(0.1,0.7)}},
      lr = {{choice([1e-3,1e-2,1e-4])}},
      l2 = {{uniform(0,0.5)}},
      num_layers = 3,
      activation = 'relu',
      dropout_flag = False,
      concat = True,
      fgp = False
      )

   train_config=dict(\
      epochs = {{choice([50,100,150])}},
      batch = {{choice([32,64,128])}}
      )
         
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

   logging.info('Model parameters:\n'+yaml.dump(model_config)) 
   logging.info('Training parameters:\n'+yaml.dump(train_config)) 
   logging.info('Top 1,2,3: ' + ' '.join(['%5.3f'%x for x in ks]))
    
   return {'loss': -ks[0], 'status': STATUS_OK, 'model': models}



logger = make_logger(args.log)
logging.info('START')

if args.trials is None or not isfile(args.trials):
   trials = Trials()
else:
   logging.info('Loading trials from %s'%args.trials)
   trials = gz_unpickle(args.trials)

best_run, best_model, space = optim.minimize(model=make_model_and_evaluate,
                                       data=data_provider,
                                       algo=tpe.suggest,
                                       max_evals=args.n_trials,
                                       trials=trials,
                                       verbose=0,
                                       return_space=True,
                                       eval_space=True,
                                       keep_temp=True)

logger.info("Best performing model chosen hyper-parameters:")
logger.info(str(best_run))
if args.trials!=None:
   gz_pickle(args.trials, trials)
logger.info('Trials saved')
