from regressor_from_config import *
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
import argparse
parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--log', type=str, default='regression_opt.log')
parser.add_argument('--trials', type=str, default=None)
parser.add_argument('--n_trials', type=int, default=10)

args = parser.parse_args()

def data_provider():
   name = 'reg_data_for_opt.pkz'
   if isfile(name):
      X, Y, splits = gz_unpickle(name)
   else:
      loader_cfg = config['loader_config']
      loader_cfg['checkpoint'] = 'regression_opt_chk.pkz'
      loader_cfg['subset'] = 0.5
      X, Y, splits = data_provider(loader_cfg)
      gz_pickle(name, (X, Y, splits))
   return X,Y,splits


def make_model_and_evaluate(X, Y, splits):
   parameters   =  dict(\
         dropout   = {{uniform(0.1,0.7)}},
         l2val     = {{uniform(0,1)}},
         batchnorm = {{choice([True,False])}},
         loss      = 'mse',
         hidden1   = {{choice([0, 50, 100])}},
         n_layers1 = {{choice([0, 1, 2, 3, 4])}},
         act1      = {{choice(['relu', 'elu', 'tanh'])}},
         merge     = 'concat',
         hidden2   = {{choice([50, 100, 150])}},
         n_layers2 = {{choice([1, 2, 3, 4])}},
         act2      = {{choice(['relu', 'elu', 'tanh'])}},
         out_act   = 'sigmoid')
   train_config=dict(\
      epochs = {{choice([50,100,150])}},
      batch_size = {{choice([32,64,128])}}
      )
   model_config = config['model_config']
   model_config.update(parameters)
         
   epoch, mean, std = evaluate_regressor(model_config, X, Y, splits, one_fold=True, train_config=train_config)
   logging.info('Stats: Epoch: %i, MSE: %8.3f'%(epoch, mean))
   logging.info('Adjustable parameters:\n'+yaml.dump(parameters)) 
   logging.info('Training parameters:\n'+yaml.dump(train_config)) 

   return {'loss': mean, 'status': STATUS_OK, 'model': None}



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
