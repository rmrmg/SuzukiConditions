import numpy as np
import gzip
import pickle as pic
import logging
import yaml
import sys
#sklearn
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
#keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Lambda, Concatenate, Multiply
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras import backend as K
from functools import partial

class LossPrinter(Callback):
     def on_train_begin(self, logs={}):
         self.losses = []
         self.count = 0
         return
              
     def on_train_end(self, logs={}):
         return
                           
     def on_epoch_begin(self, logs={}):
         self.count+=1
         return
                                        
     def on_epoch_end(self, epoch, logs={}):
        return
                                                     
     def on_batch_begin(self, batch, logs={}):
        return
                                                                  
     def on_batch_end(self, batch, logs={}):
        loss_ = logs.get('loss')
        acc_ = logs.get('acc')
        self.losses.append((loss_,acc_))
        logging.info('Epoch: %5i  loss:%8.6f   acc: %8.6f'%(self.count, loss_, acc_))
        return


def make_logger(logname=None, level=logging.INFO, logger = logging.getLogger()):
   formatter=logging.Formatter('%(asctime)s: %(levelname)s): %(message)s')

   handlers= [logging.StreamHandler(sys.stdout)]
   if logname!=None:
      handlers.append(logging.FileHandler(logname))
   for handler in handlers:
      handler.setFormatter(formatter)
      handler.setLevel(level)
      logger.addHandler(handler)
   logger.setLevel(level)
   return logger


def gz_pickle(name, stuff):
   with gzip.open(name, 'wb') as f:
      pic.dump(stuff, f)


def gz_unpickle(name):
   with gzip.open(name, 'rb') as f:
      return pic.load(f)


def load_config(name):
   if name==None:return {}
   with open(name, 'r') as f:
      raw = f.read()
   return yaml.load(raw)
