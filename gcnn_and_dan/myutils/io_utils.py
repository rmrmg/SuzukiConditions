import numpy as np
import yaml
import gzip
import pickle 

def load_yaml(name):
   with open(name, 'r') as f:
      return yaml.load(f)


def gzload(name, f32=True):
   with gzip.open(name, 'rb') as f:
      data = np.load(f)
      if f32:
         data = np.nan_to_num(data.astype(np.float32))
      return data


def gzsave(name, arr):
   with gzip.open(name, 'wb') as f:
      np.save(f, arr)


def gz_pickle(name, obj):
   with gzip.open(name, 'wb') as f:
      pickle.dump(obj, f)


def gz_unpickle(name):
   with gzip.open(name, 'rb') as f:
      return pickle.load(f)


