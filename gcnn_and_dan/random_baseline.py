import numpy as np
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')
class_counts_train = np.array([ 871., 1631., 2751., 3642., 1333., 3928.])
probs = class_counts_train/(class_counts_train.sum())
class_counts_test = np.array([ 48.,  577.,  667.,  674.,  278., 1295.])

n_k = len(class_counts_train)

def calc_top_bayes(probs):
   K= len(probs)
   t1 = np.array([x for x in probs])
   t2 = np.array([np.sum([probs[i]*probs[k]/(1-probs[i]) for i in range(K) if i!=k]) for k in range(K)]) + t1
   t3 = np.array([np.sum([[probs[i]*probs[j]*probs[k]/((1-probs[i])*(1-probs[i]-probs[j])) for i in range(K) if i!=k and i!=j] for j in range(K) if j!=k]) for k in range(K)]) + t2
   return [np.dot(probs,x) for x in [t1,t2,t3]]


top_1_naive = 1./n_k
top_1_bayes = calc_top_bayes(probs)[0]

def counts_to_synthetic_data(counts):
   result = np.zeros((int(counts.sum()), len(counts)))
   prev=0
   for i,x in enumerate(counts):
      result[prev:int(x),i]=1
      prev+=int(x)
   np.random.shuffle(result)
   return result

data = counts_to_synthetic_data(class_counts_test)

def evaluate(data, probs=np.ones(n_k)/n_k):
   Ns, Nk = data.shape
   idx = np.arange(Nk)
   top=np.zeros(3)
   for x in data:
      true_idx = np.argmax(x)
      choice_1 = np.any(true_idx==np.random.choice(idx,1, replace=False, p=probs))
      choice_2 = np.any(true_idx==np.random.choice(idx,2, replace=False, p=probs))
      choice_3 = np.any(true_idx==np.random.choice(idx,3, replace=False, p=probs))
      if choice_1:top[0]+=1
      if choice_2:top[1]+=1
      if choice_3:top[2]+=1
   top/=Ns
   return top


def arr_to_stats_str(arr):
   mean = np.mean(arr, axis=0)
   std = np.std(arr, axis=0)/np.sqrt(len(arr)-1)
   str_ = '  mean: '+str(mean.round(4))
   str_ += '\n  std: '+str(std.round(4))
   return str_

tops_naive=[]
tops_bayes=[]

if __name__=='__main__':
   
   for i in range(100):
      tops_naive.append(evaluate(data))
      tops_bayes.append(evaluate(data, probs))
      logging.info('%i done'%i)
   
   print('Naive (prior only)')
   print(' Analytic top-1: %6.4f'%top_1_naive)
   print(' Simulation:')
   print(arr_to_stats_str(tops_naive)+'\n')
   print('Bayes:')
   print(' Analytic top-1: %6.4f'%top_1_bayes)
   print(' Simulation:')
   print(arr_to_stats_str(tops_bayes)+'\n')
   
