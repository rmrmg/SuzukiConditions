import pickle
import numpy as np
import pandas as pd
import tqdm

def sigmoid(arr):
   return 1/(1+np.exp(-np.clip(arr,-50,50)))

def get_cond_to_class_map(dictionary='suzuki_dict.csv', source='rr_reactions_from_onerx_corrected.csv'):
   map_ = pd.read_csv(dictionary)
   source= pd.read_csv(source, sep=';')
   class_ = []

   for _, row in map_.iterrows():
      cat = row['category']
      if cat not in ['B', 'S']:
         class_.append('')
         continue
      name_field, class_field = ('solvent_names', 'solvent_class') if cat=='S' else ('base_names', 'base_class')
      class_name = source[source[name_field]==row['name']][class_field].values[0]
      class_.append(class_name)

   return class_

options = {
   'suzuki_rr': dict(\
      class_dict = {'M': 37, 'L': 13, 'B': 9, 'S': 39},
      class_num = 103,
      labels = ['M', 'L', 'B', 'S']),
    'suzuki_rr_ours': dict(\
         class_dict = {'M': 37, 'L': 14, 'B': 7, 'S': 5},
         class_num = 68,
         labels = ['M', 'L', 'B', 'S']),
   'suzuki_rr_classes': dict(\
         class_dict = {'M': 37, 'L': 13, 'B':3, 'S':5},
         class_num = 63,
      labels = ['M', 'L', 'B', 'S']),
   'suzuki_rr_classes_only': dict(\
         class_dict = {'M': 37, 'L': 13, 'B':3, 'S':5},
         class_num = 63,
      labels = ['M', 'L', 'B', 'S']),
    'suzuki_rr_ours_uspto': dict(\
         class_dict = {'ML': 10, 'B': 7, 'S': 6},
         class_num = 27,
         labels = ['ML', 'B', 'S']),
    'suzuki_rr_cond': dict(\
         class_dict = {'M': 37, 'L': 13, 'C':16},
         class_num = 70,
         labels = [ 'M', 'L', 'C']),
    'suzuki_rr_cond_only': dict(\
         class_dict = {'C':16},
         class_num = 18,
         labels = [ 'C'])
}

def prepare_class_indices(class_dict, key_order,pad=1):
   result = {}
   cumsum = np.cumsum([0] + [class_dict[k] for k in key_order])
   for i, key in enumerate(key_order):
      idx = list(range(cumsum[i], cumsum[i+1]))
      idx.append(pad+cumsum[-1]+i)
      result[key] = idx
   return result

def extract_blocks(array, idx_class):
   result = {}
   for label in idx_class:
      result[label] = array[:,idx_class[label]]
   return result


def get_top_k_vector(pred, reference, k=3):
   assert len(pred.shape)<=2
   if len(pred.shape)!=1:
      assert pred.shape[0]==reference.shape[0]
   if len(pred.shape)==1:
      query = pred.argsort()[-k:]
   else:
      query = pred.argsort(axis=1)[:,-k:]
   return (query==reference.argmax(axis=1).reshape(-1,1)).any(axis=1)

def combine_B_S_blocks(block_dict, max_only=False, f=10):
   B = block_dict['B']
   S = block_dict['S']
   if max_only:
      result = B.argmax(axis=-1)*f + S.argmax(axis=-1)
   else:
      Nx, Nb = B.shape
      Nx2, Ns = S.shape
      assert Nx2==Nx
      result = np.zeros((Nx,Nb*Ns))
      pivot = 0
      for i in range(Nb):
         result[:,i*Ns:(i+1)*Ns] = S * B[:,i].reshape(-1,1)
   return result

def get_top_k_vector_classes(pred, reference, k=3, class_vector=[]):
   assert len(pred.shape)<=2
   if len(pred.shape)!=1:
      assert pred.shape[0]==reference.shape[0]
   assert class_vector!=[]
   if len(pred.shape)==1:
      query = pred.argsort()[-k:]
   else:
      query = pred.argsort(axis=1)[:,-k:]
   
   result = np.zeros(reference.shape[0])
   Nclasses = len(class_vector)
   for i in range(reference.shape[0]):
      qvec = query if len(pred.shape)==1 else query[i]
      qclasses = [class_vector[x] if x<Nclasses else 'other' for x in qvec]
      argmax = reference[i].argmax()
      rclass = class_vector[argmax] if argmax<Nclasses else 'other'
      if rclass in qclasses:
         result[i] = 1

   return result.astype(bool)


def evaluate_cv(results_folder, option='suzuki_rr', use_classes=False, output='cv.csv'):
   class_dict = options[option]['class_dict']
   labels = options[option]['labels']
   class_indices = prepare_class_indices(class_dict, labels)

   result = {}
   for k in labels + ['C']:
      for t in [1,2,3]:
         result[f'{k}_top_{t}_mean'] = []
         result[f'{k}_top_{t}_std'] = []

   for e in tqdm.trange(50):
      epoch_results = {t:{k:[] for k in labels + ['C']} for t in [1,2,3]}
      
      for fold_id in range(5):
         with open(f'{results_folder}_f{fold_id}/pred_e{e+1}.pkl', 'rb') as f:
            pred = pickle.load(f)
   
         with open(f'{results_folder}_f{fold_id}/gt.pkl', 'rb') as f:
            gt = pickle.load(f)
            gt = gt.reshape((gt.shape[0], gt.shape[-1]))
   
         pred = extract_blocks(pred, class_indices)
         gt = extract_blocks(gt, class_indices)
   
         if use_classes:
            all_class_vector = get_cond_to_class_map()
      
         for k in class_indices:
            for topk in [1, 2, 3]:
               if use_classes:
                  class_vector = [all_class_vector[x] for x in class_indices[k][:-1]]
                  res = get_top_k_vector_classes(pred[k], gt[k], topk, class_vector)
               else:
                  res = get_top_k_vector(pred[k], gt[k], topk)
               res = res.mean()
               epoch_results[topk][k].append(res)

         if 'cond' not in option:
            line = f'{"B&S comb":10s}   '
            B = sigmoid(pred['B'])
            S = sigmoid(pred['S'])
            C = combine_B_S_blocks({'B':B,'S':S})
            Cgt = combine_B_S_blocks(gt)
            popularity = Cgt.mean(axis=0)
            for topk in [1,2,3]:
               res = get_top_k_vector(C, Cgt, topk)
               res = res.mean()
               epoch_results[topk]['C'].append(res)
      
      for topk in epoch_results:
         for k in epoch_results[topk]:
            m = np.mean(epoch_results[topk][k])
            s = np.std(epoch_results[topk][k])
            result[f'{k}_top_{topk}_mean'].append(m)
            result[f'{k}_top_{topk}_std'].append(s)
   results = pd.DataFrame(result)
   results.to_csv(output, sep=';', index=False)
   if 'C_top_1_mean' in results.columns:
       best = results['C_top_1_mean'].argmax()
       print('best epoch:', best+1)
       print(results.loc[best, [f'C_top_{k}_mean' for k in '123']])
       print(results.loc[best, [f'C_top_{k}_std' for k in '123']])

#refactor??
def evaluate_channels(results_folder, option='suzuki_rr', use_classes=False, conditions_baseline=''):
   with open(f'{results_folder}/pred.pkl', 'rb') as f:
      pred = pickle.load(f)
      print('pred', pred.shape)

   with open(f'{results_folder}/gt.pkl', 'rb') as f:
      gt = pickle.load(f)
      print('gt', gt.shape)
      gt = gt.reshape((gt.shape[0], gt.shape[-1]))
      print('gt', gt.shape)

   if conditions_baseline!='':
      with open(conditions_baseline, 'rb') as f:
         cb = pickle.load(f)
         print('cb', cb.shape)
   else:
      cb = None

   class_dict = options[option]['class_dict']
   labels = options[option]['labels']
   
   class_indices = prepare_class_indices(class_dict, labels)
   pred = extract_blocks(pred, class_indices)
   gt = extract_blocks(gt, class_indices)

   if use_classes:
      all_class_vector = get_cond_to_class_map()
   
   bool_masks = {}
   print(f'{"channel":10s} {"popularity-1":10s} {"result-1":10s}  {"AER":10s}| {"popularity-3":10s} {"result-3":10s} {"AER":10s}')
   for k in class_indices:
      bool_masks[k]={}
      popularity = gt[k].mean(axis=0)
      line = f'{k:10s}   '
      for topk in [1,3]:
         if use_classes:
            class_vector = [all_class_vector[x] for x in class_indices[k][:-1]]
            pop = get_top_k_vector_classes(popularity, gt[k], topk, class_vector)
            res = get_top_k_vector_classes(pred[k], gt[k], topk, class_vector)
         else:
            pop = get_top_k_vector(popularity, gt[k], topk)
            res = get_top_k_vector(pred[k], gt[k], topk)
         bool_masks[k][topk] = (pop, res)
         pop, res = pop.mean(), res.mean()
         aer = (res-pop)/(1-pop)
         line += f'{pop:10.4f} {res:10.4f} {aer:10.4f} | '
      print(line)

   if 'cond' not in option:
      line = f'{"B&S":10s}   '
      for topk in [1,3]:
         pop = bool_masks['B'][topk][0] & bool_masks['S'][topk][0]
         res = bool_masks['B'][topk][1] & bool_masks['S'][topk][1]
         pop, res = pop.mean(), res.mean()
         aer = (res-pop)/(1-pop)
         line += f'{pop:10.4f} {res:10.4f} {aer:10.4f} | '
      print(line)

      line = f'{"B&S comb":10s}   '
      B = sigmoid(pred['B'])
      S = sigmoid(pred['S'])
      C = combine_B_S_blocks({'B':B,'S':S})
      Cgt = combine_B_S_blocks(gt)
      if not isinstance(cb, type(None)):
         print('Cgt',Cgt.shape)
         popularity=cb
      else:
         popularity = Cgt.mean(axis=0)

      for topk in [1,3]:
         pop = get_top_k_vector(popularity, Cgt, topk)
         res = get_top_k_vector(C, Cgt, topk)
         pop, res = pop.mean(), res.mean()
         aer = (res-pop)/(1-pop)
         line += f'{pop:10.4f} {res:10.4f} {aer:10.4f} | '
      print(line)

def mark_pairs(dataset, cond_col='cond_class', smiles_col='Product'):
   mclass = dataset.groupby(smiles_col).agg({cond_col:lambda x:len(np.unique(x))})
   mclass = mclass[mclass.cond_class>1]
   dataset['pair_max'] = -1
   dataset['pair_min'] = -1

   par_id = 0
   for smi in mclass.index:
      view = dataset[dataset.Product==smi]
      assert len(view)>1
      Ymax = view['Yield'].max()
      cond_max = view[view['Yield']==Ymax][cond_col].values[0]
      rest =  view[(view[cond_col]!=cond_max) & (Ymax-view['Yield']>10)]
      if len(rest)==0:
         continue
      Ymin = rest['Yield'].min()
      cond_min = rest[rest['Yield']==Ymin][cond_col].values[0]

      max_mask = (dataset[smiles_col]==smi) & (dataset[cond_col]==cond_max) & (dataset['Yield']==Ymax)
      min_mask = (dataset[smiles_col]==smi) & (dataset[cond_col]==cond_min) & (dataset['Yield']==Ymin)

      max_argmax = max_mask.argmax()
      max_mask[max_argmax+1:] = False
      min_argmax = min_mask.argmax()
      min_mask[min_argmax+1:] = False
      
      assert max_mask.sum()==1
      assert min_mask.sum()==1

      dataset.loc[max_mask, 'pair_max'] = par_id
      dataset.loc[min_mask, 'pair_min'] = par_id
      par_id += 1
   print('pairs found: ', par_id)

def count_pairs(dataset, score_col, max_pair_col='pair_max', min_pair_col='pair_min'):
   max_ = dataset[dataset[max_pair_col]>=0].sort_values(max_pair_col)
   min_ = dataset[dataset[min_pair_col]>=0].sort_values(min_pair_col)
   assert len(max_)==len(min_)
   statuses = max_[score_col].values>min_[score_col].values
   return statuses.mean()


def evaluate_pairs(results_folder, option='suzuki_rr', test_core='', output='pairs.csv'):
   assert test_core != ''
   class_dict = options[option]['class_dict']
   labels = options[option]['labels']
   class_indices = prepare_class_indices(class_dict, labels)


   #collect test set arrays
   dataset = pd.read_csv(test_core%0, sep=';')
   for i in range(1,5):
      dataset = pd.concat([dataset, pd.read_csv(test_core%i, sep=';')], ignore_index=True)

   #define pairs
   dataset['cond_class'] = dataset['B'] + dataset['S']
   mark_pairs(dataset)
   results = []

   for e in tqdm.trange(100):
      epoch_results = []
      for fold_id in range(5):
         with open(f'{results_folder}_f{fold_id}/pred_e{e+1}.pkl', 'rb') as f:
            pred = pickle.load(f)
   
         with open(f'{results_folder}_f{fold_id}/gt.pkl', 'rb') as f:
            gt = pickle.load(f)
            gt = gt.reshape((gt.shape[0], gt.shape[-1]))
   
         pred = extract_blocks(pred, class_indices)
         gt = extract_blocks(gt, class_indices)
   
         assert 'cond' not in option
         B = sigmoid(pred['B'])
         S = sigmoid(pred['S'])
         C = combine_B_S_blocks({'B':B,'S':S})
         Cgt = combine_B_S_blocks(gt)
         score_for_true = (C*Cgt).max(axis=1)
         epoch_results.append(score_for_true)
      epoch_results = np.hstack(epoch_results)
      dataset['epoch_%i'%e] = epoch_results
      results.append(count_pairs(dataset, f'epoch_{e}'))
   results = pd.DataFrame({'pair_score':results})
   results.to_csv(output, sep=';', index=False)
   dataset.to_csv(f'dataset_{output}', sep=';', index=False)
   print('pair score: ', results['pair_score'].max(), 'epoch:', results['pair_score'].argmax()+1)


if __name__=='__main__':
   import argparse
   parser = argparse.ArgumentParser()
   parser.add_argument('--directory', type=str, default='result_relgcn')
   parser.add_argument('--test_core', type=str, default='')
   parser.add_argument('--use_classes', action='store_true')
   parser.add_argument('--cv', action='store_true')
   parser.add_argument('--make_popularity', action='store_true')
   parser.add_argument('--pairs', action='store_true')
   parser.add_argument('--option', type=str, default='suzuki_rr')
   parser.add_argument('--conditions_baseline', type=str, default='')
   parser.add_argument('--out', type=str, default='suzuki_rr_relgcn_cv.csv')
   args = parser.parse_args()
   print(args)
   print(options[args.option])
   if args.cv:
      evaluate_cv(args.directory, option=args.option, use_classes=args.use_classes, output=args.out)
   elif args.pairs:
      evaluate_pairs(args.directory, option=args.option, test_core=args.test_core, output=args.out)
   elif args.make_popularity:
      popularity = {}
      class_dict = options[args.option]['class_dict']
      labels = options[args.option]['labels']
   
      class_indices = prepare_class_indices(class_dict, labels)
      for i in range(5):
         with open(f'{args.directory}_f{i}/gt.pkl', 'rb') as f:
            gt = pickle.load(f)
            gt = gt.reshape((gt.shape[0], gt.shape[-1]))
   
         gt = extract_blocks(gt, class_indices)
         Cgt = combine_B_S_blocks(gt)
         popularity[i] = Cgt.sum(axis=0)
      for i in range(5):
         train_pop = sum(popularity[j] for j in range(5) if j!=i)
         train_pop/=train_pop.sum()
         with open(f'{args.out.split(".")[0]}_baseline_f{i}.pkl', 'wb') as f:
            pickle.dump(train_pop,f)
   else:
      evaluate_channels(args.directory, option=args.option, use_classes=args.use_classes, conditions_baseline=args.conditions_baseline)
#   class_indices = prepare_class_indices(class_dict, labels)
#   print(class_dict)
#   for k, v in class_indices.items():
#      print(k, v[:3], v[-3:])
