import os, sklearn, torch, argparse, itertools
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
from rxnfp.models import SmilesClassificationModel
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdChemReactions


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rxes", type=str, required=True, help='file with Suzuki reactions (full smiles with substrate conditions and products)')
    ap.add_argument("--train", type=str, required=True)
    ap.add_argument("--models", type=str, required=True)
    arguments = ap.parse_args()
    return arguments



def get_model_dirs(directory):
    model_path = [os.path.join(directory, o) for o in os.listdir(directory)
                        if os.path.isdir(os.path.join(directory,o)) and 'epoch' in o]
    return model_path

def load_model(model_path, model_type='bert'):
    print("PATH", model_path)
    use_gpu = torch.cuda.is_available()
    model = SmilesClassificationModel(model_type, model_path, num_labels=1, args={"regression": True}, use_cuda=use_gpu)
    return model



def getScale(name, sep=' '):
    df = pd.read_csv(name, sep=sep)
    df.columns = ['rxn', 'y']
    mean = df.y.mean()
    std = df.y.std()
    return mean, std

def evaluate(model, x):
    yp = model.predict(x)
    return  yp

def evaluateOLD(rxsmiArray):
    predictions = np.zeros((len(NAME_SPLIT), len(rxsmiArray)))
    for fold, namesplit in enumerate(NAME_SPLIT):
        name, split = namesplit
        df = pd.read_csv(f'../data/Suzuki-Miyaura/random_splits/{name}.tsv', sep='\t')
        train_df = df.iloc[:split][['rxn', 'y']] 
        test_df = df.iloc[split:][['rxn', 'y']] 
        train_df.columns = ['text', 'labels']
        test_df.columns = ['text', 'labels']
        mean = train_df.labels.mean()
        std = train_df.labels.std()
        model = load_model_from_results_folder(name, split)
        y_preds = model.predict(rxsmiArray)[0]
        y_preds = y_preds * std + mean
        y_preds = y_preds * 100
        y_preds = np.clip(y_preds, 0, 100)
        predictions[fold] = y_preds
    return predictions

def parse_file(fn):
    data = []
    fh = open(fn)
    for line in fh:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        data.append(line)
    fh.close()
    return data


def load_data(fname):
    rxes = []
    yldes = []
    fh = open(fname)
    for line in fh:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        rxn, yld = line.split()
        if 'y' in yld:
            continue
        rxes.append(rxn)
        yldes.append( float(yld))
    return rxes, yldes

def getTopFromBatch(ylds):
    #print("YLDS", type(ylds), ylds.shape, ylds)
    sortCond = np.argsort(ylds*-1)
    #print("sorted",   sortCond )
    #print("0::::", np.where(sortCond==0), "1!!!", np.where(sortCond==1))
    return np.amin(np.where(np.logical_or(sortCond==0, sortCond==1)))

if __name__ == "__main__":
    args = parse_args()

    mean, std  =getScale(args.train)
    print("MEAN", mean, "STD", std)
    x, y = load_data(args.rxes)
    y = np.array(y)
    fakey =  (y == -1).sum()
    usize = len(y) // (len(y)-fakey)
    print("Y", y.shape, usize, fakey, len(y))
    print(type(y), y.shape, y, type(x))

    for modeldir in get_model_dirs(args.models):
        print("DIR", modeldir)
        model = load_model(modeldir, model_type='bert')
        newy = evaluate(model, x)[0]
        newy = newy * std + mean
        beg = 0
        end = usize
        itr=1
        tops = []
        while end <= len(y):
            tops.append( getTopFromBatch(newy[beg:end]) )
            #y_preds = y_preds * 100
            beg = end
            end += usize
            itr +=1
        # print("N", newy.shape, newy )
        maes = np.abs(newy-y)
        #print(maes.shape, maes)
        print("==>", modeldir.split('/')[-1], "MEAN", np.mean(maes))
        print("==>", modeldir.split('/')[-1], "TOPS", tops.count(0)/len(tops), tops.count(1)/len(tops), tops.count(2)/len(tops) )
        print('HHHHRAW', len(tops), tops)
        #break
    """
    x_arr = np.array(x)
    y_arr = np.array(y)
    #print("DATA", x_arr.shape, type(x_arr), x_arr)
    res_arr = evaluate(x_arr)
    #with open(args.rxes+'.npy', 'wb') as f:
    #    np.save(f, res_arr)
    for poz, res in enumerate(res_arr):
        print(poz, res, y_arr[poz])
    """