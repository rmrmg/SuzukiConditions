seedNum=10
import random
random.seed(seedNum)
import numpy
np=numpy
numpy.random.seed(seedNum)

import tensorflow as tf
#tf.random.set_seed(10)


import sklearn, numpy, sys
from sklearn import preprocessing, decomposition, cluster, model_selection
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
import matplotlib.pyplot as plt
#import keras
from keras import optimizers, regularizers, utils
from keras.layers import Input, Dense
from keras.models import Model
from mpl_toolkits.mplot3d import Axes3D

def make_spys(arrs, nspy=0.1):
   spy_indices, new_arrs= [], []
   for arr in arrs:
      N = len(arr)
      r=np.ones(N)
      r*=arr
      ns = int(nspy*(arr.sum()))
      indices = np.random.choice(np.where(arr==1)[0], ns, replace=False)
      r[indices]=0
      spy_indices.append(indices)
      new_arrs.append(r)
   return spy_indices, new_arrs


def make_mask(model, X, true_arrs, epochs=5, perc=50):
   spy_indices, new_arrs = make_spys(true_arrs)
   model.fit(X, new_arrs, epochs=epochs, batch_size=100, sample_weight = [compute_sample_weight({0:1,1:3},x) for x in new_arrs], verbose=False)
   preds = model.predict(X)
   masks=[]
   for i,x in enumerate(true_arrs):
      spies = preds[i][spy_indices[i],0]
      th = np.percentile(spies, perc)
      mask = np.ones(len(x))
      idx = (x==0) & (preds[i][:,0]>th)
      mask[idx]=0
      masks.append(mask)
      print('masking: %i/%i'%(len(mask)-mask.sum(), len(mask)))
   return masks



def training(X,Y1,Y2, model_lens, nfolds=5):
    kf5=model_selection.KFold(n_splits=nfolds)
    #kf5.get_n_splits(tab)
    results, results_k = [], []
    #model = makeModel(*model_lens)
    for trainIdx, testIdx in kf5.split(X):
        model = makeModel(*model_lens)
        Xtrain, Xtest = X[trainIdx], X[testIdx]
        Y1train, Y1test = Y1[trainIdx], Y1[testIdx]
        Y2train, Y2test = Y2[trainIdx], Y2[testIdx]
        #print("Tr min max", numpy.amin(train), numpy.amax(train) )
        #print("TEST", numpy.amin(test), numpy.amax(test) )
        sample_weights =[ compute_sample_weight('balanced',  y_.argmax(axis=1)) for y_ in [Y1train, Y2train]]
        model.fit(Xtrain, [Y1train, Y2train], epochs=50, 
                  sample_weight=sample_weights,
                  batch_size=20, shuffle=False, validation_data=(Xtest, [Y1test, Y2test]))
        Y1p, Y2p = model.predict(Xtest)
        acc1 = np.mean([x.argmax()==Y1test[i].argmax() for i,x in enumerate(Y1p)])
        acc2 = np.mean([x.argmax()==Y2test[i].argmax() for i,x in enumerate(Y2p)])
        k2_acc1 = np.mean([Y1test[i].argmax() in np.argsort(x)[-2:] for i,x in enumerate(Y1p)])
        k2_acc2 = np.mean([Y2test[i].argmax() in np.argsort(x)[-2:] for i,x in enumerate(Y2p)])
        results_k.append([k2_acc1,k2_acc2])
        acc_check1 = accuracy_score(Y1test, Y1p.round(0))
        acc_v1 = [accuracy_score(Y1test[:,i], Y1p.round(0)[:,i]) for i in range(Y1.shape[1])]
        bacc_v1 = [balanced_accuracy_score(Y1test[:,i], Y1p.round(0)[:,i]) for i in range(Y1.shape[1])]
        acc_v2 = [accuracy_score(Y2test[:,i], Y2p.round(0)[:,i]) for i in range(Y2.shape[1])]
        bacc_v2 = [balanced_accuracy_score(Y2test[:,i], Y2p.round(0)[:,i]) for i in range(Y2.shape[1])]
        print(acc1, acc2)
        print('check1: ',acc_check1)
        print('vec1:',np.round(acc_v1,3))
        print('vec2:',np.round(acc_v2,3))
        print('balanced:')
        print('vec1:',np.round(bacc_v1,3))
        print('vec1:',np.round(bacc_v2,3))
        results.append([acc1,acc2])
    av = np.mean(results, axis=0)
    std = np.std(results, axis=0)
    print('Average: out1 %8.3f (%.3f)    out2 %8.3f (%.3f)'%(av[0], std[0], av[1], std[1]))
    av = np.mean(results_k, axis=0)
    std = np.std(results_k, axis=0)
    print('top-2')
    print('Average: out1 %8.3f (%.3f)    out2 %8.3f (%.3f)'%(av[0], std[0], av[1], std[1]))
        #encodedRep = encoder.predict(test)


def production(tab):
    autoencoder.fit(tab, tab, epochs=30, batch_size=20, shuffle=True)
    model_json = autoencoder.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    autoencoder.save_weights("model.h5")
    print("Saved model to disk")
     

def parseData(fn):
    tabOryg=numpy.loadtxt(fn, delimiter='\t',  )
    #inputDim=len(tabOryg[0])
    X = tabOryg[:,:-2]
    #print("X",X)
    u=numpy.unique(X, axis=0, return_counts=True)
    #print("u", type(u), u )
    #print("u1", list(u[1]) )
    #raise
    Y1= tabOryg[:,-2]
    Y1 = Y1-1
    Y2= tabOryg[:,-1]
    Y2= Y2-1
    Y1 = utils.to_categorical(Y1, int(numpy.amax(Y1)+1) )
    Y2 = utils.to_categorical(Y2, int(numpy.amax(Y2)+1) )
    print('Y1:')
    print('  shape:   :', Y1.shape)
    print('  rel count:', Y1.mean(axis=0))
    print('Y2:')
    print('  shape:   :', Y2.shape)
    print('  rel count:', Y2.mean(axis=0))
    #print(numpy.amax(Y1),  numpy.amax(Y2) )
    return X, Y1,Y2

def makeModel(inputDim, solventClassNum, baseClassNum, wide1=30, wide2=10):
    input_img = Input(shape=(inputDim,))
    hide1 = Dense(wide1, activation='elu')(input_img)
    hide9 = Dense(wide2, activation='elu')(hide1)
    solvent = Dense(solventClassNum, activation='softmax')(hide9)
    base = Dense(baseClassNum, activation='softmax')(hide9)
    model = Model(input_img, [solvent,base])
    optim = optimizers.Adam() #( clipnorm=1, lr=0.01,  amsgrad=True )
    #model.compile(optimizer=optim, loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def makeMultiLabelModel(inputDim, solventClassNum, wide1=30, wide2=10):
    input_img = Input(shape=(inputDim,))
    hide1 = Dense(wide1, activation='sigmoid')(input_img)
    hide1 = Dense(wide1, activation='sigmoid')(hide1)
    hide9 = Dense(wide2, activation='sigmoid')(hide1)
    solvents = [Dense(1, activation='sigmoid', name='solv_%i'%i)(hide9) for i in range(solventClassNum)]
    model = Model(input_img, solvents)
    optim = optimizers.Adam() #( clipnorm=1, lr=0.01,  amsgrad=True )
    model.compile(optimizer=optim, loss='binary_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model
    

def trainingMultiLabel(X,Y2, model_lens, nfolds=5):
    kf5=model_selection.KFold(n_splits=nfolds)
    #kf5.get_n_splits(tab)
    results = []
    results_k = []
    fold_idx=0
    for trainIdx, testIdx in kf5.split(X):
        print('Fold %i'%fold_idx)
        fold_idx+=1
        model = makeMultiLabelModel(*model_lens, wide1=60, wide2=60)
        Xtrain, Xtest = X[trainIdx], X[testIdx]
        Y2train, Y2test = Y2[trainIdx], Y2[testIdx]
        #print("Tr min max", numpy.amin(train), numpy.amax(train) )
        #print("TEST", numpy.amin(test), numpy.amax(test) )

        YsTrain = [Y2train[:,i] for i in range(Y2.shape[1])]
        YsTest = [Y2test[:,i] for i in range(Y2.shape[1])]

        #class_weights = [compute_class_weight('balanced',[0,1],x) for x in YsTrain]
        masks = [compute_sample_weight({0:1,1:3},x) for x in YsTrain]
        #masks = [compute_sample_weight('balanced',x) for x in YsTrain]
        masks_ =  make_mask(model, Xtrain, YsTrain, epochs=50, perc=50)
        for i,x in enumerate(masks_):
           masks[i]*=x
        model = makeMultiLabelModel(*model_lens, wide1=60, wide2=60)

        model.fit(Xtrain, YsTrain, 
                 sample_weight=masks, 
                 epochs=25, batch_size=100, shuffle=False, 
                 #class_weight=class_weights, 
                 verbose=False,
                 validation_data=(Xtest, YsTest))

        Yp = model.predict(Xtest)
        Y2p = np.array(Yp).T[0]
        print('Ytest2: ',Y2test.shape,' Y2p: ',Y2p.shape)
        acc2 = np.mean([x.argmax()==Y2test[i].argmax() for i,x in enumerate(Y2p)])

        k2_acc2 = np.mean([Y2test[i].argmax() in np.argsort(x)[-2:] for i,x in enumerate(Y2p)])

        acc_v2 = [accuracy_score(Y2test[:,i], Y2p.round(0)[:,i]) for i in range(Y2.shape[1])]
        bacc_v2 = [balanced_accuracy_score(Y2test[:,i], Y2p.round(0)[:,i]) for i in range(Y2.shape[1])]
        print('Acc:',acc2)
        print('vec2:',np.round(acc_v2,3))
        print('balanced:')
        print('vec1:',np.round(bacc_v2,3))
        print('top-2')
        print(k2_acc2)
        results.append(acc2)
        results_k.append(k2_acc2)
    av = np.mean(results)
    std = np.std(results)
    print('Average:  out2 %8.3f (%.3f)'%(av, std))
    print('top-2')
    av = np.mean(results_k)
    std = np.std(results_k)
    print('Average: out2 %8.3f (%.3f)'%(av, std))

if __name__ == "__main__":
    import gzip
    from os.path import isfile
    if isfile('x.npz') and isfile('y1.npz') and isfile('y2.npz'):
       with gzip.open('x.npz', 'rb') as f, gzip.open('y1.npz', 'rb') as g, gzip.open('y2.npz', 'rb') as h:
         X, Y1, Y2 = [np.load(fo) for fo in [f,g,h]]
    else:
      X,Y1,Y2=parseData( sys.argv[1])
      with gzip.open('x.npz', 'wb') as f, gzip.open('y1.npz', 'wb') as g, gzip.open('y2.npz', 'wb') as h:
         np.save(f,X)
         np.save(g,Y1)
         np.save(h,Y2)

    print("DIM", len(X[0]), len(Y2[0]) )
    #model=makeModel( len(X[0]), len(Y1[0]), len(Y2[0]) )
    trainingMultiLabel( X,Y2, (len(X[0]), len(Y2[0])))
    #training(X,Y1,Y2, (len(X[0]), len(Y1[0]), len(Y2[0])))


