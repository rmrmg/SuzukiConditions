seedNum=10
import random, statistics, json, sys
random.seed(seedNum)
import numpy
numpy.random.seed(seedNum)

import tensorflow as tf
tf.random.set_seed(seedNum)
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"


import sklearn, numpy
from sklearn import preprocessing, decomposition, cluster, model_selection
#import keras
from keras import optimizers, regularizers, utils
from keras import backend as K
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from mpl_toolkits.mplot3d import Axes3D


def training(X,Ybase, Ysolvent, model,  nfolds=5, epochs=70):
    kf5 = model_selection.KFold(n_splits=nfolds)
    randInit = tf.keras.initializers.RandomNormal()
    #X = preprocessing.scale(X)
    iniw = model.get_weights()
    initShapes = [ i.shape for i in iniw]
    print("shapes", initShapes)
    eachFoldData = []
    for trainIdx, testIdx in kf5.split(X):
        Xtrain, Xtest = X[trainIdx], X[testIdx]
        YtrainBase, YtestBase = Ybase[trainIdx], Ybase[testIdx]
        YtrainSolvent, YtestSolvent = Ysolvent[trainIdx], Ysolvent[testIdx]
        #print("Tr min max", numpy.amin(train), numpy.amax(train) )
        #print("TEST", numpy.amin(test), numpy.amax(test) )
        #print("XT", [x for x in Xtest[0] if x != 0])
        eachEpochData=[]
        model.set_weights( [randInit(shape=x) for x in initShapes] )
        #print("Weight", model.get_weights() )
        history= model.fit(Xtrain, [YtrainSolvent, YtrainBase], epochs=epochs, batch_size=20, shuffle=True, verbose=2, validation_data=(Xtest, [YtestSolvent, YtestBase]))
        #print("HI", history, "\nHIHIH", history.history)
        eachFoldData.append( history.history)
    for epochid in range(epochs):
        base= [ oneFold['val_base_accuracy'][epochid] for oneFold in eachFoldData]
        solvent = [ oneFold['val_solvent_accuracy'][epochid] for oneFold in eachFoldData]
        base2, base3, solvent2, solvent3 = False, False, False, False
        if 'val_solvent_top2' in eachFoldData[0]:
            base2= [ oneFold['val_base_top2'][epochid] for oneFold in eachFoldData]
            solvent2 = [ oneFold['val_solvent_top2'][epochid] for oneFold in eachFoldData]
        if 'val_solvent_top3' in eachFoldData[0]:
            base3= [ oneFold['val_base_top3'][epochid] for oneFold in eachFoldData]
            solvent3 = [ oneFold['val_solvent_top3'][epochid] for oneFold in eachFoldData]
        print(epochid+1, "base:", statistics.mean(base), statistics.stdev(base), "solvent:", statistics.mean(solvent), statistics.stdev(solvent) )
        if base2 and base3:
            print("     base_top2:",  statistics.mean(base2), statistics.stdev(base2), "top3",  statistics.mean(base3), statistics.stdev(base3))
            print("  solvent_top2:", statistics.mean(solvent2), statistics.stdev(solvent2), "top3", statistics.mean(solvent3), statistics.stdev(solvent3))

def customLoss(ytrue, ypred):
    print("\n\nXXX", ytrue, ypred, "YYYYY\n\n")
    print( dir(ytrue) )
    print( ytrue._shape, type(ytrue) )
    #print( help(ytrue) )
    #for i in ytrue:
    #    print("ONE I", i)
    #e = K.get_value(ytrue) #ytrue.eval(session=K.get_session())
    #print( type(e), e)
    return K.sum(K.log(ytrue) - K.log(ypred))

def production(tab):
    autoencoder.fit(tab, tab, epochs=30, batch_size=20, shuffle=True)
    model_json = autoencoder.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    autoencoder.save_weights("model.h5")
    print("Saved model to disk")
     

def parseData(fn, normalize=False, removeSingleValueColumn=False, encdict=None):
    print("FN", fn)
    datas = []
    for i in open(fn):
        length= len(i.split('\t') )
        if length != 11:
            print(i)
            raise
        datas.append( i.split('\t') )
    tabOryg = numpy.array(datas, dtype='str')
    #tabOryg = numpy.genfromtxt(fn, dtype='str', delimiter='\t')
    #tabOryg=numpy.loadtxt(fn, delimiter='\t', )
    encdict=json.load(open(encdict) )
    #inputDim=len(tabOryg[0])
    X = tabOryg[:,4:-1] # dont get Y
    smi1and2 = []
    for line in tabOryg:
        smi1, smi2 = line[2:4]
        smi1and2.append( encdict[smi1]+encdict[smi2])
    smi1and2 = numpy.array(smi1and2)
    X = X.astype(numpy.float)
    print(X.shape, smi1and2.shape, "zero", smi1and2[0])
    X = numpy.concatenate( [smi1and2,X], axis=1)
    Ybase= tabOryg[:,0]
    Ybase = Ybase.astype(numpy.float)
    Ybase = Ybase-1
    Ysolvent = tabOryg[:,1]
    Ysolvent = Ysolvent.astype(numpy.float)
    Ysolvent = Ysolvent-1

    Y=tabOryg[:,-1]
    Ybase = utils.to_categorical(Ybase, int(numpy.amax(Ybase)+1) ) #base
    Ysolvent = utils.to_categorical(Ysolvent, int(numpy.amax(Ysolvent)+1) ) #solvent
    #print(numpy.amax(Y1),  numpy.amax(Y2) )
    idxes = []
    zero = []
    if normalize:
        for i in range( X.shape[1]):
            if len(numpy.unique( X[:,i])) == 1:
                print(numpy.unique( X[:,i]))
                zero.append(i)
            if len(numpy.unique( X[:,i])) > 2:
                idxes.append(i)
                #X[:,i] = preprocessing.MinMaxScaler().fit_transform(X[:,i])
                #X[:,i] = preprocessing.scale( X[:,i])
        #X[:,idxes] = preprocessing.MinMaxScaler().fit_transform(X[:,idxes])
    if removeSingleValueColumn and zero:
        print("init shape", X.shape)
        X = numpy.delete(X, zero, 1)
        print("reduce shape", X.shape)
    numpy.savetxt('testX', X)
    print("ZERO", zero)
    return X, Ybase, Ysolvent

def makeModel(inputDim, solventClassNum, baseClassNum, wide1=40, wide2=10, act1='elu', act2='elu', act3='softmax'):
    input_img = Input(shape=(inputDim,))
    hide1 = Dense(wide1, activation=act1)(input_img)
    hide9 = Dense(wide2, activation=act2)(hide1)
    solvent = Dense(solventClassNum, activation=act3, name='solvent')(hide9)
    base = Dense(baseClassNum, activation=act3, name='base')(hide9)
    model = Model(input_img, [solvent,base])
    optim = optimizers.Adam()# clipnorm=5)#, lr=0.01,  amsgrad=True )
    #optim = optimizers.SGD(lr=0.001, momentum=0.9, clipvalue=5.0)
    model.compile(optimizer=optim, loss='categorical_crossentropy', 
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top2'), tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3')])
    model.summary()
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=str)
    parser.add_argument('--n1', required=True, type=int)
    parser.add_argument('--n2', required=True, type=int)
    parser.add_argument('--act1', required=True, type=str)
    parser.add_argument('--act2', required=True, type=str)
    parser.add_argument('--encdict', required=True, type=str)
    parser.add_argument('--scale', action='store_true')
    #parser.add_argument('--act3', required=True, type=str)
    args = parser.parse_args()

    X,Ybase, Ysolvent = parseData(args.input, normalize=args.scale, encdict=args.encdict)
    print("input]", args)
    #print("X", X[0], Y[0], lenx1, lenx2)
    print("DIM", len(X[0]) )
    model=makeModel( len(X[0]), len(Ysolvent[0]), len(Ybase[0]), wide1=args.n1, wide2=args.n2, act1=args.act1, act2=args.act2, act3='softmax' )
    training( X, Ybase, Ysolvent,  model)

