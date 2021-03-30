seedNum=10
import random, statistics
random.seed(seedNum)
import numpy
numpy.random.seed(seedNum)

import tensorflow as tf
tf.random.set_seed(seedNum)
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"


import sklearn, numpy, sys
from sklearn import preprocessing, decomposition, cluster, model_selection
import matplotlib.pyplot as plt
#import keras
from keras import optimizers, regularizers, utils
from keras import backend as K
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from mpl_toolkits.mplot3d import Axes3D


def training(X,Y, model,lenx1, lenx2,  nfolds=5, epochs=30):
    kf5 = model_selection.KFold(n_splits=nfolds)
    #kf5.get_n_splits(tab)
    #initWeights = model.get_weights()
    randInit = tf.keras.initializers.RandomNormal()
    #X = preprocessing.scale(X)
    iniw = model.get_weights()
    initShapes = [ i.shape for i in iniw]
    print("shapes", initShapes)
    eachFoldData = []
    for trainIdx, testIdx in kf5.split(X):
        Xtrain, Xtest = X[trainIdx], X[testIdx]
        Ytrain, Ytest = Y[trainIdx], Y[testIdx]
        #print("Tr min max", numpy.amin(train), numpy.amax(train) )
        #print("TEST", numpy.amin(test), numpy.amax(test) )
        #print("XT", [x for x in Xtest[0] if x != 0])
        eachEpochData=[]
        print("WW", type(iniw), iniw )
        model.set_weights( [randInit(shape=x) for x in initShapes] )
        for epochidx in range(epochs):
            model.fit(Xtrain, Ytrain, epochs=1, batch_size=20, shuffle=True, verbose=2, validation_data=(Xtest, Ytest))
            #encodedRep = encoder.predict(test)
            topN = []
            MAE = []
            yieldRange = []
            for testidx, xtest in enumerate(Xtest):
                allx=[ list(xtest), ] #first element in allx is from testset next are all posible combination of solvent/base class
                for i in range(lenx1):
                    for j in range(lenx2):
                        if xtest[i] > 0.99 and xtest[lenx1+j]>0.99:
                            continue
                        li =[ 0 for x in range(lenx1+lenx2)]
                        li[i]=1
                        li[lenx1+j]=1
                        li2=list(xtest)[lenx1+lenx2:]
                        allx.append( li+li2)
                allx=numpy.array(allx)
                #print("ALLX", allx.size, allx.shape)
                result2 = model.predict(allx)
                MAE.append( float(abs(result2[0]- Ytest[testidx])) )
                diffY = float(max(result2))-float(min(result2))
                #print("allY", [x[0] for x in result2])
                #print("AX", allx)
                resorted = sorted([ (x,i) for i,x in enumerate(result2)], reverse=True)
                #print("RE", resorted)
                res=[i for i,x in enumerate(resorted) if x[1] == 0]
                #print("RE", result2, res)
                #raise
                topN.append(res[0] )
                yieldRange.append( diffY)
            topNproc=[]
            for i in range(1, 6):
                s1top= len([s for s in topN if s <=i])/ len(topN)
                topNproc.append( s1top )
            print("YILE RAGE", yieldRange)
            eachEpochData.append( (statistics.mean(MAE), tuple(topNproc), statistics.mean(yieldRange) ) )
            print("last epoch", eachEpochData[-1])
        print("ALL EPOCH DONE IN FOLD", trainIdx)
        eachFoldData.append(eachEpochData)
    for epochid in range(epochs):
        thisEpoch= [ oneFold[epochid] for oneFold in eachFoldData]
        topN=[ fold[1] for fold in thisEpoch]
        aveMAE = statistics.mean([x[0] for x in thisEpoch])
        stdMAE = statistics.stdev([x[0] for x in thisEpoch])
        avgYieldRange = statistics.mean([x[2] for x in thisEpoch])
        topNproc=[ ]
        topNstdev = []
        for i in range( len(topN[0])):
            topNproc.append( statistics.mean([fold[i] for fold in topN ]) )
            topNstdev.append( statistics.stdev([fold[i] for fold in topN ]) )
        print(epochid+1, "MAE", aveMAE, stdMAE, "TOPN",  topNproc, topNstdev, "avgYrange", avgYieldRange)


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
     

def parseData(fn):
    tabOryg=numpy.loadtxt(fn, delimiter='\t',  )
    #inputDim=len(tabOryg[0])
    X0 = tabOryg[:,2:-1]
    #print("X",X)
    #u=numpy.unique(X, axis=0, return_counts=True)
    #print("u", type(u), u )
    #print("u1", list(u[1]) )
    #raise
    X1= tabOryg[:,1]
    X1 = X1-1
    X2= tabOryg[:,0]
    X2= X2-1
    Y=tabOryg[:,-1]
    X1 = utils.to_categorical(X1, int(numpy.amax(X1)+1) )
    X2 = utils.to_categorical(X2, int(numpy.amax(X2)+1) )
    #print(numpy.amax(Y1),  numpy.amax(Y2) )
    X=numpy.concatenate( [X1, X2,X0], axis=1)
    return X, Y/100, len(X1[0]), len(X2[0])

def makeModel(inputDim, wide1=90, wide2=10, act1='relu', act2='relu', act3='elu'):
    input_img = Input(shape=(inputDim,))
    hide1 = Dense(wide1, activation=act1)(input_img)
    hide9 = Dense(wide2, activation=act2)(hide1)
    hide9 = Dropout(0.05)(hide9)
    outyield = Dense(1, activation=act3)(hide9)
    model = Model(input_img, outyield)
    optim = optimizers.Adam() #( clipnorm=1, lr=0.01,  amsgrad=True )
    model.compile(optimizer=optim, loss='mean_squared_error', metrics=["mean_absolute_error",])
    #model.compile(optimizer=optim, loss='mean_squared_error', metrics=["mean_absolute_error", customLoss])
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
    parser.add_argument('--act3', required=True, type=str)
    args = parser.parse_args()
    X,Y, lenx1, lenx2 =parseData( args.input)
    print("args", args)
    print("X", X[0], Y[0], lenx1, lenx2)
    #raise
    print("DIM", len(X[0]) )
    model=makeModel( len(X[0]), wide1=args.n1, wide2=args.n2, act1=args.act1, act2=args.act2, act3=args.act3  )
    training( X,Y, model, lenx1, lenx2)

