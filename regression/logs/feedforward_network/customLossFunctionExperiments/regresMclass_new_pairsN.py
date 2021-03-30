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
#import matplotlib.pyplot as plt
#import keras
from keras import optimizers, regularizers, utils
from keras import backend as K
from keras.layers import Input, Dense, Dropout
from keras.models import Model
#from mpl_toolkits.mplot3d import Axes3D


#@tf.function
def training(X,Y, model,lenx1, lenx2, Xax, Yax, nfolds=5, batchSize=40, nepoch=30, pairs=1):
    kf5=model_selection.KFold(n_splits=nfolds)
    #kf5.get_n_splits(tab)
    #initWeights = model.get_weights()
    randInit = tf.keras.initializers.RandomNormal()
    npRand = numpy.random.default_rng()
    #X = preprocessing.scale(X)           
    iniw = model.get_weights()   
    initShapes = [ i.shape for i in iniw]
    beginPair =[x for x in range( len(Yax)) if x%2==0]
    folds=[]
    for trainIdx, testIdx in kf5.split(X):
        #Ymean=float(numpy.mean(Y))
        #Ystd=float(numpy.std(Y))
        #print("MEAN", Ymean, type(Ymean), float(Ymean), "YST", Ystd, type(Ystd))
        Xtrain, Xtest = X[trainIdx], X[testIdx]
        Ytrain, Ytest = Y[trainIdx], Y[testIdx]
        #Ytrain = (Ytrain-Ymean)/Ystd
        #Yax= (Yax-Ymean)/Ystd
        folds.append( [])
        #print("Tr min max", numpy.amin(train), numpy.amax(train) )
        #print("TEST", numpy.amin(test), numpy.amax(test) )
        #print("XT", [x for x in Xtest[0] if x != 0])
        for epoch in range(nepoch):
            #model.set_weights(initWeights)
            model.set_weights( [randInit(shape=x) for x in initShapes] )
            beginidx=0
            Ytrain2d = Ytrain.reshape( [len(Ytrain), 1])
            XYtrain = numpy.concatenate( [Xtrain,Ytrain2d], axis=1)
            npRand.shuffle(XYtrain)
            Xtrain = XYtrain[:,0:len(Xtrain[0]) ]
            Ytrain = XYtrain[:,-1]
                                    
            #res=model.fit(Xtrain, Ytrain, epochs=20, batch_size=20, shuffle=False, validation_data=(Xtest, Ytest))
            for batchNum,endidx in enumerate(range(batchSize, len(Xtrain), batchSize)):
                #print("RANGE", beginidx, endidx)
                x=Xtrain[beginidx:endidx]
                y=Ytrain[beginidx:endidx]
                for i in range(pairs):
                    beginOfPair= random.choice(beginPair) 
                    xpair=Xax[beginOfPair:beginOfPair+2]
                    ypair=Yax[beginOfPair:beginOfPair+2]
                    x=numpy.concatenate( (x,xpair), axis=0)
                    y=numpy.concatenate( (y,ypair), axis=0)
                weights=[ 1 for i in range(batchSize)]
                for i in range(pairs):
                    weights.extend( [0,0])
                #print("X", x.shape, y, ypair )
                #raise
                #print("SUMSUM bef", ([sum(x) for x in model.get_weights()]) )
                if batchNum == 0:
                    res=model.train_on_batch(x,y, sample_weight=numpy.array(weights), reset_metrics=True )
                else:
                    res=model.train_on_batch(x,y, sample_weight=numpy.array(weights) )
                #print("SUMSUM after",([sum(x) for x in model.get_weights()]) )
                beginidx=endidx
                #print("RES on train fit", res, y)
                #print( "YT", Ytrain.shape, y.shape, "X", Xtrain.shape, x.shape)
            print("EPOCH", epoch, res, end="||")

            if True: #after full epoch
                topN=[]
                MAE=[]
                for testidx, xtest in enumerate(Xtest):
                    allx=[ list(xtest), ]
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
                    result2=model.predict(allx)
                    #print("allY", [x[0] for x in result2])
                    Yerror= abs(result2[0][0]- Ytest[testidx])
                    #print("Yerror1", Yerror, "TES", result2[0][0], "T", Ytest[testIdx] )
                    #Yerror =abs( ((result2[0][0]*Ystd)+Ymean) -  ((Ytest[testIdx][0]*Ystd)+Ymean) )
                    #Yerror = (Yerror*Ystd)+Ymean
                    #print("Yerror2", Yerror, Ytest[testIdx])
                    #print("Ytest", Ytest, "\nYYY", Ytest[testidx])
                    MAE.append( Yerror  )
                    resorted = sorted([ (x,i) for i,x in enumerate(result2)], reverse=True)
                    #print("RE", resorted)
                    res=[i for i,x in enumerate(resorted) if x[1] == 0]
                    #print("RE", result2, res)
                    #raise
                    topN.append(res[0] )
                topNcount=[ topN.count(x)/len(topN) for x in range(10) ]
                topNcount = [ round(sum(topNcount[:x]),3) for x in range(1,10)]
                print("TOPN", topNcount, statistics.mean(MAE), "ENDMAE") 
                folds[-1].append( (topNcount, statistics.mean(MAE)) )

            
    for i in range( len(folds[0])):
        thisEpoch=[ cross[i] for cross in folds ]
        
        topN = [ x[0] for x in thisEpoch]
        avgTopN = [ ]
        for i in range( len(topN[0])):
            avgTopN.append( statistics.mean([ x[i] for x in topN]) )
        mae = [ x[1] for x in thisEpoch]
        print("epoch", i+1, [round(x,3) for x in avgTopN],  statistics.mean(mae) )

    #for i in range(1, 6):
    #    s1top= len([s for s in topN if s <=i])/ len(topN)
    #    print("TOP N", i, round(s1top,3) )


@tf.function
def customLoss(ytrue, ypred):
    #print("\n\nXXX", ytrue, ypred, "YYYYY\n\n")
    #print( dir(ytrue) )
    #print( ytrue._shape, type(ytrue) )
    #print( "\n\n\n\n\n", ytrue.shape)
    #print("YT", ytrue[:,:,:,0])
    #K.print_tensor(ytrue, message='y_true = ')
    #tf.print("yTrue", ytrue) #, "\nypredict", ypred)
    #print( help(ytrue) )
    #for i in ytrue:
    #    print("ONE I", i)
    #e = K.get_value(ytrue) #ytrue.eval(session=K.get_session())
    #print( type(e), e)
    #raise
    sqr = K.square(ypred - ytrue)
    #tf.print("SQR", sqr)
    rmsd = K.mean(sqr, axis=-1)
    Npairs = 3
    diff =0
    for i in range(Npairs):
        difftrue= ytrue[-2-(2*i)] - ytrue[-1-(2*i)]
        diffpred= ypred[-2-(2*i)] - ypred[-1-(2*i)]
        diff += difftrue-diffpred
    #tf.print("RMSD", len(rmsd), rmsd, rmsd+diff, difftrue, diffpred)
    #return K.sum(K.log(ytrue) - K.log(ypred))
    return rmsd +diff


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


def makeModel(inputDim, wide1=40, wide2=10):
    input_img = Input(shape=(inputDim,))
    hide1 = Dense(wide1, activation='elu')(input_img)
    hide9 = Dense(wide2, activation='sigmoid')(hide1)
    hide9 = Dropout(0.05)(hide9)
    outyield = Dense(1, activation='relu')(hide9)
    model = Model(input_img, outyield)
    optim = optimizers.Adam() #( clipnorm=1, lr=0.01,  amsgrad=True )
    #model.compile(optimizer=optim, loss='mean_squared_error', metrics=["mean_absolute_error",])
    #model.compile(optimizer=optim, loss='mean_squared_error', metrics=["mean_absolute_error", customLoss])
    model.compile(optimizer=optim, loss=customLoss, metrics=["mean_absolute_error", 'mean_squared_error', customLoss])
    model.summary()
    return model

if __name__ == "__main__":
    X,Y, lenx1, lenx2 =parseData( sys.argv[1])
    Xax, Yax, lenx1ax, lenx2ax = parseData( sys.argv[2])
    if lenx1 != lenx1ax or lenx2 != lenx2ax:
        raise
    print("X", X[0], Y[0], lenx1, lenx2, lenx1ax)
    #raise
    print("DIM", len(X[0]) )
    model=makeModel( len(X[0]) )
    training( X,Y, model, lenx1, lenx2, Xax, Yax, pairs=3)

