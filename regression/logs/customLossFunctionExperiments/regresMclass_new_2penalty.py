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

KLAS_NUM1=6
KLAS_NUM2=7

def makeAllCombination(xrow, y):
    #print(xrow[0:15])
    listOfList=[]
    yrow=[]
    for i in range(KLAS_NUM1):
        for j in range(KLAS_NUM2):
            lst= [0,]*(KLAS_NUM1+KLAS_NUM2)
            lst[i]=1
            lst[j]=1
            lst = lst + list(xrow[KLAS_NUM1+KLAS_NUM2:])
            listOfList.append( lst )
            #yrow.append( [y,] )
            yrow.append(y, )
    retX = numpy.array(listOfList)
    retY = numpy.array(yrow)
    #print(retX.shape, "Y", retY.shape)
    return retX, retY

#@tf.function
def training(X,Y, model,lenx1, lenx2, Xax, Yax, nfolds=5, batchSize=20, nepoch=30):
    kf5 = model_selection.KFold(n_splits=nfolds)
    #kf5.get_n_splits(tab)
    initWeights = model.get_weights()
    #beginPair = [x for x in range( len(Yax)) if x%2==0]
    randInit = tf.keras.initializers.RandomNormal()
    initShapes = [ i.shape for i in initWeights]

    folds = []
    Ymean = float(numpy.mean(Y))
    Ystd = float(numpy.std(Y))
    #print("Y std", Ystd, "MEAN", Ymean)
    npRand = numpy.random.default_rng()
    fold = 0
    multipletIdxes = [(f1,f2) for (f1,f2) in kf5.split(Xax)]
    multipletResults = []
    for trainIdx, testIdx in kf5.split(X):
        multipletResults.append( [] )
        beginPair = [ x for x in multipletIdxes[fold][0] if x%2==0]
        beginTestPair = [ x for x in multipletIdxes[fold][1] if x%2==0]
        fold += 1
        #print("MEAN", Ymean, type(Ymean), float(Ymean), "YST", Ystd, type(Ystd))
        Xtrain, Xtest = X[trainIdx], X[testIdx]
        Ytrain, Ytest = Y[trainIdx], Y[testIdx]
        Ytrain = (Ytrain-Ymean)/Ystd
        Ytest = (Ytest - Ymean)/Ystd
        Yax = (Yax-Ymean)/Ystd
        folds.append( [])
        #print("Tr min max", numpy.amin(train), numpy.amax(train) )
        #print("TEST", numpy.amin(test), numpy.amax(test) )
        #print("XT", [x for x in Xtest[0] if x != 0])
        model.set_weights([randInit(x) for x in initShapes ])
        for epoch in range(nepoch):
            #do shuffle
            Ytrain2d = Ytrain.reshape( [len(Ytrain), 1])
            #print("SHAPRES1", Xtrain.shape, Ytrain.shape, Ytrain2d.shape)
            XYtrain = numpy.concatenate( [Xtrain,Ytrain2d], axis=1)
            npRand.shuffle(XYtrain)
            Xtrain = XYtrain[:,0:len(Xtrain[0]) ]
            Ytrain = XYtrain[:,-1]
            beginidx = 0
            #print("SHAPRES2", Xtrain.shape, Ytrain.shape)
            #raise
            #res=model.fit(Xtrain, Ytrain, epochs=20, batch_size=20, shuffle=False, validation_data=(Xtest, Ytest))
            checkedPairs=set()
            for batchNum,endidx in enumerate(range(batchSize, len(Xtrain), batchSize)):
                #print("RANGE", beginidx, endidx)
                beginOfPair= random.choice(beginPair)  #random Pair
                checkedPairs.add( beginOfPair)
                x=Xtrain[beginidx:endidx]
                y=Ytrain[beginidx:endidx]
                xpair=Xax[beginOfPair:beginOfPair+2]
                ypair=Yax[beginOfPair:beginOfPair+2]
                x=numpy.concatenate( (x,xpair), axis=0)
                y=numpy.concatenate( (y,ypair), axis=0)
                weights=[ 1 for i in range(batchSize)]
                weights.extend( [0,0])

                for xrow, yrow in zip(x,y):
                    xnew, ynew = makeAllCombination(xrow, yrow)
                    x=numpy.concatenate( (x,xnew), axis=0)
                    #print("Y", y.shape)
                    y=numpy.concatenate( (y,ynew), axis=0)
                    weights.extend( [0,]* len(ynew) )
                    #print("WE", len(weights), weights)
                #print("X", x.shape, y, ypair )
                #raise
                #print("SUMSUM bef", ([sum(x) for x in model.get_weights()]) )
                if batchNum == 0:
                    res=model.train_on_batch(x,y, sample_weight=numpy.array(weights), reset_metrics=True )
                else:
                    res=model.train_on_batch(x,y, sample_weight=numpy.array(weights) )
                #print("SUMSUM after",([sum(x) for x in model.get_weights()]) )
                beginidx=endidx
                #print("RES on train fit", res, type(res[0]), float(res[0]))
                if not( float(res[0]) >0 or float(res[0]) < 0):
                    raise
                #if float(res[0]) != 999.9:
                #    pass
                #print( "YT", Ytrain.shape, y.shape, "X", Xtrain.shape, x.shape)
            #model.save('zachowanyModel_'+str(epoch)+'epoch'+str(fold)+'fold.save')
            print("EPOCH", epoch, res, end="||")
            doCheckPairs= True
            if doCheckPairs:
                resultsPairs=[0,0]
                YpairPred=model.predict(Xax)
                for i in range(0, len(YpairPred), 2):
                    if not(i in beginTestPair):
                        continue
                    firstHigher = Yax[i] > Yax[i+1]
                    predFirstHigher = YpairPred[i] > YpairPred[i+1]
                    resultsPairs[1] +=1
                    if firstHigher == predFirstHigher:
                        resultsPairs[0] +=1
                        print( "Ok", YpairPred[i], YpairPred[i+1] )
                print("Pairs correctenss", round(resultsPairs[0]/resultsPairs[1],3), resultsPairs, len(checkedPairs) )
                multipletResults[-1].append( resultsPairs)
            print("MR", multipletResults)
            if True: #after full epoch
                topN=[]
                topClass1=[]
                topClass2=[]
                MAE=[]
                trueXY=[0,0]
                usedXY=[ 'True', ]
                for idxTestX, xtest in enumerate(Xtest):
                    allx=[ list(xtest), ]
                    for i in range(lenx1):
                        for j in range(lenx2):
                            #print("XTI", i,j, xtest[i], "\nXT:", xtest.shape, xtest)
                            if xtest[i] > 0.99 and xtest[lenx1+j]>0.99: 
                                #first elements of vecotrs is one-hot-encoded classes
                                trueXY = [i,j, lenx1+j]
                                continue
                                #raise
                            li = [ 0 for x in range(lenx1+lenx2)]
                            li[i] = 1
                            li[lenx1+j] = 1
                            li2 = list(xtest)[lenx1+lenx2:]
                            allx.append( li+li2)
                            usedXY.append( (i,j) )
                    allx = numpy.array(allx)
                    #print("ALLX", allx.size, allx.shape)
                    result2 = model.predict(allx)
                    klass1=[]
                    for i in range(lenx1):
                        klass1.append([ float(pred) for pos,pred in enumerate(result2) if usedXY[pos][0] == i] )
                    klass2=[]
                    for i in range(lenx2):
                        klass2.append([ float(pred) for pos,pred in enumerate(result2) if usedXY[pos][1] == i] )
                    #print("TRIE", trueXY, "MEAN", [statistics.mean(x) for x in klass1],  [statistics.mean(x) for x in klass2] )
                    #print("TRIE", trueXY, "MAX:", [max(x) for x in klass1],  [max(x) for x in klass2] )
                    #print("allY", [x[0] for x in result2])
                    Yerror = abs(result2[0][0] - Ytest[idxTestX])
                    #print("Yerror1", Yerror, "TES", result2[0][0], "T", Ytest[testIdx] )
                    #if testidx > 2580:
                    #    print("SHAPES", Xtest.shape, Ytest.shape, idxTestX)
                    Yerror =abs( ((result2[0][0]*Ystd)+Ymean) -  ((Ytest[idxTestX]*Ystd)+Ymean) )
                    #Yerror = (Yerror*Ystd)+Ymean
                    #print("Yerror2", Yerror, Ytest[testIdx])
                    #print("Ytest", Ytest, "\nYYY", Ytest[testidx])
                    MAE.append( Yerror  )
                    resorted = sorted([ (x,i) for i,x in enumerate(result2)], reverse=True)
                    #print("RE", resorted)
                    res = [i for i,x in enumerate(resorted) if x[1] == 0]
                    res1top = [i for i,x in enumerate(resorted) if x[1] == 0 or usedXY[x[1]][0] == trueXY[0] ]
                    res2top = [i for i,x in enumerate(resorted) if x[1] == 0 or usedXY[x[1]][1] == trueXY[1]]
                    #print("RE", result2, res)
                    #raise
                    topN.append(res[0] )
                    topClass1.append( res1top[0])
                    topClass2.append( res2top[0])
                topNcount = [ topN.count(x)/len(topN) for x in range(10) ]
                topC1count = [ topClass1.count(x)/len(topClass1) for x in range(10) ]
                topC2count = [ topClass2.count(x)/len(topClass2) for x in range(10) ]
                topNcount = [ round(sum(topNcount[:x]),3) for x in range(1,10)]
                topC1count = [ round(sum(topC1count[:x]),3) for x in range(1,10)]
                topC2count = [ round(sum(topC2count[:x]),3) for x in range(1,10)]
                print("TOPN", topNcount, statistics.mean(MAE), "ENDMAE" "C!", topC1count, "C2", topC2count) 
                folds[-1].append( (topNcount, statistics.mean(MAE)) )

            
    for i in range( len(folds[0])):
        thisEpoch=[ cross[i] for cross in folds ]
        
        topN = [ x[0] for x in thisEpoch]
        avgTopN = [ ] 
        for j in range( len(topN[0])):
            avgTopN.append( statistics.mean([ x[j] for x in topN]) )
        mae = [ x[1] for x in thisEpoch]
        print("epoch", i+1, [round(x,3) for x in avgTopN],  statistics.mean(mae) )
        epom = [fold[i] for fold in multipletResults]
        percentm = sum([x[0] for x in epom])/sum([ x[1] for x in epom])
        print("    ", round(percentm,3), epom)
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
    sqr = K.square(ypred[0:20] - ytrue[0:20])
    #tf.print("SQR", sqr)
    rmsd = K.mean(sqr, axis=-1)
    difftrue= ytrue[20] - ytrue[21]
    diffpred= ypred[20] - ypred[21]
    diff = difftrue-diffpred
    Yrange=0
    for idx, i in enumerate(range(22, len(ypred), 30)):
        #print( "RED", tf.reduce_max( ypred[i:i+30]) )
        #print("SHAPE", ypred[i:i+30], ypred.shape, ypred[i:i+30].shape, "REDSHAPE", type(red), red, red.shape, dir(red), "ZERO", red[0] )
        #print("YP", ypred[i:i+30], float(tf.reduce_max(ypred[i:i+30])) , float(tf.reduce_max(ypred)) )
        YrangePart= tf.reduce_max( ypred[i:i+30]) - tf.reduce_min(ypred[i:i+30])
        #print("YR", YrangePart)
        YrangePart = (100.0*ytrue[idx]) / (YrangePart+0.000001)
        Yrange += YrangePart
    #tf.print("RMSD", len(rmsd), rmsd, rmsd+diff, difftrue, diffpred)
    #return K.sum(K.log(ytrue) - K.log(ypred))
    res= (rmsd +diff+Yrange)/len(ytrue)
    #print("RES", [res,] )
    return [res,]*len(ytrue)


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
    #print("MAX 1 2", numpy.amax(X1), numpy.amax(X2), "both +1", int(numpy.amax(X1)+1), int(numpy.amax(X2)+1))
    X1 = utils.to_categorical(X1, int(numpy.amax(X1)+1) )
    X2 = utils.to_categorical(X2, int(numpy.amax(X2)+1) )
    #print(numpy.amax(Y1),  numpy.amax(Y2) )
    X=numpy.concatenate( [X1, X2,X0], axis=1)
    #raise
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
    print("run command", sys.argv)
    model=makeModel( len(X[0]) )
    training( X,Y, model, lenx1, lenx2, Xax, Yax)

