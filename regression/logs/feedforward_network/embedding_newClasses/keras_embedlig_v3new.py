seedNum=10
import random, statistics
random.seed(seedNum)
import numpy
numpy.random.seed(seedNum)
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
tf.random.set_seed(seedNum)


import sklearn, numpy, sys
from sklearn import preprocessing, decomposition, cluster, model_selection
import matplotlib.pyplot as plt
#import keras
from keras import optimizers, regularizers, utils
from keras import backend as K
from keras.layers import Input, Dense, Dropout, Add , Embedding, Concatenate, Flatten
from keras.models import Model




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
    solv1 =tabOryg[:,0]
    base1 =tabOryg[:,1]
    ligand1=tabOryg[:,2]
    ligand2=tabOryg[:,3]
    temp = tabOryg[:,4]
    sbs1 = tabOryg[:, 5:5+512]
    sbs2 = tabOryg[:, 5+512:5+512+512]
    yld = tabOryg[:,-1]
    return {'solvents':[solv1,], 'bases':[base1, ], 'ligands':[ligand1, ligand2], 'temp':temp, 'sbses':[sbs1,sbs2], 'yield':yld }


def makeModel(inputDim, wide1=90, wide2=10, embDim=3, solventClasses=1+6, baseClasses=1+7, ligandClasses=1+81, act1='relu', act2='relu', act3='elu' ):
    subs1 = Input(shape=(inputDim,))
    subs2 = Input(shape=(inputDim,))
    temper = Input(shape=(1,) )

    sol1 = Input(shape=(1,) )
    base_1 = Input(shape=(1,) )
    lgand1 = Input(shape=(1,) )
    lgand2 = Input(shape=(1,) )

    solventEmbd = Embedding(solventClasses, embDim, input_length=1)
    #solventEmbd = Dense(2, activation='relu')
    solvent1 = solventEmbd(sol1)

    baseEmbd = Embedding(baseClasses, embDim, input_length=1)
    #baseEmbd = Dense(2, activation='relu')
    base1 = baseEmbd(base_1)

    ligandEmbd = Embedding(ligandClasses, embDim, input_length=1)
    #ligandEmbd = Dense(2, activation='relu')
    ligand1 = ligandEmbd(lgand1)
    ligand2 = ligandEmbd(lgand2)

    #solvent = Add()([solvent1, solvent2, solvent3, solvent4])
    #base = Add()([base1, base2])
    ligand = Add()([ligand1, ligand2])
    conditions =Concatenate()([solvent1,base1, ligand])
    conditions =Flatten()(conditions)
    conditions = Concatenate()([conditions, temper])
    sbs1 = Dense(wide1, activation=act1)(subs1)
    sbs2 = Dense(wide1, activation=act1)(subs2)
    conditionsAndSubstrate = Concatenate() ([conditions, sbs1,sbs2])

    hide9 = Dense(wide2, activation=act2)(conditionsAndSubstrate)
    hide9 = Dropout(0.05)(hide9)
    outyield = Dense(1, activation=act3)(hide9)
    model = Model((sol1,base_1,lgand1,lgand2, temper, subs1, subs2), outyield)

    optim = optimizers.Adam() # lr=0.0005) #( clipnorm=1, lr=0.01,  amsgrad=True ) lr:=default:=0.001
    model.compile(optimizer=optim, loss='mean_squared_error', metrics=["mean_absolute_error",])
    #model.compile(optimizer=optim, loss='mean_squared_error', metrics=["mean_absolute_error", customLoss])
    model.summary()
    return model


def training(data, model,  nfolds=5, epochs=30, klas1=6, klas2=7):
    kf5=model_selection.KFold(n_splits=nfolds)
    #initWeights = model.get_weights()
    randInit = tf.keras.initializers.RandomNormal()
    #X = preprocessing.scale(X)
    iniw = model.get_weights()
    initShapes = [ i.shape for i in iniw]
    eachFoldData=[]
    print("LEN", len(data['sbses'][0]), len(data['yield']) )
    histories=[]
    for trainIdx, testIdx in kf5.split(data['sbses'][0]):
        # Model((solvent1,solvent2,solvent3,solvent4,base1,base2, ligand1,ligand2, temper, sbs1, sbs2), outyield)
        solvent1train= data['solvents'][0][trainIdx]
        solvent1test= data['solvents'][0][testIdx]

        base1train= data['bases'][0][trainIdx]
        base1test= data['bases'][0][testIdx]

        ligand1train= data['ligands'][0][trainIdx]
        ligand1test= data['ligands'][0][testIdx]
        ligand2train= data['ligands'][1][trainIdx]
        ligand2test= data['ligands'][1][testIdx]

        temptrain = data['temp'][trainIdx]
        temptest = data['temp'][testIdx]

        sbs1train = data['sbses'][0][trainIdx]
        sbs1test = data['sbses'][0][testIdx]

        sbs2train = data['sbses'][1][trainIdx]
        sbs2test = data['sbses'][1][testIdx]

        Yldtrain, Yldtest = data['yield'][trainIdx], data['yield'][testIdx]
        eachEpochData=[]
        #model.set_weights(initWeights)
        model.set_weights( [randInit(shape=x) for x in initShapes] )
        #for epochidx in range(epochs):
        inputTrain = [ solvent1train, base1train, ligand1train, ligand2train, temptrain, sbs1train, sbs2train]
        inputTest = [solvent1test, base1test, ligand1test, ligand2test, temptest, sbs1test, sbs2test]

        #history=model.fit(inputTrain, Yldtrain, epochs=epochs, batch_size=20, shuffle=True, validation_data=(inputTest, Yldtest), verbose=2)
        #histories.append( history.history)


        eachEpochData=[]
        for epochidx in range(epochs):
            model.fit(inputTrain, Yldtrain, epochs=1, batch_size=20, shuffle=True, verbose=2, validation_data=(inputTest, Yldtest))
            topN = []
            MAE = []
            yieldRange = []
            for testidx in testIdx:
                thisSolvClasses = numpy.zeros((klas1*klas2, 1))
                thisBasesClasses = numpy.zeros((klas1*klas2, 1))
                thisSolvClasses[0][0]= data['solvents'][0][testidx]
                thisBasesClasses[0][0]= data['bases'][0][testidx]
                thisLigand1 = numpy.array([ [data['ligands'][0][testidx],] for x in range(klas1*klas2)])
                thisLigand2 = numpy.array([ [data['ligands'][1][testidx],] for x in range(klas1*klas2)])
                thisTemp = numpy.array([ [data['temp'][testidx],] for x in range(klas1*klas2)])
                thisSbs1 = numpy.array([ data['sbses'][0][testidx] for x in range(klas1*klas2)])
                thisSbs2 = numpy.array([ data['sbses'][1][testidx] for x in range(klas1*klas2)])
                pos =1
                #print("XXX",data['solvents'][0][testidx],  data['bases'][0][testidx])
                for i in range(1, klas1+1):
                    for j in range(1,klas2+1):
                        if abs(i - data['solvents'][0][testidx]) < 0.01 and abs(j - data['bases'][0][testidx]) < 0.01:
                            continue
                        #print(i,j)
                        thisSolvClasses[pos][0] = i
                        thisBasesClasses[pos][0] = j
                        pos +=1
                result2 = model.predict( [thisSolvClasses, thisBasesClasses, thisLigand1, thisLigand2, thisTemp, thisSbs1, thisSbs2  ])
                
                MAE.append( float(abs(result2[0]- data['yield'][testidx])) )
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
            #print("YILE RAGE", yieldRange)
            eachEpochData.append( (statistics.mean(MAE), tuple(topNproc), statistics.mean(yieldRange) ) )
            print("last epoch", eachEpochData[-1])
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

    #for i in range(epochs):
    #    print("epoch", i, statistics.mean([f['val_mean_absolute_error'][i] for f in histories]), "stdev", statistics.stdev([f['val_mean_absolute_error'][i] for f in histories]) )


def parseArgs():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=str)
    parser.add_argument('--w1', required=True, type=int)
    parser.add_argument('--w2', required=True, type=int)
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    arg=parseArgs()
    print("ARGS", arg)
    data =parseData( arg.input)
    model=makeModel( 512, wide1=arg.w1, wide2=arg.w2 )
    training( data, model, epochs=30)

