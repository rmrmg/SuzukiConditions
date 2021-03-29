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
    solv2 =tabOryg[:,1]
    solv3 =tabOryg[:,2]
    solv4 =tabOryg[:,3]
    base1 =tabOryg[:,4]
    base2 =tabOryg[:,5]
    ligand1=tabOryg[:,6]
    ligand2=tabOryg[:,7]
    temp = tabOryg[:,8]
    sbs1 = tabOryg[:, 9:9+512]
    sbs2 = tabOryg[:, 9+512:9+512+512]
    yld = tabOryg[:,-1]
    return {'solvents':[solv1, solv2, solv3, solv4], 'bases':[base1, base2], 'ligands':[ligand1, ligand2], 'temp':temp, 'sbses':[sbs1,sbs2], 'yield':yld }


def makeModel(inputDim, wide1=90, wide2=10, embDim=3, solventClasses=1+54, baseClasses=1+72, ligandClasses=1+81, act1='relu', act2='relu', act3='elu' ):
    subs1 = Input(shape=(inputDim,))
    subs2 = Input(shape=(inputDim,))
    temper = Input(shape=(1,) )
    sol1 = Input(shape=(1,) )
    sol2 = Input(shape=(1,) )
    sol3 = Input(shape=(1,) )
    sol4 = Input(shape=(1,) )

    base_1 = Input(shape=(1,) )
    base_2 = Input(shape=(1,) )

    lgand1 = Input(shape=(1,) )
    lgand2 = Input(shape=(1,) )

    solventEmbd = Embedding(solventClasses, embDim, input_length=1)
    #solventEmbd = Dense(2, activation='relu')
    solvent1 = solventEmbd(sol1)
    solvent2 = solventEmbd(sol2)
    solvent3 = solventEmbd(sol3)
    solvent4 = solventEmbd(sol4)
    baseEmbd = Embedding(baseClasses, embDim, input_length=1)
    #baseEmbd = Dense(2, activation='relu')
    base1 = baseEmbd(base_1)
    base2 = baseEmbd(base_2)
    ligandEmbd = Embedding(ligandClasses, embDim, input_length=1)
    #ligandEmbd = Dense(2, activation='relu')
    ligand1 = ligandEmbd(lgand1)
    ligand2 = ligandEmbd(lgand2)

    solvent = Add()([solvent1, solvent2, solvent3, solvent4])
    base = Add()([base1, base2])
    ligand = Add()([ligand1, ligand2])
    conditions =Concatenate()([solvent,base, ligand])
    conditions =Flatten()(conditions)
    conditions = Concatenate()([conditions, temper])
    sbs1 = Dense(wide1, activation=act1)(subs1)
    sbs2 = Dense(wide1, activation=act1)(subs2)
    conditionsAndSubstrate = Concatenate() ([conditions, sbs1,sbs2])

    hide9 = Dense(wide2, activation=act2)(conditionsAndSubstrate)
    hide9 = Dropout(0.05)(hide9)
    outyield = Dense(1, activation=act3)(hide9)
    model = Model((sol1,sol2,sol3,sol4,base_1,base_2, lgand1,lgand2, temper, subs1, subs2), outyield)

    optim = optimizers.Adam() # lr=0.0005) #( clipnorm=1, lr=0.01,  amsgrad=True ) lr:=default:=0.001
    model.compile(optimizer=optim, loss='mean_squared_error', metrics=["mean_absolute_error",])
    #model.compile(optimizer=optim, loss='mean_squared_error', metrics=["mean_absolute_error", customLoss])
    model.summary()
    return model


def training(data, model,  nfolds=5, epochs=30):
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
        solvent2train= data['solvents'][1][trainIdx]
        solvent2test= data['solvents'][1][testIdx]
        solvent3train= data['solvents'][2][trainIdx]
        solvent3test= data['solvents'][2][testIdx]
        solvent4train= data['solvents'][3][trainIdx]
        solvent4test= data['solvents'][3][testIdx]

        base1train= data['bases'][0][trainIdx]
        base1test= data['bases'][0][testIdx]
        base2train= data['bases'][1][trainIdx]
        base2test= data['bases'][1][testIdx]

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
        inputTrain = [ solvent1train, solvent2train, solvent3train, solvent4train, base1train, base2train, ligand1train, ligand2train, temptrain, sbs1train, sbs2train]
        inputTest = [solvent1test, solvent2test, solvent3test, solvent4test, base1test, base2test, ligand1test, ligand2test, temptest, sbs1test, sbs2test]

        history=model.fit(inputTrain, Yldtrain, epochs=epochs, batch_size=20, shuffle=True, validation_data=(inputTest, Yldtest), verbose=2)
        histories.append( history.history)
    for i in range(epochs):
        print("epoch", i, statistics.mean([f['val_mean_absolute_error'][i] for f in histories]), "stdev", statistics.stdev([f['val_mean_absolute_error'][i] for f in histories]) )


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

