seedNum=10
import random, os, statistics, argparse, json
random.seed(seedNum)
import numpy
numpy.random.seed(seedNum)

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow as tf
tf.random.set_seed(seedNum)
from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn import preprocessing, decomposition, cluster, model_selection
#import keras
from keras import optimizers, regularizers
from keras.layers import Input, Dense
from keras.models import Model


def training(model, X, nfolds=5, bsize=20, epochs=50, fn='testresult'):
    kf5=model_selection.KFold(n_splits=nfolds)
    #kf5.get_n_splits(tab)
    randInit = tf.keras.initializers.RandomNormal()
    #X = preprocessing.scale(X)
    iniw = model.get_weights()
    initShapes = [ i.shape for i in iniw]
    #print("shapes", initShapes)
    eachFoldData = []
    for trainIdx, testIdx in kf5.split(X):
        train, test = X[trainIdx], X[testIdx]
        model.set_weights( [randInit(shape=x) for x in initShapes] )
        history= model.fit(train, train, epochs=epochs, batch_size=bsize, shuffle=True, validation_data=(test, test), verbose=2)
        eachFoldData.append( history.history)
    bestval=[None, None, None]
    fw=open(fn, 'w')
    for epochid in range(epochs):
        val= [ oneFold['val_mean_absolute_error'][epochid] for oneFold in eachFoldData]
        meanv = statistics.mean(val)
        stdv =  statistics.stdev(val)
        print(epochid+1, "val:", meanv, stdv, file=fw)
        if not(bestval[0]) or (meanv < bestval[0]):
            bestval = [meanv, stdv, epochid]
    fw.close()
    return bestval

def produce(model, X, encDim=2, hide1=30, act1='elu', act2='relu', epochs=None, bsize=20):
    model.fit(X, X, epochs=epochs, batch_size=bsize, shuffle=True, verbose=2)
    encoder = modelEnc(model, len(X[0]), encDim=encDim, hide1=hide1, act1=act1, act2=act2)
    res =  encoder.predict(X)
    #numpy.savetxt('encoded.smiles.csv', res, delimiter='\t' )
    return res

def getSmilesRep(fn, fplen=512, radius=3):
    data =[]
    for smi in open(fn):
        row = [ float(x) for x in AllChem.GetMorganFingerprintAsBitVect(AllChem.MolFromSmiles(smi),radius, nBits=fplen ).ToBitString() ]
        data.append(row)
    res= numpy.array( data)
    return res


def makeModel(inputDim, encDim=2, hide1=30, hide2=30, act1='elu', act2='relu'):
    #encoding_dim = int(sys.argv[2])  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
    # this is our input placeholder
    input_img = Input(shape=(inputDim,))
    # "encoded" is the encoded representation of the input
    encoded1 = Dense(hide1, activation=act1)(input_img)
    #encoded1 = Dropout(0.05)(encoded1)
    encoded = Dense(encDim, activation=act2)(encoded1)
    # "decoded" is the lossy reconstruction of the input
    decoded1 = Dense(hide2, activation=act2)(encoded)
    #decoded1 = Dropout(0.05)(decoded1)
    decoded = Dense(inputDim, activation=act1)(decoded1)
    encoder = Model(input_img, encoded)
    #raise
    # this model maps an input to its reconstruction
    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)

    #optim = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #optim = optimizers.RMSprop(lr=0.1, clipnorm=1.2)
    #optim = optimizers.Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
    optim = optimizers.Adam() #( clipnorm=1, lr=0.01,  amsgrad=True )
    autoencoder.compile(optimizer=optim, loss="mean_squared_error", metrics=['mean_absolute_error',] )
    return autoencoder

def modelEnc(model, inputDim, encDim=2, hide1=30, act1='elu', act2='relu'):
    input_img = Input(shape=(inputDim,))
    # "encoded" is the encoded representation of the input
    encoded1 = Dense(hide1, activation=act1)(input_img) 
    #encoded1 = Dropout(0.05)(encoded1)
    encoded = Dense(encDim, activation=act2)(encoded1)
    encModel = Model(input_img, encoded)
    print("ENNN", [x.shape for x in encModel.get_weights()])
    print("L", [x.shape for x in model.get_weights()], "LAY", model.layers[0].get_weights() )
    #print(model.layers[0].get_weights().shape, model.layers[1].get_weights().shape, model.layers[2].get_weights().shape )
    encModel.set_weights([x for i, x in enumerate(model.get_weights()) if i < len(encModel.get_weights())])
    return encModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=str)
    parser.add_argument('--n1', required=True, type=int)
    parser.add_argument('--n2', required=True, type=int)
    parser.add_argument('--act1', required=True, type=str)
    parser.add_argument('--act2', required=True, type=str)
    parser.add_argument('--encdim', required=True, type=int)
    parser.add_argument('--radius', required=True, type=int)
    parser.add_argument('--fplen', required=True, type=int)
    parser.add_argument('--produce', required=False, type=int)
    #parser.add_argument('--act3', required=True, type=str)
    args = parser.parse_args()
    print("ARGS", args)
    #raise
    data = getSmilesRep(args.input, fplen=args.fplen, radius=args.radius)
    print("DATA", data)
    if args.produce:
        model = makeModel(len(data[0]), encDim=args.encdim, hide1=args.n1, hide2=args.n2, act1=args.act1, act2=args.act2)
        numres = produce(model, data, encDim=args.encdim, hide1=args.n1, act1=args.act1, act2=args.act2, epochs=args.produce)
        res = dict()
        for idx,smi in enumerate(open(args.input)):
            smi=smi.strip()
            res[smi] = tuple([float(x) for x in numres[idx]])
        json.dump(res, open('smiEncoded.json', 'w'))
    else:
        model = makeModel(len(data[0]), encDim=args.encdim, hide1=args.n1, hide2=args.n2, act1=args.act1, act2=args.act2)
        fname =f"wyniki_autoenc_{args.encdim}dims_{args.n1}_{args.n2}_{args.act1}_{args.act2}_fp{args.fplen}_radius{args.radius}"
        res= training(model, data, nfolds=5, bsize=20, epochs=70, fn=fname)
        print(fname, res)