from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Lambda
from keras.regularizers import l2
from keras.optimizers import Adam
from keras import backend as K
from functools import partial
import logging

def make_dataset(test_config={}):
   noise = test_config.get('noise', 0.15)
   size = test_config.get('size',10000)
   factor = test_config.get('factor',0.3)
   x,y = make_moons(size, noise=noise)
   x2, y2 = make_circles(size, noise=noise, factor=factor)
   return np.hstack([x,x2]), np.hstack([y.reshape(-1,1), y2.reshape(-1,1)])


def relabel_positive(y, fraction=0.1):
   if len(y.shape)==2:
      Ncol=y.shape[1]
      return np.hstack([relabel_positive(y[:,i], fraction).reshape(-1,1) for i in range(Ncol)])
   positive_idx = np.where(y==1)[0]
   sample = int(len(positive_idx)*fraction)
   to_change = np.random.choice(positive_idx, sample, replace=False)
   y2 = np.ones(y.shape).astype(int)*y
   y2[to_change] = 0
   return y2


def Jpen(y_pred, y_true):
   log_phi_p = K.sum(K.log(y_pred)*y_true, axis=0)/(K.sum(y_true,axis=0)+K.epsilon())
   log_phi = K.log(K.mean(y_pred, axis=0))
   return K.sum((1-log_phi_p)/(K.epsilon() + K.relu(log_phi_p-log_phi)))

log4 = K.log(4.0)

#for phi, D goes as sample_weights
def phi_loss(y_true, y_pred, sample_weights):
   J = (K.sum((y_pred*K.log(1-sample_weights)), axis=0) + log4)/(K.sum(y_pred, axis=0)+K.epsilon())
   J = K.abs(J)*Jpen(y_pred, y_true)
   return K.sum(J)

nprelu = lambda x: np.where(x>0,x,0)

def Jpen2(y_pred, y_true):
   log_phi_p = np.sum(np.log(y_pred+K.epsilon())*y_true, axis=0)/(np.sum(y_true,axis=0)+K.epsilon())
   log_phi = np.log(np.mean(y_pred, axis=0)+K.epsilon())
   return (1-log_phi_p)/(K.epsilon() + nprelu(log_phi_p-log_phi))


def phi_loss2(y_true, y_pred, sample_weights):
   J = (np.sum((y_pred*np.log(1-sample_weights+K.epsilon())), axis=0) + log4)/(np.sum(y_pred, axis=0)+K.epsilon())
   J = np.abs(J)*Jpen2(y_pred, y_true)
   return J


#for D, phi goes as sample_weights
def d_loss(y_true, y_pred, sample_weights):
   BP = K.sum(K.log(y_pred)*y_true, axis=0)/(K.sum(y_true, axis=0) + K.epsilon())
   BX = K.sum(K.log(1-y_pred)*sample_weights, axis=0)/(K.sum(sample_weights, axis=0) + K.epsilon())
   return K.sum(-(BP+BX))


def d_loss2(y_true, y_pred, sample_weights):
   BP = np.sum(np.log(y_pred+K.epsilon())*y_true, axis=0)/(np.sum(y_true, axis=0)+K.epsilon())
   BX = np.sum(np.log(1-y_pred+K.epsilon())*sample_weights, axis=0)/(np.sum(sample_weights, axis=0)+K.epsilon())
   return -(BP+BX)


def make_uncompiled_mlp(model_config):
   input_size = model_config.get('input_size', 2048)
   output_size = model_config.get('output_size', 1)
   n_layers = model_config.get('layers', 1)
   n_hidden = model_config.get('hidden',128)
   drp = model_config.get('dropout', 0.2)
   l2_val = model_config.get('l2val', 0.001)
   act_hidden = model_config.get('activation_hidden', 'relu')
    
   I = Input(shape=(input_size,))
   h = Dense(n_hidden, kernel_regularizer=l2(l2_val), activation=act_hidden)(I)
   h = Dropout(drp)(h)
   for _ in range(n_layers-1):
      h = Dense(n_hidden, kernel_regularizer=l2(l2_val), activation=act_hidden)(h)
      h = Dropout(drp)(h)
   out = Dense(output_size, activation='sigmoid', kernel_regularizer=l2(l2_val))(h)

   return Model(inputs=I, outputs=out)


def make_dan_pair(model_config):
   phi_model = make_uncompiled_mlp(model_config)
   #phi_model.compile(optimizer='adam', loss=phi_loss)
   phi_w = Input(shape=(None,))
   philoss = partial(phi_loss, sample_weights=phi_w)
   phi_wrapped = Model([phi_model.inputs[0], phi_w], phi_model.outputs[0])
   phi_wrapped.compile(optimizer='adam', loss=philoss)

   d_w = Input(shape=(None,))
   dloss = partial(d_loss, sample_weights=d_w)
   d_mlp = make_uncompiled_mlp(model_config)
   d_model = Model([d_mlp.inputs[0], d_w], d_mlp.outputs[0])
   d_model.compile(optimizer='adam', loss=dloss)
    
   return phi_model, phi_wrapped, d_mlp, d_model

   
def make_dan_iteration(model_phi, model_phi_wrapped, _, model_d, Xtrain, ytrain, train_config={}):
   epochs = train_config.get('epochs', 1)
   batch = train_config.get('batch', 100)
    
   phi_pred = model_phi.predict(Xtrain)
   
   #print(phi_pred.mean())
   #print(model_d.evaluate([Xtrain, phi_pred], ytrain))
   pd=model_d.predict([Xtrain, phi_pred])
   #print('disc: ',pd.mean())
   #print('dloss: ',d_loss2(ytrain, pd, phi_pred))

   #logging.info('Discriminator:')
   model_d.fit([Xtrain, phi_pred], ytrain, epochs=epochs, batch_size=batch, verbose=False)
   d_pred = model_d.predict([Xtrain, phi_pred])
   
   #logging.info('Predictor:')
   #print(d_pred.mean())
   #print(Jpen2(ytrain,phi_pred))
   #print(phi_loss2(ytrain, phi_pred, d_pred))

   history = model_phi_wrapped.fit([Xtrain, d_pred], ytrain, epochs=epochs, batch_size=batch, verbose=False)
   return history.history['loss'][-1]


#more accurate

def dummy_loss(y_true, y_pred):
   return y_pred

def make_dan_pair2(model_config):
   #PHI
   input_size = model_config.get('input_size', 2)
   input_phi_P, input_phi_X, input_phi_D_on_X = [Input(shape=(size,)) for size in [input_size,input_size,1]]
    
   phi_model = make_uncompiled_mlp(model_config)
   phi_on_P = phi_model(input_phi_P)
   phi_on_X = phi_model(input_phi_X)
   phi_Jdist = Lambda(lambda x: K.sum(K.log(1-x[1])*x[0])/K.sum(x[0]))([phi_on_X, input_phi_D_on_X])
    
   #penalty
   mean_P_log_phi = Lambda(lambda x: K.mean(K.log(x)))(phi_on_P)
   log_mean_X_phi = Lambda(lambda x: K.log(K.mean(x)))(phi_on_X)
   phi_Jpen = Lambda(lambda x: (1-x[0])/(K.relu(x[0]-x[1])+K.epsilon()))([mean_P_log_phi, log_mean_X_phi])
   phi_loss_val = Lambda(lambda x: K.abs(x[0]+log4)*x[1])([phi_Jdist, phi_Jpen])
    
   phi_wrapped = Model([input_phi_P, input_phi_X, input_phi_D_on_X], phi_loss_val)
   phi_wrapped.compile(optimizer=Adam(beta_1=0.5, beta_2=0.99), loss=dummy_loss)
     
   #D
   input_d_P, input_d_X, input_d_phi_on_X = [Input(shape=(size,)) for size in [input_size,input_size,1]]
   d_mlp = make_uncompiled_mlp(model_config)
   d_on_X = d_mlp(input_d_X)
   d_on_P = d_mlp(input_d_P)
   d_loss_val = Lambda(lambda x: -(K.mean(K.log(x[0])) + K.sum(x[1]*x[2])/K.sum(x[2])) )([d_on_P, d_on_X, input_d_phi_on_X])
     
   d_model = Model([input_d_P, input_d_X, input_d_phi_on_X], d_loss_val)
   d_model.compile(optimizer=Adam(beta_1=0.5, beta_2=0.99), loss=dummy_loss)
    
   return phi_model, phi_wrapped, d_mlp, d_model


def make_dan_iteration2(model_phi, model_phi_wrapped, d_mlp, model_d, Xtrain, ytrain, train_config={}):
   #select batch
   batch = train_config.get('batch', 100)
   P_idx = np.where(ytrain==1)[0]
   all_idx = np.arange(len(ytrain))
    
   Xi = np.random.choice(all_idx, batch, replace=False)
   Pi = np.random.choice(P_idx, batch, replace=False)
   X = Xtrain[Xi]
   P = Xtrain[Pi]

   phi_pred = model_phi.predict(X)
   
   model_d.train_on_batch([P, X, phi_pred], Xi)
   d_pred = d_mlp.predict(X)
   
   val_loss = model_phi_wrapped.train_on_batch([P, X, d_pred], Xi)
   return val_loss


def tpr_tnr_stats(ytrue, p):
   CM = confusion_matrix(ytrue, p.round(0))

   TN = CM[0][0]
   FN = CM[1][0]
   TP = CM[1][1]
   FP = CM[0][1]

   tpr = TP/(TP+FN)
   tnr = TN/(TN+FP)
   bacc = 0.5*(tpr+tnr)
   tot = (TP+TN+FP+FN)
   acc = (TP+TN)/tot
   p = (TP+FP)/tot
   f = tpr**2/p
   return tpr, tnr, acc, bacc,f


def baseline(model_config, *train_stuff, train_config={}):
   epochs = train_config.get('epochs', 100)
   batch = train_config.get('batch', 100)
   xtrain, ytrain, xtest, ytest = train_stuff

   model = make_uncompiled_mlp(model_config)
   w= compute_sample_weight('balanced', ytrain)
   model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
   model.fit(xtrain, ytrain, sample_weight=w, epochs=epochs, batch_size=batch, verbose=False)
   p=model.predict(xtest)
   d = model.evaluate(xtest, ytest)[-1]
   logging.info('   Accuracy: %8.3f'%d)

   if len(ytest.shape)==2:
      ys = [ytest[:,i] for i in range(ytest.shape[1])]
      ps = [p[:,i] for i in range(ytest.shape[1])]
   else:
      ys=[ytest]
      ps=[p]
   for i, y in enumerate(ys):   
      tpr, tnr, acc, bacc, f = tpr_tnr_stats(y, ps[i])
      logging.info('  OUT : %i'%(i))
      logging.info('   TPR: %8.3f'%(tpr))
      logging.info('   TNR: %8.3f'%(tnr))
      logging.info('     F: %8.3f'%(f))
      logging.info('  g_av: %8.3f'%(np.sqrt(tnr*tpr)))


if __name__=='__main__':
   import argparse
   parser = argparse.ArgumentParser()
   parser.add_argument('--th', type=float, default=0.2 )
   parser.add_argument('--epochs', type=int, default=100 )
   parser.add_argument('--report', type=int, default=10 )
   parser.add_argument('--baseline', action='store_true' )
   parser.add_argument('--random', action='store_true' )
   parser.add_argument('--mode', type=int, default=1, choices=[1,2] )
   args=parser.parse_args()

   from sklearn.ensemble import RandomForestClassifier
   from scipy.spatial.distance import pdist, squareform
   logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(message)s')   
   #make dataset
   X, Y = make_dataset(dict(noise=0.2))
   #Yfake = relabel_positive(Y, fraction=0.2)

   # use distances to relabel stuff close to the decision boundary
   if args.random:
      Yfake = relabel_positive(Y, fraction=0.2)
   else:
      Yfake = np.ones(Y.shape)*Y
      for i in range(Y.shape[1]):
         D = squareform(pdist(X[:,2*i:2*(i+1)]))
         D[:, Y[:,i]==1] = 0.0
         D = np.where((D>0) & (D<args.th), 1, 0)
         D = D.sum(axis=1) 
         idx = (D>0) & (Y[:,i]==1)
         Yfake[idx,i]=0
   logging.info('True: '+str(Y.sum(axis=0)))
   logging.info('Fake: '+str(Yfake.sum(axis=0)))
   logging.info('Corruption:  ' +str((1-Yfake.sum(axis=0)/Y.sum(axis=0))))
   
   model_config = {'input_size':X.shape[1], 'output_size':Y.shape[1]}

   #true_baseline 
   Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, stratify=Y)
   if args.baseline:
      logging.info('Non-relabeled baseline')
      baseline(model_config, Xtrain, Ytrain, Xtest, Ytest)

   #relabelled_baseline
   Xtrain, Xtest, Ytrain, Ytest, Yf_train, Yf_test = train_test_split(X, Y, Yfake, test_size=0.2, stratify=Yfake)
   if args.baseline:
      logging.info('Relabeled baseline')
      baseline(model_config, Xtrain, Yf_train, Xtest, Ytest)

   #DAN part
   if args.mode==2:
      make_dan_pair = make_dan_pair2
      make_dan_iteration = make_dan_iteration2

   phi, phi_wr, d_mlp, disc = make_dan_pair(model_config)
   phi.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
   phi.fit(Xtrain, Yf_train, epochs=5, batch_size=100, verbose=False)
   logging.info('order: tpr tnr bacc f')
   for meta_epoch in range(args.epochs):
      loss= make_dan_iteration(phi, phi_wr, d_mlp, disc, Xtrain, Yf_train)
      ptrain = phi.predict(Xtrain)
      ptest = phi.predict(Xtest)
      #norm = ptrain.max()
      #print(norm)
      #ptrain/=norm
      #ptest/=norm
      if len(Ytest.shape)==2:
         set_train = [(Ytrain[:,i], ptrain[:,i]) for i in range(Ytest.shape[1])]
         set_test = [(Ytest[:,i], ptest[:,i]) for i in range(Ytest.shape[1])]
      else:
         set_train = [(Ytrain, ptrain)]
         set_test = [(Ytest, ptest)]

      for i, train_result in enumerate(set_train):
         train_stats = tpr_tnr_stats(*train_result)
         test_stats = tpr_tnr_stats(*set_test[i])
         if meta_epoch%args.report==0:
            logging.info('ME: %5i out:%i loss:%7.4f  train: %5.3f %5.3f %5.3f f:%5.3f  test: %5.3f %5.3f %5.3f f:%5.3f'%(
            meta_epoch, i, loss, train_stats[0], train_stats[1], train_stats[3], train_stats[4], test_stats[0], test_stats[1], test_stats[3], test_stats[4]))

