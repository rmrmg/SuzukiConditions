from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix
import gzip
import pickle as pic
from sklearn.utils.class_weight import compute_sample_weight
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Lambda, Concatenate, Multiply
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
from functools import partial
import logging
from my_neural_fgp_test_reduced import *
import yaml
import sys

from tensorflow import keras
 
class LossPrinter(Callback):
     def on_train_begin(self, logs={}):
         self.losses = []
         self.count = 0
         return
              
     def on_train_end(self, logs={}):
         return
                           
     def on_epoch_begin(self, logs={}):
         self.count+=1
         return
                                        
     def on_epoch_end(self, epoch, logs={}):
        return
                                                     
     def on_batch_begin(self, batch, logs={}):
        return
                                                                  
     def on_batch_end(self, batch, logs={}):
        loss_ = logs.get('loss')
        acc_ = logs.get('acc')
        self.losses.append((loss_,acc_))
        logging.info('Epoch: %5i  loss:%8.6f   acc: %8.6f'%(self.count, loss_, acc_))
        return

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
      h = BatchNormalization()(h)
      h = Dropout(drp)(h)
   out = Dense(output_size, activation='sigmoid', kernel_regularizer=l2(l2_val))(h)

   return Model(inputs=I, outputs=out)

def make_input(shape):
   if len(shape)==1:
      to_use=(None,)
   else:
      to_use=shape[1:]
   return Input(shape=to_use)


def make_gcnn_classifier(input_shapes, output_shape, model_config):
   ''' order: X_input, filters_input, nums_input, identity_input, adjacency_input 
   '''
   #training_data = ([X, graph_conv_filters, lens], Y)
   #X_input, filters_input, nums_input, identity_input, adjacency_input 
   N=len(input_shapes)
   assert N%2==0
   graph1_shapes = input_shapes[:int(N/2)]
   graph2_shapes = input_shapes[int(N/2):]

   graph1_inputs = [make_input(x) for x in graph1_shapes]
   graph2_inputs = [make_input(x) for x in graph2_shapes]

   max_atoms1 = graph1_shapes[1][1]
   max_atoms2 = graph2_shapes[1][1]

   #control parameters
   N_H = model_config.get('hidden_units', 128)
   N_H_mlp = model_config.get('hidden_units_mlp', 100)
   fgp_size = model_config.get('fgp_size', 50)
   dropout_prob = model_config.get('dropout', 0.031849402173891934)
   lr = model_config.get('lr', 1e-3)
   l2_val = model_config.get('l2', 1e-3)
   N_it = model_config.get('num_layers', 4)
   activation = model_config.get('activation', 'relu')
   drp_flag = model_config.get('dropout_flag', False)
 
   config_str = '\n'+yaml.dump(model_config)
   logging.info('Echo model_config:')
   logging.info(config_str)

   gcnn1_config = {'hidden_units':N_H, 'dropout':dropout_prob, 'lr':lr, 'l2':l2_val, 'num_layers':N_it, 'activation':activation, 'dropout_flag':drp_flag}
   gcnn1_config['concat'] = model_config.get('concat', True)
   gcnn1_config['fgp'] = model_config.get('fgp', True)
   gcnn1_config['fgp_size'] = fgp_size

   gcnn2_config = {}
   gcnn2_config.update(gcnn1_config)

   gcnn1_config['shapes']={'max_atoms':max_atoms1, 'features':graph1_shapes[0]}
   gcnn2_config['shapes']={'max_atoms':max_atoms2, 'features':graph2_shapes[0]}

   output1 = make_reduced_gcnn_module(graph1_inputs[0], graph1_inputs, gcnn1_config)
   output2 = make_reduced_gcnn_module(graph2_inputs[0], graph2_inputs, gcnn2_config)

   output = Concatenate()([output1, output2])

   mlp_hidden = Dense(N_H_mlp, activation='relu', kernel_regularizer=l2(l2_val))(output)
   if len(output_shape)==2:
      N_output=output_shape[1]
   else:
      N_output=1
   output_activation='sigmoid'
   metric='accuracy'
   loss_f='binary_crossentropy'
   output = Dropout(dropout_prob)(mlp_hidden, training=drp_flag)
   output = Dense(N_output, activation=output_activation)(mlp_hidden)#output)
   
   model = Model(inputs=graph1_inputs+graph2_inputs, outputs=output)
   model.compile(loss=loss_f, optimizer=Adam(lr=lr), metrics=[metric])

   return model, metric


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
   phi_Jdist = Lambda(lambda x: K.sum(K.log(1-x[1])*x[0], axis=0)/K.sum(x[0], axis=0))([phi_on_X, input_phi_D_on_X])
    
   #penalty
   mean_P_log_phi = Lambda(lambda x: K.mean(K.log(x), axis=0))(phi_on_P)
   log_mean_X_phi = Lambda(lambda x: K.log(K.mean(x, axis=0)))(phi_on_X)
   phi_Jpen = Lambda(lambda x: (1-x[0])/(K.relu(x[0]-x[1])+K.epsilon()))([mean_P_log_phi, log_mean_X_phi])
   phi_loss_val = Lambda(lambda x: K.mean(K.abs(x[0]+log4)*x[1]))([phi_Jdist, phi_Jpen])
    
   phi_wrapped = Model([input_phi_P, input_phi_X, input_phi_D_on_X], phi_loss_val)
   phi_wrapped.compile(optimizer=Adam(beta_1=0.5, beta_2=0.99), loss=dummy_loss)
     
   #D
   input_d_P, input_d_X, input_d_phi_on_X = [Input(shape=(size,)) for size in [input_size,input_size,1]]
   d_mlp = make_uncompiled_mlp(model_config)
   d_on_X = d_mlp(input_d_X)
   d_on_P = d_mlp(input_d_P)
   d_loss_val = Lambda(lambda x: -K.mean((K.mean(K.log(x[0]),axis=0) + K.sum(x[1]*x[2], axis=0)/K.sum(x[2],axis=0))) )([d_on_P, d_on_X, input_d_phi_on_X])
     
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
   try:
      CM = confusion_matrix(ytrue, p.round(0))
   except:
      return 0,0,0,0,0

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


def baseline(model_config, *train_stuff, train_config={}, graph=False):
   epochs = train_config.get('epochs', 50)
   batch = train_config.get('batch', 100)
   xtrain, ytrain, xtest, ytest = train_stuff

   if graph:
      model, _ = make_gcnn_classifier([xx.shape for xx in xtrain], ytrain.shape, model_config)
   else:
      model = make_uncompiled_mlp(model_config)
      model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
   w= compute_sample_weight('balanced', ytrain)
   if len(ytrain.shape)==1:
      w = np.where(ytrain==1,2*w, w)
   printer = LossPrinter()
   model.fit(xtrain, ytrain, validation_data=(xtest, ytest), sample_weight=w, epochs=epochs, batch_size=batch, verbose=True)#, callbacks=[printer])
   p = model.predict(xtest)
   d = model.evaluate(xtest, ytest)[-1]
   logging.info('   Accuracy: %8.3f'%d)

   if len(ytest.shape)==2:
      topk = categorical_stats(ytest,p)
      for i,k in enumerate(topk):
         logging.info(' top-%i: %8.3f'%(i,k))
      ys = [ytest[:,i] for i in range(ytest.shape[1])]
      ps = [p[:,i] for i in range(ytest.shape[1])]
   else:
      ys=[ytest]
      ps=[p]
   for i, y in enumerate(ys):   
      tpr, tnr, acc, bacc, f = tpr_tnr_stats(y, ps[i])
      #logging.info('  OUT : %i'%(i))
      #logging.info('   TPR: %8.3f'%(tpr))
      #logging.info('   TNR: %8.3f'%(tnr))
      #logging.info('     F: %8.3f'%(f))
      logging.info('  g_av: %8.3f'%(np.sqrt(tnr*tpr)))
   return p, model


def categorical_stats(ytrue, p):
   topk=[0,0,0]
   for k in range(len(topk)):
      k_acc = np.mean([ytrue[i].argmax() in np.argsort(x)[-(k+1): ] for i,x in enumerate(p)])  
      topk[k]=k_acc
   return topk


def prepare_data(name='rr_processed_data.pkz', desc=['ecfp6'], inp=['boronic','halogen'], out='solvent'):
   with gzip.open(name, 'rb') as f: data=pic.load(f)
   output = data['%s_enc'%out]
   arrays=[]
   for i in inp:
      if i in ['solvent', 'base']:
         arrays.append(data['%s_enc'%i])
      else:
         for d in desc:
            arrays.append(data['%s_%s'%(i,d)])
   if 'graphs' in desc:
      result = arrays[0][0], arrays[1][0]
   else:
      result = np.hstack(arrays)
   return result, output


def make_logger(logname=None, level=logging.INFO, logger = logging.getLogger()):
   formatter=logging.Formatter('%(asctime)s: %(levelname)s): %(message)s')

   handlers= [logging.StreamHandler(sys.stdout)]
   if logname!=None:
      handlers.append(logging.FileHandler(logname))
   for handler in handlers:
      handler.setFormatter(formatter)
      handler.setLevel(level)
      logger.addHandler(handler)
   logger.setLevel(level)
   return logger


def gz_pickle(name, stuff):
   with gzip.open(name, 'wb') as f:
      pic.dump(stuff, f)


def gz_unpickle(name):
   with gzip.open(name, 'rb') as f:
      return pic.load(f)


def load_config(name):
   if name==None:return {}
   with open(name, 'r') as f:
      raw = f.read()
   return yaml.load(raw)
