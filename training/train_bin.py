import warnings
warnings.filterwarnings("ignore")
from training.generator import Generator
import math
import numpy as np
import tensorflow as tf
from keras.regularizers import l2
from keras import Input
from keras.layers import Dropout
from keras.layers import Dense
from keras import Model
import keras
import tensorflow as tf
import time
import multiprocessing #to use all the system cpu cores
import xgboost as xgb
import pickle


from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

METRICS = [
      keras.metrics.AUC(name='auc'),
]


class NNDropout():
    """
            Constructor for the class implementing a neural network architecture
            @param mc             True in case of MC Dropout architecture and False for standard
                                  nurel network.
            @param activn_fn      Activation function to be used
    """
    def __init__(self, mc, actvn):
        self.mc = mc
        self.activn_fn = actvn

        
    def architecture(self, n_hidden, input_dim, dropout, reg):
        
        """
        Function to create the NN architecture
            @param n_hidden       Vector with the number of neurons for each
                                  hidden layer.
            @param input_dim      Dimension of the input features
            @param reg            Regularization parameter, can be defined using dropout_prob
            @param dropout_prob   Dropout rate for all the dropout layers in the
                                  network.
        """
        
        inputs = Input(shape=(input_dim,))
        inter = Dropout(dropout)(inputs, training=True)
        inter = Dense(n_hidden[0], activation=self.activn_fn, W_regularizer=l2(reg))(inter)
        for i in range(len(n_hidden) - 1):
            inter = Dropout(dropout)(inter, training=True)
            inter = Dense(n_hidden[i+1], activation=self.activn_fn, W_regularizer=l2(reg))(inter)
        inter = Dropout(dropout)(inter, training=True)
        outputs = Dense(1, W_regularizer=l2(reg), activation='sigmoid')(inter)
        model = Model(inputs, outputs)
        return model

    def model_runner(self, X_train, y_train, dropout_prob=0.10, n_epochs=10, tau=1.0, batch_size=1024, 
                lengthscale=1e-2, n_hidden=[100,150, 100]):
        
        """
        Function to run the model
            @param X_train      Matrix with the features for the training data.
            @param y_train      Vector with the target variables for the
                                training data.
            @param n_hidden     Vector with the number of neurons for each
                                hidden layer.
            @param n_epochs     Numer of epochs for which to train the
                                network. The recommended value 10 should be
                                enough.
            @param tau          Tau value used for regularization
        """
  
        input_dim = X_train.shape[1]
        N = X_train.shape[0]
        reg = lengthscale**2 * (1 - dropout_prob) / (2. * N * tau)


        print('Fitting the Dropout NN architecture...')

        model_mc_dropout = self.architecture(n_hidden=n_hidden, input_dim=input_dim, 
                                        dropout=dropout_prob, reg=reg)
        model_mc_dropout.compile(optimizer='adam', loss='binary_crossentropy', metrics=METRICS)
        
        #model_mc_dropout.fit(X_train, y_train, batch_size=batch_size, nb_epoch=n_epochs, verbose=0)
        
        
        train_generator = Generator(X_train, y_train, batch_size).generate()
        
        # Iterate the learning process
        start_time = time.time()
        
        model_mc_dropout.fit_generator(
                    generator = train_generator, 
                    steps_per_epoch = math.floor(X_train.shape[0]/batch_size), 
                    epochs = n_epochs,  
                    max_queue_size = 10, 
                    workers = multiprocessing.cpu_count(),
                    use_multiprocessing = True, 
                    shuffle = True,
                    initial_epoch = 0
        )
        
        self.running_time = time.time() - start_time
        print('Running Time for Training: ', self.running_time)

        return model_mc_dropout



#class deep ensemble
class DeepArch_clf():
    """
            Constructor for the class implementing a neural network architecture for deep ensemble
            @param X_train      Matrix with the features for the training data.
            @param y_train      Vector with the target variables for the
                                training data.
    """
    def __init__(self, X_train,  y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.mc = False
        self.activn_fn = 'relu'
        
    def gaussian_nll(self, y_true, y_pred):
        
        """
            Gaussian negative log likelihood

            Note: to make training more stable, we optimize
            a modified loss by having our model predict log(sigma^2)
            rather than sigma^2. 
        """

        y_true = tf.reshape(y_true, [-1])
        mu = y_pred[:, 0]
        si = y_pred[:, 1]
        loss = (si + tf.square(y_true - mu)/tf.math.exp(si)) / 2.0
        return tf.reduce_mean(loss)
    
    def architecture(self, n_hidden, input_dim, dropout_prob, reg):
        
        """
        Function to create the NN architecture
            @param n_hidden       Vector with the number of neurons for each
                                  hidden layer.
            @param input_dim      Dimension of the input features
            @param reg            Regularization parameter, can be defined using dropout_prob
            @param dropout_prob   Dropout rate for all the dropout layers in the
                                  network.
        """
        
        inputs = tf.keras.layers.Input(shape=(input_dim,))
        inter = tf.keras.layers.Dropout(dropout_prob)(inputs, training=self.mc)
        inter = tf.keras.layers.Dense(n_hidden[0], activation=self.activn_fn, 
                          kernel_regularizer=tf.keras.regularizers.l2(reg))(inter)
            
        for i in range(len(n_hidden) - 1):
            inter = tf.keras.layers.Dropout(dropout_prob)(inter, training=self.mc)
            inter = tf.keras.layers.Dense(n_hidden[i+1], activation=self.activn_fn,
                              kernel_regularizer=tf.keras.regularizers.l2(reg))(inter)
        
        inter = tf.keras.layers.Dropout(dropout_prob)(inter, training=self.mc)
        
        h_expec = tf.keras.layers.Dense(1)(inter)
        h_expec = tf.keras.layers.Activation('sigmoid')(h_expec)
        
        h_var = tf.keras.layers.Dense(1)(inter)
        h_var = tf.keras.layers.Activation('softplus')(h_var)
        #h_var = tf.keras.layers.Lambda(lambda x: x + 1e-6, output_shape=(1,))(h_var)
        
        oup = tf.keras.layers.Concatenate(axis=-1)([h_expec, h_var])

        # model
        model = tf.keras.Model(inputs=inputs, outputs=oup)

        return model

    def fit_model(self,dropout_prob=0.10, n_epochs=2, tau=1.0, batch_size=1024, 
                lengthscale=1e-2, n_hidden=[100,150, 100]):
        
        """
        Function to run the model
            @param n_hidden     Vector with the number of neurons for each
                                hidden layer.
            @param n_epochs     Numer of epochs for which to train the
                                network. The recommended value 10 should be
                                enough.
            @param tau          Tau value used for regularization
        """
  
        input_dim = self.X_train.shape[1]
        N = self.X_train.shape[0]
        
        reg =  4.5000000000000003e-07


        print('Fitting NN...')

        model_de = self.architecture(n_hidden=n_hidden, input_dim=input_dim, 
                                        dropout_prob=dropout_prob, reg=reg)
        model_de.compile(loss=self.gaussian_nll, optimizer='nadam', metrics=['accuracy'])
        



        train_generator = Generator(self.X_train, self.y_train, batch_size).generate()
        
        # Iterate the learning process
        start_time = time.time()
        
        model_de.fit_generator(
                    generator = train_generator, 
                    steps_per_epoch = math.floor(self.X_train.shape[0]/batch_size), 
                    epochs = n_epochs,  
                    max_queue_size = 10, 
                    workers = multiprocessing.cpu_count(),
                    use_multiprocessing = True, 
                    shuffle = True,
                    initial_epoch = 0
        )
        
        self.running_time = time.time() - start_time
        print('Running Time for Training: ', self.running_time)

        return model_de


import xgboost as xgb
from xgboost import XGBClassifier

class XGBMulti():
    """
            Constructor for the class implementing a XGBoost architecture
    """
    def __init__(self, X_train, y_train, rsampls, csampls):
        self.rsampls = rsampls
        self.csampls = csampls
        self.X_train = X_train
        self.y_train = y_train

        
    def architecture(self, lr, nes, max_depth, min_child_weight, rs, cs, savePATH):
        model = XGBClassifier(
            learning_rate=lr,
            n_estimators=nes,
            max_depth=max_depth,
            min_child_weight=min_child_weight, 
            gamma=3,
            subsample=rs,
            colsample_bytree=cs,
            objective= 'binary:logistic',
            scale_pos_weight=15,
            seed=27,
            )
        model.fit(self.X_train, self.y_train, verbose=True)
        pickle.dump(model, open(savePATH, 'wb'))
        print('model saved at: ', savePATH)
        
    def get_models(self, lr, nes, max_depth=6, min_child_weight=5):
        countr = 0
        for subsample in self.rsampls:
            for colsample in self.csampls:
                print("Training with subsample={}, colsample={}".format( subsample, colsample))
                model_path = 'trained_models/XGBclf/'+ str(countr)+'_th.h5'
                print('model saved at::', model_path)
                countr+=1
                self.architecture(lr, nes, max_depth, min_child_weight, subsample, colsample, model_path)