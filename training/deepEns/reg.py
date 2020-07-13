import warnings
warnings.filterwarnings("ignore")

import time
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

class DeepArch_reg():
    """
            Constructor for the class implementing a Deep Ensemble architecture
    """
    def __init__(self, X_train,  y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.mc = False
        self.activn_fn = 'tanh'

        
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
        outputs = tf.keras.layers.Dense(2, activation=None)(inter) 
        model = tf.keras.Model(inputs, outputs)
        return model

    def fit_model(self,dropout_prob=0.10, n_epochs=10, tau=1.0, batch_size=1024, 
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
  
        input_dim = self.X_train.shape[1]
        N = self.X_train.shape[0]
        
        reg =  4.5000000000000003e-07


        print('Fitting NN...')

        model_mc_dropout = self.architecture(n_hidden=n_hidden, input_dim=input_dim, 
                                        dropout_prob=dropout_prob, reg=reg)
        model_mc_dropout.compile(optimizer='adam', loss=self.gaussian_nll, metrics=['mean_squared_error'])
        
        
        # Iterate the learning process
        start_time = time.time()
        
        model_mc_dropout.fit(
            self.X_train, self.y_train,
            batch_size=batch_size,
            epochs=n_epochs,
            verbose=1,
            validation_data=(self.X_val, self.y_val),
            callbacks=[es]
        )
        
        self.running_time = time.time() - start_time
        print('Running Time: ', self.running_time)

        return model_mc_dropout
