import warnings
warnings.filterwarnings("ignore")
from training.generator import Generator

import time
import math
import multiprocessing #to use all the system cpu cores
import math
import numpy as np
import tensorflow as tf


class NNDropout_rgr():
    """
            Constructor for the class implementing a neural network architecture
            @param mc             True in case of MC Dropout architecture and False for standard
                                  nurel network.
            @param activn_fn      Activation function to be used
    """
    def __init__(self, mc, actvn):
        self.mc = mc
        self.activn_fn = actvn

        
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
        outputs = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(reg))(inter) 
        model = tf.keras.Model(inputs, outputs)
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


        print('Fitting NN...')

        model_mc_dropout = self.architecture(n_hidden=n_hidden, input_dim=input_dim, 
                                        dropout_prob=dropout_prob, reg=reg)
        model_mc_dropout.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
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
        print('Running Time: ', self.running_time)

        return model_mc_dropout