import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np
import tensorflow as tf


class NNDropout():
    """
            Constructor for the class implementing a neural network architecture
            @param n_hidden       Vector with the number of neurons for each
                                  hidden layer.
            @param input_dim      Dimension of the input features
            @param reg            Regularization parameter, can be defined using dropout_prob

            @param dropout_prob   Dropout rate for all the dropout layers in the
                                  network.
            @param mc             True in case of MC Dropout architecture and False for standard
                                  nurel network.
            @param activn_fn      Activation function to be used
    """
    def __init__(self, n_hidden, input_dim, reg, dropout_prob, mc, actvn):
        self.n_hidden = n_hidden
        self.input_dim = input_dim
        self.reg = reg
        self.dropout_prob = dropout_prob
        self.mc = mc
        self.activn_fn = actvn

    def _dropout(self, X):
        if self.mc:
            return tf.keras.layers.Dropout(self.dropout_prob)(X, training=True)
        else:
            return tf.keras.layers.Dropout(self.dropout_prob)(X)

    def architecture(self, optimizer, loss_fn, evl_metrics):
        
        """
            Function for creating the network architecture.
            @return model   The the constructed network
        """

        inputs = tf.keras.layers.Input(shape=(self.input_dim, ))
        model = self._dropout(inputs)
        model = tf.keras.layers.Dense(self.n_hidden[0], activation=self.activn_fn, W_regularizer=l2(self.reg))(model)

        for i in range(len(self.n_hidden)-1):
            model = self._dropout(model)
            model = tf.keras.layers.Dense(self.n_hidden[i+1], activation = self.activn_fn, W_regularizer=l2(self.reg))(model)
            
        model = self._dropout(model)
        #out = tf.keras.layers.Dense(1, activation='softmax')(model)
        out = tf.keras.layers.Dense(1, W_regularizer=l2(self.reg))(model)

        model = tf.keras.Model(inputs=inputs,outputs=model)
        model.compile(optimizer = optimizer, loss = loss_fn, metrics = evl_metrics)

        return model
    
