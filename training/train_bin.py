#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings("ignore")

import math
from scipy.misc import logsumexp
import numpy as np
import tensorflow as tf

class NNDropout():
    def __init__(self, dropout_prob, mc):
        self.dropout_prob = dropout_prob
        self.mc = mc


    def _dropout(self, X):
        if self.mc:
            return tf.keras.layers.Dropout(self.dropout_prob)(X, training=True)
        else:
            return tf.keras.layers.Dropout(p)(X)

    def architecture(n_hidden, input_dim, dropout_prob, reg, actvn, ):

        inputs = tf.keras.layers.Input(shape=(input_dim, ))
        model = _dropout(inputs)

        for i in range(len(n_hidden)):
            model = tf.keras.layers.Dense(n_hidden[i], activation=actvn, W_regularizer=l2(reg))(model)
            model = _dropout(model)


        out = tf.keras.layers.Dense(1, activation='softmax')(model)

        model = tf.keras.Model(inputs=inputs,outputs=model)

        return model
