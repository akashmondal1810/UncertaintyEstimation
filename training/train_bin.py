#!/usr/bin/env python
# coding: utf-8


"""
This script deals with the preprocessing of data and getting it 
ready for the machine learning algorithms. The main topics are:
1. Dealing with missing data
2. Handling categorical data
3. Partitioning dataset into training and test subsets.
4. Bringing features on the same scale.
5. Selecting meaningful features.
6. Regularizing data.
"""

import warnings
warnings.filterwarnings("ignore")

import math
from scipy.misc import logsumexp
import numpy as np
import tensorflow as tf


"""

def get_dropout(input_tensor, p=0.5, mc=False):
    if mc:
        return Dropout(p)(input_tensor, training=True)
    else:
        return Dropout(p)(input_tensor)

def generate_dropout_model(mc=False, act="relu"):
    # We construct the network
    inp = Input(shape=(784,))
    x = get_dropout(inp, p=0.05, mc=mc)
    x = Dense(1000, activation=act)(x)
    x = get_dropout(x, p=0.05, mc=mc)
    x = Dense(500, activation=act)(x)
    x = get_dropout(x, p=0.05, mc=mc)
    x = Dense(100, activation=act)(x)
    x = get_dropout(x, p=0.05, mc=mc)
    out = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inp, outputs=out)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

"""

class NNDropout():
    """
    Preprocess strategies defined and exected in this class
    """
    def __init__(self, dropout_prob, mc):
        self.dropout_prob = dropout_prob
        self.mc = mc


    def _dropout(self, X, p=self.dropout_prob, mc = self.mc):
        if mc:
            return tf.keras.layers.Dropout(p)(X, training=True)
        else:
            return tf.keras.layers.Dropout(p)(X)

    def mc_dropout(n_hidden, input_dim, dropout_prob, reg, mc=True, actvn):

        inputs = tf.keras.layers.Input(shape=(input_dim, ))
        model = _dropout(inputs, p=dropout_prob, mc=mc)

        for i in range(len(n_hidden)):
            model = tf.keras.layers.Dense(n_hidden[i], activation=actvn, W_regularizer=l2(reg))(model)
            model = _dropout(model, p=dropout_prob, mc=mc)


        out = tf.keras.layers.Dense(num_classes, activation='softmax')(model)

        model = tf.keras.Model(inputs=inputs,outputs=model)

        return model