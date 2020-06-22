#!/usr/bin/env python
# coding: utf-8

from training.mc_dropout_multi import NNexperiment_multi

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
import sys
import json
import tensorflow as tf

class LoadMNIST():

    def __init__(self):
    	self.x_train = None
    	self.x_val = None
    	self.x_test = None

    	self.y_train = None
    	self.y_val = None
    	self.y_test = None

    def loadData(self):
    	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    	num_classes = 10

    	x_train = x_train.reshape(60000, 784)
    	x_test = x_test.reshape(10000, 784)
    	x_train = x_train.astype('float32')
    	x_test = x_test.astype('float32')
    	x_train /= 255
    	x_test /= 255

    	# convert class vectors to binary class matrices
    	y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    	y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    	x_val, x_test = np.array_split(x_test, 2)
    	y_val, y_test = np.array_split(y_test, 2)

    	print(x_train.shape[0], 'train samples')
    	print(x_test.shape[0], 'test samples')
    	print(x_val.shape[0], 'val samples')

    	self.x_train = x_train
    	self.x_val = x_val
    	self.x_test = x_test

    	self.y_train = y_train
    	self.y_val = y_val
    	self.y_test = y_test

    	np.save('data/mnist/x_test.npy', x_test)
    	np.save('data/mnist/y_test.npy', y_test)

    	return x_train, y_train, x_val, y_val



if __name__ == '__main__':
	datasetName = sys.argv[1]

	if datasetName=='MNIST':
		x_train, y_train, x_val, y_val = LoadMNIST().loadData()

		nnarch = NNexperiment_multi(x_train, y_train, x_val, y_val, 'training/training_strategy.json', num_class=10)
		t=100
		model = nnarch.find_best_network()
    