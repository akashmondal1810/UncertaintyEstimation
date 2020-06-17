#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd


class Generator():
    """
    A custom data generator
    """
    def __init__(self, X_data, y_data, batch_size):
        self.X_sample = X_data
        self.y_sample = y_data
        self.batch_size = batch_size

    def generate(self):

    	samples_per_epoch = self.X_sample.shape[0]
    	number_of_batches = samples_per_epoch/self.batch_size
    
	    counter=0
	    
	    while True:

	    	X_batch = self.X_sample[self.batch_size*counter:self.batch_size*(counter+1)]
	        X_batch = np.array(X_batch).astype('float32')

	        y_batch = self.y_sample[self.batch_size*counter:self.batch_size*(counter+1)]
	        y_batch = np.array(y_batch).astype('float32')
	        
	        counter += 1
	        yield X_batch,y_batch

	    #restart counter to yeild data in the next epoch as well
	    if counter >= number_of_batches:
	    	counter = 0
