#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import numpy as np
import argparse
import sys


parser=argparse.ArgumentParser()

parser.add_argument('--dir', '-d', required=True, help='Name of the UCI Dataset directory. Eg: bostonHousing')

args=parser.parse_args()

data_directory = args.dir


# We delete previous results

from subprocess import call


_DATA_DIRECTORY_PATH = "./UCI_Datasets/" + data_directory + "/data/"
_DATA_FILE = _DATA_DIRECTORY_PATH + "data.txt"
_HIDDEN_UNITS_FILE = _DATA_DIRECTORY_PATH + "n_hidden.txt"
_EPOCHS_FILE = _DATA_DIRECTORY_PATH + "n_epochs.txt"
_INDEX_FEATURES_FILE = _DATA_DIRECTORY_PATH + "index_features.txt"
_INDEX_TARGET_FILE = _DATA_DIRECTORY_PATH + "index_target.txt"
_N_SPLITS_FILE = _DATA_DIRECTORY_PATH + "n_splits.txt"

def _get_index_train_test_path(split_num, train = True):
    """
       Method to generate the path containing the training/test split for the given
       split number (generally from 1 to 20).
       @param split_num      Split number for which the data has to be generated
       @param train          Is true if the data is training data. Else false.
       @return path          Path of the file containing the requried data
    """
    if train:
        return _DATA_DIRECTORY_PATH + "index_train_" + str(split_num) + ".txt"
    else:
        return _DATA_DIRECTORY_PATH + "index_test_" + str(split_num) + ".txt" 

# We fix the random seed

np.random.seed(1)

print ("Loading data and other hyperparameters...")
# We load the data

data = np.loadtxt(_DATA_FILE)

# We load the indexes for the features and for the target

index_features = np.loadtxt(_INDEX_FEATURES_FILE)
index_target = np.loadtxt(_INDEX_TARGET_FILE)

X = data[ : , [int(i) for i in index_features.tolist()] ]
y = data[ : , int(index_target.tolist()) ]

# We iterate over the training test splits

n_splits = np.loadtxt(_N_SPLITS_FILE)
print ("Done.")


# In[2]:


import xgboost as xgb
def get_model(rs, cs, model_path, X_train, y_train, X_val, y_val):
    
    print('training....')
    model = xgb.XGBRegressor(
        learning_rate=0.5,
        n_estimators=100,
        max_depth=3,
        min_child_weight=1, 
        subsample=rs,
        colsample_bytree=cs,
        objective= 'reg:linear',
        seed=27,
    )
    model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric='mae', eval_set=[(X_val, y_val)], verbose=True)
    
    model.save_model(model_path)
    print('model saved at: ', model_path)
    
def model_pred(X_out1, tergt1):
    dout1 = xgb.DMatrix(X_out1, label=tergt1)
    #print('Starting prediction..')
    probs_mc_dropout = []
    
    for i in range(109):
        #print('predicting model:',i+1)
        load_filename = 'models/'+ str(i)+'_th.model'
    
        loaded_model = xgb.Booster()
        loaded_model.load_model(load_filename)

        y_prd = loaded_model.predict(dout1)
        probs_mc_dropout += [y_prd]
        
    predictive_mean = np.mean(probs_mc_dropout, axis=0)
    predictive_std = np.std(probs_mc_dropout, axis=0)
    return predictive_mean, predictive_std


# In[3]:


import random
import time
import pandas as pd
import xgboost as xgb
from scipy.stats import norm
from sklearn import preprocessing

gridsearch_params = [
    (subsample, colsample)
    for subsample in [0.01, 0.2, 0.26, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for colsample in [0.01, 0.1, 0.22, 0.3, 0.4, 0.5, 0.55, 0.7, 0.8, 0.9, 1]
]

class multiXGB():
    """
            Constructor for the class implementing a XGBoost architecture
    """
    def __init__(self, X_train, y_train, X_val, y_val):
        
        scaler_trx = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler_trx.transform(X_train)
        countr = 0
    
        
        for subsample, colsample in reversed(gridsearch_params):
            #print("Training with subsample={}, colsample={}".format( subsample, colsample))
            model_path = 'models/'+ str(countr)+'_th.model'
            countr+=1

            get_model(subsample, colsample, model_path, X_train, y_train, X_val, y_val)
    
    
    def predict(self, x_testss, y_testsss):
        scaler_testx = preprocessing.StandardScaler().fit(x_testss)
        x_testss = scaler_testx.transform(x_testss)
        y_pred, yhat_std = model_pred(x_testss, y_testsss)
        #trp = [i/j for i,j in zip(y_pred, y_testsss)]
        #print(trp)
        
        ll = []
        for i in range(y_testsss.shape[0]):
            ll.append(norm.logpdf(y_testsss[i], y_pred[i]/5, yhat_std[i]))
        new_test_ll = np.mean(ll)
        
        return -(new_test_ll)


# In[4]:


lls = []
for split in range(int(n_splits)):

    # We load the indexes of the training and test sets
    print ('Loading file: ' + _get_index_train_test_path(split, train=True))
    print ('Loading file: ' + _get_index_train_test_path(split, train=False))
    index_train = np.loadtxt(_get_index_train_test_path(split, train=True))
    index_test = np.loadtxt(_get_index_train_test_path(split, train=False))

    X_train = X[ [int(i) for i in index_train.tolist()] ]
    y_train = y[ [int(i) for i in index_train.tolist()] ]
    
    X_test = X[ [int(i) for i in index_test.tolist()] ]
    y_test = y[ [int(i) for i in index_test.tolist()] ]

    X_train_original = X_train
    y_train_original = y_train
    num_training_examples = int(0.8 * X_train.shape[0])
    X_validation = X_train[num_training_examples:, :]
    y_validation = y_train[num_training_examples:]
    X_train = X_train[0:num_training_examples, :]
    y_train = y_train[0:num_training_examples]
    
    # Printing the size of the training, validation and test sets
    print ('Number of training examples: ' + str(X_train.shape[0]))
    print ('Number of validation examples: ' + str(X_validation.shape[0]))
    print ('Number of test examples: ' + str(X_test.shape[0]))
    print ('Number of train_original examples: ' + str(X_train_original.shape[0]))
    


    loglk = multiXGB(X_train_original, y_train_original, X_validation, y_validation).predict(X_test, y_test)


    print ("Tests on split " + str(split) + " complete. "+"with ll: "+str(loglk))

    lls += [loglk]
    print()
    print('*********************************************************************')
    print()


# In[5]:

print('Log Likelihood of all the splits::')
print(lls)


# In[6]:


print('Mean Log Likelihood:: ', np.mean(lls))
print('Std. Dev of the Log Likelihood:: 'np.std(lls))
        

