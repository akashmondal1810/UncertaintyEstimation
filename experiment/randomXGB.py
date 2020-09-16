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


import random
import time
import pandas as pd
import xgboost as xgb
from scipy.stats import norm
from sklearn import preprocessing

class XGBRandom2():
    """
            Constructor for the class implementing a XGBoost architecture
    """
    def __init__(self, X_train, y_train, param, num_round):
        
        scaler_trx = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler_trx.transform(X_train)
    
        
        self.dtrain = xgb.DMatrix(X_train, label=y_train)
        self.num_round=num_round
        
        print('Training using rounds::', self.num_round)
        self.model = xgb.train(param, self.dtrain, self.num_round)
        print('Training finished')
        
    def random_subset(self, iterator, K):
        #randomly choosing k value from iterator using reservoir-sampling
        result = []
        N = 0

        for item in iterator:
            N += 1
            if len( result ) < K:
                result.append( item )
            else:
                s = int(random.random() * N)
                if s < K:
                    result[ s ] = item

        return result
    
    def get_tree_pred(self, X_testss, y_testss):
        scaler_testx = preprocessing.StandardScaler().fit(X_testss)
        X_testss = scaler_testx.transform(X_testss)
        
        self.dtest = xgb.DMatrix(X_testss, label=y_testss)
        
        print('Predicting scores..')
        
        all_tree_pred = []
        all_tree_pred.append(self.model.predict(self.dtest, output_margin=False, ntree_limit=1))
        for i in range(1, self.num_round):
            pred = self.model.predict(self.dtest, output_margin=False, ntree_limit=i+1)-all_tree_pred[i-1]
            all_tree_pred.append(pred)
            
        all_tree_pred = np.array(all_tree_pred)
        
        print('Getting mean and var..')
        start_time = time.time()
        predction_mean = []
        predction_var = []
        for i in range(len(all_tree_pred[0])):
            randomPredcts = []
            for _ in range(self.num_round//5):
                pp = self.random_subset(all_tree_pred[:, i], self.num_round//5)
                randomPredcts.append(np.sum(pp))
            predction_mean.append(np.mean(randomPredcts))
            predction_var.append(np.var(randomPredcts))
            
        print("time taken for prediction %s seconds ---" % (time.time() - start_time))

            
        return predction_mean, predction_var
    
    def predict(self, x_testss, y_testsss):
        y_pred, yhat_std = self.get_tree_pred(x_testss, y_testsss)
        #trp = [i/j for i,j in zip(y_pred, y_testsss)]
        #print(trp)
        
        ll = []
        for i in range(y_testsss.shape[0]):
            ll.append(norm.logpdf(y_testsss[i], y_pred[i]/5, yhat_std[i]))
        new_test_ll = np.mean(ll)
        
        return -(new_test_ll)


# In[3]:


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
    

    # List of hyperparameters which we will try out using grid-search
    
    mxds = [2, 3, 4, 5]
    etas = [0.05, 0.1, 0.5]
    param = {'max_depth': 2, 'eta': 0.1, 'objective':'reg:linear',
         'min_child_weight':1, 'subsample':0.7, 'colsample_bytree':0.8}
    num_round = 50

    # We perform grid-search to select the best hyperparameters based on the highest log-likelihood value
    best_network = None
    best_ll = -float('inf')
    best_mxd = 0
    best_eta = 0
    for mxd in mxds:
        for eta in etas:
            print ('Grid search step: max_depth: ' + str(mxd) + ' eta: ' + str(eta))
            param['max_depth'] = mxd
            param['eta'] = eta
            ll = XGBRandom2(X_train, y_train, param, num_round).predict(X_validation, y_validation)
            
            if (ll > best_ll):
                best_ll = ll
                best_eta = eta
                best_mxd = mxd
                print ('Best log_likelihood changed to: ' + str(best_ll))
                print ('Best eta changed to: ' + str(best_eta))
                print ('Best max deapth changed to: ' + str(best_mxd))
            

    # Storing test results
    param['max_depth'] = best_mxd
    param['eta'] = best_eta
    loglk = XGBRandom2(X_train_original, y_train_original, param, num_round).predict(X_test, y_test)


    print ("Tests on split " + str(split) + " complete. "+"with ll: "+str(loglk))

    lls += [loglk]
    print()
    print('*********************************************************************')
    print()


# In[9]:
print('Log Likelihood of all the splits::')
print(lls)


# In[6]:


print('Mean Log Likelihood:: ', np.mean(lls))
print('Std. Dev of the Log Likelihood:: 'np.std(lls))