#!/usr/bin/env python
# coding: utf-8

from autotuning.mc_dropout_multi import NNexperiment_multi
from autotuning.mc_dropout import NNexperiment
from autotuning.mc_dropout_rgr import NNexperiment_rgr

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
import sys
import json
import tensorflow as tf
from sklearn.datasets import load_breast_cancer

class GetRequiredVal():

    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.problem_type = None


        _train_str_file_PATH = 'autotuning/training_strategy.json'
        with open(_train_str_file_PATH) as f:
            training_param = json.load(f)

        self.train_parameters = training_param

        self.train_data = pd.read_csv(self.train_parameters["train_data_Path"])
        self.val_data = pd.read_csv(self.train_parameters["val_data_Path"])

        self.target_col = self.train_parameters["target_col"]

        
        self.y_train = self.train_data[self.target_col]
        self.X_train = self.train_data.drop([self.target_col], 1)

        self.y_val = self.val_data[self.target_col]
        self.X_val = self.val_data.drop([self.target_col], 1)

    def findProblemType(self):
        if(len(set(self.y_train))/len(self.y_train))>.15:
            self.problem_type = 'regression'
            return 'regression'

        if len(set(self.y_train))==2:
            self.problem_type = 'binary_classification'
            return 'binary_classification'

        self.problem_type = 'multi_classification'
        return 'multi_classification'

    def get_data(self):
        return self.X_train, self.y_train, self.X_val, self.y_val




class LoadMNIST():

    def __init__(self):
        print("Loading the MNIST Data...")
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
        nnarch = NNexperiment_multi(x_train, y_train, x_val, y_val, 'autotuning/training_strategy.json', num_class=10)
        model = nnarch.find_best_network()
    
    elif datasetName=='LOAN':
        train = pd.read_csv("data/loan/train.csv")
        val = pd.read_csv("data/loan/validate.csv")
        
        train["profit_rate"] = train.apply(lambda x: ((x['total_pymnt'] - x['loan_amnt'])/x['loan_amnt']), axis = 1)
        train["class_type"] = [0 if i > 0 else 1 for i in train["profit_rate"]]

        val["profit_rate"] = val.apply(lambda x: ((x['total_pymnt'] - x['loan_amnt'])/x['loan_amnt']), axis = 1)
        val["class_type"] = [0 if i > 0 else 1 for i in val["profit_rate"]]

        y_train = train["class_type"]
        y_val = val["class_type"]

        X_train = train.drop(["profit_rate", "class_type"], 1)
        X_val = val.drop(["profit_rate", "class_type"], 1)

        nndrop = NNexperiment(X_train, y_train, X_val, y_val, 'autotuning/training_strategy.json')
        model = nndrop.find_best_network()
    
    else:
        X_train, y_train, X_val, y_val = GetRequiredVal().get_data()
        prob_typ = GetRequiredVal().findProblemType()

        if prob_typ=='binary_classification':
            nndrop = NNexperiment(X_train, y_train, X_val, y_val, 'autotuning/training_strategy.json')
            model = nndrop.find_best_network()

        elif prob_typ=='regression':
            nndrop = NNexperiment_rgr(X_train, y_train, X_val, y_val, 'autotuning/training_strategy.json')
            model = nndrop.find_best_network()