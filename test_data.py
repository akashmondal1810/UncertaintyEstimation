#!/usr/bin/env python
# coding: utf-8


from testing.test_bin import NNtesting_bin
from testing.test_rgr import NNtesting_rgr

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
import sys
import json
import tensorflow as tf

class GetRequiredVal():

    def __init__(self):
        self.x_test = None
        self.y_test = None
        self.problem_type = None


        _train_str_file_PATH = 'autotuning/training_strategy.json'
        with open(_train_str_file_PATH) as f:
            training_param = json.load(f)

        self.train_parameters = training_param

        self.test_data = pd.read_csv(self.train_parameters["test_data_Path"])

        self.target_col = self.train_parameters["target_col"]
        self.batch_size = self.train_parameters["batch_size"]

        
        self.y_test = self.test_data[self.target_col]
        self.x_test = self.test_data.drop([self.target_col], 1)


    def findProblemType(self):
        if(len(set(self.y_test))/len(self.y_test))>.15:
            self.problem_type = 'regression'
            return 'regression'

        if len(set(self.y_test))==2:
            self.problem_type = 'binary_classification'
            return 'binary_classification'

        self.problem_type = 'multi_classification'
        return 'multi_classification'

    def get_data(self):
        return self.x_test, self.y_test, self.batch_size


if __name__ == '__main__':
    datasetName = sys.argv[1]
    model_path = sys.argv[2]
    
    if datasetName=='MNIST':
        pass
        
    elif datasetName=='LOAN':
        pass
    
    else:
        x_test, y_test, batch_size = GetRequiredVal().get_data()
        prob_typ = GetRequiredVal().findProblemType()

        if prob_typ=='binary_classification':
            nndtst = NNtesting_bin(model_path, x_test, y_test, batch_size)
            nndtst.test_score()

        elif prob_typ=='regression':
            nndtstr = NNtesting_rgr(model_path, x_test, y_test, batch_size)
            nndtstr.test_score()
            