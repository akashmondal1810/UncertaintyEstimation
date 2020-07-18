#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings("ignore")

from training.train_bin import NNDropout

import numpy as np
import json
import pandas as pd
from io import StringIO
import sys
import json
import tensorflow as tf


import argparse
import sys

parser=argparse.ArgumentParser()

parser.add_argument('--dataDir', '-dr', required=True, help='Dataset path for taining')
parser.add_argument('--tergetCol', '-tc', required=True, help='Name of the terget column')
parser.add_argument('--hidden', '-nh', required=True, nargs='+', type=int, help='nodes in each layers')
parser.add_argument('--epochx','-e', default=4, type=int, help='number of epochs for training.')
parser.add_argument('--batch_size', '-bts', default=1, type=int, help='Batch Size')
parser.add_argument('--dropout', '-dp', default=0.5, type=float, help='Dropout rate')
parser.add_argument('--tau', '-tu', default=0.4, type=float, help='Tau value used for regularization')


args=parser.parse_args()

data_directory = args.dataDir
tcol = args.tergetCol
nhid = args.hidden
epochs = args.epochx
btsz = args.batch_size
drp = args.dropout
tau = args.tau

class GetRequiredVal():

    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.problem_type = None

        self.train_data = pd.read_csv(data_directory)
        
        self.y_train = self.train_data[tcol]
        self.X_train = self.train_data.drop([tcol], 1)


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
        return self.X_train, self.y_train


if __name__ == '__main__':
	best_val = {'tau': tau, 'dropout': drp, 'terget': tcol, 'batch_size':btsz}
	with open('param.json', 'w') as fp:
		json.dump(best_val, fp)

	print('Reading Data...')
	x_train, y_train = GetRequiredVal().get_data()
	prob_typ = GetRequiredVal().findProblemType()

	if prob_typ=='binary_classification':
		print('Detected problem type:: binary_classification')
		nndrop = NNDropout(mc = True, actvn = 'relu')
		model = nndrop.model_runner(X_train=x_train, y_train=y_train, dropout_prob=drp, n_epochs=epochs, tau=tau, batch_size=btsz, n_hidden=nhid)
		model.save('MC_Dropout_trained.h5')
		print('model saved at:: MC_Dropout_trained.h5')

	elif prob_typ=='regression':
		pass
        