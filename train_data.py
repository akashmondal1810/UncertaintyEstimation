#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings("ignore")

from training.train_bin import NNDropout, DeepArch_clf, XGBMulti
from training.train_rgr import NNDropout_rgr, DeepArch_reg


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

parser.add_argument('--algo', '-alg', required=True, help='Algorith to be used')
parser.add_argument('--dataDir', '-dr', required=True, help='Dataset path for taining')
parser.add_argument('--tergetCol', '-tc', required=True, help='Name of the terget column')


args=parser.parse_args()

algo = args.algo
data_directory = args.dataDir
tcol = args.tergetCol


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

	print('Reading Data...')
	x_train, y_train = GetRequiredVal().get_data()
	prob_typ = GetRequiredVal().findProblemType()

	if algo=='MCD':
		with open('training/training_strategy_mc.json') as f:
			training_param = json.load(f)

		drp = training_param['dropout_rates']
		epochs = training_param['n_epochs']
		tau = training_param['tau']
		btsz = training_param['batch_size']
		nhid = training_param['hidden_units']

		training_param['tergetCol'] = tcol
		with open('training/training_strategy_mc.json', 'w') as fp:
			json.dump(training_param, fp)


		if prob_typ=='binary_classification':
			print('Detected problem type:: binary_classification')
			nndrop = NNDropout(mc = True, actvn = 'relu')
			model = nndrop.model_runner(X_train=x_train, y_train=y_train, dropout_prob=drp, n_epochs=epochs, tau=tau, batch_size=btsz, n_hidden=nhid)
			model.save('MC_Dropout_trained.h5')
			print('model saved at:: MC_Dropout_trained.h5')

		elif prob_typ=='regression':
			print('Detected problem type:: Regression')
			mdldrp = NNDropout_rgr(mc = True, actvn = 'relu')
			model = mdldrp.model_runner(X_train=x_train, y_train=y_train, dropout_prob=drp, n_epochs=epochs, tau=tau, batch_size=btsz, n_hidden=nhid)
			model.save('MC_Dropout_trained_reg.h5')
			print('model saved at:: MC_Dropout_trained_reg.h5')


	if algo=='DeepEnsmb':
		with open('training/training_strategy_de.json') as f:
			training_param = json.load(f)

		epochs = training_param['n_epochs']
		btsz = training_param['batch_size']
		nhid = training_param['hidden_units']

		training_param['tergetCol'] = tcol
		with open('training/training_strategy_de.json', 'w') as fp:
			json.dump(training_param, fp)


		if prob_typ=='binary_classification':
			print('Detected problem type:: binary_classification')

			n_members = 5
			for i in range(n_members):
				nndrop = DeepArch_clf(X_train=x_train,  y_train=y_train)
				model = nndrop.fit_model(n_epochs=epochs, batch_size=btsz, n_hidden=nhid)
				savepath = 'trained_models/DEnsmb_trained_'+str(i)+'.h5'
				model.save(savepath)
				print('model saved at::', savepath)

		elif prob_typ=='regression':
			print('Detected problem type:: Regression')

			n_members = 5
			for i in range(n_members):
				nndrop = DeepArch_reg(X_train=x_train,  y_train=y_train)
				model = nndrop.fit_model(n_epochs=epochs, batch_size=btsz, n_hidden=nhid)
				savepath = 'trained_models/DEnsmb_trained_reg_'+str(i)+'.h5'
				model.save(savepath)
				print('model saved at::', savepath)

	if algo=='MultiXGB':
		with open('training/training_strategy_xgb.json') as f:
			training_param = json.load(f)

		rsmp = training_param['subsample']
		clsmp = training_param['colsample']
		learning_rate = training_param['learning_rate']
		nestim = training_param['nes']
		max_depth = training_param['max_depth']
		min_child_weight = training_param['min_child_weight']

		training_param['tergetCol'] = tcol
		with open('training/training_strategy_xgb.json', 'w') as fp:
			json.dump(training_param, fp)


		if prob_typ=='binary_classification':
			print('Detected problem type:: binary_classification')
			models = XGBMulti(X_train, y_train, rsmp, clsmp).get_models(lr=learning_rate, nes=nestim, max_depth=max_depth, min_child_weight=min_child_weight)

		elif prob_typ=='regression':
			print('Regression setup is not defined for the multiple XGBoost method')
			pass
        