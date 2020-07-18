import json
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd

import argparse
import sys

parser=argparse.ArgumentParser()

parser.add_argument('--valDataDir', '-vdr', required=True, help='Dataset path for validation')

args=parser.parse_args()

data_directory = args.valDataDir

class NNCutoff_bin():
    
    def __init__(self, model_PATH, X_test, y_test, batch_size):
        print('Loading the saved model..')
        self.model = load_model(model_PATH)
        self.X_test = X_test
        self.y_test = y_test
        self.batch_size = batch_size
        
        
    def find_cutoff(self, T_val=100):
        probs_mc_dropout = []
        T = T_val
        print('predicting for Evaluation...')
        for t_i in range(T):
        	print('sch round::', t_i)
        	probs_mc_dropout += [self.model.predict(self.X_test, batch_size=self.batch_size, verbose=1)]
        predictive_mean = np.mean(probs_mc_dropout, axis=0)[:,0]
        predictive_variance = np.var(probs_mc_dropout, axis=0)[:,0]
        
        return np.mean(predictive_variance)
        
        #return self.ll, auc_score, self.tpr, self.fpr, classification_report(y_test, predictions)
if __name__ == '__main__':
	with open('param.json') as f:
		param = json.load(f)

	print('Reading val data..')
	val_data = pd.read_csv(data_directory)
	y_val = val_data[param['terget']]
	X_val = val_data.drop([param['terget']], 1)

	mdl = NNCutoff_bin('MC_Dropout_trained.h5', X_val, y_val, param['batch_size']).find_cutoff()
	print('Cutoff value::',mdl)
	param['cutoff']=float(mdl)
	with open('param.json', 'w') as fp:
		json.dump(param, fp)