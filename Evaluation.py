import json
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd

import argparse
import sys
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()

parser.add_argument('--algo', '-alg', required=True, help='Algorith to be used')
parser.add_argument('--evalDataDir', '-edr', required=True, help='Dataset path for Evaluation')
parser.add_argument('--evalSaveDir', '-esd', required=True, help='file name to save the results')

args=parser.parse_args()

algo = args.algo
data_directory = args.evalDataDir
save_directory = args.evalSaveDir

class MCDEval_bin():
    
    def __init__(self, model_PATH, X_test, y_test, batch_size):
        print('Loading the saved model..')
        self.model = load_model(model_PATH)
        self.X_test = X_test
        self.y_test = y_test
        self.batch_size = batch_size
        self.pm = None
        self.pv = None
        
        
    def find_mv(self, T_val=100):
        probs_mc_dropout = []
        T = T_val
        print('predicting for Evaluation...')
        for t_i in range(T):
            print('sch round::', t_i)
            probs_mc_dropout += [self.model.predict(self.X_test, batch_size=self.batch_size, verbose=1)]
        predictive_mean = np.mean(probs_mc_dropout, axis=0)[:,0]
        predictive_variance = np.var(probs_mc_dropout, axis=0)[:,0]
        self.pm = predictive_mean
        self.pv = predictive_variance

    def prdct(self, mn, var, cutoff):
        if var>cutoff:
            return 'Uncertain'
        return int(round(mn))

    def eval(self, outpth, cutoff):
        self.find_mv()
        df = pd.DataFrame(
             {'mean': self.pm,
              'variance': self.pv,
              'y_true': self.y_test
             })
        
        dfval = df.loc[df['variance'] <cutoff]
        fpr, tpr, _ = roc_curve(dfval['y_true'], dfval['mean'])
        model_auc = np.round(auc(fpr, tpr), 4)

        # visualization
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, label='AUC: ' + str(model_auc))
        ax.plot(fpr, fpr, 'k:')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True)
        plt.tight_layout()
        plt.show()

        df['prediction']=[self.prdct(p, v, cutoff) for p, v in zip(df['mean'], df['variance'])]
        df.to_csv(outpth, encoding='utf-8', index=False)
        print('Results saved at::', outpth)


class DeepEnsmbEval_bin():
    
    def __init__(self, X_test, y_test, batch_size):
        print('Loading the saved model..')
        self.models = []

        for i in range(5):
        	self.models.append('trained_models/DEnsmb_trained_'+str(i)+'.h5')
        self.models = [load_model(mp, custom_objects={'gaussian_nll': self.gaussian_nll}) for mp in self.models]
        self.X_test = X_test
        self.y_test = y_test
        self.batch_size = batch_size


    def gaussian_nll(self, y_true, y_pred):
        
        """
            Gaussian negative log likelihood

            Note: to make training more stable, we optimize
            a modified loss by having our model predict log(sigma^2)
            rather than sigma^2. 
        """

        y_true = tf.reshape(y_true, [-1])
        mu = y_pred[:, 0]
        si = y_pred[:, 1]
        loss = (si + tf.square(y_true - mu)/tf.math.exp(si)) / 2.0
        return tf.reduce_mean(loss)

    def predict(self, x):
	    '''
		    Args:
		        models: The trained keras model ensemble
		        x: the input tensor with shape [N, M]
		        samples: the number of monte carlo samples to collect
		    Returns:
		        y_mean: The expected value of our prediction
		        y_std: The standard deviation of our prediction
	    '''
	    mu_arr = []
	    si_arr = []

	    for model in self.models:
	        y_pred = model.predict(x)
	        mu = y_pred[:, 0]
	        si = y_pred[:, 1]

	        mu_arr.append(mu)
	        si_arr.append(si)

	    mu_arr = np.array(mu_arr)
	    si_arr = np.array(si_arr)
	    var_arr = np.exp(si_arr)

	    y_mean = np.mean(mu_arr, axis=0)
	    y_variance = np.mean(var_arr + mu_arr**2, axis=0) - y_mean**2
	    y_std = np.sqrt(y_variance)
	    return y_mean, y_std, y_variance

    def prdct(self, mn, var, cutoff):
        if var>cutoff:
            return 'Uncertain'
        return int(round(mn))

    def eval(self, outpth, cutoff):
        yhat_mean, yhat_std, predictive_variance = self.predict(self.X_test)
        df = pd.DataFrame(
             {'mean': yhat_mean,
              'variance': predictive_variance,
              'y_true': self.y_test
             })
        
        dfval = df.loc[df['variance'] <cutoff]
        print(dfval)
        fpr, tpr, _ = roc_curve(dfval['y_true'], dfval['mean'])
        model_auc = np.round(auc(fpr, tpr), 4)

        # visualization
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, label='AUC: ' + str(model_auc))
        ax.plot(fpr, fpr, 'k:')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True)
        plt.tight_layout()
        plt.show()

        df['prediction']=[self.prdct(p, v, cutoff) for p, v in zip(df['mean'], df['variance'])]
        df.to_csv(outpth, encoding='utf-8', index=False)
        print('Results saved at::', outpth)
        
        #return self.ll, auc_score, self.tpr, self.fpr, classification_report(y_test, predictions)
if __name__ == '__main__':
	if algo=='MCD':
		with open('training/training_strategy_mc.json') as f:
			param = json.load(f)

		print('Reading data..')
		val_data = pd.read_csv(data_directory)
		y_val = val_data[param['tergetCol']]
		X_val = val_data.drop([param['tergetCol']], 1)
		mdl = MCDEval_bin('MC_Dropout_trained.h5', X_val, y_val, param['batch_size']).eval(save_directory, param['cutoff'])

	if algo=='DeepEnsmb':
		with open('training/training_strategy_de.json') as f:
			param = json.load(f)

		print('Reading data..')
		val_data = pd.read_csv(data_directory)
		y_val = val_data[param['tergetCol']]
		X_val = val_data.drop([param['tergetCol']], 1)

		mdl = DeepEnsmbEval_bin(X_val, y_val, param['batch_size']).eval(save_directory, param['cutoff'])