import json
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd

import argparse
import sys
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt
import xgboost as xgb
import pickle
import random
import time

parser=argparse.ArgumentParser()

parser.add_argument('--algo', '-alg', required=True, help='Algorith to be used')
parser.add_argument('--predDataDir', '-pdr', required=True, help='Dataset path for predictions')
parser.add_argument('--predSaveDir', '-psd', required=True, help='file name to save the results')

args=parser.parse_args()

algo = args.algo
data_directory = args.predDataDir
save_directory = args.predSaveDir

class MCDpred():
    
    def __init__(self, model_PATH, X_test, batch_size):
        print('Loading the saved model..')
        self.model = load_model(model_PATH)
        self.X_test = X_test
        self.batch_size = batch_size
        self.pm = None
        self.pv = None
        
        
    def find_mv(self):
        probs_mc_dropout = []
        T = 100
        print('predicting for Evaluation...')
        for t_i in range(T):
            print('sch round::', t_i)
            probs_mc_dropout += [self.model.predict(self.X_test, batch_size=self.batch_size, verbose=1)]
        predictive_mean = np.mean(probs_mc_dropout, axis=0)[:,0]
        predictive_variance = np.var(probs_mc_dropout, axis=0)[:,0]
        self.pm = predictive_mean
        self.pv = predictive_variance


    def predictions(self, outpth):
        self.find_mv()

        self.X_test['prediction']= self.pm
        self.X_test['uncertainty']= self.pv
        self.X_test.to_csv(outpth, encoding='utf-8', index=False)
        print('Results saved at::', outpth)


class DeepEnsmbPred():
    
    def __init__(self, X_test, batch_size):
        print('Loading the saved model..')
        self.models = []

        for i in range(5):
        	self.models.append('trained_models/DEnsmb_trained_reg_'+str(i)+'.h5')
        self.models = [load_model(mp, custom_objects={'gaussian_nll': self.gaussian_nll}) for mp in self.models]
        self.X_test = X_test
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
	        y_pred = model.predict(x, verbose=1)
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

    def predictions(self, outpth):
        yhat_mean, yhat_std, predictive_variance = self.predict(self.X_test)
   

        self.X_test['prediction']= yhat_mean
        self.X_test['uncertainty']= predictive_variance
        self.X_test.to_csv(outpth, encoding='utf-8', index=False)
        print('Results saved at::', outpth)
        dftp = self.X_test[['prediction', 'uncertainty']]
        print(dftp)
        
        #return self.ll, auc_score, self.tpr, self.fpr, classification_report(y_test, predictions)

class multiXGB_pred():
    
    def __init__(self, X_test, no_models):
        self.X_test = X_test

        print('Starting prediction using validation data..')
        self.probs_mc_dropout = []

        for i in range(no_models):
            print('predicting model:',i+1)
            load_filename = 'trained_models/XGBclf/'+ str(i)+'_th.h5'
            loaded_model = pickle.load(open(load_filename, 'rb'))
            y_prob = loaded_model.predict_proba(self.X_test)
            y_prob = [i[1] for i in y_prob]

            self.probs_mc_dropout += [y_prob]

        self.mean = np.mean(self.probs_mc_dropout, axis=0)
        self.var = np.var(self.probs_mc_dropout, axis=0)

    def prdct(self, mn, var, cutoff):
        if var>cutoff:
            return 'Uncertain'

        return int(round(mn))

    def predictions(self, outpth, cutoff):

        self.X_test['mean']= self.mean
        self.X_test['uncertainty']= self.var
        df['prediction']=[self.prdct(p, v, cutoff) for p, v in zip(df['mean'], df['uncertainty'])]
        self.X_test.to_csv(outpth, encoding='utf-8', index=False)
        print('Results saved at::', outpth)
        dftp = self.X_test[['prediction', 'uncertainty']]
        print(dftp)


class XGBRandom_pred():
    def __init__(self, X_test, num_round):
        self.X_test = X_test
        self.dtest = xgb.DMatrix(X_test)
        self.num_round = num_round
        print('Loading the trained model..')
        load_filename = 'trained_models/XGBrandom.h5'
        self.model = pickle.load(open(load_filename, 'rb'))

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

    def get_tree_pred(self):
        
        print('Predicting scores..')
        
        all_tree_pred = []
        all_tree_pred.append(self.model.predict(self.dtest, output_margin=True, ntree_limit=1))
        for i in range(1, self.num_round):
            pred = self.model.predict(self.dtest, output_margin=True, ntree_limit=i+1)-all_tree_pred[i-1]
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
    
    def prdct(self, mn, var, cutoff):
        if var>cutoff:
            return 'Uncertain'

        return int(round(1/(1+np.exp(-mn))))
    
    def predictions(self, outpth, cutoff):
        yhat_mean, predictive_variance = self.get_tree_pred()
        
        self.X_test['mean']= yhat_mean
        self.X_test['uncertainty']= predictive_variance
        df['prediction']=[self.prdct(p, v, cutoff) for p, v in zip(df['mean'], df['uncertainty'])]
        self.X_test.to_csv(outpth, encoding='utf-8', index=False)
        print('Results saved at::', outpth)
        dftp = self.X_test[['prediction', 'uncertainty']]
        print(dftp)


if __name__ == '__main__':
	if algo=='MCD':
		with open('training/training_strategy_mc.json') as f:
			param = json.load(f)

		print('Reading data..')
		val_data = pd.read_csv(data_directory)
		#y_val = val_data[param['tergetCol']]
		#X_val = val_data.drop([param['tergetCol']], 1)
		mdl = MCDpred('MC_Dropout_trained_reg.h5', val_data, param['batch_size']).predictions(save_directory)

	if algo=='DeepEnsmb':
		with open('training/training_strategy_de.json') as f:
			param = json.load(f)

		print('Reading data..')
		val_data = pd.read_csv(data_directory)
		#y_val = val_data[param['tergetCol']]
		#X_val = val_data.drop([param['tergetCol']], 1)

		mdl = DeepEnsmbPred(val_data, param['batch_size']).predictions(save_directory)

    if algo=='multiXGB_pred':
        with open('training/training_strategy_xgb.json') as f:
            param = json.load(f)

        print('Reading data..')
        val_data = pd.read_csv(data_directory)
        no_models = no_models['num_round']

        mdl = multiXGB_pred(val_data, no_models).predictions(save_directory, param['cutoff'])

    if algo=='RandomXGB':
        with open('training/training_strategy_xgbRndm.json') as f:
            param = json.load(f)

        print('Reading data..')
        val_data = pd.read_csv(data_directory)
        num_round = param['num_round']

        mdl = XGBRandom_pred(val_data, num_round).predictions(save_directory, param['cutoff'])