import json
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt


class NNtesting_rgr():
    
    def __init__(self, model_PATH, X_test, y_test, batch_size):
        print('Loading the saved model..')
        self.model = load_model(model_PATH)
        self.X_test = X_test
        self.y_test = y_test
        self.batch_size = batch_size
        self.ll = None

    
    def logsumexp(self, X):
        r = 0.0
        for x in X:
            r += np.exp(x)  
        return np.log(r)
    
    def log_likelihood(self, y_test, predictive_mean, tau_val, T_val):
        diffl = []
        for yt, yp in zip(y_test, predictive_mean):
            diffl.append(-0.5 * tau_val * (yt - yp)**2)
        
        ll = (
            self.logsumexp(diffl) 
            - np.log(T_val) 
            - 0.5*np.log(2*np.pi) + 0.5*np.log(tau_val)
        )

        return np.mean(ll)
        
        
    def test_score(self, T_val=100):
        probs_mc_dropout = []
        T = T_val
        print('predicting for Evaluation...')
        for t_i in range(T):
            probs_mc_dropout += [self.model.predict(self.X_test, batch_size=self.batch_size, verbose=1)]
        predictive_mean = np.mean(probs_mc_dropout, axis=0)
        predictive_variance = np.var(probs_mc_dropout, axis=0)

        
        # obtained the test ll from the validation sets
        self.ll = self.log_likelihood(self.y_test, predictive_mean, 0.05, 100)
        print('log likelihood: ', self.ll)
        
        rmse = sqrt(mean_squared_error(self.y_test, predictive_mean))
        print('RMSE: ', rmse)
        
        #return self.ll, auc_score, self.tpr, self.fpr, classification_report(y_test, predictions)