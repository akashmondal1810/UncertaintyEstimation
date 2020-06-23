import json
from tensorflow.keras.models import load_model
import numpy as np
import pickle

class NNtesting_bin():
    """
    Preprocess strategies defined and exected in this class
    """
    def __init__(self, model_PATH, X_test, y_test, batch_size):
        self.model = load_model(model_PATH)
        self.X_test = X_test
        self.y_test = y_test
        self.batch_size = batch_size

    
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
        for t_i in range(T):
            probs_mc_dropout += [self.model.predict(self.X_val, batch_size=self.batch_size, verbose=1)]
        predictive_mean = np.mean(probs_mc_dropout, axis=0)
        predictive_variance = np.var(probs_mc_dropout, axis=0)
        
        # obtained the test ll from the validation sets
        ll = self.log_likelihood(self.y_val, predictive_mean, tau, T)
        
        return ll