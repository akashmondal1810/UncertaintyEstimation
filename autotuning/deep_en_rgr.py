import json
from training.deepEns.de_reg import DeepEn_reg
import numpy as np
import pickle

class DENexperiment_rgr():
    """
    Preprocess strategies defined and exected in this class
    """
    def __init__(self, X_train, y_train, X_val, y_val, activn_fn):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        self.de_model = DeepEn_reg(
            X_train=self.X_train,  
            y_train=self.y_train, 
            X_val=self.X_val, 
            y_val=self.y_val,
            activn_fn=activn_fn
            )
        self.model = None
    
    def log_likelihood(self, y_test, m, v):

        ll = np.mean(-0.5 * np.log(2 * np.pi * v) - 0.5 * (y_test - m)**2 / v)

        return ll

    def predict(self, models, x):
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

        for model in models:
            y_pred = model(x)
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
        return y_mean, y_std, mu_arr
        
    def train_network(self, n_members, n_hidden, n_epochs, batch_size):
        model_stack =[]

        for i in range(n_members):
            print('Training model:', i+1)
            model = self.de_model.fit_model(
                        n_epochs=n_epochs,
                        batch_size=batch_size,
                        n_hidden=n_hidden
                    )
            model_stack.append(model)
        return model_stack