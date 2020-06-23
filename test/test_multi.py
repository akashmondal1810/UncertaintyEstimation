import json
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model

class NNtesting_multi():

    def __init__(self, model_PATH, X_test, y_test, batch_size):
        self.model = load_model(model_PATH)
        self.X_test = np.load(X_test)
        self.y_test = np.load(y_test)
        self.batch_size = batch_size
        
    
    def brier_multi(self, targets, probs):
        """
        Brier score implementation follows 
        https://papers.nips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles.pdf.
        The lower the Brier score is for a set of predictions, the better the predictions are calibrated.
        """        

        return np.mean(np.sum((probs - targets)**2, axis=1))
        
    def test_score(self, T_val=100):
        
        probs_mc_dropout = []
        T = T_val
        for t_i in range(T):
            print('T: ', t_i)
            probs_mc_dropout += [self.model.predict(self.X_test, batch_size=self.batch_size, verbose=1)]
        predictive_mean = np.mean(probs_mc_dropout, axis=0)
        predictive_variance = np.var(probs_mc_dropout, axis=0)
        
        # obtained the test brier_score from the validation sets
        brier_score = self.brier_multi(self.y_test, predictive_mean)
        
        return brier_score