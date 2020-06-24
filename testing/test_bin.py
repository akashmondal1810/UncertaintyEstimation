import json
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score

class NNtesting_bin():
    
    def __init__(self, model_PATH, X_test, y_test, batch_size):
        print('Loading the saved model..')
        self.model = load_model(model_PATH)
        self.X_test = X_test
        self.y_test = y_test
        self.batch_size = batch_size
        self.y_hat = None
        self.tpr = None
        self.fpr = None
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
    
    def find_TPR_FPR(self, y_actual, y_hat):
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for a,b in zip(y_actual, y_hat):
            if a==1 and b==1:
                TP += 1
            if b==1 and a!=b:
                FP += 1
            if a==b==0:
                TN += 1
            if b==0 and a!=b:
                FN += 1

        self.tpr = (TP / (TP + FN))
        self.fpr = (FP / (FP + TN))
        
        
    def test_score(self, T_val=100):
        probs_mc_dropout = []
        T = T_val
        print('predicting for Evaluation...')
        for t_i in range(T):
            probs_mc_dropout += [self.model.predict(self.X_test, batch_size=self.batch_size, verbose=1)]
        predictive_mean = np.mean(probs_mc_dropout, axis=0)
        predictive_variance = np.var(probs_mc_dropout, axis=0)
        
        self.y_hat = [1 if i>0.5 else 0 for i in predictive_mean]
        
        # obtained the test ll from the validation sets
        self.ll = self.log_likelihood(self.y_test, predictive_mean, 0.05, 100)
        print('log likelihood: ', self.ll)
        
        self.find_TPR_FPR(self.y_test, self.y_hat)
        print('TPR: ', self.tpr)
        print('Fpr: ', self.fpr)
        
        auc_score = roc_auc_score(self.y_test, predictive_mean)
        print('AUC Score: ', auc_score)
        print('Classification Report: ', classification_report(self.y_test, self.y_hat))
        
        #return self.ll, auc_score, self.tpr, self.fpr, classification_report(y_test, predictions)