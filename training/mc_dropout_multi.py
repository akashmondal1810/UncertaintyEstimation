import json
from train_multi import NNDropout_multi
import numpy as np
import pickle

class NNexperiment_multi():
    """
    Preprocess strategies defined and exected in this class
    """
    def __init__(self, X_train, y_train, X_val, y_val, dir_path, num_class):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.num_class = num_class
        self._HP_DATA_DIRECTORY_PATH = dir_path
        
        self._DROPOUT_RATES_ = None
        self._TAU_VALUES = None
        self._HIDDEN_UNITS_FILE = None
        self.n_epochs = None
        self.batch_size = None
        self.MCDmodel_output_PATH = None
        self.hp_output_PATH = None

        self.best_tau_val = None
        self.best_dropout_val = None
        self.best_HIDDEN_UNITS_lay = None
        self.best_MSD_model = None

        
        self.mcd_model = NNDropout_multi(mc=True, actvn='relu', num_class=self.num_class)
        self.model = None

    
    def brier_multi(self, targets, probs):
        """
        Brier score implementation follows 
        https://papers.nips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles.pdf.
        The lower the Brier score is for a set of predictions, the better the predictions are calibrated.
        """        

        return np.mean(np.sum((probs - targets)**2, axis=1))
    
    def _open_hp_files(self):
        print('Importing training strategies for Autotuning...')
    
        with open(self._HP_DATA_DIRECTORY_PATH) as f:
            _hp_param = json.load(f)

        self._DROPOUT_RATES = _hp_param["dropout_rates"]
        self._TAU_VALUES = _hp_param["tau_val"]
        self._HIDDEN_UNITS_FILE = _hp_param["hidden_units"]
        self.n_epochs = _hp_param["n_epochs"]
        self.batch_size = _hp_param["batch_size"]
        self.MCDmodel_output_PATH = _hp_param["saved_MCDmodel_output_PATH"]
        self.hp_output_PATH = _hp_param["best_hp_output_PATH"]
        
    def find_best_network(self, T_val=100):
        self._open_hp_files()
        
        best_network = None
        best_bs = float('inf')
        best_tau = 0
        best_dropout = 0
        best_HIDDEN_UNITS = []
        for dropout_rate in self._DROPOUT_RATES:
            for tau in self._TAU_VALUES:
                for n_hidden in self._HIDDEN_UNITS_FILE:
                    
                    print ('Grid search step: Tau: ' + str(tau) + ' Dropout rate: ' + str(dropout_rate)+ ' Hidden units : ' + str(n_hidden))

                    network = self.mcd_model.model_runner(
                        self.X_train, self.y_train,
                        dropout_prob=dropout_rate,
                        n_epochs=self.n_epochs,
                        tau=tau,
                        batch_size=self.batch_size,
                        lengthscale=1e-2,
                        n_hidden=n_hidden
                    )
                    
                    print('Starting prediction using validation data..')
                    probs_mc_dropout = []
                    self.model = network
                    T = T_val
                    for t_i in range(T):
                        print('T: ', t_i)
                        probs_mc_dropout += [self.model.predict(self.X_val, batch_size=self.batch_size, verbose=1)]
                    predictive_mean = np.mean(probs_mc_dropout, axis=0)
                    predictive_variance = np.var(probs_mc_dropout, axis=0)
                    
                    

                    # obtained the test brier_score from the validation sets
                    brier_score = self.brier_multi(self.y_val, predictive_mean)
                    
                    if (brier_score < best_bs):
                        best_bs = brier_score
                        best_network = network
                        best_tau = tau
                        best_dropout = dropout_rate
                        best_HIDDEN_UNITS = n_hidden
                        print ('Best log_likelihood changed to: ' + str(best_bs))
                        print ('Best tau changed to: ' + str(best_tau))
                        print ('Best dropout rate changed to: ' + str(best_dropout))


        self.best_tau_val = best_tau
        self.best_dropout_val = best_dropout
        self.best_HIDDEN_UNITS_lay = best_HIDDEN_UNITS
        self.best_MSD_model = best_network

        best_val = {
            'best_tau': self.best_tau_val,
            'best_dropout': self.best_dropout_val,
            'best_HIDDEN_UNITS': self.best_HIDDEN_UNITS_lay

        }
        print(best_val)

        with open(self.hp_output_PATH, 'w') as fp:
            json.dump(best_val, fp)
            
        self.best_MSD_model.save(self.MCDmodel_output_PATH)