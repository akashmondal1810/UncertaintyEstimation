from train_bin import NNDropout
from generator import Generator
import time
import tensorflow as tf
import math
import multiprocessing
from scipy.misc import logsumexp

import keras

METRICS = [
      keras.metrics.AUC(name='auc'),
]

class MCDropout():
    """
            Constructor for the class implementing a MC Dropout neural network.
            @param X_train      Matrix with the features for the training data.
            @param y_train      Vector with the target variables for the
                                training data.
            @param n_hidden     Vector with the number of neurons for each
                                hidden layer.
            @param n_epochs     Numer of epochs for which to train the
                                network. The recommended value 40 should be
                                enough.
            @param tau          Tau value used for regularization
            @param dropout      Dropout rate for all the dropout layers in the
                                network.
    """
    def __init__(self, X_train, y_train, n_hidden, n_epochs = 40, tau = 1.0, dropout = 0.05):
        self.X_train = X_train
        self.y_train = y_train
        self.n_hidden = n_hidden
        self.n_epochs = n_epochs
        self.dropout_prob = dropout
        self.tau = tau
        
        # for network construction
        self.input_dim = X_train.shape[1]
        N = X_train.shape[0]
        batch_size = 128
        lengthscale = 1e-2
        self.reg = lengthscale**2 * (1 - self.dropout_prob) / (2. * N * self.tau)
        
        #importing the NN architecture
        archt = NNDropout(self.n_hidden, self.input_dim, self.reg, self.dropout_prob, True, 'relu')
        model = archt.architecture('adam', 'binary_crossentropy', METRICS)
        
        #importing the generator
        train_generator = Generator(self.X_train, self.y_train, batch_size).generate()
        
        # Iterate the learning process
        start_time = time.time()
        
        model.fit_generator(
                    generator = train_generator, 
                    steps_per_epoch = math.floor(self.X_train.shape[0]/batch_size), 
                    epochs = self.n_epochs,  
                    max_queue_size = 10, 
                    workers = multiprocessing.cpu_count(),
                    use_multiprocessing = True, 
                    shuffle = True,
                    initial_epoch = 0
        )
        
        self.model = model
        self.running_time = time.time() - start_time
        
        
    def predict(self, X_test, y_test):

        """
            Function for making predictions with the Bayesian neural network.
            @param X_test   The matrix of features for the test data
            
    
            @return m       The predictive mean for the test target variables.
            @return v       The predictive variance for the test target
                            variables.
        """
        
        probs_mc_dropout = []
        model = self.model
        T = 100
        for _ in range(T):
            probs_mc_dropout += [model.predict(X_test, batch_size=512, verbose=1)]
        predictive_mean = np.mean(probs_mc_dropout, axis=0)
        predictive_variance = np.var(probs_mc_dropout, axis=0)
        
        
        # We compute the test log-likelihood
        ll = (logsumexp(-0.5 * self.tau * (y_test - predictive_mean)**2., 0) - np.log(T) 
            - 0.5*np.log(2*np.pi) + 0.5*np.log(self.tau))
        
        test_ll = np.mean(ll)

        return predictive_mean, predictive_variance, test_ll