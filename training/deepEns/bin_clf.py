from keras.models import Sequential, Model, Input
from keras.layers import Dense, Dropout
import keras
from keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# simple early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

METRICS = [
      keras.metrics.AUC(name='auc'),
]

class DeepArch():
    """
            Constructor for the class implementing a neural network architecture
            @param mc             True in case of MC Dropout architecture and False for standard
                                  nurel network.
            @param activn_fn      Activation function to be used
    """
    def __init__(self, X_train,  y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val


    def fit_model(self, n_hidden, batch_size, dp):
        input_dim = self.X_train.shape[1]
        class_weights = compute_class_weight('balanced', np.unique(self.y_train), self.y_train)

        # define model
        model = Sequential()
        model.add(Dense(n_hidden[0], input_dim=input_dim, activation='relu'))
        model.add(Dropout(dp))

        for i in range(len(n_hidden) - 1):
            model.add(Dense(n_hidden[i+1], activation='relu'))
            model.add(Dropout(dp))

        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=METRICS)

        model.fit(self.X_train, self.y_train,
                  batch_size=batch_size,
                  epochs=10,
                  verbose=1,
                  validation_data=(self.X_val, self.y_val),
                  class_weight=class_weights,
                 callbacks=[es])
        return model