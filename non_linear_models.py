import numpy as np
from sklearn import svm
import keras 
from keras.layers import LSTM, Dense



class SVM():
    def __init__(self, kernel='rbf', C=1, gamma='auto', epsilon=0.1):
        self.model = svm.SVR(kernel=kernel, C=C)
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)

class LSTM():
    def __init__(self, input_shape, output_shape):
        self.isfit = False
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.LSTM(units=125, input_shape=input_shape, return_sequences=True, activation='relu'))
        self.model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.LSTM(units=125, return_sequences=True, activation='relu'))
        self.model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.LSTM(units=125, return_sequences=False, activation='relu'))
        self.model.add(Dense(output_shape))
    
    def compile(self, loss='mean_squared_error', optimizer='adam'):
        self.model.compile(loss=loss, optimizer=optimizer)
    
    def fit(self, X, y, epochs=100, batch_size=32):
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size)
        self.isfit = True
    
    def check_is_fitted(self):
        return self.isfit
    
    def predict(self, X):
        return self.model.predict(X)

class Conv_LSTM():
    def __init__(self, input_shape, output_shape):
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.ConvLSTM2D(50, (1, 3), input_shape=input_shape))
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(output_shape))