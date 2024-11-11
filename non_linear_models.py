import numpy as np
from sklearn import svm
import keras 
from keras.layers import LSTM, Dense
from sklearn.ensemble import RandomForestRegressor



class SVM():
    def __init__(self, kernel='rbf', C=1, gamma='auto', epsilon=0.1):
        self.model = svm.SVR(kernel=kernel, C=C)
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)

class RF():
    def __init__(self, n_estimators=1000):
        self.model = RandomForestRegressor(n_estimators=n_estimators)
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)

class DNN():
    def __init__(self, input_shape, output_shape):
        self.isfit = False
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Dense(units=64, input_shape=input_shape, activation='relu'))
        self.model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.Dense(units=128, activation='relu'))
        self.model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.Dense(units=128, activation='relu'))
        self.model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.Dense(units=64, activation='relu'))
        self.model.add(Dense(output_shape, activation='linear'))
    
    def compile(self, loss='mean_squared_error', optimizer='adam'):
        self.model.compile(loss=loss, optimizer=optimizer)
    
    def fit(self, X, y, epochs=500, batch_size=32, verbose=1):
        # Early stopping call back
        early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=10)
        # Learning rate callback
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.0001)
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[early_stopping, reduce_lr])
        self.isfit = True
    
    def check_is_fitted(self):
        return self.isfit
    
    def predict(self, X):
        return self.model.predict(X)
class LSTM():
    def __init__(self, input_shape, output_shape):
        self.isfit = False
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.LSTM(units=128, input_shape=input_shape, return_sequences=True, activation='tanh'))
        self.model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.LSTM(units=128, return_sequences=True, activation='tanh'))
        self.model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.LSTM(units=128, return_sequences=True, activation='tanh'))
        self.model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.LSTM(units=128, return_sequences=True, activation='tanh'))
        self.model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.LSTM(units=128, return_sequences=True, activation='tanh'))
        self.model.add(keras.layers.LSTM(units=32, return_sequences=False, activation='tanh'))
        self.model.add(Dense(output_shape))
    
    def compile(self, loss='mean_squared_error', optimizer='adam'):
        self.model.compile(loss=loss, optimizer=optimizer)
    
    def fit(self, X, y, epochs=200, batch_size=16, verbose=1):
        # Early stopping call back
        early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=10)
        # Learning rate callback
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.0001)
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[reduce_lr, early_stopping])
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