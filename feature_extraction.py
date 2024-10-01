import pandas as pd
import numpy as np
import os
import fnmatch
import zipfile as zip
import keras
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from pred import load_data, load_data_for_keras
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from keras.layers import Input, Dense
from keras.models import Model

import pred

class Autoencoder:
    def __init__(self, encoding_dim, input_shape):
        self.encoding_dim = encoding_dim
        self.input_shape = input_shape
        self.model = self.autoencoder_build()
        pass

    def fit(self, tX, epochs=100, batch_size=256, shuffle=True, validation_split=0.2):
        tX = tX.reshape(tX.shape[0], tX.shape[1]*tX.shape[2])
        self.model.fit(tX, tX, epochs=epochs, batch_size=batch_size, 
                       shuffle=shuffle, validation_split=validation_split)
    
    def save(self, path):
        self.model.save(path)

    def autoencoder_build(self):
        encoding_dim= self.encoding_dim*self.input_shape[-1]

        # Reshape the data
        combined_shape = self.input_shape[1]*self.input_shape[2]

        # Fit and transform the data
        input_layer = Input(shape=(combined_shape,))
        encoded = Dense(encoding_dim, activation='relu')(input_layer)
        decoded = Dense(combined_shape, activation='sigmoid')(encoded)

        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        print(autoencoder.summary())

        return autoencoder 

    def predict(self, valX):
        """
        Transforms the input into expected shape and passes it through the encoder

        Input shape: (samples, TSTEP, moneyness)
        Output shape: (samples, encoding_dim*tX.shape[-1])
        """
        # Transform the data to predict
        encoder = Model(inputs=self.model.input, outputs=self.model.layers[1].output)
        valX = valX.reshape(valX.shape[0], valX.shape[1]*valX.shape[2])
        return encoder.predict(valX)


def autoencoder_fit(tX, ty, encoding_dim, TSTEPS=32):
    """
    Uses an autoencoder to extract features from the data 
    and then uses a regression model to predict the implied volatility

    Features extracted will be in the shape of (samples, components*tX.shape[-1])
    i.e. component number of skews for each sample

    Output: encoder, ridge model 
    """
    encoder = Autoencoder(encoding_dim, tX.shape)
    encoder.fit(tX)
    tX_transform = encoder.predict(tX)
    # Fit regression model
    model = Ridge()
    model.fit(tX_transform, ty)
    return encoder, model


def pca_predict(valX, model, n_components, TSTEPS):
    valX_transform = pca_transform(valX, n_components, TSTEPS)
    return model.predict(valX_transform)

def pca_transform(tX, components=3, TSTEPS=32):
    """
    uses PCA to extract features from the data

    Input shape: (samples, TSTEP, moneyness)

    Ouput shape: (samples, components*tX.shape[-1])
    """
    n_components = components*tX.shape[-1]

    # Reshape the data
    tX = tX.reshape(tX.shape[0], tX.shape[1]*tX.shape[2])

    # Fit and transform the data
    pca = PCA(n_components=n_components)
    tX_transform = pca.fit_transform(tX)

    return tX_transform
    
def pca_fit(tX, ty, components=3 , TSTEPS=32):
    """
    Uses PCA to extract features from the data 
    and then uses a regression model to predict the implied volatility

    Features extracted will be in the shape of (samples, components*tX.shape[-1])
    i.e. component number of skews for each sample
    """
    tX_transform = pca_transform(tX, components=components, TSTEPS=TSTEPS) 
    # Fit regression model
    model = Ridge()
    model.fit(tX_transform, ty)
    return model


def har_transform(tX, TSTEPS=32):
    """
    Transform the given input data to the HAR method of feature extraction

    Input shape: (samples, TSTEP, moneyness)
    TSTEP must be of size 32 as a month of IV data is required for the HAR method

    Output shape: (samples, tx.shape[-1]*3)
    """
    if TSTEPS != 32:
        raise ValueError('TSTEP must be 32 for HAR method of feature extraction')

    # Get average skew for 32 days
    skew1 = np.mean(tX[:, :, :], axis=1)
    skew2 = np.mean(tX[:, -5:, :], axis=1)
    skew3 = np.mean(tX[:, -1:, :], axis=1)

    tX = np.concatenate([skew1, skew2, skew3], axis=1)
    return tX 


def har_features(tX, tY, TSTEPS=32):
    """
    Extracts har features of implied volatility which includes an averaged skew for the 1 month, 1 week and 1 day lagging features
    """
    tX = har_transform(tX, TSTEPS=TSTEPS)

    # XXX: New shape is samples, 3 features concatenated
    
    # Fit regression model
    model = Ridge()
    model.fit(tX, tY)
    return model


def har_predict(valX, model, TSTEPS=32):
    valX_transform = har_transform(valX, TSTEPS=TSTEPS)
    return model.predict(valX_transform)    

def tskew_pred(model_name='pca', TSTEPS=10):
    # Load data
    tX, tY, vX, vY, _ = load_data(TSTEPS=TSTEPS)
    tX = tX.reshape(tX.shape[:-1]) 
    vX = vX.reshape(vX.shape[:-1])

    mms = np.arange(pred.LM, pred.UM+pred.MSTEP, pred.MSTEP)

    count = 0

    for j, m in enumerate(mms):
        if count % 10 == 0:
            print('Done: ', count)
        count += 1

        # XXX: shape = samples, TSTEPS, moneyness, term structure
        # No need to reshape as har requires the tstep feature
        tskew = tX[:, :, j]
        tYY = tY[:, j]

        vtskew= vX[:, :, j]
        
        # Fit the model
        if model_name == 'har':
            if TSTEPS != 32:
                continue
            model = har_features(tskew, tYY, TSTEPS=TSTEPS)
            ypred = har_predict(vtskew, model, TSTEPS=TSTEPS)
            # Fit the model
        elif model_name == 'autoencoder':
            encoding_dim = TSTEPS//2 
            encoder, model = autoencoder_fit(tskew, tYY, encoding_dim=encoding_dim, TSTEPS=TSTEPS)
            # transform and validate 
            valX_transform = encoder.predict(vtskew) 
            ypred = model.predict(valX_transform)
        else: # PCA
            n_components = TSTEPS//2 
            model = pca_fit(tskew, tYY, components=n_components, TSTEPS=TSTEPS)
            ypred = pca_predict(vtskew, model, n_components=n_components, TSTEPS=TSTEPS)
            pass

        # Evaluate the model
        rmse = root_mean_squared_error(vY[:, j], ypred, multioutput='raw_values')
        mapes = mean_absolute_percentage_error(vY[:, j], ypred, multioutput='raw_values')
        r2sc = r2_score(vY[:, j], ypred, multioutput='raw_values')

        print('RMSE mean: ', np.mean(rmse), 'RMSE std: ', np.std(rmse))
        print('MAPE mean: ', np.mean(mapes), 'MAPE std: ', np.std(mapes))
        print('R2 mean: ', np.mean(r2sc), 'R2 std: ', np.std(r2sc))

        import pickle
        # Check if directory exists
        if not os.path.exists('./tskew_feature_models'):
            os.makedirs('./tskew_feature_models')
        with open('./tskew_feature_models/%s_ts_%s_%s.pkl' % (model_name, TSTEPS, m), 'wb') as f:
            pickle.dump(model, f)


def mskew_pred(model_name='pca', TSTEPS=10):
    # Load data
    tX, tY, vX, vY, _ = load_data(TSTEPS=TSTEPS)
    tX = tX.reshape(tX.shape[:-1]) 
    vX = vX.reshape(vX.shape[:-1])

    tts = [i/pred.DAYS for i in range(pred.LT, pred.UT+pred.TSTEP, pred.TSTEP)]

    count = 0

    for j, t in enumerate(tts):
        if count % 10 == 0:
            print('Done: ', count)
        count += 1
        # XXX: shape = samples, TSTEPS, moneyness, term structure
        mskew = tX[:, :, :, j]
        tYY = tY[:, :, j]

        vmskew= vX[:, :, :, j]
        
        # Fit the model
        if model_name == 'har':
            if TSTEPS != 32:
                continue
            model = har_features(mskew, tYY, TSTEPS=TSTEPS)
            ypred = har_predict(vmskew, model, TSTEPS=TSTEPS)
            # Fit the model
        elif model_name == 'autoencoder':
            encoding_dim = TSTEPS//2
            encoder, model = autoencoder_fit(mskew, tYY, encoding_dim=encoding_dim, TSTEPS=TSTEPS)
            # transform and validate 
            valX_transform = encoder.predict(vmskew) 
            ypred = model.predict(valX_transform)
        else: # PCA
            n_components = TSTEPS//2 
            model = pca_fit(mskew, tYY, components=n_components, TSTEPS=TSTEPS)
            ypred = pca_predict(vmskew, model, n_components=n_components, TSTEPS=TSTEPS)
            pass

        # Evaluate the model
        rmse = root_mean_squared_error(vY[:, :, j], ypred, multioutput='raw_values')
        mapes = mean_absolute_percentage_error(vY[:, :, j], ypred, multioutput='raw_values')
        r2sc = r2_score(vY[:, :, j], ypred, multioutput='raw_values')

        print('RMSE mean: ', np.mean(rmse), 'RMSE std: ', np.std(rmse))
        print('MAPE mean: ', np.mean(mapes), 'MAPE std: ', np.std(mapes))
        print('R2 mean: ', np.mean(r2sc), 'R2 std: ', np.std(r2sc))

        import pickle
        # Check if directory exists
        if not os.path.exists('./mskew_feature_models'):
            os.makedirs('./mskew_feature_models')

        with open('./mskew_feature_models/%s_ts_%s_%s.pkl' % (model_name, TSTEPS, t), 'wb') as f:
            pickle.dump(model, f)




if __name__ == "__main__":
    # tskew_pred(model='autoencoder', TSTEPS=10) 
    # mskew_pred(model="autoencoder", TSTEPS=20)
    for k in ['autoencoder']:
        for j in [5, 10, 20]:
            mskew_pred(model_name=k, TSTEPS=j)
            tskew_pred(model_name=k, TSTEPS=j)
    # Do a run for the HAR method
    tskew_pred(model='har', TSTEPS=32)
    mskew_pred(model='har', TSTEPS=32)