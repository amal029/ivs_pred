import pandas as pd
import numpy as np
import os
import fnmatch
import zipfile as zip
import keras
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from pred import load_data, load_data_for_keras, regression_predict
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

import pred


def pca_predict(valX, model, n_components, TSTEP):
    valX_transform = pca_transform(valX, n_components, TSTEP)
    return model.predict(valX_transform)

def pca_transform(tX, components=3, TSTEP=32):
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
    
def pca_fit(tX, ty, components=3 , TSTEP=32):
    """
    Uses PCA to extract features from the data 
    and then uses a regression model to predict the implied volatility

    Features extracted will be in the shape of (samples, components*tX.shape[-1])
    i.e. component number of skews for each sample
    """
    tX_transform = pca_transform(tX, components=components, TSTEP=TSTEP) 
    # Fit regression model
    model = Ridge()
    model.fit(tX_transform, ty)
    return model


def har_transform(tX, TSTEP=32):
    """
    Transform the given input data to the HAR method of feature extraction

    Input shape: (samples, TSTEP, moneyness)
    TSTEP must be of size 32 as a month of IV data is required for the HAR method

    Output shape: (samples, tx.shape[-1]*3)
    """
    if TSTEP != 32:
        raise ValueError('TSTEP must be 32 for HAR method of feature extraction')

    # Get average skew for 32 days
    skew1 = np.mean(tX[:, :, :], axis=1)
    skew2 = np.mean(tX[:, -5:, :], axis=1)
    skew3 = np.mean(tX[:, -1:, :], axis=1)

    tX = np.concatenate([skew1, skew2, skew3], axis=1)
    return tX 


def har_features(tX, tY, TSTEP=32):
    """
    Extracts har features of implied volatility which includes an averaged skew for the 1 month, 1 week and 1 day lagging features
    """
    tX = har_transform(tX, TSTEP=TSTEP)

    # XXX: New shape is samples, 3 features concatenated
    
    # Fit regression model
    model = Ridge()
    model.fit(tX, tY)
    return model


def har_predict(valX, model, TSTEP=32):
    valX_transform = har_transform(valX, TSTEP=TSTEP)
    return model.predict(valX_transform)    

     
def main():
    TSTEPS = 20 
    model = 'pca'
    # Load data
    tX, tY, vX, vY, _ = load_data(TSTEPS=TSTEPS)
    tX = tX.reshape(tX.shape[:-1]) 
    vX = vX.reshape(vX.shape[:-1])

    tts = [i/pred.DAYS for i in range(pred.LT, pred.UT+TSTEPS, TSTEPS)]

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
        if model == 'har':
            model = har_features(mskew, tYY, TSTEP=TSTEPS)
            ypred = har_predict(vmskew, model, TSTEP=TSTEPS)
            # Fit the model
        else: # PCA
            model = pca_fit(mskew, tYY, components=3, TSTEP=TSTEPS)
            ypred = pca_predict(vmskew, model, n_components=3, TSTEP=TSTEPS)
            pass

        # Evaluate the model
        rmse = root_mean_squared_error(vY[:, :, j], ypred, multioutput='raw_values')
        mapes = mean_absolute_percentage_error(vY[:, :, j], ypred, multioutput='raw_values')
        r2sc = r2_score(vY[:, :, j], ypred, multioutput='raw_values')

        print('RMSE mean: ', np.mean(rmse), 'RMSE std: ', np.std(rmse))
        print('MAPE mean: ', np.mean(mapes), 'MAPE std: ', np.std(mapes))
        print('R2 mean: ', np.mean(r2sc), 'R2 std: ', np.std(r2sc))

        import pickle
        # with open('./mskew_har_feature_models/%s_ts_%s_%s.pkl' % (model, TSTEPS, t), 'wb') as f:
        #     pickle.dump(model, f)




if __name__ == "__main__":
    main()