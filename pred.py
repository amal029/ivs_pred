#!/usr/bin/env python

import pandas as pd
import os
import fnmatch
import zipfile as mzip
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from statsmodels.nonparametric.kernel_regression import KernelReg
# import tensorflow.math as K
# import tensorflow as tf
import xgboost as xgb
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import glob
from keras.layers import Input, ConvLSTM2D, BatchNormalization, Conv2D
from keras.models import Model
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
# from sklearn.utils.validation import check_array, FLOAT_DTYPES
from sklearn.utils.validation import check_is_fitted
import keras
# from PIL import Image
# XXX: For plotting only
import matplotlib.pyplot as plt
# from sklearn.linear_model import SGDRegressor
from sklearn.cross_decomposition import PLSSVD
from scipy.stats import norm
# from scipy.optimize import curve_fit
from scipy.stats import ttest_1samp
from scipy.optimize import minimize

# XXX: Moneyness Bounds inclusive
LM = 0.9
UM = 1.1
MSTEP = 0.00333

# XXX: Tau Bounds inclusive
LT = 14
UT = 366
TSTEP = 5                       # days

DAYS = 365


def cr2_score_pval(ytrue, y1, y2, greater=True):
    """ytrue: The true series
    y1: prediction 1: (sample, len(moneyness)*len(termstructure)).
    This should be one model that you want to compare against.

    y2: prediction 2: (sample, len(moneyness)*len(termstructure)). This
    is the second model that you want to compare against.

    """
    f = np.sum((ytrue - y1)**2, axis=1)
    s = np.sum((ytrue - y2)**2 - (y2 - y1)**2, axis=1)
    v = f - s
    if greater:
        resg = ttest_1samp(v, 0.0, alternative='greater')
    else:
        resg = ttest_1samp(v, 0.0)
    return resg.pvalue


def cr2_score(ytrue, y1, y2):
    """Own r2score from paper: Are there gains from using information
    over the surface of implied volatilies?
    ytrue: true time series
    y1: prediction from model 1(benchmark to compare against)
    y2  prediction from model 2

    """
    assert (ytrue.shape == y1.shape)
    assert (ytrue.shape == y2.shape)
    assert (len(ytrue.shape) == 2)
    num = np.sum((ytrue - y2)**2)
    den = np.sum((ytrue - y1)**2)
    return 1 - (num/den)


# XXX: Curve fit to get the average (expected lambda)
def ym(m, b0, b1, b2, lam):
    return (b0 +
            b1*((1-np.exp(-lam*m))/(lam*m)) +
            b2*(((1-np.exp(-lam*m))/(lam*m))-np.exp(-lam*m)))


# XXX: The class to perform Nelson-Siegel prediction of term structure
class NS:
    def __init__(self, xdf, TSTEPS, model):
        if model == 'tsknsridge' or model == 'msknsridge':
            self.reg = Ridge()
        elif model == 'tsknslasso' or model == 'msknslasso':
            self.reg = Lasso()
        elif model == 'tsknsenet' or model == 'msknsenet':
            self.reg = ElasticNet()
        self.xdf = xdf
        self.TSTEPS = TSTEPS

    def _getcoefs(self, vec, y=None):
        vec = vec[:, :-1]
        vec = vec.reshape(vec.shape[0], self.TSTEPS,
                          vec.shape[1]//self.TSTEPS)

        # XXX: There are 3 coefficients in the latent NS space
        def ffit(i):
            res = list()
            for j in range(vec.shape[1]):
                lreg = LinearRegression(n_jobs=-1).fit(self.xdf, vec[i, j])
                res += [lreg.coef_[0], lreg.coef_[1], lreg.intercept_]
            return res

        # XXX: For each sample and each lag convert to latent NS space.
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=-1)(delayed(ffit)(i)
                                  for i in range(vec.shape[0]))
        xcoefs = np.array(res)
        if y is not None:
            ycoefs = np.array([1.0]*y.shape[0]*3).reshape(y.shape[0], 3)
            # XXX: Get the betas for the output too!
            for i in range(y.shape[0]):
                lreg = LinearRegression().fit(self.xdf, y[i])
                ycoefs[i] = [lreg.coef_[0], lreg.coef_[1], lreg.intercept_]
            return xcoefs, ycoefs
        else:
            return xcoefs

    def fit(self, vec, y):
        # XXX: Fit the model for x to y coefficients
        xcoefs, ycoefs = self._getcoefs(vec, y)
        self.reg = self.reg.fit(xcoefs, ycoefs)
        return self.reg

    def _predict(self, pycoefs):
        yp = np.array([1.0]*pycoefs.shape[0]*self.xdf.shape[0]).reshape(
            pycoefs.shape[0], self.xdf.shape[0])
        for i in range(pycoefs.shape[0]):
            yp[i] = np.dot(self.xdf, pycoefs[i, :2]) + pycoefs[i, 2]
        return yp

    def score(self, vec, y):
        check_is_fitted(self.reg)
        xcoefs = self._getcoefs(vec)
        pycoefs = self.reg.predict(xcoefs)
        yp = self._predict(pycoefs)
        return r2_score(y, yp)

    def predict(self, vec):
        # XXX: Predit the output given the input
        xcoefs = self._getcoefs(vec)
        pycoefs = self.reg.predict(xcoefs)
        return self._predict(pycoefs)


# XXX: The class to perform pls based regression
class MPls:
    def __init__(self, n_components, intercept, t='plsridge'):
        self.__n_components = n_components
        self._reg_svd = PLSSVD(n_components=n_components)
        self.xmean = 0
        self.ymean = 0
        self.xstd = 0
        self.ystd = 0
        if (t == 'plsridge' or t == 'pmplsridge' or t == 'mskplsridge' or (
                t == 'tskplsridge')):
            # print('Doing %s' % t)
            self._reg = Ridge(fit_intercept=intercept)
        elif t == 'plslasso' or t == 'pmplslasso' or t == 'mskplslasso' or (
                t == 'tskplslasso'):
            # print('Doing %s' % t)
            self._reg = Lasso(fit_intercept=intercept)
        elif t == 'plsenet' or t == 'pmplsenet' or t == 'mskplsenet' or (
                t == 'tskplsenet'):
            # print('Doing %s' % t)
            self._reg = ElasticNet(fit_intercept=intercept)

    def _center_scale_xy(self, X, Y, scale=True):
        # print(X.shape, Y.shape)
        # center
        x_mean = X.mean(axis=0)
        X -= x_mean
        y_mean = Y.mean(axis=0)
        Y -= y_mean
        # scale
        if scale:
            x_std = X.std(axis=0, ddof=1)
            x_std[x_std == 0.0] = 1.0
            X /= x_std
            y_std = Y.std(axis=0, ddof=1)
            if type(y_std) is not np.float64:
                y_std[y_std == 0.0] = 1.0
            else:
                y_std = 1.0 if y_std == 0.0 else y_std
                Y /= y_std
        else:
            x_std = np.ones(X.shape[1])
            y_std = np.ones(Y.shape[1])
        return x_mean, y_mean, x_std, y_std

    def fit(self, tX, tY):
        (self.xmean, self.ymean,
         self.xstd, self.ystd) = self._center_scale_xy(np.copy(tX),
                                                       np.copy(tY))
        tXX, tYY = self._reg_svd.fit_transform(tX, tY)
        # print(tXX.shape, tYY.shape)
        self._reg = self._reg.fit(tXX, tYY)
        # print('Ridge r2: ', self._reg.score(tXX, tYY))
        # tXX = check_array(tXX, input_name='X', dtype=FLOAT_DTYPES)
        # tXX1 = tXX @ self._reg_svd.x_weights_.T * self.xstd + self.xmean
        # tYY = check_array(tYY, input_name='y', dtype=FLOAT_DTYPES)
        # tYY1 = tYY @ self._reg_svd.y_weights_.T * self.ystd + self.ymean
        # print('X SVD r2: ', r2_score(tX, tXX1))
        # print('Y SVD r2: ', r2_score(tY, tYY1))
        return self

    def predict(self, tX):
        tXX = self._reg_svd.transform(tX)
        check_is_fitted(self._reg_svd)
        check_is_fitted(self._reg)
        # print(tXX.shape)
        vYY = self._reg.predict(tXX)
        # print(vYY.shape)
        # vY = vYY @ self._reg_svd.y_weights_.T * self.ystd + self.ymean
        vYY = vYY.reshape(vYY.shape[0], self._reg_svd.y_weights_.T.shape[0])
        vY = np.dot(vYY, self._reg_svd.y_weights_.T) * self.ystd + self.ymean
        return vY

    def score(self, tX, tY):
        tXX = self._reg_svd.transform(tX)
        vYY = self._reg.predict(tXX)
        vYY = vYY.reshape(vYY.shape[0], self._reg_svd.y_weights_.T.shape[0])
        vY = np.dot(vYY, self._reg_svd.y_weights_.T) * self.ystd + self.ymean
        return r2_score(tY, vY)


# XXX: Chalamandaris and Tsekrekos (2015) model
class CT:
    def __init__(self, model, mms, TTS, TSTEPS, LAMBDA=0.0147):
        self.TSTEPS = TSTEPS
        self.MMS = len(mms)
        self.TTS = len(TTS)
        # XXX: Make the data frame for fitting
        self.df = pd.DataFrame({'m': np.array([[m]*len(TTS)
                                               for m in mms]).reshape(
                                                       len(mms)*len(TTS)),
                                't': np.array(TTS*len(mms))})
        self.df['mlt1'] = (self.df['m'] < 1.0).astype(int, copy=False)
        self.df['mgeq1'] = (self.df['m'] >= 1.0).astype(int, copy=False)
        self.df['m2'] = self.df['m']**2
        self.df['mt'] = self.df['m']*self.df['t']
        # XXX: The required model values
        self.df['b0'] = 1
        self.df['b1'] = self.df['m2']*self.df['mgeq1']
        self.df['b2'] = self.df['m2']*self.df['mlt1']
        texp = (self.df['t']*-LAMBDA).apply(np.exp)
        lt = self.df['t']*LAMBDA
        self.df['b3'] = (1 - texp)/lt
        self.df['b4'] = self.df['b3'] - texp
        self.df['b5'] = self.df['mgeq1']*self.df['mt']
        self.df['b6'] = self.df['mlt1']*self.df['mt']

        if model == 'ctridge':
            self.reg = Ridge()
        elif model == 'ctlasso':
            self.reg = Lasso()
        elif model == 'ctenet':
            self.reg = ElasticNet()

    def fitX(self, X, features):
        # XXX: Do this TSTEPS time
        res = list()
        for i in range(X.shape[0]):
            lreg = LinearRegression(n_jobs=-1,
                                    fit_intercept=False).fit(features,
                                                             X[i])
            res.append(lreg.coef_)
        return np.array(res)

    def fitY(self, Y, features):
        lreg = LinearRegression(n_jobs=-1,
                                fit_intercept=False).fit(features, Y)
        return lreg.coef_

    def fit(self, tX, tY):
        tX = tX.reshape(tX.shape[0], self.TSTEPS, self.MMS*self.TTS)
        tY = tY.reshape(tX.shape[0], self.MMS*self.TTS)
        features = self.df[['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6']]
        from joblib import Parallel, delayed
        resY = np.array(Parallel(n_jobs=-1)(delayed(self.fitY)(tY[i], features)
                                            for i in range(tY.shape[0])))
        # XXX: Fit the X
        resX = np.array(Parallel(n_jobs=-1)(delayed(self.fitX)(tX[i], features)
                                            for i in range(tX.shape[0])))
        resX = resX.reshape(resX.shape[0], resX.shape[1]*resX.shape[2])

        # XXX: Now fit the coefficients learning model
        self.reg.fit(resX, resY)
        return self

    def score(self, tX, tY):
        check_is_fitted(self.reg)
        tX = tX.reshape(tX.shape[0], self.TSTEPS, self.MMS*self.TTS)
        tY = tY.reshape(tX.shape[0], self.MMS*self.TTS)
        features = self.df[['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6']]
        from joblib import Parallel, delayed
        # XXX: Fit the X
        resX = np.array(Parallel(n_jobs=-1)(delayed(self.fitX)(tX[i], features)
                                            for i in range(tX.shape[0])))
        resX = resX.reshape(resX.shape[0], resX.shape[1]*resX.shape[2])
        # XXX: Predict the next day' coefficients
        presY = self.reg.predict(resX)
        # XXX: do a dot product to get the pY
        pY = np.dot(presY, features.T)
        return r2_score(tY, pY)

    def predict(self, tX):
        check_is_fitted(self.reg)
        tX = tX.reshape(tX.shape[0], self.TSTEPS, self.MMS*self.TTS)
        pass
        features = self.df[['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6']]
        from joblib import Parallel, delayed
        # XXX: Fit the X
        resX = np.array(Parallel(n_jobs=-1)(delayed(self.fitX)(tX[i], features)
                                            for i in range(tX.shape[0])))
        resX = resX.reshape(resX.shape[0], resX.shape[1]*resX.shape[2])
        # XXX: Predict the next day' coefficients
        presY = self.reg.predict(resX)
        # XXX: do a dot product to get the pY
        pY = np.dot(presY, features.T)
        return pY


# XXX: The static arb free (risk neutral measure) fit called surface
# stochastic volatility inspired (SSVI) by Gatheral 2013.
class SSVI:
    # SSVI
    def SSVI(self, theta, T, rho, nu):
        phi = nu / (theta**0.5)  # power law
        result = (0.5 * theta) * (1 + rho * phi * self.k +
                                  np.sqrt((phi * self.k + rho)**2 +
                                          1 - rho**2))
        return np.sqrt(result/T)

    def __init__(self, model, TSTEPS):
        self.isfitted = False
        self.bnds = [(-1+1e-6, 1-1e-6), (0+1e-6, np.inf)]
        # self.cons2 = [{'type': 'ineq', 'fun': self.Heston_condition}]
        self.mms = np.arange(LM, UM+MSTEP, MSTEP)
        self.tts = np.array([i/365 for i in range(LT, UT+TSTEP, TSTEP)])
        self.k = np.log(self.mms)         # log moneyness
        self.MMS = self.mms.size
        self.TTS = self.tts.size
        self.TSTEPS = TSTEPS
        self.atmi = np.where(np.isclose(self.mms, 0.9999))[0][0]

        # XXX: The models
        if model == 'ssviridge':
            self.reg = Ridge()
        elif model == 'ssvilasso':
            self.reg = Lasso()
        elif model == 'ssvienet':
            self.reg = ElasticNet()

        # XXX: For predicting the ATM IV
        self.atmreg = Ridge()

    # XXX: Fit for a given Day
    def fitADay(self, params, Y):
        rho, nu = params
        THETAS = Y[self.atmi]**2*self.tts
        pY = list()
        # XXX: Go slice by slice
        for ti, T in enumerate(self.tts):
            theta = THETAS[ti]
            pY.append(self.SSVI(theta, T, rho, nu))
        # XXX: Now do a transpose of res
        pY = np.array(pY).T
        # XXX: Now do a sum of square differences
        return np.sum((Y - pY)**2)

    def fitY(self, tY):
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=-1)(
            delayed(minimize)(self.fitADay,
                              x0=[0.4, 0.1],
                              args=(tY[i]),
                              bounds=self.bnds,
                              method='COBYQA',
                              options={'disp': False})
            for i in range(tY.shape[0]))

        # XXX: Add the target THETAS for prediciting the ATM IV
        tthetas = list()
        for y in tY:
            tthetas.append(y[self.atmi])

        # XXX: These are the targets for predicting
        res = np.array([[i.x[0], i.x[1]] for i in res])
        return res, np.array(tthetas)

    def fitX(self, tX):
        def __fitX(D):
            res = list()
            for d in D:
                m = minimize(self.fitADay, x0=[0.4, 0.1], args=(d),
                             bounds=self.bnds,
                             method='COBYQA',
                             options={'disp': False})
                m = [m.x[0], m.x[1]]
                res.append(m)
            return res

        from joblib import Parallel, delayed
        res = Parallel(n_jobs=-1)(delayed(__fitX)(tX[i])
                                  for i in range(tX.shape[0]))
        fthetas = list()
        for x in tX:
            thetas = list()
            for x1 in x:
                thetas.append(x1[self.atmi])
            fthetas.append(thetas)

        return np.array(res), np.array(fthetas)

    def check_is_fitted(self):
        return self.isfitted

    def fit(self, tX, tY):
        tX = tX.reshape(tX.shape[0], self.TSTEPS, self.MMS, self.TTS)
        tY = tY.reshape(tX.shape[0], self.MMS, self.TTS)

        # XXX: Fit the features
        features, fthetas = self.fitX(tX)
        features = features.reshape(features.shape[0],
                                    features.shape[1]*features.shape[2])

        # XXX: Fit the targets
        targets, tthetas = self.fitY(tY)

        # from statsmodels.graphics.tsaplots import plot_pacf
        # plot_pacf(features[:, 0])
        # plt.show()

        # XXX: Fit the model -- this score is very low!
        self.reg = self.reg.fit(features, targets)
        # print('reg score: ', self.reg.score(features, targets))

        # XXX: Fit the theta prediction model
        fthetas = fthetas.reshape(fthetas.shape[0],
                                  fthetas.shape[1]*fthetas.shape[2])
        self.atmreg = self.atmreg.fit(fthetas, tthetas)
        # print('atm score: ', self.atmreg.score(fthetas, tthetas))
        # XXX: I am now fitted
        self.isfitted = True
        return self

    def score(self, tX, tY):
        pY = self.predict(tX)
        return r2_score(tY, pY)

    def predict1(self, params, THETAS):
        rho, nu = params[0], params[1]
        THETAS = THETAS**2*self.tts
        pY = list()
        for ti, T in enumerate(self.tts):
            theta = THETAS[ti]
            pY.append(self.SSVI(theta, T, rho, nu))
        pY = np.array(pY).T
        return pY

    def predict(self, tX):
        tX = tX.reshape(tX.shape[0], self.TSTEPS, self.MMS, self.TTS)
        features, fthetas = self.fitX(tX)
        features = features.reshape(features.shape[0],
                                    features.shape[1]*features.shape[2])
        check_is_fitted(self.reg)
        pftargets = self.reg.predict(features)
        check_is_fitted(self.atmreg)
        pttargets = self.atmreg.predict(fthetas.reshape(fthetas.shape[0],
                                                        (fthetas.shape[1] *
                                                         fthetas.shape[2])))
        from joblib import Parallel, delayed
        pY = Parallel(n_jobs=-1)(
            delayed(self.predict1)(pftargets[i], pttargets[i])
            for i in range(pttargets.shape[0]))
        pY = np.array(pY)
        pY = pY.reshape(pY.shape[0], pY.shape[1]*pY.shape[2])
        return pY


def getr(row, mk):
    """otype: the type of option
       row: row of the dataframe
       row[delta]: the delta greek
       row[tau]: the time to expiry (days left)/365
       row[S]: current underlying price
       row[K]: strike price
       row[sigma]: volatility
    return: risk free interest rate
    """

    delta = row['Delta']
    sigma = row['IV']
    t = row['tau']
    S = row['UnderlyingPrice']
    K = row['Strike']
    otype = row['Type']

    if otype == 'call':
        d1 = norm.ppf(delta)
    else:
        d1 = norm.ppf(1 + delta)
        # XXX: Now compute the interest rate
    r = ((d1 * sigma * np.sqrt(t)) - np.log(S/K))/t - (sigma**2/2)
    # XXX: DEBUG
    dd1 = 1/(sigma*np.sqrt(t)) * (np.log(S/K) + (r + sigma**2/2)*t)
    if otype == 'call':
        if not np.isclose(delta, norm.cdf(dd1)):
            print('call:', delta, norm.cdf(dd1), mk)
            print('d1:', d1)
            print(row)
    else:
        if not np.isclose(delta, norm.cdf(dd1)-1):
            print('put: ', delta, norm.cdf(dd1)-1, mk)
            print('d1:', d1)
            print(row)
    return r


# XXX: 20220322 has a number of nans
def interest_rates(dfs: dict):
    for k in dfs.keys():
        df = dfs[k]
        # XXX: First only get those that have volume > 0
        df = df[df['Volume'] > 0].reset_index(drop=True)
        # XXX: Make the log of K/UnderlyingPrice
        df['m'] = (df['Strike']/df['UnderlyingPrice'])
        # XXX: Moneyness is not too far away from ATM
        df = df[(df['m'] >= LM) & (df['m'] <= UM)]
        # XXX: Make the days to expiration
        df['Expiration'] = pd.to_datetime(df['Expiration'])
        df['DataDate'] = pd.to_datetime(df['DataDate'])
        df['tau'] = (df['Expiration'] - df['DataDate']).dt.days
        # XXX: Only those that are greater than at least 2 weeks ahead
        # and also not too ahead
        df = df[(df['tau'] >= LT) & (df['tau'] <= UT)]
        df['tau'] = df['tau']/DAYS

        # XXX: implied volatility is not zero!
        df = df[df['IV'] != 0]

        # XXX: Compute the interest rates
        dfr = df[['Delta', 'tau', 'UnderlyingPrice', 'Strike', 'IV', 'Type']]
        df['InterestR'] = dfr.apply(lambda d: getr(d, k), axis=1)
        df['ForwardP'] = df['UnderlyingPrice']*np.exp(
            df['InterestR']*df['tau'])
        df['Mid'] = (df['Ask'] + df['Bid'])/2
        # XXX: Numpy array
        dfr = df.drop(['AKA', 'Exchange'], axis=1)
        dfr = dfr.reset_index(drop=True)
        dfr.to_csv('./interest_rates/%s.csv' % (k))


def preprocess_ivs_df(dfs: dict, otype):
    toret = dict()
    for k in dfs.keys():
        df = dfs[k]
        # XXX: First only get those that have volume > 0
        df = df[(df['Volume'] > 0) &
                (df['Type'] == otype)].reset_index(drop=True)
        # XXX: Make the log of K/UnderlyingPrice
        df['m'] = (df['Strike']/df['UnderlyingPrice'])
        # XXX: Moneyness is not too far away from ATM
        df = df[(df['m'] >= LM) & (df['m'] <= UM)]
        # XXX: Make the days to expiration
        df['Expiration'] = pd.to_datetime(df['Expiration'])
        df['DataDate'] = pd.to_datetime(df['DataDate'])
        df['tau'] = (df['Expiration'] - df['DataDate']).dt.days
        # XXX: Only those that are greater than at least 2 weeks ahead
        # and also not too ahead
        df = df[(df['tau'] >= LT) & (df['tau'] <= UT)]
        df['tau'] = df['tau']/DAYS
        df['m2'] = df['m']**2
        df['tau2'] = df['tau']**2
        df['mtau'] = df['m']*df['tau']
        # XXX: This is the final dataframe
        dff = df[['IV', 'm', 'tau', 'm2', 'tau2', 'mtau']]
        toret[k] = dff.reset_index(drop=True)
    return toret


def plot_hmap(ivs_hmap, mrows, tcols, otype, dd='figs'):
    for k in ivs_hmap.keys():
        np.save('/tmp/%s_%s/%s.npy' % (otype, dd, k), ivs_hmap[k])


def plot_ivs(ivs_surface, IVS='IVS', view='XY'):
    for k in ivs_surface.keys():
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        X = ivs_surface[k]['m']
        Y = ivs_surface[k]['tau']
        if IVS == 'IVS':
            Z = ivs_surface[k][IVS]*100
        else:
            Z = ivs_surface[k][IVS]
            # viridis = cm.get_cmap('gist_gray', 256)
        _ = ax.plot_trisurf(X, Y, Z, cmap='afmhot',
                            linewidth=0.2, antialiased=True)
        # ax.set_xlabel('m')
        # ax.set_ylabel('tau')
        # ax.set_zlabel(IVS)
        # ax.view_init(azim=-45, elev=30)
        # ax.invert_xaxis()
        # ax.invert_yaxis()
        if view == 'XY':
            ax.view_init(elev=90, azim=-90)
        elif view == 'XZ':
            ax.view_init(elev=0, azim=-90)
        elif view == 'YZ':
            ax.view_init(elev=0, azim=0)
            ax.axis('off')
            # ax.zaxis.set_major_formatter('{x:.02f}')
            # fig.colorbar(surf, shrink=0.5, aspect=5)
            # plt.show()
            # fig.subplots_adjust(bottom=0)
            # fig.subplots_adjust(top=0.00001)
            # fig.subplots_adjust(right=1)
            # fig.subplots_adjust(left=0)
        plt.savefig('/tmp/figs/{k}_{v}.png'.format(k=k, v=view),
                    bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        # XXX: Convert to gray scale 1 channel only
        # img = Image.open('/tmp/figs/{k}.png'.format(k=k)).convert('LA')
        # img.save('/tmp/figs/{k}.png'.format(k=k))


def load_data_for_keras(otype, dd='./figs', START=0, NUM_IMAGES=1000, TSTEP=1):

    Xs = list()               # Training inputs [0..TEST_IMAGES-1]
    Ys = list()               # Training outputs [1..TEST_IMAGES]
    Ysdates = list()
    ff = sorted(glob.glob('./'+otype+'_'+dd.split('/')[1]+'/*.npy'))
    # XXX: Load the first TEST_IMAGES for training
    # print('In load image!')
    for i in range(START, START+NUM_IMAGES):
        # print('i is: ', i)
        for j in range(TSTEP):
            # print('j is: ', j)
            img = np.load(ff[i+j])   # PIL image
            np.all((img > 0) & (img <= 1))
            # print('loaded i, j: X(i+j)', i, j,
            #       ff[i+j].split('/')[-1].split('.')[0])
            Xs += [img]

        # XXX: Just one output image to compare against
        # XXX: Now do the same thing for the output label image
        # img = Image.open(ff[i+TSTEP]).convert('LA')
        img = np.load(ff[i+TSTEP])   # PIL image
        np.all((img > 0) & (img <= 1))
        # print('loaded Y: i, TSTEP: (i+TSTEP)', i, TSTEP,
        #       ff[(i+TSTEP)].split('/')[-1].split('_')[0])
        Ysdates.append(ff[(i+TSTEP)].split('/')[-1].split('.')[0])
        Ys += [img]

    # XXX: Convert the lists to np.array
    Xs = np.array(Xs)
    Ys = np.array(Ys)
    np.expand_dims(Xs, axis=-1)
    np.expand_dims(Ys, axis=-1)
    return Xs, Ys, Ysdates


def load_image_for_keras(dd='./figs', START=0, NUM_IMAGES=1000, NORM=255,
                         RESIZE_FACTOR=8, TSTEP=1):

    Xs = list()               # Training inputs [0..TEST_IMAGES-1]
    Ys = list()               # Training outputs [1..TEST_IMAGES]
    Ysdates = list()
    ff = sorted(glob.glob(dd+'/*.png'))
    # XXX: Load the first TEST_IMAGES for training
    # print('In load image!')
    for i in range(START, START+NUM_IMAGES):
        # print('i is: ', i)
        for j in range(TSTEP):
            # print('j is: ', j)
            # img = Image.open(ff[i+j]).convert('LA')
            img = load_img(ff[i+j])   # PIL image
            # print('loaded i, j: X(i+j)', i, j,
            #       ff[i+j].split('/')[-1].split('_')[0])
            w, h = img.size
            img = img.resize((w//RESIZE_FACTOR, h//RESIZE_FACTOR))
            img_gray = img.convert('L')
            img_array = img_to_array(img_gray)
            # img_array = rgb2gray(img_array)  # make it gray scale
            Xs += [img_array/NORM]

        # XXX: Just one output image to compare against
        # XXX: Now do the same thing for the output label image
        # img = Image.open(ff[i+TSTEP]).convert('LA')
        img = load_img(ff[i+TSTEP])   # PIL image
        # print('loaded Y: i, TSTEP: (i+TSTEP)', i, TSTEP,
        #       ff[(i+TSTEP)].split('/')[-1].split('_')[0])
        Ysdates.append(ff[(i+TSTEP)].split('/')[-1].split('_')[0])
        w, h = img.size
        img = img.resize((w//RESIZE_FACTOR, h//RESIZE_FACTOR))
        img_gray = img.convert('L')
        img_array1 = img_to_array(img_gray)
        # img_array = rgb2gray(img_array1)
        Ys += [img_array1/NORM]

        # DEBUG
        # XXX: Plot the images
        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # ax1.title.set_text('color')
        # ax2.title.set_text('gray')
        # ax1.imshow(img, cmap='gray')
        # ax2.imshow(img_gray, cmap='gray')
        # plt.show()
        # plt.close(fig)

    # XXX: Convert the lists to np.array
    Xs = np.array(Xs)
    Ys = np.array(Ys)
    np.expand_dims(Xs, axis=-1)
    np.expand_dims(Ys, axis=-1)
    return Xs, Ys, Ysdates


def main(mdir, years, months, instrument, dfs: dict):
    ff = []
    for f in os.listdir(mdir):
        for y in years:
            for m in months:
                # XXX: Just get the year and month needed
                tosearch = "*_{y}_{m}*.zip".format(y=y, m=m)
                if fnmatch.fnmatch(f, tosearch):
                    ff += [f]
                    # print(ff)
                    # XXX: Read the csvs
    for f in ff:
        z = mzip.ZipFile(mdir+f)
        ofs = [i for i in z.namelist() if 'options_' in i]
        # print(ofs)
        # XXX: Now read just the option data files
        for f in ofs:
            key = f.split(".csv")[0].split("_")[2]
            df = pd.read_csv(z.open(f))
            df = df[df['UnderlyingSymbol'] == instrument].reset_index(
                drop=True)
            dfs[key] = df


def build_gird_and_images_gaussian(df, otype):
    # print('building grid and fitting')
    # XXX: Now fit a multi-variate linear regression to the dataset
    # one for each day.
    df = dict(sorted(df.items()))
    fitted_dict = dict()
    grid = dict()
    scores = list()
    for k in df.keys():
        # print('doing key: ', k)
        y = df[k]['IV']
        X = df[k][['m', 'tau']]
        # print('fitting')
        reg = KernelReg(endog=y, exog=X, var_type='cc',
                        reg_type='lc')
        # reg.fit()               # fit the model
        # print('fitted')
        fitted_dict[k] = reg
        scores += [reg.r_squared()]
        # XXX: Now make the grid
        ss = []
        mms = np.arange(LM, UM+MSTEP, MSTEP)
        tts = [i/DAYS for i in range(LT, UT+TSTEP, TSTEP)]
        # print('making grid: ', len(mms), len(tts))
        for mm in mms:
            for tt in tts:
                # XXX: Make the feature vector
                ss.append([mm, tt])

        grid[k] = pd.DataFrame(ss, columns=['m', 'tau'])
        # print('made grid and output')

    print("average fit score: ", sum(scores)/len(scores))
    # XXX: Now make the smooth ivs surface for each day
    ivs_surf_hmap = dict()
    ivs_surface = dict()
    for k in grid.keys():
        # XXX: This ivs goes m1,t1;m1,t2... then
        # m2,t1;m2,t2,m2,t3.... this is why reshape for heat map as
        # m, t, so we get m rows and t cols. Hence, x-axis is t and
        # y-axis is m.
        pivs, _ = fitted_dict[k].fit(grid[k])
        ivs_surface[k] = pd.DataFrame({'IVS': pivs,
                                       'm': grid[k]['m'],
                                       'tau': grid[k]['tau']})
        ivs_surface[k]['IVS'] = ivs_surface[k]['IVS']
        # print('IVS len:', len(ivs_surface[k]['IVS']))
        mcount = len(mms)
        tcount = len(tts)
        # print('mcount%s, tcount%s: ' % (mcount, tcount))
        ivs_surf_hmap[k] = ivs_surface[k]['IVS'].values.reshape(mcount,
                                                                tcount)
        # print('ivs hmap shape: ', ivs_surf_hmap[k].shape)

    # XXX: Plot the heatmap
    plot_hmap(ivs_surf_hmap, mms, tts, otype, dd='gfigs')


def build_gird_and_images(df, otype):
    # print('building grid and fitting')
    # XXX: Now fit a multi-variate linear regression to the dataset
    # one for each day.
    df = dict(sorted(df.items()))
    fitted_dict = dict()
    grid = dict()
    scores = list()
    for k in df.keys():
        # print('doing key: ', k)
        y = df[k]['IV']
        X = df[k][['m', 'tau', 'm2', 'tau2', 'mtau']]
        # print('fitting')
        reg = LinearRegression(n_jobs=-1).fit(X, y)
        # print('fitted')
        fitted_dict[k] = reg
        scores += [reg.score(X, y)]

        # XXX: Now make the grid
        ss = []
        mms = np.arange(LM, UM+MSTEP, MSTEP)
        tts = [i/DAYS for i in range(LT, UT+TSTEP, TSTEP)]
        # print('making grid: ', len(mms), len(tts))
        for mm in mms:
            for tt in tts:
                # XXX: Make the feature vector
                ss.append([mm, tt, mm**2, tt**2, mm*tt])

        grid[k] = pd.DataFrame(ss, columns=['m', 'tau', 'm2', 'tau2', 'mtau'])
        # print('made grid and output')

    print("average fit score: ", sum(scores)/len(scores))
    # XXX: Now make the smooth ivs surface for each day
    ivs_surf_hmap = dict()
    ivs_surface = dict()
    for k in grid.keys():
        # XXX: This ivs goes m1,t1;m1,t2... then
        # m2,t1;m2,t2,m2,t3.... this is why reshape for heat map as
        # m, t, so we get m rows and t cols. Hence, x-axis is t and
        # y-axis is m.
        pivs = fitted_dict[k].predict(grid[k])
        ivs_surface[k] = pd.DataFrame({'IVS': pivs,
                                       'm': grid[k]['m'],
                                       'tau': grid[k]['tau']})
        ivs_surface[k]['IVS'] = ivs_surface[k]['IVS'].clip(0.01, None)
        # print('IVS len:', len(ivs_surface[k]['IVS']))
        mcount = len(mms)
        tcount = len(tts)
        # print('mcount%s, tcount%s: ' % (mcount, tcount))
        ivs_surf_hmap[k] = ivs_surface[k]['IVS'].values.reshape(mcount,
                                                                tcount)
        # print('ivs hmap shape: ', ivs_surf_hmap[k].shape)

    # XXX: Plot the heatmap
    plot_hmap(ivs_surf_hmap, mms, tts, otype)

    # XXX: Plot the ivs surface
    # plot_ivs(ivs_surface, view='XY')


def excel_to_images(dvf=True, otype='call', ironly=False):
    dir = '../../HistoricalOptionsData/'
    years = [str(i) for i in range(2002, 2024)]
    months = [
        'January', 'February',
        'March',
        'April', 'May', 'June', 'July',
        'August', 'September', 'October', 'November', 'December'
    ]
    instrument = ["SPX"]
    dfs = dict()
    # XXX: The dictionary of all the dataframes with the requires
    # instrument ivs samples
    for i in instrument:
        # XXX: Load the excel files
        main(dir, years, months, i, dfs)

        if ironly:
            interest_rates(dfs)
        else:
            # XXX: Now make ivs surface for each instrument
            df = preprocess_ivs_df(dfs, otype)

            if dvf:
                # XXX: Build the 2d matrix with DVF
                build_gird_and_images(df, otype)
            else:
                # XXX: Build the 2d matrix with NW
                build_gird_and_images_gaussian(df, otype)


def build_keras_model(shape, inner_filters, bs, LR=1e-3):
    inp = Input(shape=shape[1:], batch_size=bs)
    x = ConvLSTM2D(
        filters=32,
        kernel_size=(7, 7),
        padding="same",
        data_format='channels_last',
        activation='relu',
        # dropout=0.2,
        # recurrent_dropout=0.1,
        # stateful=True,
        return_sequences=True)(inp)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(
        filters=64,
        kernel_size=(5, 5),
        data_format='channels_last',
        padding='same',
        # dropout=0.2,
        # recurrent_dropout=0.1,
        activation='relu',
        # stateful=True,
        return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(
        filters=inner_filters,
        kernel_size=(3, 3),
        data_format='channels_last',
        padding='same',
        # dropout=0.2,
        # recurrent_dropout=0.1,
        activation='relu',
        # stateful=True,
        return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(
        filters=inner_filters,
        kernel_size=(1, 1),
        data_format='channels_last',
        padding='same',
        # dropout=0.1,
        # recurrent_dropout=0.1,
        activation='relu',
        # stateful=True
        )(x)
    # XXX: 3D layer for images, 1 for each timestep
    x = Conv2D(
        filters=1, kernel_size=(1, 1), activation="relu",
        padding="same")(x)
    # XXX: Take the average in depth
    # x = keras.layers.AveragePooling3D(pool_size=(shape[1], 1, 1),
    #                                   padding='same',
    #                                   data_format='channels_last')(x)
    # XXX: Flatten the output
    # x = keras.layers.Flatten()(x)
    # # XXX: Dense layer for 1 output image
    # tot = 1
    # for i in shape[2:]:
    #     tot *= i
    # # print('TOT:', tot)
    # x = keras.layers.Dense(units=tot, activation='relu')(x)

    # # XXX: Reshape the output
    x = keras.layers.Reshape(shape[2:4])(x)

    # XXX: The complete model and compiled
    model = Model(inp, x)
    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=keras.optimizers.Adam(learning_rate=LR))

    # def loss(ytrue, ypred):
    #     ytrue = tf.reshape(ytrue, (ytrue.shape[0],
    #                                ytrue.shape[1]*ytrue.shape[2]))
    #     ypred = tf.reshape(ypred, (ypred.shape[0],
    #                                ypred.shape[1]*ypred.shape[2]))
    #     num = K.reduce_sum(K.square(ytrue-ypred), axis=-1)
    #     den = K.reduce_sum(K.square(ytrue - K.reduce_mean(ytrue)),
    #                        axis=-1)
    #     res = 1 - (num/den)
    #     return (-res)

    # model.compile(loss=loss,
    #               optimizer=keras.optimizers.Adam(learning_rate=LR))
    return model


def keras_model_fit(model, trainX, trainY, valX, valY, batch_size):
    # Define some callbacks to improve training.
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss",
                                                   patience=10,
                                                   restore_best_weights=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                  patience=5)

    # Define modifiable training hyperparameters.
    epochs = 500
    # batch_size = 2

    # Fit the model to the training data.
    history = model.fit(
        trainX,                 # this is not a 5D tensor right now!
        trainY,                 # this is not a 5D tensor right now!
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(valX, valY),
        verbose=1,
        callbacks=[early_stopping, reduce_lr])
    # callbacks=[reduce_lr])
    return history


def load_data(otype, dd='./figs', TSTEPS=10):
    NIMAGES1 = 2000
    # XXX: This is very important. If too long then changes are not
    # shown. If too short then too much influence from previous lags.
    TSTEPS = TSTEPS
    START = 0

    # Load, process and learn a ConvLSTM2D network
    trainX, trainY, _ = load_data_for_keras(otype, dd=dd,
                                            START=START, NUM_IMAGES=NIMAGES1,
                                            TSTEP=TSTEPS)
    # print(trainX.shape, trainY.shape)
    trainX = trainX.reshape(trainX.shape[0]//TSTEPS, TSTEPS,
                            *trainX.shape[1:], 1)
    # trainY = trainY.reshape(trainY.shape[0]//TSTEPS, TSTEPS,
    #                         *trainY.shape[1:])
    # print(trainX.shape, trainY.shape)

    NIMAGES2 = 1000
    START = START+NIMAGES1

    valX, valY, _ = load_data_for_keras(otype, dd=dd, START=START,
                                        NUM_IMAGES=NIMAGES2,
                                        TSTEP=TSTEPS)
    # print(valX.shape, valY.shape)
    valX = valX.reshape(valX.shape[0]//TSTEPS, TSTEPS, *valX.shape[1:], 1)
    # valY = valY.reshape(valY.shape[0]//TSTEPS, TSTEPS, *valY.shape[1:])
    # print(valX.shape, valY.shape)
    return (trainX, trainY, valX, valY, TSTEPS)


def plot_predicted_outputs_reg(vY, vYP, TSTEPS):

    # XXX: The moneyness
    MS = np.arange(LM, UM+MSTEP, MSTEP)
    # XXX: The term structure
    TS = np.array([i/DAYS
                   for i in
                   range(LT, UT+TSTEP, TSTEP)])
    # XXX: Reshape the outputs
    vY = vY.reshape(vY.shape[0], len(MS), len(TS))
    vYP = vYP.reshape(vYP.shape[0], len(MS), len(TS))
    print(vY.shape, vYP.shape)

    for i in range(vY.shape[0]):
        y = vY[i]*100
        yp = vYP[i]*100
        fig, axs = plt.subplots(1, 2,
                                subplot_kw=dict(projection='3d'))
        axs[0].title.set_text('Truth')
        # XXX: Make the y dataframe
        ydf = list()
        for cm, m in enumerate(MS):
            for ct, t in enumerate(TS):
                ydf.append([m, t, y[cm, ct]])
                ydf = np.array(ydf)
                axs[0].plot_trisurf(ydf[:, 0], ydf[:, 1], ydf[:, 2],
                                    cmap='afmhot', linewidth=0.2,
                                    antialiased=True)
                axs[0].set_xlabel('Moneyness')
                axs[0].set_ylabel('Term structure')
                axs[0].set_zlabel('Vol %')
                axs[1].title.set_text('Predicted')
                ypdf = list()
        for cm, m in enumerate(MS):
            for ct, t in enumerate(TS):
                ypdf.append([m, t, yp[cm, ct]])
                ypdf = np.array(ypdf)
                axs[1].plot_trisurf(ypdf[:, 0], ypdf[:, 1], ypdf[:, 2],
                                    cmap='afmhot', linewidth=0.2,
                                    antialiased=True)
                axs[1].set_xlabel('Moneyness')
                axs[1].set_ylabel('Term structure')
                axs[1].set_zlabel('Vol %')

        plt.show()
        plt.close()


def clean_data(tX, tY):
    # print("Cleaning data for gfigs")
    mask = np.isnan(tX)
    tX[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask),
                         tX[~mask])
    mask = np.isnan(tY)
    tY[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask),
                         tY[~mask])
    return tX, tY


def regression_predict(otype, dd='./figs', model='Ridge', TSTEPS=10):
    # XXX: We will need to do steps 5, 10 and 20
    tX, tY, vX, vY, lags = load_data(otype, dd=dd, TSTEPS=TSTEPS)
    tX = tX.reshape(tX.shape[:-1])
    vX = vX.reshape(vX.shape[:-1])
    # tX = np.append(tX, vX, axis=0)
    # tY = np.append(tY, vY, axis=0)
    # print('tX, tY: ', tX.shape, tY.shape)
    tX = tX.reshape(tX.shape[0], tX.shape[1]*tX.shape[2]*tX.shape[3])
    tY = tY.reshape(tY.shape[0], tY.shape[1]*tY.shape[2])
    # print('tX, tY:', tX.shape, tY.shape)

    # XXX: Validation set
    vX = vX.reshape(vX.shape[0], vX.shape[1]*vX.shape[2]*vX.shape[3])
    vY = vY.reshape(vY.shape[0], vY.shape[1]*vY.shape[2])
    # print('vX, vY:', vX.shape, vY.shape)

    # XXX: Intercept?
    intercept = True

    # Fill in NaN's... required for non-parametric regression
    if dd == './gfigs':
        tX, tY = clean_data(tX, tY)
        vX, vY = clean_data(vX, vY)

    # XXX: Make a LinearRegression
    if model == 'Lasso':
        treg = 'lasso'  # overfits
        reg = Lasso(fit_intercept=intercept, alpha=1,
                    selection='random')

    if model == 'Ridge':        # gives the best results
        treg = 'ridge'
        reg = Ridge(fit_intercept=intercept, alpha=1)

    if model == 'OLS':
        treg = 'ols'
        reg = LinearRegression(fit_intercept=intercept, n_jobs=-1)

    if model == 'ElasticNet':
        treg = 'enet'               # overfits
        reg = ElasticNet(fit_intercept=intercept, alpha=1,
                         selection='random')

    if model == 'RF':
        treg = 'rf'
        reg = RandomForestRegressor(n_jobs=10, max_features='sqrt',
                                    n_estimators=150,
                                    bootstrap=True, verbose=1)

    if model == 'XGBoost':
        treg = 'xgboost'
        reg = MultiOutputRegressor(
            xgb.XGBRegressor(n_jobs=12,
                             tree_method='hist',
                             multi_strategy='multi_output_tree',
                             n_estimators=100,
                             verbosity=2))

    if model == 'plsridge' or model == 'plslasso' or model == 'plsenet':
        if dd != './gfigs':
            tokeep = cca_comps(tX, tY)
        else:
            tokeep = cca_comps(tX, tY, N_COMP=20)
            # XXX: Get the score and predict using scores then get it back
        treg = model
        reg = MPls(tokeep, intercept, model)

    if model == 'ctridge' or model == 'ctlasso' or model == 'ctenet':
        mms = np.arange(LM, UM+MSTEP, MSTEP)
        TTS = [i for i in range(LT, UT+TSTEP, TSTEP)]
        reg = CT(model, mms, TTS, TSTEPS)
        treg = model

    if model == 'ssviridge' or model == 'ssvilasso' or model == 'ssvienet':
        treg = model
        reg = SSVI(model, TSTEPS)

    reg.fit(tX, tY)
    print('Train set R2: ', reg.score(tX, tY))

    # XXX: Predict (Validation)
    vYP = reg.predict(vX)
    print(vY.shape, vYP.shape)
    rmses = root_mean_squared_error(vY, vYP, multioutput='raw_values')
    mapes = mean_absolute_percentage_error(vY, vYP, multioutput='raw_values')
    r2sc = r2_score(vY, vYP, multioutput='raw_values')
    print('RMSE mean: ', np.mean(rmses), 'RMSE std-dev: ', np.std(rmses))
    print('MAPE mean: ', np.mean(mapes), 'MAPE std-dev: ', np.std(mapes))
    print('R2 score mean:', np.mean(r2sc), 'R2 score std-dev: ', np.std(r2sc))

    # XXX: Plot some outputs
    # plot_predicted_outputs_reg(vY, vYP, TSTEPS)

    # XXX: Save the model
    import pickle
    if dd != './gfigs':
        with open('./surf_models/model_%s_ts_%s_%s.pkl' %
                  (treg, lags, otype), 'wb') as f:
            pickle.dump(reg, f)
    else:
        with open('./surf_models/model_%s_ts_%s_%s_gfigs.pkl' %
                  (treg, lags, otype), 'wb') as f:
            pickle.dump(reg, f)


def convlstm_predict(dd='./figs'):
    TSTEPS = 20
    trainX, trainY, valX, valY, _ = load_data(dd=dd, TSTEPS=TSTEPS)

    # Fill in NaN's... required for non-parametric regression
    if dd == './gfigs':
        print("fixing gfigs data")
        mask = np.isnan(trainX)
        trainX[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask),
                                 trainX[~mask])
        mask = np.isnan(trainY)
        trainY[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask),
                                 trainY[~mask])

        print('tX, tY:', trainX.shape, trainY.shape)
        mask = np.isnan(valX)
        valX[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask),
                               valX[~mask])
        mask = np.isnan(valY)
        valY[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask),
                               valY[~mask])
        print('vX, vY:', valX.shape, valY.shape)

    # XXX: Inner number of filters
    inner_filters = 64

    # XXX: Now fit the model
    batch_size = 20

    # XXX: Now build the keras model
    model = build_keras_model(trainX.shape, inner_filters, batch_size)
    print(model.summary())

    history = keras_model_fit(model, trainX, trainY, valX, valY, batch_size)

    # XXX: Save the model after training
    if dd == './gfigs':
        model.save('modelcr_bs_%s_ts_%s_filters_%s_%s.keras' %
                   (batch_size, TSTEPS, inner_filters, 'gfigs'))
    else:
        model.save('modelcr_bs_%s_ts_%s_filters_%s_%s.keras' %
                   (batch_size, TSTEPS, inner_filters, 'figs'))

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('history_bs_%s_ts_%s_filters_%s.pdf' % (batch_size, TSTEPS,
                                                        inner_filters))


def mskew_pred(otype, dd='./figs', model='mskridge', TSTEPS=5):
    # XXX: We will need to do steps 5, 10 and 20
    tX, tY, vX, vY, lags = load_data(otype, dd=dd, TSTEPS=TSTEPS)
    tX = tX.reshape(tX.shape[:-1])
    vX = vX.reshape(vX.shape[:-1])

    # Fill in NaN's... required for non-parametric regression
    if dd == './gfigs':
        tX, tY = clean_data(tX, tY)
        vX, vY = clean_data(vX, vY)

    # XXX: Now go through the MS and TS
    tts = [i/DAYS for i in range(LT, UT+TSTEP, TSTEP)]

    count = 0

    # XXX: First get the LAMBDA
    if model == 'msknsridge' or model == 'msknslasso' or model == 'msknsenet':
        mms = np.arange(LM, UM+MSTEP, MSTEP)
        LAMBDA = 0.7822641809107665  # obtained from the code below
        # LAMBDA = [None]*len(tts)
        # for j in range(len(tts)):
        #     tYY = tY[:, :, j]
        #     params = np.array([curve_fit(ym, mms, tYY[k],
        #                                  p0=[0, 0, 0, 0.0047],
        #                                  bounds=(-1, 1),
        #                                  maxfev=100000)[0]
        #                        for k in range(tYY.shape[0])])
        #     LAMBDA[j] = np.mean(params, axis=0)[-1]
        # LAMBDA = np.mean(LAMBDA)
        x1s = [(1-np.exp(-LAMBDA*i))/(LAMBDA*i) for i in mms]
        x2s = [i-(np.exp(-LAMBDA*t)) for (i, t) in zip(x1s, mms)]
        xdf = pd.DataFrame({'x1s': x1s, 'x2s': x2s})
        # print('LAMBDA overall:', LAMBDA)

    # XXX: Now we go moneyness skew
    for j, t in enumerate(tts):
        if count % 10 == 0:
            print('Done: ', count)

        count += 1
        # XXX: shape = samples, TSTEPS, moneyness, term structure
        mskew = tX[:, :, :, j]
        tYY = tY[:, :, j]
        mskew = mskew.reshape(mskew.shape[0], mskew.shape[1]*mskew.shape[2])
        # XXX: Add t to the sample set
        ts = np.array([t]*mskew.shape[0]).reshape(mskew.shape[0], 1)
        mskew = np.append(mskew, ts, axis=1)
        # XXX: Fit the ridge model
        if model == 'mskridge':
            reg = Ridge(fit_intercept=True, alpha=1)
        elif model == 'msklasso':
            reg = Lasso(fit_intercept=True, alpha=1)
        elif model == 'mskenet':
            reg = ElasticNet(fit_intercept=True, alpha=1)
        elif (model == 'mskplslasso' or model == 'mskplsridge' or
              model == 'mskplsenet'):
            tokeep = cca_comps(mskew, tYY)
            reg = MPls(tokeep, True, model)
        elif (model == 'msknsridge' or model == 'msknslasso' or
              model == 'msknsenet'):
            # XXX: Now fit the xdf and call NS
            reg = NS(xdf, TSTEPS, model)

        reg.fit(mskew, tYY)
        # print('train r2score:', reg.score(mskew, tYY))

        import pickle
        if dd != './gfigs':
            with open('./mskew_models/%s_ts_%s_%s_%s.pkl' %
                      (model, lags, t, otype), 'wb') as f:
                pickle.dump(reg, f)
        else:
            with open('./mskew_models/%s_ts_%s_%s_%s_gfigs.pkl' %
                      (model, lags, t, otype), 'wb') as f:
                pickle.dump(reg, f)


def tskew_pred(otype, dd='./figs', model='tskridge', TSTEPS=5):
    # XXX: We will need to do steps 5, 10 and 20
    tX, tY, vX, vY, lags = load_data(otype, dd=dd, TSTEPS=TSTEPS)
    tX = tX.reshape(tX.shape[:-1])
    vX = vX.reshape(vX.shape[:-1])

    # Fill in NaN's... required for non-parametric regression
    if dd == './gfigs':
        tX, tY = clean_data(tX, tY)
        vX, vY = clean_data(vX, vY)

    # XXX: Now go through the MS and TS
    mms = np.arange(LM, UM+MSTEP, MSTEP)

    count = 0
    if model == 'tsknsridge' or model == 'tsknslasso' or model == 'tsknsenet':
        TTS = [i for i in range(LT, UT+TSTEP, TSTEP)]
        LAMBDA = 0.0147     # from Guo 2014.
        x1s = [(1-np.exp(-LAMBDA*i))/(LAMBDA*i) for i in TTS]
        x2s = [i-(np.exp(-LAMBDA*t)) for (i, t) in zip(x1s, TTS)]
        xdf = pd.DataFrame({'x1s': x1s, 'x2s': x2s})

    # XXX: Now we go term structure skew
    for j, m in enumerate(mms):
        if count % 10 == 0:
            print('Done: ', count)
        count += 1

        # XXX: shape = samples, TSTEPS, moneyness, term structure
        tskew = tX[:, :, j]
        # tskew1 = np.copy(tskew)  # needed for NS model
        tskew = tskew.reshape(tskew.shape[0], tskew.shape[1]*tskew.shape[2])
        # # XXX: Add m to the sample set
        ms = np.array([m]*tskew.shape[0]).reshape(tskew.shape[0], 1)
        tskew = np.append(tskew, ms, axis=1)
        tYY = tY[:, j]
        # XXX: Fit the ridge model
        if model == 'tskridge':
            reg = Ridge(fit_intercept=True, alpha=1)
        elif model == 'tsklasso':
            reg = Lasso(fit_intercept=True, alpha=1)
        elif model == 'tskenet':
            reg = ElasticNet(fit_intercept=True, alpha=1)
        elif (model == 'tskplsridge' or model == 'tskplslasso' or
              model == 'tskplsenet'):
            tokeep = cca_comps(tskew, tYY)
            reg = MPls(tokeep, True, model)
            # reg = PLSRegression(tokeep)
        elif (model == 'tsknsridge' or model == 'tsknslasso' or
              model == 'tsknsenet'):
            # XXX: First we want to get the betas for each day using
            # OLS. Next, we predict the betas for t using t-1,...t-N
            # lags using one of the models.
            reg = NS(xdf, TSTEPS, model)

        reg.fit(tskew, tYY)
        # print(reg.score(tskew, tYY))

        import pickle
        if dd != './gfigs':
            with open('./tskew_models/%s_ts_%s_%s_%s.pkl' %
                      (model, lags, m, otype), 'wb') as f:
                pickle.dump(reg, f)
        else:
            with open('./tskew_models/%s_ts_%s_%s_%s_gfigs.pkl' %
                      (model, lags, m, otype), 'wb') as f:
                pickle.dump(reg, f)


def cca_comps(X, y, N_COMP=None):
    if N_COMP is None:
        N_TARGETS = 1 if len(y.shape) == 1 else y.shape[1]
        N_COMP_UB = min(X.shape[0], X.shape[1], N_TARGETS)
        N_COMP = max(1, N_COMP_UB)

    reg = PLSSVD(n_components=N_COMP)
    reg.fit(X, y)
    ypir, yir = reg.transform(X, y)
    tokeep = 0
    # print(N_COMP)
    for i in range(N_COMP):
        corr = np.corrcoef(ypir[:, i], yir[:, i])[0, 1]
        # print(corr**2)
        if corr**2 < 0.15:
            break
        else:
            tokeep += 1
    # XXX: The number of valid components to keep
    tokeep = 1 if tokeep == 0 else tokeep
    return tokeep


def point_pred(otype, dd='./figs', model='pmridge', TSTEPS=10):
    # XXX: We will need to do steps 5, 10 and 20
    tX, tY, vX, vY, lags = load_data(otype, dd=dd, TSTEPS=TSTEPS)
    tX = tX.reshape(tX.shape[:-1])
    vX = vX.reshape(vX.shape[:-1])
    # tX = np.append(tX, vX, axis=0)
    # tY = np.append(tY, vY, axis=0)
    # print('tX, tY: ', tX.shape, tY.shape)

    # XXX: Validation set
    # print('vX, vY:', vX.shape, vY.shape)

    # Fill in NaN's... required for non-parametric regression
    if dd == './gfigs':
        tX, tY = clean_data(tX, tY)
        vX, vY = clean_data(vX, vY)

    # XXX: Now go through the MS and TS
    mms = np.arange(LM, UM+MSTEP, MSTEP)
    tts = [i/DAYS for i in range(LT, UT+TSTEP, TSTEP)]

    count = 0
    for i, s in enumerate(mms):
        for j, t in enumerate(tts):
            if count % 50 == 0:
                print('Done: ', count)
            count += 1
            # XXX: Make the vector for training
            k = np.array([s, t]*tX.shape[0]).reshape(tX.shape[0], 2)
            train_vec = np.append(tX[:, :, i, j], k, axis=1)
            # print(train_vec.shape, tY[:, i, j].shape)

            # XXX: Fit the ridge model
            if model == 'pmridge':
                reg = Ridge(fit_intercept=True, alpha=1)
            elif model == 'pmlasso':
                reg = Lasso(fit_intercept=True, alpha=1)
            elif (model == 'pmplsridge' or model == 'pmplslasso' or
                  model == 'pmplsenet'):
                tokeep = cca_comps(train_vec, tY[:, i, j])
                reg = MPls(tokeep, True)
                # reg = PLSRegression(n_components=tokeep)
            else:
                reg = ElasticNet(fit_intercept=True, alpha=1,
                                 selection='random')
            reg.fit(train_vec, tY[:, i, j])
            # print('Train set R2: ', reg.score(train_vec, tY[:, i, j]))
            # assert (False)

            # XXX: Predict (Validation)
            # print(vX.shape, vY.shape)
            # k = np.array([s, t]*vX.shape[0]).reshape(vX.shape[0], 2)
            # val_vec = np.append(vX[:, :, i, j], k, axis=1)
            # vYP = reg.predict(val_vec)
            # vvY = vY[:, i, j]
            # r2sc = r2_score(vvY, vYP, multioutput='raw_values')
            # print('Test R2:', np.mean(r2sc))

            # XXX: Save the model
            import pickle
            if dd != './gfigs':
                with open('./point_models/%s_ts_%s_%s_%s_%s.pkl' %
                          (model, lags, s, t, otype), 'wb') as f:
                    pickle.dump(reg, f)
            else:
                with open('./point_models/%s_ts_%s_%s_%s_%s_gfigs.pkl' %
                          (model, lags, s, t, otype), 'wb') as f:
                    pickle.dump(reg, f)


def linear_fit(otype):
    # Surface regression prediction (RUN THIS WITH OMP_NUM_THREADS=10 on
    # command line)

    # XXX: Moneyness skew regression
    for j in ['./figs', './gfigs']:
        for i in [5, 20, 10]:
            for k in ['ssviridge', 'ssvilasso', 'ssvienet',
                      # 'ctridge', 'ctlasso', 'ctenet',
                      # 'plsenet', 'plsridge', 'plslasso',
                      # 'Ridge', 'Lasso', 'ElasticNet'
                      ]:
                print('Doing: %s_%s_%s' % (k, j, i))
                regression_predict(otype, model=k, dd=j, TSTEPS=i)

            # for k in ['tsknsridge', 'tsknslasso', 'tsknsenet',
            #           'tskplsridge', 'tskplslasso', 'tskplsenet',
            #           'tskridge', 'tsklasso', 'tskenet'
            #           ]:
            #     print('Doing: %s_%s_%s' % (k, j, i))
            #     tskew_pred(otype, dd=j, model=k, TSTEPS=i)

            # for k in ['pmplsridge', 'pmplslasso', 'pmplsenet',
            #           'pmridge', 'pmlasso', 'pmenet'
            #           ]:
            #     print('Doing: %s_%s_%s' % (k, j, i))
            #     point_pred(otype, dd=j, model=k, TSTEPS=i)

            # for k in ['mskplslasso', 'msknsridge', 'msknsenet', 'msknslasso',
            #           'mskplsridge', 'mskplsenet',
            #           'mskridge', 'msklasso', 'mskenet'
            #           ]:
            #     print('Doing: %s_%s_%s' % (k, j, i))
            #     mskew_pred(otype, dd=j, model=k, TSTEPS=i)


if __name__ == '__main__':
    # XXX: Excel data to images
    # excel_to_images(otype='call')  # call options with linear fit
    # excel_to_images(otype='put')  # put options with linear fit

    # XXX: Non-parametric regression for calls
    # excel_to_images(otype='call', dvf=False)
    # XXX: Non-parametric regression for puts
    # excel_to_images(otype='put', dvf=False)

    # XXX: Fit the linear models
    for otype in ['call', 'put']:
        linear_fit(otype)

    # XXX: Get interest rates and forward prices
    # excel_to_images(ironly=True)

    # XXX: ConvLSTM2D prediction
    # convlstm_predict(dd='./gfigs')
