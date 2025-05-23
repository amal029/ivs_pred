#!/usr/bin/env python
import glob
import numpy as np
from model_load_test import date_to_num, num_to_date
import pandas as pd
from pred import SSVI
import matplotlib.pyplot as plt
# from statsmodels.stats.diagnostic import het_arch
from sklearn.linear_model import Ridge, RidgeCV
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
from joblib import Parallel, delayed
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from pred import cr2_score, cr2_score_pval
from keras.layers import Input, LSTM, Dropout
from keras.models import Model
import keras
from scipy.optimize import minimize
import mpu
from collections import namedtuple
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.varmax import VARMAX


# XXX: This function loads the real data
def load_real_data(dd='./interest_rates', otype='call'):
    toret = dict()
    for d in range(2002, 2024):
        ff = sorted(glob.glob(dd+'/'+str(d)+'*.csv'))
        for i in ff:
            toret[i.split('/')[-1].split('.')[0]] = pd.read_csv(i)
    return toret


def load_data(otype, dd='./figs', START='20020208', NUM_IMAGES=2000):
    Xs = list()               # Training inputs [0..TEST_IMAGES-1]
    Xdates = list()
    ff = sorted(glob.glob('./'+otype+'_'+dd.split('/')[1]+'/*.npy'))
    START = date_to_num(otype, START, dd)
    for i in range(START, START+NUM_IMAGES):
        img = np.load(ff[i])
        assert (np.all((img > 0) & (img <= 1)))
        Xs += [img]
        Xdates.append(num_to_date(otype, i, dd))
    Xs = pd.DataFrame({'Date': Xdates, 'IVS': Xs})
    return Xs


def param_summary(params):
    # XXX: Get stationarity of the prameters themselves
    from arch.unitroot import ADF
    print(ADF(np.diff(params[:, 0]), trend='n', method='bic').summary())
    print(ADF(np.diff(params[:, 1]), trend='n', method='bic').summary())
    # XXX: Now predict the next param using GARCH
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(np.diff(params[:, 0])**2)
    plot_acf(np.diff(params[:, 1])**2)
    plt.show(block=False)


def pacount(x, BREAK=0.1, var=False, nlags=10, diff=True):
    import numpy as np
    from statsmodels.tsa.stattools import pacf, acf
    xd = np.diff(x) if diff else x
    if not var:
        pa = pacf(xd, nlags=nlags)
    else:
        pa = pacf(xd**2, nlags=nlags)
    pcount = 0
    for i in pa[1:]:
        if np.abs(i) < BREAK:
            break
        pcount += 1

    if not var:
        aa = acf(xd, nlags=nlags)
    else:
        aa = acf(xd**2, nlags=nlags)
    acount = 0
    for i in aa[1:]:
        if np.abs(i) < BREAK:
            break
        acount += 1

    return pcount, acount


def fitandforecastARIMA(Y, trend='n', N=1000000):

    Y = Y.values.reshape(Y.shape[0],)
    rpm, ram = pacount(Y, BREAK=0.1)

    SCALE = 1
    dY = np.diff(Y)
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    # XXX: ARMA on differenced series := ARIMA
    # df = rpm + ram
    model_fit = SARIMAX(dY*SCALE, order=(rpm, 0, ram), trend=trend,
                        ).fit(maxiter=10000, disp=False,
                              method='lbfgs')
    # XXX: This is the mean of the model
    retmean = model_fit.forecast(steps=1)[0]/SCALE

    # XXX: The residual fitting
    from arch.univariate import ZeroMean, GARCH, StudentsT
    # XXX: AR on differenced series
    rp, ra = pacount(Y, BREAK=0.1, var=True)
    rp = 1 if rp <= 0 else rp
    ra = 1 if ra <= 0 else ra
    vol_model = GARCH(p=ra, q=rp)
    model_fit = ZeroMean(model_fit.resid, volatility=vol_model,
                         rescale=True, distribution=StudentsT())
    model_fit = model_fit.fit(update_freq=0, disp='off',
                              options={'maxiter': 10000})
    SCALE = model_fit.scale
    retvar = (model_fit.forecast(horizon=1).variance.iloc[0, 0] /
              np.power(SCALE, 2))**0.5

    # XXX: We have removed auto-corellation, but the std_resid still
    # have kurtosis -- so we use a StudentsT distribution.

    # XXX: Total return for the shifted and scaled StudentsT
    # distribution
    ret = np.mean(retmean + retvar *
                  np.random.standard_t(df=model_fit.params['nu'], size=N))

    # from statsmodels.stats.diagnostic import acorr_ljungbox
    # ress = acorr_ljungbox(model_fit.std_resid, model_df=df)
    # print('correlation: ', ress)
    # from statsmodels.stats.stattools import jarque_bera
    # jb, jbp, skew, kur = jarque_bera(model_fit.std_resid)
    # print('JB stat:%s, JBp-val:%s, skew:%s, kurtosis:%s' %
    #       (jb, jbp, skew, kur))
    # plt.hist(model_fit.std_resid, bins=100)
    # plt.show()

    return ret + Y[-1]


def doARIMA(params, WINDOW):
    import warnings
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

    rho = pd.DataFrame(params[:, 0])
    nu = pd.DataFrame(params[:, 1])

    prho = rho.rolling(WINDOW).apply(
        lambda x:
        fitandforecastARIMA(x, trend='n')
    ).dropna()
    prho = prho.values.reshape(prho.shape[0],)

    pnu = nu.rolling(WINDOW).apply(
        lambda x:
        fitandforecastARIMA(x, trend='n')
    ).dropna()
    pnu = pnu.values.reshape(pnu.shape[0],)

    print(prho.shape, pnu.shape)
    return pd.DataFrame({'rho': prho[:-1], 'nu': pnu[:-1]})


def doLGBM(params, WINDOW, TSTEP):
    import warnings
    warnings.filterwarnings(action='ignore', category=UserWarning)

    def fitandforecastLGBM(aa, TSTEP):
        da = np.diff(aa, axis=0)
        da = np.vstack((da[0, :], da))  # Just replicated the first one.
        Y = list()
        for i in range(da.shape[0]):
            if i > 0 and i % TSTEP == 0:
                Y.append(da[i])
        Y = np.array(Y)
        Y1 = Y[:, 0]            # rho
        Y2 = Y[:, 1]            # nu
        X1 = da[:, 0]           # rho lags
        X1 = X1.reshape(X1.shape[0]//TSTEP, TSTEP)
        X2 = da[:, 1]           # nu lags
        X2 = X2.reshape(X2.shape[0]//TSTEP, TSTEP)
        # XXX: Differenced lags with regularisation
        m1 = XGBRegressor(verbosity=0, subsample=0.1,
                          booster='gblinear').fit(X1[:-1], Y1)
        m2 = XGBRegressor(verbosity=0, subsample=0.1,
                          booster='gblinear').fit(X2[:-1], Y2)
        r1 = m1.predict(X1[-1].reshape(1, X1[-1].shape[0]))
        r2 = m2.predict(X2[-1].reshape(1, X2[-1].shape[0]))
        return r1[0]+aa[-1, 0], r2[0]+aa[-1, 1]

    # rho = params[:, 0]
    # nu = params[:, 1]

    prho = list()
    pnu = list()
    # assert (rho.shape == nu.shape)
    N = params.shape[0]
    for i in range(N-WINDOW):
        if i % 100 == 0:
            print('Rho/Nu Done: ', i)
        aa = params[i:WINDOW+i]
        yy1, yy2 = fitandforecastLGBM(aa, TSTEP)
        prho.append(yy1)
        pnu.append(yy2)

    return pd.DataFrame({'rho': prho, 'nu': pnu})


def doLSTM(params, WINDOW, TSTEP):

    def buildKeras(TSTEP, batch_size, trainX, trainY, epochs=500,
                   LR=1e-3, validation_split=0.05,
                   activation='tanh'):
        inp = Input(shape=(TSTEP, 1), batch_size=batch_size)
        l1 = LSTM(1, stateful=True, return_sequences=True,
                  activation=activation)(inp)
        # XXX: Just one final output
        l2 = LSTM(1, stateful=True, return_sequences=False,
                  activation=activation)(l1)
        # l3 = keras.layers.Dense(1)(l1)
        model = Model(inp, l2)
        model.compile(loss=keras.losses.mean_squared_error,
                      optimizer=keras.optimizers.Adam(learning_rate=LR))
        print(model.summary())
        # Define some callbacks to improve training.
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=100, restore_best_weights=True)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                      patience=5)
        # Fit the model to the training data.
        for i in range(epochs):
            history = model.fit(
                trainX,
                trainY,
                batch_size=1,
                epochs=1,
                verbose=0,
                validation_split=validation_split,
                shuffle=False,
                callbacks=[early_stopping, reduce_lr])
            model.get_layer(index=1).reset_states()
            # model.get_layer(index=2).reset_states()
        return model, history

    def fitandforecastLSTM(aa, TSTEP, m1, m2):
        # from sklearn.preprocessing import MinMaxScaler
        # scaler = MinMaxScaler(feature_range=(-1, 1))
        # da = np.diff(aa, axis=0)
        # da = np.vstack((da[0, :], da))  # Just replicated the first one.
        # da = scaler.fit_transform(aa)   # Transform to between 1 and -1
        da = aa   # Transform to between 1 and -1
        Y = list()
        for i in range(da.shape[0]):
            if i > 0 and i % TSTEP == 0:
                Y.append(da[i])
        Y = np.array(Y)
        Y1 = Y[:, 0]            # diff-rho
        Y2 = Y[:, 1]            # diff-nu

        X1 = da[:, 0]           # rho lags
        # XXX: shape = (batch_size, timesteps, features)
        X1 = X1.reshape(X1.shape[0]//TSTEP, TSTEP, 1)

        X2 = da[:, 1]           # nu lags
        # XXX: shape = (batch_size, timesteps, features)
        X2 = X2.reshape(X2.shape[0]//TSTEP, TSTEP, 1)

        if m1 is None:
            print('m1 is None so training the LSTM')
            # XXX: Differenced lags with regularisation
            m1, h1 = buildKeras(TSTEP, 1, X1[:-1], Y1)
        # XXX: Predict the output
        r1 = m1.predict(X1[-1].reshape(X1[-1].shape[0], 1),
                        batch_size=1, verbose=0)
        # m1.get_layer(index=1).reset_states()

        # m1.get_layer(index=2).reset_states()
        if m2 is None:
            print('m2 is None so training the LSTM')
            m2, h2 = buildKeras(TSTEP, 1, X2[:-1], Y2,
                                activation='tanh')

        r2 = m2.predict(X2[-1].reshape(X2[-1].shape[0], 1),
                        batch_size=1, verbose=0)
        # m2.get_layer(index=1).reset_states()
        # m2.get_layer(index=2).reset_states()
        # rs = scaler.inverse_transform([[r1[0, 0], r2[0, 0]]])
        return r1[-1, 0], r2[-1, 0], m1, m2

    # rho = params[:, 0]
    # nu = params[:, 1]

    prho = list()
    pnu = list()
    # assert (rho.shape == nu.shape)
    N = params.shape[0]
    m1 = None
    m2 = None
    for i in range(N-WINDOW):
        if i % 100 == 0:
            print('LSTM done: ', i)
        aa = params[i:WINDOW+i]
        yy1, yy2, m1, m2 = fitandforecastLSTM(aa, TSTEP, m1, m2)
        prho.append(yy1)
        pnu.append(yy2)

    return pd.DataFrame({'rho': prho, 'nu': pnu})


def doATMIV(atmiv, WINDOW, TSTEP):
    def RidgeFit(atmiv):
        Y = list()
        for i in range(atmiv.shape[0]):
            if i > 0 and i % TSTEP == 0:
                Y.append(atmiv[i])
        Y = np.array(Y)
        X = atmiv.reshape(atmiv.shape[0]//TSTEP,
                          TSTEP, atmiv.shape[1])
        Xf = X[:-1]
        Xf = Xf.reshape(Xf.shape[0], Xf.shape[1]*Xf.shape[2])
        model = Ridge().fit(Xf, Y)
        Xp = X[-1]
        return model.predict(Xp.reshape(1, Xp.shape[0]*Xp.shape[1]))

    pY = list()
    N = atmiv.shape[0]
    for i in range(N-WINDOW):
        aa = atmiv[i:WINDOW+i, :]
        yy = RidgeFit(aa)
        pY.append(yy.reshape(yy.shape[1],))

    return np.array(pY)


def doSSVI(X, WINDOW, TSTEP):
    def doit(ivs, ssvi):
        # XXX: Get the outputs
        Y = list()
        for i in range(ivs.shape[0]):
            if i > 0 and i % TSTEP == 0:
                Y.append(ivs[i])
        Y = np.array(Y)

        X = np.array([ivs[i] for i in range(ivs.shape[0])])
        # XXX: Reshape the ivs
        X = X.reshape(X.shape[0]//TSTEP, TSTEP, X.shape[1],
                      X.shape[2])

        if not ssvi.check_is_fitted():
            ssvi = ssvi.fit(X[:-1], Y)
        # XXX: Predict the output
        pY = ssvi.predict(X[-1].reshape(1, *X[-1].shape))
        return pY.reshape(pY.shape[1],), ssvi

    # XXX: Roll through the ivs and doit
    X = X['IVS'].values
    N = X.shape[0]
    pY = list()
    ssvi = SSVI('ssviridge', TSTEP)
    for i in range(N-WINDOW):
        if i % 100 == 0:
            print('Done %s' % i)
        aa = X[i:WINDOW+i]
        yy, ssvi = doit(aa, ssvi)
        pY.append(yy)
    return np.array(pY)


def doModel(X, WINDOW, TSTEP):
    def doit(ivs, ridge):
        # XXX: Get the outputs
        Y = list()
        for i in range(ivs.shape[0]):
            if i > 0 and i % TSTEP == 0:
                Y.append(ivs[i])
        Y = np.array(Y)

        X = np.array([ivs[i] for i in range(ivs.shape[0])])
        # XXX: Reshape the ivs
        X = X.reshape(X.shape[0]//TSTEP, TSTEP, X.shape[1],
                      X.shape[2])
        X = X.reshape(X.shape[0], X.shape[1]*X.shape[2]*X.shape[3])
        Y = Y.reshape(Y.shape[0], Y.shape[1]*Y.shape[2])
        try:
            check_is_fitted(ridge)
        except NotFittedError:
            ridge = ridge.fit(X[:-1], Y)

        # XXX: Predict the output
        pY = ridge.predict(X[-1].reshape(1, *X[-1].shape))
        return pY.reshape(pY.shape[1],), ridge

    # XXX: Roll through the ivs and doit
    X = X['IVS'].values
    N = X.shape[0]
    pY = list()
    ridge = Ridge()
    for i in range(N-WINDOW):
        if i % 100 == 0:
            print('Done %s' % i)
        aa = X[i:WINDOW+i]
        yy, ridge = doit(aa, ridge)
        pY.append(yy)
    return np.array(pY)


def main():
    for otype in ['call']:
        WINDOW = 1000
        TSTEP = 5
        START_DATE = '20140109'
        END_DATE = '20221230'
        START = date_to_num(otype, START_DATE, dd='./figs')
        # XXX: Go back for training
        START = START-WINDOW
        START_DATE = num_to_date(otype, START)
        END = date_to_num(otype, END_DATE, dd='./figs') - TSTEP
        NIMAGES = END - START

        print('start date: ', START_DATE)
        X = load_data(otype, START=START_DATE, NUM_IMAGES=NIMAGES)
        print(X.shape)

        # XXX: Get the true values from IVS
        yT = np.array([i for i in X['IVS'][WINDOW:]])
        yT = yT.reshape(yT.shape[0], yT.shape[1]*yT.shape[2])

        # XXX: Fit the SSVI model to each day separately
        ssvi = SSVI('ssviridge', TSTEP)
        params, ATM_IV = ssvi.fitY(X['IVS'])

        # XXX: Get the ATM_IV predictions FIXME: We need to make sure
        # that dATM_IV/dt >= 0, so first first fit the term structure @
        # ATM_IV with NS and then predict the parameters of NS.
        pY = doATMIV(ATM_IV, WINDOW, TSTEP=TSTEP)
        print(ATM_IV[WINDOW:].shape, pY.shape)
        print('R2 score ATM: ', r2_score(ATM_IV[WINDOW:], pY))

        # XXX: Predict the parameters using LSTM NN
        pparams = doLSTM(params, WINDOW, TSTEP)
        print('R2 score rho: ', r2_score(params[WINDOW:, 0], pparams['rho']))
        print('R2 score nu: ', r2_score(params[WINDOW:, 1], pparams['nu']))

        # XXX: Predict the IVS using LSTM models
        plY = Parallel(n_jobs=-1)(delayed(ssvi.predict1)(
            pparams.loc[i, :], pY[i])
                                  for i in range(pY.shape[0]))
        plY = np.array(plY)
        plY = plY.reshape(plY.shape[0], plY.shape[1]*plY.shape[2])
        # XXX: interpolate nan values!
        mask = np.isnan(plY)
        if (mask.sum() > 0):
            print('Nan values in prediction: ', mask.sum())
            plY[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask),
                                  plY[~mask])
        print('R2 score LSTM: ', r2_score(yT, plY))

        # XXX: Predict the parameters using light gbm/XGBBoost
        pparams = doLGBM(params, WINDOW, TSTEP)
        print('R2 score rho: ', r2_score(params[WINDOW:, 0], pparams['rho']))
        print('R2 score nu: ', r2_score(params[WINDOW:, 1], pparams['nu']))

        # XXX: Predict the IVS using LGBM models
        pgY = Parallel(n_jobs=-1)(delayed(ssvi.predict1)(
            pparams.loc[i, :], pY[i])
                                  for i in range(pY.shape[0]))
        pgY = np.array(pgY)
        pgY = pgY.reshape(pgY.shape[0], pgY.shape[1]*pgY.shape[2])
        # XXX: interpolate nan values!
        mask = np.isnan(pgY)
        if (mask.sum() > 0):
            print('Nan values in prediction: ', mask.sum())
            pgY[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask),
                                  pgY[~mask])
        print('R2 score XGBoost: ', r2_score(yT, pgY))

        # XXX: Now try this same thing with ARIMA
        pparams = doARIMA(params, WINDOW)
        print('R2 score rho: ', r2_score(params[WINDOW:, 0], pparams['rho']))
        print('R2 score nu: ', r2_score(params[WINDOW:, 1], pparams['nu']))

        # XXX: Predict the IVS using ARIMA models
        paY = Parallel(n_jobs=-1)(delayed(ssvi.predict1)(
            pparams.loc[i, :], pY[i])
                                  for i in range(pY.shape[0]))
        paY = np.array(paY)
        paY = paY.reshape(paY.shape[0], paY.shape[1]*paY.shape[2])
        print('R2 score ARIMA-GARCH: ', r2_score(yT, paY))

        # XXX: Model fit (AR model)
        prY = doModel(X, WINDOW, TSTEP=TSTEP)
        print('R2 score %s:' % r2_score(yT, prY))

        # XXX: Fit using the standard technique
        psY = doSSVI(X, WINDOW, TSTEP=TSTEP)

        # XXX: interpolate nan values!
        mask = np.isnan(psY)
        if (mask.sum() > 0):
            print('Nan values in prediction: ', mask.sum())
            psY[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask),
                                  psY[~mask])

        # XXX: Get the R2 for both predictions
        print('R2 score SSVI: ', r2_score(yT, psY))

        # XXX: Compare the R2 scores VAR R2 vs SSVI ARIMA
        print('Ridge vs ARIMA-GARCH SSVI: ', cr2_score(yT, prY, paY))
        print('Ridge vs ARIMA-GARCH SSVI p-val: ',
              cr2_score_pval(yT, prY, paY))

        # XXX: Compare the R2 scores Ridge SSVI vs SSVI ARIMA
        print('Ridge SSVI vs ARIMA-GARCH SSVI: ', cr2_score(yT, psY, paY))
        print('Ridge SSVI vs ARIMA-GARCH SSVI p-val: ',
              cr2_score_pval(yT, psY, paY))

        # XXX: Compare XGBoost vs ARIMA-GARCH
        print('XGBoost SSVI vs ARIMA-GARCH SSVI: ', cr2_score(yT, pgY, paY))
        print('XGBoost SSVI vs ARIMA-GARCH SSVI p-val: ',
              cr2_score_pval(yT, pgY, paY))

        # XXX: Compare LSTM vs ARIMA-GARCH
        print('LSTM SSVI vs ARIMA-GARCH SSVI: ', cr2_score(yT, plY, paY))
        print('LSTM SSVI vs ARIMA-GARCH SSVI p-val: ',
              cr2_score_pval(yT, plY, paY))


# XXX: For fitting a single slice (at a given maturity across log moneyness)
def sviraw(k, t, param):
    a = param[0]
    b = param[1]
    m = param[2]
    rho = param[3]
    sigma = param[4]

    totalvariance = a + b * (rho * (k - m) +
                             np.sqrt((k - m) ** 2 + sigma**2))
    return totalvariance


# XXX: Fitting the ATM term structure
def ATMTS_fit(ATMTS: np.array, taus, k,
              f=lambda a, beta, mu, t: mu + a*(1 - np.exp(-beta*t))):
    """1) The ATM term structure data points themselves
    2) The func objective that should be fitted
    """
    def obj(params: np.array, sigma: np.array, taus: list):
        alpha = params[0]
        beta = params[1]
        mu = params[2]
        res = np.array([f(alpha, beta, mu, t) for t in taus])
        return np.sum((sigma - res)**2)

    bounds = [(1e-6, np.inf), (1e-6, np.inf),
              (-np.inf, np.inf)]
    res = minimize(
        obj,
        bounds=bounds,
        tol=1e-8,
        # method='Nelder-Mead',
        options={'disp': False,
                 'maxiter': 100000},
        x0=(0.1, 0.0047, 0),
        args=(ATMTS, taus)
    )
    if res.success:
        alpha = res.x[0]
        beta = res.x[1]
        mu = res.x[2]
        # print('alpha: %s beta: %s mu: %s' % (alpha, beta, mu))
        # x = [i/365 for i in list(range(14, 365*2))]
        # TS = [f(alpha, beta, mu, t) for t in x]
        # plt.plot(x, TS)
        # plt.plot(taus, ATMTS, marker='o', linestyle='none')
        # plt.savefig('/tmp/%s_ATMTS.pdf' % k, bbox_inches='tight')
        # plt.close()
        return alpha, beta, mu


def lprocess_data(dfs, otype):
    def inner_opt(params, veck, w):
        def mobj(params, X, Y):
            return np.sum((np.dot(X, params) - Y)**2)

        def con1(params):
            return params[2]-np.abs(params[1])

        def con2(params):
            return 4*sigma - params[2] - np.abs(params[1])

        sigma = params[0]
        m = params[1]
        X = np.ones(veck.shape[0]*3).reshape(veck.shape[0], 3)
        Y = w                 # vector of real total variance
        for i in range(veck.shape[0]):
            yy = (veck.iloc[i]-m)/sigma
            X[i] = [1, yy, np.sqrt(yy**2+1)]
        # XXX: This gets the initial values using the paper:
        # XXX: ADAM OHMAN thesis (KTH)
        try:
            beta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))
        except Exception:
            # XXX: Catch any exceptions
            beta = [0, 0, 0]

        mbounds = [(0, w.max()),  # a
                   (-np.inf, np.inf),  # d
                   (0, 4*sigma)       # c
                   ]
        res = minimize(mobj,
                       args=(X, Y),
                       bounds=mbounds,
                       tol=1e-8,
                       method='COBYQA',
                       options={'disp': False,
                                'maxiter': 100000},
                       x0=(beta[0], beta[1], beta[2]),
                       constraints=[{'type': 'ineq', 'fun': con1},
                                    {'type': 'ineq', 'fun': con2}]
                       )
        return np.dot(X, res.x), res.x

    # XXX: The target optimisation function
    def obj(params, veck, w):
        pw, _ = inner_opt(params, veck, w)
        return np.sum((pw - w)**2)

    def compute(df, k):
        thetaFits = {'alpha': list(), 'beta': list(), 'mu': list(),
                     'tau': list(), 'theta': list(), 'date': list()}
        print('Doing: ', k)
        # XXX: Tenors to consider
        df = df[(df['Type'] == otype) & (df['tau'] >= 14/365) &
                (df['tau'] <= 2)]
        # XXX: Moneyness to consider
        df = df[(df['m'] >= 0.8) & (df['m'] <= 1.2) & (df['Volume'] > 1)]
        taus = sorted(df['tau'].unique())
        # XXX: For each given tau fit the SVI param
        thetats = list()
        ttaus = list()
        for t in taus:
            dfw = df[df['tau'] == t][['m', 'IV', 'Strike']]
            dfw['w'] = dfw['IV']**2 * t
            dfw['lnm'] = np.log(dfw['m'])
            if dfw['lnm'].shape[0] <= 1:
                continue
            bounds = [(0.005, np.inf),  # sigma, 0.005 from paper
                      # Zelaide systems
                      (dfw['lnm'].min(), dfw['lnm'].max())]
            res = minimize(obj,
                           args=(dfw['lnm'],
                                 dfw['w']),
                           bounds=bounds,
                           tol=1e-8,
                           method='COBYQA',
                           options={'disp': False,
                                    'maxiter': 100000},
                           # XXX: Make this 10-100 restarts with
                           # randomly chosen points
                           x0=(0.5,  # sigma
                               dfw['lnm'].max())  # m
                           )
            if res.success:
                fparams = res.x
                sigma = fparams[0]
                m = fparams[1]
                # XXX: Get the linear param values
                pw, lparams = inner_opt(fparams, dfw['lnm'], dfw['w'])
                a, d, c = lparams[0], lparams[1], lparams[2]
                assert (lparams[0] >= 0)
                assert (sigma > 0)
                vv = np.sqrt(a + d*(-m/sigma) + c*np.sqrt((m/sigma)**2+1))
                thetats.append(vv)
                ttaus.append(t)
                # FIXME: Fix static arbitrage here
        # XXX: Fit the \Sigma to alpha*(1-exp(-lambda*t))
        (a, b, m) = ATMTS_fit(np.array(thetats), ttaus, k.split('/')[-1])
        thetaFits['date'].append(k.split('/')[-1].split('.')[0])     # date
        thetaFits['alpha'].append(a)     # alpha
        thetaFits['beta'].append(b)     # beta
        thetaFits['mu'].append(m)     # mu
        thetaFits['tau'].append(ttaus)  # taus
        thetaFits['theta'].append(thetats)  # raw theta values
        return thetaFits

    from joblib import Parallel, delayed
    res = Parallel(n_jobs=-1)(delayed(compute)(df, k)
                              for k, df in dfs.items())
    return res


def MSSVI(params, XK, TY):
    alpha = params[0]
    beta = params[1]
    mu = params[2]
    rho = params[3]
    nu = params[4]
    res = list()
    for i in range(TY.shape[0]):
        ty = TY[i][0]
        theta = (mu + alpha * (1-np.exp(-beta*ty)))**2
        phi = nu/(theta**0.5)
        k = XK[i]
        result = (0.5 * theta) * (1 + rho * phi * k +
                                  np.sqrt((phi * k + rho)**2 +
                                          1 - rho**2))
        result = np.sqrt(result/ty)*100
        res.append(result)
    # XXX: This should be a 2D array
    return np.array(res)


def main_raw(dfs, otype, ff='./thetaFits_SPX_call.json'):
    def ssvi(params, theta, t, k):
        rho = params[0]
        nu = params[1]
        phi = nu / (theta**0.5)  # power law
        result = (0.5 * theta) * (1 + rho * phi * k +
                                  np.sqrt((phi * k + rho)**2 +
                                          1 - rho**2))
        # XXX: Giving back the "IV" slice at "t"
        return np.sqrt(result/t)

    def ssvi_plot(all_params):
        # XXX: This is just for plotting
        K = np.linspace(-1.5, 1.5, 100)
        T = np.array([i/365 for i in range(14, 730)])
        XK, YT = np.meshgrid(K, T, indexing='xy')
        # XXX: Don't include the date param in there
        Z = MSSVI(all_params[:-1], XK, YT)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(XK, YT, Z, antialiased=False, linewidth=0,
                        cmap='viridis')
        ax.view_init(elev=27, azim=-129)
        ax.set_zlabel('IV (%)')
        ax.set_xlabel('Log moneyness')
        ax.set_ylabel('Term structure')
        plt.show()
        plt.close(fig)

    def fitssvi(alpha, beta, mu, taus, df, date):
        def obj(params, taus, df, alpha, beta, mu):
            pY = list()
            Y = list()
            for T in taus:
                dfw = df[df['tau'] == T][['m', 'IV']]
                theta = (mu + alpha*(1-np.exp(-beta*T)))**2
                Y.append(dfw['IV'])
                pY.append(ssvi(params, theta, T, np.log(dfw['m'].values)))
            # [print(i.shape) for i in Y]
            # [print(i.shape) for i in pY]
            # XXX: Now do a sum of square differences
            h1 = np.array([j for i in Y for j in i])
            h2 = np.array([j for i in pY for j in i])
            return np.sum((h1 - h2)**2)

        print('Doing: ', date)
        # XXX: Tenors to consider
        df = df[(df['Type'] == otype) & (df['tau'] >= 14/365) &
                (df['tau'] <= 2)]
        # XXX: Moneyness to consider
        df = df[(df['m'] >= 0.8) & (df['m'] <= 1.2) & (df['Volume'] > 1)]
        res = minimize(
            obj,
            x0=[0.4, 0.1],
            args=(taus, df, alpha, beta, mu),
            bounds=[(-1+1e-6, 1-1e-6), (0+1e-6, np.inf)],
            method='COBYQA',
            options={'disp': False, 'maxiter': 100000}
        )
        if res.success:
            # XXX: Now we can plot the real result vis-a-vis the predicted
            # result
            rho = res.x[0]
            nu = res.x[1]
            all_params = [alpha, beta, mu, rho, nu,
                          pd.to_datetime(date, format='%Y%m%d')]
            # XXX: We can call plot here if we want
            return all_params

    # XXX: Read the data that you need from the fitted svi_raw
    data = mpu.io.read(ff)
    instr = list(dfs.values())[0]['UnderlyingSymbol'][0]+'_'+otype
    res = Parallel(n_jobs=-1)(delayed(fitssvi)(
        d['alpha'][0],
        d['beta'][0],
        d['mu'][0],
        d['tau'][0],
        dfs[d['date'][0]],
        d['date'][0]
    ) for d in data)
    res = np.array(res)
    res = pd.DataFrame(data=res, columns=['alpha', 'beta', 'mu', 'rho', 'nu',
                                          'date'])
    res.to_csv('./ssvi_params_%s.csv' % instr)

    # XXX: Fit the var model
    # var_fits(df, ORDER_CRITERION='aic')


def build_samples(df, lag):
    """Makes input and output samples (AR style) for training models
    given dataframe df and lag L.

    """

    train_sample = df.shape[0]
    samples = list()
    response = list()
    assert (train_sample - lag > 0)
    for i in range(train_sample-lag):
        samples.append(df.iloc[i:lag+i].values)
        res = df.iloc[lag + i]
        res = res.values if len(df.shape) > 1 else res
        response.append(res)
        # response.append(df.iloc[lag+i].values)

    samples = np.array(samples)
    response = np.array(response)
    return samples, response


class MYLSTM:
    def __init__(self, TSTEP, batch_size, LR=1e-3, activation='tanh',
                 LSTMOUT=40, TT='AR'):
        self.TSTEP = TSTEP
        self.batch_size = batch_size
        self.LR = LR
        self.activation = activation
        self.LSTMOUT = LSTMOUT
        self.TT = TT
        self.model = self.__kerasLSTM()

    def __kerasLSTM(self):
        if self.TT == 'AR':
            inp = Input(shape=(self.TSTEP, 1), batch_size=self.batch_size)
        elif self.TT == 'VAR':
            # XXX: This is for VAR, thus 5 SSVI features being used all
            # at once
            inp = Input(shape=(self.TSTEP, 5), batch_size=self.batch_size)
        else:
            assert ("Cannot determine the type of regression for LSTM")

        l1 = LSTM(self.LSTMOUT, stateful=False, return_sequences=True,
                  activation=self.activation)(inp)
        dl = Dropout(0.1)(l1)
        l2 = LSTM(self.LSTMOUT, stateful=False, return_sequences=False,
                  activation=self.activation)(dl)
        dl2 = Dropout(0.1)(l2)
        l3 = keras.layers.Dense(1, activation=self.activation)(dl2)
        model = Model(inp, l3)
        model.compile(loss=keras.losses.mean_squared_error,
                      optimizer=keras.optimizers.Adam(learning_rate=self.LR))
        # print(model.summary())
        return model

    def __fitLSTM(self, trainX, trainY, batch_size,
                  epochs=5000, validation_split=0.15):
        # Define some callbacks to improve training.
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=50, restore_best_weights=True)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                      patience=5)
        # Fit the model to the training data -- online training
        # for i in range(epochs):
        _ = self.model.fit(
            trainX,
            trainY,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0,
            validation_split=validation_split,
            shuffle=True,
            callbacks=[early_stopping, reduce_lr])
        # self.model.get_layer(index=1).reset_states()
        # self.model.get_layer(index=2).reset_states()
        return self

    def fit(self, trainX, trainY, batch_size=128):
        # XXX: Make the trainX and trainY shape as needed
        if self.TT == 'AR':
            trainX = trainX.reshape(*trainX.shape, 1)
        trainX = np.asarray(trainX).astype('float32')
        trainY = trainY.reshape(*trainY.shape, 1)
        trainY = np.asarray(trainY).astype('float32')
        return self.__fitLSTM(trainX, trainY, batch_size)

    def predict(self, testX):
        # XXX: Make the trainX and trainY shape as needed
        if self.TT == 'AR':
            testX = testX.reshape(*testX.shape, 1)
        testX = np.asarray(testX).astype('float32')
        res = self.model.predict(testX, batch_size=1, verbose=0)
        res = res.reshape(res.shape[0],)
        # self.model.get_layer(index=1).reset_states()
        # self.model.get_layer(index=2).reset_states()
        return res


def predict_ssvi_params(ff='./ssvi_params_SPX_call.csv', train_sample=3000,
                        CV=1000, har_fit=True, var_fit=True, ar_fit=True,
                        dolstm=False, doridgexg=True,
                        sarimax_fit=True, varmax_fit=True,
                        glsar_fit=True, bayesian_fit=True):
    """Different prediction methods for predicting the parameters of
    SSVI IV surface.

    """

    def scores(Y, YP, insample=True):
        if insample:
            print('****************In sample scores********************')
        else:
            print('****************Out sample scores********************')
        alphas = pd.DataFrame({'alpha': Y['alpha'].values,
                               'palpha': YP['alpha'].values}).dropna()
        betas = pd.DataFrame({'beta': Y['beta'].values,
                              'pbeta': YP['beta'].values}).dropna()
        mus = pd.DataFrame({'mu': Y['mu'].values,
                            'pmu': YP['mu'].values}).dropna()
        rhos = pd.DataFrame({'rho': Y['rho'].values,
                             'prho': YP['rho'].values}).dropna()
        nus = pd.DataFrame({'nu': Y['nu'].values,
                            'pnu': YP['nu'].values}).dropna()
        ascore, bscore, muscore, rscore, nscore = (-np.inf, -np.inf,
                                                   -np.inf, -np.inf,
                                                   -np.inf)
        if alphas.shape[0] > 0:
            ascore = r2_score(alphas['alpha'], alphas['palpha'])
            print('alpha r2 score: ', ascore)
        if betas.shape[0] > 0:
            bscore = r2_score(betas['beta'], betas['pbeta'])
            print('beta r2 score: ', bscore)
        if mus.shape[0] > 0:
            muscore = r2_score(mus['mu'], mus['pmu'])
            print('mu r2 score: ', muscore)
        if rhos.shape[0] > 0:
            rscore = r2_score(rhos['rho'], rhos['prho'])
            print('rho r2 score: ', rscore)
        if nus.shape[0] > 0:
            nscore = r2_score(nus['nu'], nus['pnu'])
            print('nu r2 score: ', nscore)
        return (ascore+bscore+muscore+rscore+nscore)/5
        # return r2_score(Y.drop(['date'], axis=1), YP.drop(['date'], axis=1))

    def ar_fit_ridge_xgboost_lstm(X, Y, dates,
                                  best: dict,
                                  lags,
                                  VARS=5,
                                  pp=False,
                                  model_name='ridge'):

        def get_ars(offset, X):
            assert (offset < 5)
            assert (offset >= 0)
            ii = list(range(offset, X.shape[1], 5))
            indices = np.array([i in ii
                                for i in
                                range(X.shape[1])]*X.shape[0]).reshape(
                                    X.shape[0], X.shape[1])
            toret = X[indices]
            toret = toret.reshape(X.shape[0], toret.shape[0]//X.shape[0])
            return toret

        # names = {0: 'alpha', 1: 'beta', 2: 'mu', 3: 'rho', 4: 'nu'}
        names = {i: j for i, j in enumerate(colnames)}
        X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])
        alphas = np.array([0.1, 0.5, 1, 2, 5, 10])
        if model_name == 'ridge':
            model = RidgeCV(alphas, scoring='neg_mean_squared_error', cv=5)
        elif model_name == 'xgboost':
            model = XGBRegressor(n_jobs=-1, booster='gblinear')
        elif model_name == 'lstm':
            model = MYLSTM(lags, batch_size=None,
                           activation=activations[names[VARS[0]]])
        # XXX: For each get the variables and perform AR-Ridge and XGBBoost
        VARS = range(VARS) if type(VARS) is int else VARS
        for i in VARS:
            if model_name == 'lstm':
                print('Doing: ', names[i])
            Xtrains = get_ars(i, X)
            Ytrains = Y[:, i]
            model = model.fit(Xtrains, Ytrains)
            # XXX: Test the results
            df_test = df[colnames][train_sample-lags-1:
                                   train_sample-lags-1+CV]  # the -1 is needed
            testX, responseY = build_samples(df_test, lags)
            testX = testX[:, :, :-1]
            testX = testX.reshape(testX.shape[0],
                                  testX.shape[1]*testX.shape[2])
            testX = get_ars(i, testX)
            testYP = model.predict(testX)
            testYP = pd.DataFrame(testYP, columns=[colnames[i]])
            testYP['date'] = responseY[:, -1]
            testY = pd.DataFrame(responseY[:, i], columns=[colnames[i]])
            testY['date'] = responseY[:, -1]
            testYP.index = pd.to_datetime(testYP['date'])
            # print(testY.head())
            # print(testYP.head())
            score = r2_score(testY[names[i]], testYP[names[i]])
            if best[names[i]][0] < score:
                best[names[i]] = BAR(score, lags, model, testYP[names[i]])
            # print('%s: %s for lags: %s' % (names[i], score, lags))

    def var_fit_ridge_xgboost_lstm(X, Y, response, dates, lags,
                                   reg_score, xgboost_score,
                                   lstm_score, dolstm, doridgexg):
        if dolstm:
            # XXX: The test vector
            df_test = df[colnames][train_sample-lags-1:
                                   train_sample-lags-1+CV]  # the -1 is needed
            testX, responseY = build_samples(df_test, lags)
            testX = testX[:, :, :-1]
            # XXX: Training and testing the LSTM
            print('-------------------LSTM-------------------')
            for k, v in enumerate(colnames[:-1]):
                print('Doing var: ', v)
                model = MYLSTM(lags, batch_size=None,
                               activation=activations[v],
                               TT='VAR').fit(X, Y[:, k])
                testYP = model.predict(testX)
                testYP = testYP.reshape(testYP.shape[0],)
                testYP = pd.DataFrame(testYP, columns=[v])
                testYP['date'] = responseY[:, -1]
                testYP.index = pd.to_datetime(testYP['date'])
                testY = pd.DataFrame(responseY[:, k], columns=[v])
                score = r2_score(testY[v], testYP[v])
                if lstm_score[v][0] < score:
                    # XXX: The tscore is the average score
                    lstm_score[v] = BVAR(score, lags, model, testYP[v])
        if doridgexg:
            # flattned for Ridge
            X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])
            # XXX: Fit a RidgeCV model
            alphas = np.array([0.1, 0.5, 1, 2, 5, 10])
            # XXX: Uses 5 fold Cross-validation for Ridge regression
            ridge = RidgeCV(alphas, scoring='neg_mean_squared_error',
                            cv=5, fit_intercept=False).fit(X, Y)
            # YP = pd.DataFrame(ridge.predict(X), columns=colnames[:-1])
            # YP['date'] = dates
            # response = pd.DataFrame(response, columns=colnames)
            print('-------------------Ridge CV-------------------')
            # scores(response, YP)               # in sample scores

            # XXX: Testing out of samples -- Ridge regression
            df_test = df[colnames][train_sample-lags-1:
                                   train_sample-lags-1+CV]  # the -1 is needed
            testX, responseY = build_samples(df_test, lags)
            testX = testX[:, :, :-1]
            testX = testX.reshape(testX.shape[0],
                                  testX.shape[1]*testX.shape[2])
            # XXX: This is for the RidgeCV
            testYP = pd.DataFrame(ridge.predict(testX), columns=colnames[:-1])
            testYP['date'] = responseY[:, -1]
            testYP.index = pd.to_datetime(testYP['date'])
            testY = pd.DataFrame(responseY, columns=colnames)
            # out of sample scores
            tscore = scores(testY, testYP, insample=False)
            if reg_score[colnames[0]][0] < tscore:
                for k, v in enumerate(colnames[:-1]):
                    reg_score[v] = BVAR(tscore, lags, ridge, testYP[v])

            print('-------------------XGBoost-------------------')
            # XXX: This is XGBRegressor
            for k, v in enumerate(colnames[:-1]):
                model = XGBRegressor(n_jobs=-1,
                                     booster='gblinear').fit(X, Y[:, k])
                testYP = model.predict(testX)
                testYP = pd.DataFrame(testYP, columns=[v])
                testYP['date'] = responseY[:, -1]
                testYP.index = pd.to_datetime(testYP['date'])
                score = r2_score(testY[v], testYP[v])
                if xgboost_score[v][0] < score:
                    xgboost_score[v] = BVAR(score, lags, model, testYP[v])
        return reg_score, xgboost_score, lstm_score

    def arma_ssvi_fit(train_df, train_dates, test_df, test_df_i,
                      arma_best,
                      v, ORDER=(1, 0, 1)):
        model = SARIMAX(train_df, order=ORDER, trend='n')
        mres = model.fit(method='nm', maxiter=1000000,
                         disp=False, return_params=False)
        arparams = mres.arparams[::-1]
        maparams = mres.maparams[::-1]
        predict_resids = mres.resid
        predict_resids.index = train_dates

        def arma_predict(df):
            assert (df.shape[0] == arparams.shape[0])
            vv = np.dot(arparams, df)
            if maparams.shape[0] > 0:
                # XXX: Add the MA components too
                vv += np.dot(maparams, predict_resids[-maparams.shape[0]:])
            predict_resids.loc[len(predict_resids)] = (
                df_test.loc[df.index[-1]+1] - vv)
            return vv

        # XXX: Predict on a rolling basis
        res = test_df[:len(test_df)-1].rolling(
            arparams.shape[0]).apply(lambda x:
                                     arma_predict(x)).dropna()
        test_df = test_df[arparams.shape[0]:]
        assert (res.dropna().shape == df_test[arparams.shape[0]:].shape)
        res.index = test_df_i[arparams.shape[0]:]
        test_df.index = test_df_i[arparams.shape[0]:]
        score = r2_score(test_df, res)
        print(v, ' arma R2 score: ', score, 'order: ', ORDER)

        if arma_best[v][0] < score and score >= 0:
            arma_best[v] = BARMA(score, (arparams.shape[0], 0,
                                         maparams.shape[0]),
                                 mres, res)

    def glsar_ssvi_fit(train_df, train_dates, test_df, test_df_i,
                       glsar_best, v, ORDER=1):
        from statsmodels.regression.linear_model import GLSAR
        model = GLSAR(train_df, rho=ORDER)
        mres = model.iterative_fit(maxiter=100000, tol=1e-8)
        glsarparams = model.rho[::-1]
        # const = mres.params.values[0]

        def glsar_predict(df):
            assert (df.shape[0] == glsarparams.shape[0])
            # XXX: Check if we have to add the const?
            vv = np.dot(glsarparams, df)
            return vv

        res = test_df[:len(test_df)-1].rolling(
            glsarparams.shape[0]).apply(lambda x:
                                        glsar_predict(x)).dropna()
        test_df = test_df[glsarparams.shape[0]:]
        assert (res.dropna().shape == df_test[glsarparams.shape[0]:].shape)
        res.index = test_df_i[glsarparams.shape[0]:]
        test_df.index = test_df_i[glsarparams.shape[0]:]
        score = r2_score(test_df, res)
        print(v, ' GLSAR R2 score: ', score, 'order: ', ORDER)
        if glsar_best[v][0] < score and score >= 0:
            glsar_best[v] = GLSARB(score, glsarparams.shape[0], mres, res)

    def varma_ssvi_fit(train_df, train_dates, test_df, test_df_i,
                       varma_best, ORDER=(1, 0)):
        model = VARMAX(train_df, order=ORDER, trend='n')
        mres = model.fit(method='lbfgs', maxiter=1000000,
                         disp=False, return_params=False)
        arparams = mres.coefficient_matrices_var[::-1]
        if ORDER[1] > 0:
            maparams = mres.coefficient_matrices_vma[::-1]
            predict_resids = mres.resid

        def varma_predict(df):
            assert (df.shape[0] == arparams.shape[0])
            vv = list()
            for i in range(df.shape[0]):
                vv.append(np.dot(arparams[i], df.iloc[i].T))

            vv = np.sum(vv, axis=0)
            if ORDER[1] > 0:
                maresids = predict_resids[-maparams.shape[0]:]
                # XXX: Add the MA components too
                for i in range(maparams.shape[0]):
                    vv += np.dot(maparams[i], maresids.iloc[i].T)
                # XXX: Add the new error to predict_resids
                predict_resids.loc[len(predict_resids)] = (
                    df_test.loc[df.index[-1]+1] - vv)
            return vv.T         # make it (1, #vars)

        # XXX: Predict on a rolling basis
        res = list()
        # XXX: Perform a rolling prediction
        for i in range(test_df.shape[0]-arparams.shape[0]):
            res.append(varma_predict(test_df[i:i+arparams.shape[0]]))
        # Turn it into a dataframe
        res = pd.DataFrame(res, columns=colnames[:-1])
        test_df = test_df[arparams.shape[0]:]
        assert (res.shape == test_df.shape)
        res.index = test_df_i[arparams.shape[0]:]
        test_df.index = test_df_i[arparams.shape[0]:]
        score = r2_score(test_df, res)
        print('VARMAX R2 score: ', score, 'order: ', ORDER)

        if varma_best[0] < score and score >= 0:
            maparams = maparams.shape[0] if ORDER[1] > 0 else 0
            varma_best = BVARMA(score, (arparams.shape[0], maparams),
                                mres, res, test_df)
        return varma_best

    def har_ssvi_fit(train_df, train_dates, test_df, har_best,
                     VARS, LAGS=[1, 5, 21]):
        from arch.univariate import HARX
        train_df.index = pd.to_datetime(train_dates)
        test_df.index = pd.to_datetime(test_df['date'])
        test_df = test_df[train_df.columns]
        for var in [VARS]:
            harx_model = HARX(train_df[var], lags=LAGS, rescale=False)
            harx_fit_res = harx_model.fit(disp='off', update_freq=0)
            params = harx_fit_res.params[:-1].values.T  # just the mean model
            # XXX: Now test it
            forecast = test_df[var].rolling(LAGS[-1]).apply(
                lambda x:
                np.dot(params, np.array([1,
                                         x[:LAGS[0]].mean(),
                                         x[:LAGS[1]].mean(),
                                         x[:LAGS[2]].mean()]))
            ).dropna()
            score = r2_score(test_df[var][LAGS[-1]-1:], forecast)
            # print('%s R2 score: %s' % (var, score))
            if har_best[var][0] < score and score >= 0:
                har_best[var] = BHAR(score, LAGS, harx_fit_res, forecast)

    def bayesian_ar(df_train, p, v, mws, df_test, df_test_i, barmabest):
        samples, response = build_samples(df_train, p)
        # XXX: Now perform Bayesian estimation
        with pm.Model() as model:
            tsamples = pm.Data("tsamples", samples)
            tresponse = pm.Data("tresponse", response)
            # XXX: variance for LL distribution
            sigma = pm.HalfCauchy('sigma', beta=100**2)
            # XXX: The likelihood function
            if mws == "uniform":
                ws = [pm.Uniform('w%s' % w, lower=-1, upper=1)
                      for w in range(p)]
                intercept = pm.Uniform("intercept", lower=-1,
                                       upper=1)
            else:
                ws = [pm.Normal('w%s' % w) for w in range(p)]
                intercept = pm.Normal("intercept")
            # XXX: The dof for LL distribution
            dof = pm.HalfCauchy('dof', beta=100**2)
            _ = pm.StudentT('response',
                            nu=dof,
                            mu=intercept + tsamples.dot(ws),
                            sigma=sigma,
                            observed=tresponse)
        # XXX: For testing to get the best from CV
        test_samples, test_response = build_samples(df_test, p)
        import pymc.sampling.jax as pmjax
        with model:
            idata = pmjax.sample_numpyro_nuts(chains=10,
                                              target_accept=0.97,
                                              progressbar=False)
            # XXX: Sample posterior and return the prediction
            pm.set_data({"tsamples": test_samples,
                         "tresponse": np.zeros(test_response.shape[0])})
            idata.extend(pm.sample_posterior_predictive(idata,
                                                        predictions=False,
                                                        progressbar=False))
        res = idata.posterior_predictive['response'].median(['chain', 'draw'])
        res = res.to_pandas()
        res.index = df_test_i[p:]
        score = r2_score(test_response, res)
        print(v, ' Bayesian ARMA R2 score: ', score, 'order: ', (p, 0, 0, mws))
        if barmabest[v][0] < score:
            barmabest[v] = BAYAR(score, (p, 0, 0, mws), model, res)

    def bayesian_arma(df_train, p, q, v, mws, df_test, df_test_i,
                      barmabest):
        def step(*args):
            """This is the step function for ARMA scan
            """
            ytrue = args[0]
            ys = list(args[1:p+1])
            errs = list(args[p+1:p+1+q])
            intercept, w, ew, sigma, dof = args[p+1+q:]
            yp = w.dot(ys) + intercept + ew.dot(errs)
            # Likelihood function
            yhat = pm.StudentT.dist(mu=yp, sigma=sigma, nu=dof)
            epsilon = ytrue - yp
            return (epsilon, yhat), collect_default_updates([yhat, epsilon])

        import pytensor as pt
        from pymc.pytensorf import collect_default_updates
        arlags = [-1*i for i in range(p+1)]
        malags = [-1*i for i in range(1, q+1)]
        with pm.Model() as model:
            dof = pm.HalfCauchy("dof", beta=10**2)
            sigma = pm.HalfCauchy('sigma', beta=10)
            if mws == "uniform":
                w = pm.Uniform("ws", lower=-1, upper=1, shape=p)
                ew = pm.Uniform("ew", lower=-1, upper=1, shape=q)
                intercept = pm.Uniform("intercept", lower=-1, upper=1)
            else:
                w = pm.Normal("ws", shape=p)
                ew = pm.Normal("ew", shape=q)
                intercept = pm.Normal("intercept")

            Y = pm.Data("Y", df_train)
            err0 = (pt.tensor.zeros(shape=q)[0] if q == 1 else
                    pt.tensor.zeros(shape=q))
            initials = [{'initial': err0, 'taps': malags}, None]
            [epsilons, yhat], update = pt.scan(
                fn=step,
                sequences=[{'input': Y, 'taps': arlags}],
                outputs_info=initials,
                non_sequences=[intercept, w, ew, sigma, dof],
                strict=True, allow_gc=False)

            tresponse = pm.Data("tresponse", df_train[p:train_sample])
            model.register_rv(yhat, name='response', observed=tresponse)
        import pymc.sampling.jax as pmjax
        with model:
            idata = pmjax.sample_numpyro_nuts(chains=5,
                                              target_accept=0.97,
                                              progressbar=False)
            pm.set_data({"Y": df_test,
                         "tresponse": np.zeros(df_test.shape[0])})
            idata.extend(pm.sample_posterior_predictive(idata,
                                                        predictions=False))
            res = idata.posterior_predictive['response'].median(
                ['chain', 'draw'])
            res = res.to_pandas()
            res.index = df_test_i[p:]
        # XXX: Notice the addition of 'p', because of lags
        score = r2_score(df_test[p:], res)
        print(v, ' Bayesian ARMA R2 score: ', score, 'order: ', (p, 0, q, mws))
        if barmabest[v][0] < score:
            barmabest[v] = BAYAR(score, (p, 0, q, mws), model, res)

    def bayesian_fit_f(df_train, train_dates, test_df, test_df_i,
                       barmabest, v, p, q, ws):
        if q == 0:
            res = bayesian_ar(df_train, p, v, ws, test_df, test_df_i,
                              barmabest)
        else:
            res = bayesian_arma(df_train, p, q, v, ws, test_df, test_df_i,
                                barmabest)
        return res

    # XXX: Calling the main functions
    colnames = ['rho', 'beta', 'mu', 'alpha', 'nu', 'date']
    activations = {'rho': 'tanh', 'beta': 'tanh', 'mu': 'tanh',
                   'alpha': 'tanh', 'nu': 'tanh'}
    df = pd.read_csv(ff)

    # XXX: The original dataset
    dfo = df.copy()
    dfo = dfo[colnames]
    dfo.index = pd.to_datetime(dfo['date'])
    dfo = dfo.drop('date', axis=1)

    if dolstm:
        # XXX: Scaling alpha for outliers
        from sklearn.preprocessing import MinMaxScaler
        alpha_scaler = MinMaxScaler((-1, 1)).fit(
            np.log(df['alpha']).values.reshape(-1, 1))
        res = alpha_scaler.fit_transform(
            np.log(df['alpha'].values).reshape(-1, 1))
        df['alpha'] = res

        beta_scaler = MinMaxScaler((-1, 1)).fit(
            df['beta'].values.reshape(-1, 1))
        df['beta'] = beta_scaler.fit_transform(df['beta'].values.reshape(-1,
                                                                         1))

        nu_scaler = MinMaxScaler((-1, 1)).fit(df['nu'].values.reshape(-1, 1))
        df['nu'] = nu_scaler.fit_transform(df['nu'].values.reshape(-1, 1))

        mu_scaler = MinMaxScaler((-1, 1)).fit(df['mu'].values.reshape(-1, 1))
        df['mu'] = mu_scaler.fit_transform(df['mu'].values.reshape(-1, 1))

        scalers = {'alpha': alpha_scaler,
                   'beta': beta_scaler,
                   'nu': nu_scaler,
                   'mu': mu_scaler}
    else:
        scalers = {}

    # from statsmodels.stats.descriptivestats import describe
    # print(describe(df[1:]))
    df_train = df[colnames][:train_sample]

    if glsar_fit:
        GLSARB = namedtuple("GLSAR", ['score', 'lags', 'model',
                                      'tForecast'])
        glsar_best = {i: GLSARB(score=-np.inf, lags=(0, 0, 0),
                                model=None, tForecast=None)
                      for i in colnames[:-1]}
        for v in glsar_best.keys():
            df_test = df[v].iloc[train_sample:
                                 train_sample+CV]
            df_test.index = df['date'].iloc[train_sample:
                                            train_sample+CV]
            for ar in range(1, 40):
                df_test = df[v].iloc[train_sample-ar:train_sample-ar+CV]
                df_test_i = df['date'].iloc[train_sample-ar:
                                            train_sample-ar+CV]
                glsar_ssvi_fit(df_train[v], df_train['date'],
                               df_test, df_test_i, glsar_best,
                               v, ORDER=ar)
    if sarimax_fit:
        import warnings
        warnings.filterwarnings(action='ignore', category=UserWarning)
        BARMA = namedtuple("BARMA", ['score', 'lags', 'model',
                                     'tForecast'])
        arma_best = {i: BARMA(score=-np.inf, lags=(0, 0, 0),
                              model=None, tForecast=None)
                     for i in colnames[:-1]}
        for v in arma_best.keys():
            df_test = df[v].iloc[train_sample:
                                 train_sample+CV]
            df_test.index = df['date'].iloc[train_sample:
                                            train_sample+CV]
            for ar in range(1, 10):
                df_test = df[v].iloc[train_sample-ar:train_sample-ar+CV]
                df_test_i = df['date'].iloc[train_sample-ar:
                                            train_sample-ar+CV]
                for ma in range(0, 5):
                    arma_ssvi_fit(df_train[v], df_train['date'],
                                  df_test, df_test_i, arma_best,
                                  v, ORDER=(ar, 0, ma))

    if varmax_fit:
        import warnings
        warnings.filterwarnings(action='ignore', category=UserWarning)
        BVARMA = namedtuple("BVARMA", ['score', 'lags', 'model',
                                       'tForecast', 'rForecast'])
        varma_best = BVARMA(score=-np.inf, lags=(0, 0),
                            model=None, tForecast=None,
                            rForecast=None)

        df_test = df.iloc[train_sample:train_sample+CV]
        df_test.index = df['date'].iloc[train_sample:
                                        train_sample+CV]
        for ar in range(1, 11):
            df_test = df[colnames[:-1]].iloc[train_sample-ar:
                                             train_sample-ar+CV]
            df_test_i = df['date'].iloc[train_sample-ar:
                                        train_sample-ar+CV]
            # XXX: Anything above 4 here results is very bad results
            for ma in range(0, 4):
                varma_best = varma_ssvi_fit(df_train[colnames[:-1]],
                                            df_train['date'],
                                            df_test, df_test_i,
                                            varma_best,
                                            ORDER=(ar, ma))
        # XXX: Now just make the new varma_best for each column
        # separately
        varma_best = {i: BVARMA(model=varma_best.model,
                                lags=varma_best.lags,
                                tForecast=varma_best.tForecast[i],
                                score=r2_score(varma_best.rForecast[i],
                                               varma_best.tForecast[i]),
                                rForecast=varma_best.rForecast[i])
                      for i in colnames[:-1]}

    if har_fit:
        # XXX: HAR model fit for the parameters
        print('----------------------------HAR--------------------')
        L2 = list(range(2, 5))
        L3 = list(range(5, 30))
        BHAR = namedtuple('BHAR',
                          ['score', 'lags', 'model', 'tForecast'])
        har_best = {i: BHAR(score=-np.inf, lags=-1,
                            model=None, tForecast=None)
                    for i in colnames[:-1]}
        for l2 in L2:
            for l3 in L3:
                LAGS = [1, l2, l3]
                for v in har_best.keys():
                    # print('Doing %s with lags %s' % (v, LAGS))
                    har_ssvi_fit(df_train[colnames[:-1]], df_train['date'],
                                 df.iloc[train_sample-LAGS[-1]:
                                         train_sample-LAGS[-1]+CV],
                                 LAGS=LAGS,
                                 har_best=har_best, VARS=v)

    if var_fit:
        # XXX: Grid search for VAR models with Ridge and XGBoost
        BVAR = namedtuple('BVAR',
                          ['score', 'lags', 'model', 'tForecast'])
        r_score = {i: BVAR(-np.inf, 0, None, None) for i in colnames[:-1]}
        x_score = {i: BVAR(-np.inf, 0, None, None) for i in colnames[:-1]}
        l_score = {i: BVAR(-np.inf, 0, None, None) for i in colnames[:-1]}
        print('-----------VAR------------')
        for i in [1, 5, 10, 20, 30, 40]:
            print('Doing VAR lag: ', i)
            # LAGS = [i]
            samples, response = build_samples(df_train, i)

            # XXX: Get the samples that you need
            X = samples[:, :, :-1]  # removed the date
            Y = response[:, :-1]   # remove the date
            dates = response[:, -1]
            r_score, x_score, l_score = var_fit_ridge_xgboost_lstm(
                np.copy(X), np.copy(Y), np.copy(response), np.copy(dates),
                i, r_score, x_score, l_score, dolstm, doridgexg)

    if bayesian_fit:
        import warnings
        warnings.filterwarnings(action='ignore', category=UserWarning)
        # import arviz as az
        import pymc as pm
        # XXX: Performing grid search for lags in AR with Ridge and XGBoost
        BAYAR = namedtuple('BAYAR',
                           ['score', 'lags', 'model', 'tForecast'])
        barmabest = {i: BAYAR(-np.inf, 0, None, None) for i in colnames[:-1]}
        print('-----------BAYESIAN AR------------')

        for v in colnames[:-1]:
            df_test = df[v].iloc[train_sample:
                                 train_sample+CV]
            df_test.index = df['date'].iloc[train_sample:
                                            train_sample+CV]
            for ar in range(1, 10):
                df_test = df[v].iloc[train_sample-ar:train_sample-ar+CV]
                df_test_i = df['date'].iloc[train_sample-ar:
                                            train_sample-ar+CV]
                for ma in range(0, 5):
                    for mws in ['uniform', 'normal']:
                        bayesian_fit_f(df_train[v], df_train['date'],
                                       df_test, df_test_i, barmabest,
                                       v, ar, ma, mws)

    if ar_fit:
        # XXX: Performing grid search for lags in AR with Ridge,
        # XGBoost, and LSTM
        BAR = namedtuple('BAR',
                         ['score', 'lags', 'model', 'tForecast'])
        best_ridge = {i: BAR(-np.inf, 0, None, None) for i in colnames[:-1]}
        best_xgboost = {i: BAR(-np.inf, 0, None, None) for i in colnames[:-1]}
        best_lstm = {i: BAR(-np.inf, 0, None, None) for i in colnames[:-1]}
        print('-----------AR------------')
        for i in [1, 5, 10, 20, 30, 40]:
            print('Doing AR lags: ', i)
            # LAGS = [i]
            samples, response = build_samples(df_train, i)

            # XXX: Get the samples that you need
            X = samples[:, :, :-1]  # removed the date
            Y = response[:, :-1]   # remove the date
            dates = response[:, -1]

            for v in [[0], [1], [2], [3], [4]]:
                if dolstm:
                    # XXX: This is the LSTM
                    ar_fit_ridge_xgboost_lstm(np.copy(X), np.copy(Y),
                                              # np.copy(response),
                                              np.copy(dates), best_lstm, i,
                                              pp=True,
                                              VARS=v,
                                              model_name='lstm')
                if doridgexg:
                    # XXX: This is AR Ridge
                    ar_fit_ridge_xgboost_lstm(np.copy(X), np.copy(Y),
                                              # np.copy(response),
                                              np.copy(dates), best_ridge, i,
                                              VARS=v,
                                              pp=True,
                                              model_name='ridge')
                    # XXX: This is the XGboost
                    ar_fit_ridge_xgboost_lstm(np.copy(X), np.copy(Y),
                                              # np.copy(response),
                                              np.copy(dates), best_xgboost, i,
                                              pp=True,
                                              VARS=v,
                                              model_name='xgboost')
        print('\n')

    def scale_back(df, scaler, var):
        df = scaler.inverse_transform(df)
        if var == 'alpha':
            df = np.exp(df)
        return df

    def print_best_res(dd: dict):
        """Prints the best results after inverse_transform to original
        values.

        """
        for k in dd:
            print(k, ' best R2: %s, lags: %s' % (dd[k].score,
                                                 dd[k].lags))
            # XXX: Conver the scaled values to real values
            start = dd[k].tForecast.index[0]
            end = dd[k].tForecast.index[-1]
            dfc = dfo[k].loc[start:end]
            if k in scalers.keys():
                pdfc = scale_back(dd[k].tForecast.values.reshape(-1, 1),
                                  scalers[k], k)
            else:
                pdfc = dd[k].tForecast
            print('Orig value R2: ', r2_score(dfc, pdfc))
            print('Orig value RMSE: ', root_mean_squared_error(dfc, pdfc))

    if glsar_fit:
        print('--------------GLSAR best results-------------------')
        print_best_res(glsar_best)
    if sarimax_fit:
        print('--------------ARMA SARIMAX best results-------------------')
        print_best_res(arma_best)
    if bayesian_fit:
        print('--------------ARMA Bayesian best results-------------------')
        print_best_res(barmabest)
    if varmax_fit:
        print('----------------VARMA best results----------------')
        print_best_res(varma_best)
    # XXX: Best HAR results
    if har_fit:
        print('--------------HAR best results-------------------')
        print_best_res(har_best)
    # XXX: The best models for VAR-Ridge and VAR-XGBBoost
    if var_fit:
        if doridgexg:
            print('------------------VAR best Ridge results----------------')
            print_best_res(r_score)
            print('------------------VAR best XGBoost results----------------')
            print_best_res(x_score)
        if dolstm:
            print('------------------VAR best LSTM results----------------')
            print_best_res(l_score)
    # XXX: The best models for AR-Ridge AR-XGBoost
    if ar_fit:
        if doridgexg:
            print('------------------AR best Ridge results----------------')
            print_best_res(best_ridge)
            print('------------------AR best XGBoost results----------------')
            print_best_res(best_xgboost)
        if dolstm:
            print('------------------AR best LSTM results----------------')
            print_best_res(best_lstm)


if __name__ == '__main__':
    # XXX: Read the real world data
    # dfs = load_real_data()

    # XXX: Create the required theta_t curves (takes a day on 28 cores)
    # thetaFits = lprocess_data(dfs, 'call')
    # mpu.io.write('/tmp/thetaFits.json', thetaFits)

    # XXX: Fit the raw SSVI parameters for each day
    # XXX: This is pretty fast
    # main_raw(dfs, 'call')

    # XXX: Ridge prediction for the AR and VAR models
    predict_ssvi_params(har_fit=False, ar_fit=True, dolstm=True,
                        var_fit=False, doridgexg=False,
                        sarimax_fit=False, varmax_fit=False,
                        glsar_fit=False, bayesian_fit=False)

    # XXX: Predict the next day SSVI parameters
    # main()
