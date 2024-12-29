#!/usr/bin/env python
import glob
import numpy as np
from model_load_test import date_to_num, num_to_date
import pandas as pd
from pred import SSVI
import matplotlib.pyplot as plt
# from statsmodels.stats.diagnostic import het_arch
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from joblib import Parallel, delayed
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from pred import cr2_score, cr2_score_pval
from keras.layers import Input, LSTM
from keras.models import Model
import keras
from scipy.optimize import minimize
import mpu


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
    res = Parallel(n_jobs=12)(delayed(compute)(df, k)
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


def ssvi_parm_explore(ff='./ssvi_params_SPX_call.csv'):

    def sarimax_fit(df, TRAIN_SAMPLE=2000):
        df = df.iloc[: TRAIN_SAMPLE, :]
        # XXX: Always use the differenced series, because of
        # optimisation failure.
        df = df.diff().dropna()

        stationarity(df)        # check if series is stationary

        # XXX: This is the order for diff series from acf and pacf
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        for i in range(df.shape[1]):
            Y = df[df.columns[i]]
            arc, mac = pacount(Y, nlags=100, diff=False)
            model = SARIMAX(endog=Y, order=(arc, 0, mac))
            model_res = model.fit(disp=False, method='nm',
                                  maxiter=1000000)
            print(model_res.summary())

            # XXX: Fit the residuals for hetroscadicity
            from arch.univariate import ZeroMean, GARCH, StudentsT
            # XXX: AR on differenced series
            # rp, ra = pacount(Y, BREAK=0.1, var=True, diff=False)
            # rp = 1 if rp <= 0 else rp
            # ra = 1 if ra <= 0 else ra
            vol_model = GARCH(p=1, q=1)
            model_res = ZeroMean(model_res.resid, volatility=vol_model,
                                 rescale=False, distribution=StudentsT())
            model_res = model_res.fit(update_freq=0, disp='off',
                                      options={'maxiter': 10000})
            assert (model_res.optimization_result.success)
            print(model_res.summary())
            # SCALE = model_res.scale

            from statsmodels.stats.diagnostic import het_arch
            lm, lmpval, fval, fpval = het_arch(model_res.std_resid,
                                               ddof=(arc+mac))
            # XXX: No hetroscadicity left in the model
            assert (lmpval > 0.05)
            assert (fpval > 0.05)
            print('%s het_arch test: lm:%s,lmpval:%s,fval:%s,fpval:%s' %
                  (df.columns[i], lm, lmpval, fval, fpval))
            from statsmodels.stats.stattools import jarque_bera
            jb, jbp, skew, kur = jarque_bera(model_res.std_resid)
            print('JB stat:%s, JBp-val:%s, skew:%s, kurtosis:%s' %
                  (jb, jbp, skew, kur))

    def vecm_fits(df, ORDER_CRITERION='aic', TRAIN_SAMPLE=2000):
        from statsmodels.tsa.vector_ar.vecm import select_coint_rank
        from statsmodels.tsa.vector_ar.vecm import VECM
        from statsmodels.tsa.vector_ar.vecm import select_order
        vecm_order = select_order(df.iloc[:TRAIN_SAMPLE, :], maxlags=100)
        assert (vecm_order.vecm)    # make sure this is vecm model
        vecm_nlags = vecm_order.selected_orders[ORDER_CRITERION]
        print('vecm lags: ', vecm_nlags)
        coint_rank = select_coint_rank(df, det_order=0,
                                       k_ar_diff=vecm_nlags,
                                       signif=0.01,
                                       method='maxeig')
        print('Co-integration rank @ 1% significance: ', coint_rank.rank)
        vecm_model = VECM(df.iloc[:TRAIN_SAMPLE, :],
                          k_ar_diff=vecm_nlags,
                          coint_rank=coint_rank.rank)
        vecm_res = vecm_model.fit()
        print(vecm_res.summary())
        print(vecm_res.test_normality().summary())
        print(vecm_res.test_whiteness(nlags=vecm_nlags+1).summary())

    def stationarity(df):
        # XXX: Are they all stationary?
        from statsmodels.tsa.stattools import adfuller
        adf, pval, _, _, cv, _ = adfuller(df['alpha'])
        assert (pval <= 0.05)
        adf, pval, _, _, cv, _ = adfuller(df['beta'])
        assert (pval <= 0.05)
        adf, pval, _, _, cv, _ = adfuller(df['mu'])
        assert (pval <= 0.05)
        adf, pval, _, _, cv, _ = adfuller(df['rho'])
        assert (pval <= 0.05)
        adf, pval, _, _, cv, _ = adfuller(df['nu'])
        assert (pval <= 0.05)

    def var_fits(df, ORDER_CRITERION='aic', TRAIN_SAMPLE=2000):
        df = df.iloc[:TRAIN_SAMPLE, :]  # Samples to test

        stationarity(df)        # First always check for stationarity

        # XXX: Then fit the model
        from statsmodels.tsa.vector_ar.var_model import VAR
        var_model = VAR(df)
        var_nlags = var_model.select_order(
            maxlags=100, trend='c').selected_orders[ORDER_CRITERION]
        var_model_res = var_model.fit(maxlags=100,
                                      ic=ORDER_CRITERION, trend='c')
        print(var_model_res.summary())
        print(var_model_res.test_normality().summary())
        print(var_model_res.test_whiteness(nlags=var_nlags+1).summary())

        # XXX: Is the model stable?
        print(var_model_res.is_stable())

        # XXX: Plot the auto correlation
        # var_model_res.plot_acorr(nlags=var_nlags+1)
        # plt.show()

        # XXX: Get the residuals and do het_arch test
        resids = var_model_res.resid
        print(resids.shape)
        # XXX: Fix this:
        # https://stats.stackexchange.com/questions/153017/how-should-i-test-for-multivariate-arch-effects-in-r
        from statsmodels.stats.diagnostic import het_arch
        params = ['alpha', 'beta', 'mu', 'rho', 'nu']
        for i in range(resids.shape[1]):
            lm, lmpval, fval, fpval = het_arch(resids.iloc[:, i],
                                               nlags=var_nlags+1,
                                               ddof=var_nlags)
            print('Het test ', params[i], ': ', lm, lmpval, fval, fpval)

    import warnings
    from statsmodels.tools.sm_exceptions import ValueWarning
    warnings.filterwarnings(action='ignore', category=ValueWarning)
    df = pd.read_csv(ff)
    df = df[['date', 'alpha', 'beta', 'mu', 'rho', 'nu']]
    df.index = pd.to_datetime(df['date'], format='%Y%m%d')
    df = df.drop('date', axis=1)

    # XXX: General descriptive statistics (Normally distributed?)
    from statsmodels.stats.descriptivestats import Description
    dd = Description(df)
    print(dd.summary())

    # XXX: Fit the SARIMAX model
    sarimax_fit(df)

    # XXX: Fit the var model
    # var_fits(df, ORDER_CRITERION='aic')


if __name__ == '__main__':
    # XXX: Read the real world data
    dfs = load_real_data()

    # XXX: Create the required theta_t curves (takes a day on 28 cores)
    # thetaFits = lprocess_data(dfs, 'call')
    # mpu.io.write('/tmp/thetaFits.json', thetaFits)

    # XXX: Fit the raw SSVI parameters for each day
    # XXX: This is pretty fast
    main_raw(dfs, 'call')

    # XXX: Explore the parameters of SSVI fit
    # ssvi_parm_explore()

    # XXX: Predict the next day SSVI parameters
    # main()
