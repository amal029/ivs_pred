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
from scipy.optimize import differential_evolution
from scipy.optimize import NonlinearConstraint


# XXX: This function loads the real data
def load_real_data(dd='./interest_rates', otype='call'):
    toret = dict()
    for d in range(2002, 2003):
        ff = sorted(glob.glob(dd+'/'+str(d)+'*.csv'))
        for i in ff:
            toret[i] = pd.read_csv(i)
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


def pacount(x, BREAK=0.1, var=False):
    import numpy as np
    from statsmodels.tsa.stattools import pacf, acf
    xd = np.diff(x)
    if not var:
        pa = pacf(xd, nlags=10)
    else:
        pa = pacf(xd**2, nlags=10)
    pcount = 0
    for i in pa[1:]:
        if np.abs(i) < BREAK:
            break
        pcount += 1

    if not var:
        aa = acf(xd, nlags=10)
    else:
        aa = acf(xd**2, nlags=10)
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
    for otype in ['put', 'call']:
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


def lprocess_data(dfs, otype):
    # XXX: The function to compute the optimal transformed problem
    def linear_obj(params, veck, w):
        sigma = params[0]
        m = params[1]
        X = np.ones(veck.shape[0]*3).reshape(veck.shape[0], 3)
        Y = w                 # vector of real total variance
        for i in range(veck.shape[0]):
            yy = (veck.iloc[i]-m)/sigma
            X[i] = [1, yy, np.sqrt(yy**2+1)]
        # XXX: Solve using linear algebra
        beta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))
        # toret = np.dot(X, beta)
        # print('toret: ', toret, 'beta: ', beta)
        # print('X: ', X)
        # XXX: These are the linearized constants
        return np.dot(X, beta), beta

    # XXX: The target optimisation function
    def obj(params, veck, w):
        pw, _ = linear_obj(params, veck, w)
        return np.sum((pw - w)**2)

    # count = 0
    for k, df in dfs.items():
        # count += 1
        # if count < 10:
        #     continue
        print('Doing: ', k)
        df = df[df['Type'] == otype]
        taus = sorted(df['tau'].unique())
        # XXX: For each given tau fit the SVI param
        for t in taus:
            dfw = df[df['tau'] == t][['m', 'IV', 'Strike']]
            dfw['w'] = dfw['IV']**2 * t
            dfw['lnm'] = np.log(dfw['m'])
            bounds = [(1e-6, np.inf),
                      (dfw['lnm'].min(), dfw['lnm'].max())]
            res = minimize(obj,
                           args=(dfw['lnm'],
                                 dfw['w']),
                           bounds=bounds,
                           tol=1e-8,
                           method='Nelder-Mead',
                           options={'disp': True,
                                    'maxiter': 100000},
                           # XXX: Make this 10-100 restarts with
                           # randonly chosen points
                           x0=(0.5,  # sigma
                               dfw['lnm'].min())  # m
                           )
            print(res)
            if res.success:
                fparams = res.x
                sigma = fparams[0]
                m = fparams[1]
                # XXX: Get the linear param values
                pw, lparams = linear_obj(fparams, dfw['lnm'], dfw['w'])
                # print('fparams: ', fparams)
                # print('lparams: ', lparams)
                # print('pw: ', pw)
                # print('w: ', dfw['w'])
                # print('error: ', np.sum((pw - dfw['w'])**2))
                # XXX: Now just plot the graph
                K = np.linspace(-1.5, 1.5, 100)
                pIVS = [(lparams[0] +
                         lparams[1]*((i-m)/sigma) +
                         lparams[2]*np.sqrt(((i-m)/sigma)**2+1))
                        for i in K]
                # print('PIVS: ', pIVS)
                plt.plot(K, pIVS)
            plt.plot(dfw['lnm'], dfw['w'], marker='o', linestyle='none')
            plt.show()
            plt.close()
            assert (False)


if __name__ == '__main__':
    # XXX: Read the real world data
    dfs = load_real_data()
    lprocess_data(dfs, 'call')
    # main()
