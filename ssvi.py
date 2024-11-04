#!/usr/bin/env python
import glob
import multiprocessing
import numpy as np
from model_load_test import date_to_num, num_to_date
import pandas as pd
from pred import SSVI
import matplotlib.pyplot as plt
# from statsmodels.stats.diagnostic import het_arch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from joblib import Parallel, delayed
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
import dask.dataframe as dd
import keras
from pred import cr2_score, cr2_score_pval
from non_linear_models import LSTM, DNN


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


def fitandforecastARIMA(Y, trend='n'):
    def pacount(x, BREAK=0.2):
        import numpy as np
        from statsmodels.tsa.stattools import pacf, acf
        pa = pacf(np.diff(x), nlags=10)
        pcount = 0
        for i in pa[1:]:
            if np.abs(i) < BREAK:
                break
            pcount += 1
        # pcount = 1 if pacount == 0 else pcount

        aa = acf(np.diff(x), nlags=10)
        acount = 0
        for i in aa[1:]:
            if np.abs(i) < BREAK:
                break
            acount += 1
        # acount = 1 if acount == 0 else acount

        return pcount, acount

    Y = Y.values.reshape(Y.shape[0],)
    rp, ra = pacount(Y)

    SCALE = 10
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    model_fit = SARIMAX(Y*SCALE, order=(rp, 1, ra), trend=trend,
                        ).fit(maxiter=10000, disp=False,
                              method='bfgs')
    ret = model_fit.forecast(steps=1)[0]/SCALE
    return ret


def doARIMA(params, WINDOW):

    rho = pd.DataFrame(params[:, 0])
    nu = pd.DataFrame(params[:, 1])

    # XXX: For nu p=4, d=1, q=1
    # XXX: For rho p=3, d=1, q=1

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

def innerSVM(ivs, TSTEP):
    # XXX: Get the outputs
    Y = list()
    for i in range(ivs.shape[0]):
        if i > 0 and i % TSTEP == 0:
            Y.append(ivs[i])
    Y = np.array(Y)

    X = np.array([ivs[i] for i in range(ivs.shape[0])])

    # XXX: Reshape the ivs
    X = X.reshape(X.shape[0]//TSTEP, TSTEP, X.shape[1])
    X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])
    svm_rho = svm.SVR()
    svm_nu = svm.SVR()

    svm_rho.fit(X[:-1, 0].reshape((X[:-1].shape[0], 1)), Y[:, 0])
    svm_nu.fit(X[:-1, 1].reshape((X[:-1].shape[0], 1)), Y[:, 1])

    # XXX: Predict the output
    pY = svm_rho.predict(X[-1, 0].reshape(1, 1))
    pYY = svm_nu.predict(X[-1, 0].reshape(1, 1))
    return np.array([pY, pYY])



def doSVM(params, WINDOW, TSTEP):

    # XXX: Roll through the ivs and doit
    N = params.shape[0]
    pY = list()
    input = []
    # Create iterable input with (params[N:WINDOW], TSTEP)
    for i in range(N-WINDOW):
        input.append((params[i:WINDOW+i], TSTEP))
    cpu_count = multiprocessing.cpu_count()
    with multiprocessing.Pool(cpu_count) as p:
        print("Starting multiprocessing")
        output = p.starmap(innerSVM, input) 
        print("Done")
        for x in output:
            pY.append(x)
    return np.array(pY)

def innerRF(ivs, TSTEP):
    # XXX: Get the outputs
    Y = list()
    for i in range(ivs.shape[0]):
        if i > 0 and i % TSTEP == 0:
            Y.append(ivs[i])
    Y = np.array(Y)

    X = np.array([ivs[i] for i in range(ivs.shape[0])])
    # XXX: Reshape the ivs
    X = X.reshape(X.shape[0]//TSTEP, TSTEP, X.shape[1])
    X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])

    rf = RandomForestRegressor(n_estimators=1000)
    
    rf.fit(X[:-1], Y)

    # XXX: Predict the output
    pY = rf.predict(X[-1].reshape(1, *X[-1].shape))
    return pY.reshape(pY.shape[1],) 

def doRF(params, WINDOW, TSTEP):
    print("Doing RF")
    # XXX: Roll through the ivs and doit
    N = params.shape[0]
    pY = list()
    input = []
    # Create iterable input with (params[N:WINDOW], TSTEP)
    for i in range(N-WINDOW):
        input.append((params[i:WINDOW+i], TSTEP))
    cpu_count = multiprocessing.cpu_count()
    with multiprocessing.Pool(cpu_count) as p:
        print("Starting multiprocessing")
        output = p.starmap(innerRF, input) 
        print("Done")
        for x in output:
            pY.append(x)
    return np.array(pY)

def innerDNN(dnn, ivs, TSTEP):
    # XXX: Get the outputs
    Y = list()
    for i in range(ivs.shape[0]):
        if i > 0 and i % TSTEP == 0:
            Y.append(ivs[i])
    Y = np.array(Y)

    X = np.array([ivs[i] for i in range(ivs.shape[0])])
    # XXX: Reshape the ivs
    X = X.reshape(X.shape[0]//TSTEP, TSTEP, X.shape[1])
    X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])
    # XXX: Check if the model is fitted
    if not dnn.check_is_fitted():
        dnn.fit(X[:-1], Y)
    # XXX: Predict the output
    pY = dnn.predict(X[-1].reshape(1, *X[-1].shape))
    return pY.reshape(pY.shape[1],)

def doDNN(params, WINDOW, TSTEP):
    # XXX: Roll through the ivs and doit
    N = params.shape[0]
    pY = list()

    # XXX: Create the DNN model
    dnn = DNN((TSTEP*2,), 2)
    dnn.compile(optimizer=keras.optimizers.Adam(0.001))

    # XXX: Parallel processing
    # input = []
    # # Create iterable input with (params[N:WINDOW], TSTEP)
    # for i in range(N-WINDOW):
    #     input.append((params[i:WINDOW+i], TSTEP))
    # cpu_count = multiprocessing.cpu_count()
    # with multiprocessing.Pool(6) as p:
    #     print("Starting multiprocessing")
    #     output = p.starmap(innerDNN, input) 
    #     print("Done")
    #     for x in output:
    #         pY.append(x)

    # XXX: Loop through the inputs
    for i in range(N-WINDOW): 
        if i % 100 == 0:
            print('Done %s' % i)
        aa = params[i:WINDOW+i]
        yy = innerDNN(aa, TSTEP)
        pY.append(yy)

    return np.array(pY)

def innerLSTM(lstm, ivs, TSTEP):
    # XXX: Get the outputs
    Y = list()
    for i in range(ivs.shape[0]):
        if i > 0 and i % TSTEP == 0:
            Y.append(ivs[i])
    Y = np.array(Y)

    X = np.array([ivs[i] for i in range(ivs.shape[0])])
    # XXX: Reshape the ivs
    X = X.reshape(X.shape[0]//TSTEP, TSTEP, X.shape[1])
    # XXX: Check if the model is fitted
    if not lstm.check_is_fitted():
        lstm.fit(X[:-1], Y)
    # XXX: Predict the output
    pY = lstm.predict(X[-1].reshape(1, *X[-1].shape))
    return pY.reshape(pY.shape[1],) 

def doLSTM(params, WINDOW, TSTEP):

    
    # XXX: Roll through the ivs and doit
    N = params.shape[0]
    pY = list()

    # XXX: Create the LSTM model
    lstm = LSTM((TSTEP, 2), 2)
    lstm.compile(optimizer=keras.optimizers.Adam(0.001))

    # XXX: Multi processing
    # input = []
    # for i in range(N-WINDOW):
    #     input.append((params[i:WINDOW+i], TSTEP))
    # cpu_count = multiprocessing.cpu_count()
    # with multiprocessing.Pool(cpu_count) as p:
    #     print("Starting multiprocessing")
    #     output = p.starmap(innerLSTM, input) 
    #     print("Done")
    #     for x in output:
    #         pY.append(x)
    
    for i in range(N-WINDOW):
        if i % 100 == 0:
            print('Done %s' % i)
        aa = params[i:WINDOW+i]
        yy, lstm= innerLSTM(aa, lstm)
        pY.append(yy)

    return pd.DataFrame({'rho': pY[:, 0], 'nu': pY[:, 1]}) 

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


def doRidge(X, WINDOW, TSTEP):
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

def non_linear():
    # turn off gpu support for tensorflow
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
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

        # XXX: DNN fit
        paramsDNN = doDNN(params, WINDOW, TSTEP) 
        with open('./ssvi_results.txt', 'a') as f:
            f.write('R2 score DNN rho: %s\n' % r2_score(params[WINDOW:, 0], paramsDNN[:, 0]))
            f.write('R2 score DNN nu: %s\n' % r2_score(params[WINDOW:, 1], paramsDNN[:, 1]))
        print('R2 score DNN rho: ', r2_score(params[WINDOW:, 0], paramsDNN[:, 0]))
        print('R2 score DNN nu: ', r2_score(params[WINDOW:, 1], paramsDNN[:, 1]))

        # XXX: SVM fit 
        paramsSVM = doSVM(params, WINDOW, TSTEP)
        with open('./ssvi_results.txt', 'a') as f:
            f.write('R2 score SVM rho: %s\n' % r2_score(params[WINDOW:, 0], paramsSVM[:, 0]))
            f.write('R2 score SVM nu: %s\n' % r2_score(params[WINDOW:, 1], paramsSVM[:, 1]))
        print('R2 score SVM rho: ', r2_score(params[WINDOW:, 0], paramsSVM[:, 0]))
        print('R2 score SVM nu: ', r2_score(params[WINDOW:, 1], paramsSVM[:, 1]))

        # XXX: RF fit
        paramsRF = doRF(params, WINDOW, TSTEP)
        with open('./ssvi_results.txt', 'a') as f:
            f.write('R2 score RF rho: %s\n' % r2_score(params[WINDOW:, 0], paramsRF[:, 0]))
            f.write('R2 score RF nu: %s\n' % r2_score(params[WINDOW:, 1], paramsRF[:, 1]))
        print('R2 score RF rho: ', r2_score(params[WINDOW:, 0], paramsRF[:, 0]))
        print('R2 score RF nu: ', r2_score(params[WINDOW:, 1], paramsRF[:, 1]))

        # XXX: LSTM fit
        paramsLSTM = doLSTM(params, WINDOW, TSTEP)
        with open('./ssvi_results.txt', 'a') as f:
            f.write('R2 score LSTM rho: %s\n' % r2_score(params[WINDOW:, 0], paramsLSTM['rho']))
            f.write('R2 score LSTM nu: %s\n' % r2_score(params[WINDOW:, 1], paramsLSTM['nu']))
        print('R2 score LSTM rho: ', r2_score(params[WINDOW:, 0], paramsLSTM['rho']))
        print('R2 score LSTM nu: ', r2_score(params[WINDOW:, 1], paramsLSTM['nu']))

        # XXX: Now try this same thing with ARIMA
        pparams = doARIMA(params, WINDOW)
        with open('./ssvi_results.txt', 'a') as f:
            f.write('R2 score rho: %s\n' % r2_score(params[WINDOW:, 0], pparams['rho']))
            f.write('R2 score nu: %s\n' % r2_score(params[WINDOW:, 1], pparams['nu']))
        print(params[WINDOW:].shape, pparams.shape)
        print('R2 score rho: ', r2_score(params[WINDOW:, 0], pparams['rho']))
        print('R2 score nu: ', r2_score(params[WINDOW:, 1], pparams['nu']))


        print('R2 score SVM rho: ', r2_score(params[WINDOW:, 0], paramsSVM[:, 0]))
        print('R2 score SVM nu: ', r2_score(params[WINDOW:, 1], paramsSVM[:, 1]))
        print('R2 score RF rho: ', r2_score(params[WINDOW:, 0], paramsRF[:, 0]))
        print('R2 score RF nu: ', r2_score(params[WINDOW:, 1], paramsRF[:, 1]))
        print('R2 score rho: ', r2_score(params[WINDOW:, 0], pparams['rho']))
        print('R2 score nu: ', r2_score(params[WINDOW:, 1], pparams['nu']))
        print('R2 score LSTM rho: ', r2_score(params[WINDOW:, 0], paramsLSTM['rho']))
        print('R2 score LSTM nu: ', r2_score(params[WINDOW:, 1], paramsLSTM['nu']))




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

        # XXX: LSTM fit
        plstmY = doLSTM(params, WINDOW, TSTEP)
        print('R2 score LSTM rho: ', r2_score(params[WINDOW:, 0], plstmY['rho']))
        print('R2 score LSTM nu: ', r2_score(params[WINDOW:, 1], plstmY['nu']))

        # XXX: Ridge fit (VAR model)
        prY = doRidge(X, WINDOW, TSTEP)
        print('R2 score Ridge: ', r2_score(yT, prY))

        # XXX: Get the ATM_IV predictions
        pY = doATMIV(ATM_IV, WINDOW, TSTEP=TSTEP)
        print(ATM_IV[WINDOW:].shape, pY.shape)
        print('R2 score ATM: ', r2_score(ATM_IV[WINDOW:], pY))

        # XXX: Now try this same thing with ARIMA
        pparams = doARIMA(params, WINDOW)
        print(params[WINDOW:].shape, pparams.shape)
        print('R2 score rho: ', r2_score(params[WINDOW:, 0], pparams['rho']))
        print('R2 score nu: ', r2_score(params[WINDOW:, 1], pparams['nu']))

        # XXX: Predict the IVS using ARIMA models
        paY = Parallel(n_jobs=-1)(delayed(ssvi.predict1)(
            pparams.loc[i, :], pY[i])
                                  for i in range(pY.shape[0]))
        paY = np.array(paY)
        paY = paY.reshape(paY.shape[0], paY.shape[1]*paY.shape[2])
        print('R2 score ARIMA: ', r2_score(yT, paY))

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
        print('Ridge vs ARIMA SSVI: ', cr2_score(yT, prY, paY))
        print('Ridge vs ARIMA SSVI p-val: ', cr2_score_pval(yT, prY, paY))

        # XXX: Compare the R2 scores Ridge SSVI vs SSVI ARIMA
        print('Ridge SSVI vs ARIMA SSVI: ', cr2_score(yT, psY, paY))
        print('Ridge vs ARIMA SSVI p-val: ', cr2_score_pval(yT, psY, paY))


if __name__ == '__main__':
    non_linear()
