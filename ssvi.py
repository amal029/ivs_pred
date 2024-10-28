#!/usr/bin/env python
import glob
import numpy as np
from model_load_test import date_to_num, num_to_date
import pandas as pd
from pred import SSVI
import matplotlib.pyplot as plt
# from statsmodels.stats.diagnostic import het_arch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


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


def fitandforecastARIMA(Y, order=(4, 1, 1), trend='n', fiterrors=False):
    from statsmodels.tsa.arima.model import ARIMA
    model_fit = ARIMA(Y, order=order, trend=trend).fit()
    ret = model_fit.forecast(steps=1).values[0]

    # XXX: Test for hetroscadicity of fitted errors
    # from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
    # plot_acf(model_fit.resid**2)
    # plot_pacf(model_fit.resid**2)
    # plt.show()

    # # XXX: Test resids for hetroscadicity
    # lmr, lmpr, fvalr, fpr = het_arch(model_fit.resid, nlags=10,
    #                                  ddof=sum(order))
    return ret


def doARIMA(params, WINDOW):
    # XXX: For nu p=3, d=1, q=1
    # XXX: For rho p=4, d=1, q=1
    rho = pd.DataFrame(params[:, 0])
    nu = pd.DataFrame(params[:, 1])

    prho = rho.rolling(WINDOW).apply(
        lambda x:
        fitandforecastARIMA(x, order=(4, 1, 1), trend='n')
    ).dropna()

    pnu = nu.rolling(WINDOW).apply(
        lambda x:
        fitandforecastARIMA(x, order=(3, 1, 1), trend='n')
    ).dropna()

    return pd.DataFrame({'rho': prho[:-1], 'nu': pnu[:-1]})


def doATMIV(atmiv, WINDOW, TSTEP=5):
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


def main():
    for otype in ['call']:
        X = load_data(otype, START='20090102')

        WINDOW = 1000
        # XXX: Fit the SSVI model to each day separately
        ssvi = SSVI('ssviridge', 0)
        params, ATM_IV = ssvi.fitY(X['IVS'])

        # XXX: Get the ATM_IV predictions
        pY = doATMIV(ATM_IV, WINDOW)
        print(ATM_IV[WINDOW:].shape, pY.shape)
        print('R2 score ATM: ', r2_score(ATM_IV[WINDOW:], pY))

        # XXX: Now try this same thing with ARIMA
        pparams = doARIMA(params, WINDOW)
        print(params[WINDOW:].shape, pparams.shape)
        print('R2 score rho: ', r2_score(params[WINDOW:, 0], pparams['rho']))
        print('R2 score nu: ', r2_score(params[WINDOW:, 1], pparams['nu']))


if __name__ == '__main__':
    main()
