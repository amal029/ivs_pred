#!/usr/bin/env python
import glob
import numpy as np
from model_load_test import date_to_num, num_to_date
import pandas as pd
from pred import SSVI
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import het_arch
from sklearn.linear_model import HuberRegressor


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

    # XXX: Test resids for hetroscadicity
    lmr, lmpr, fvalr, fpr = het_arch(model_fit.resid, nlags=10,
                                     ddof=sum(order))
    if lmpr <= 0.05 and fiterrors:
        LAGS = 10
        SCALE = 100
        # XXX: Do Ridge regression here
        X = np.array(model_fit.resid.values*SCALE)
        Y = list()
        for i in range(X.shape[0]):
            if (i > 0) and (i % LAGS == 0):
                Y.append(X[i])
        Y = np.array(Y)**2
        X = X.reshape(X.shape[0]//LAGS, LAGS)**2
        # print(Y.shape, X.shape)
        # XXX: Fit the Ridge model for the error residuals
        var_model = HuberRegressor(max_iter=100000).fit(X[:-1], Y)
        # print('OLS score: ', var_model.score(X[:-1], Y))
        var = np.sqrt(var_model.predict(X[-1].reshape(1, -1))[0])/SCALE
        # print('ret before: ', ret)
        # print('var^2: ', var_model.predict(X[-1].reshape(1, -1))[0]/SCALE**2)
        # print('var :', var)
        if not np.isnan(var):
            ret += var
        # ret = ret + var if var is not np.nan else ret
        # print('ret after: ', ret)
        # assert (False)

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

    from sklearn.metrics import r2_score
    print('R2 score rho: ', r2_score(params[WINDOW:, 0], prho[:-1]))
    print('R2 score nu: ', r2_score(params[WINDOW:, 1], pnu[:-1]))

    # XXX: Get the error vector between prediction and reality
    erho = params[WINDOW:, 0] - prho.values[:-1].reshape(
        params[WINDOW:, 0].shape)
    enu = params[WINDOW:, 1] - pnu.values[:-1].reshape(
        params[WINDOW:, 1].shape)

    # XXX: Test if the out of sample errors are normally distributed.
    from statsmodels.stats.stattools import jarque_bera
    jbr, jbpr, skewr, kurr = jarque_bera(erho)
    print('JB test rho; jb:%s, jbp:%s, skew:%s, kur:%s' % (jbr, jbpr,
                                                           skewr, kurr))
    jbn, jbpn, skewn, kurn = jarque_bera(enu)
    print('JB test nu; jb:%s, jbp:%s, skew:%s, kur:%s' % (jbn, jbpn,
                                                          skewn, kurn))
    # # XXX: Test for hetroscadicity of out of sample errors
    # lmr, lmpr, fvalr, fpr = het_arch(erho, ddof=6)
    # print('ARCH test rho: lm%s, lmp:%s, fr:%s, fpr:%s' % (lmr, lmpr,
    #                                                       fvalr, fpr))
    # lmr, lmpr, fvalr, fpr = het_arch(pnu, ddof=5)
    # print('ARCH test rho: lm%s, lmp:%s, fr:%s, fpr:%s' % (lmr, lmpr,
    #                                                       fvalr, fpr))

    # XXX: Plot of out of sample errors
    # fig, axs = plt.subplots(nrows=2, ncols=1)
    # axs[0].plot(erho)
    # axs[1].plot(enu)
    # plt.show()
    # plt.close(fig)

    # XXX: Plot the out of sample prediction vs truth
    # fig, axs = plt.subplots(nrows=2, ncols=1)
    # axs[0].plot(params[:, 0], label=r'$\rho$')
    # axs[0].plot(prho, label=r'$\hat{\rho}$')
    # axs[0].legend()
    # axs[1].plot(params[:, 1], label=r'$\nu$')
    # axs[1].plot(pnu, label=r'$\hat{\nu}$')
    # axs[1].legend()
    # plt.show()
    # plt.close(fig)


def main():
    for otype in ['call']:
        X = load_data(otype, START='20090102')

        WINDOW = 1000
        # XXX: Fit the SSVI model to each day separately
        ssvi = SSVI('ssviridge', 0)
        params, ATM_IV = ssvi.fitY(X['IVS'])

        # XXX: Fit a volatility (GARCH style) model and forecast
        # doGARCH(params, WINDOW, SCALE)

        # XXX: Now try this same thing with ARIMA
        doARIMA(params, WINDOW)


if __name__ == '__main__':
    main()
