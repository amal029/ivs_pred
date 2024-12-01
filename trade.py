#!/usr/bin/env python

import pred
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import blosc2
from blackscholes import BlackScholesCall, BlackScholesPut


def getpreds_trading(name1, otype):
    mms = np.arange(pred.LM, pred.UM+pred.MSTEP, pred.MSTEP)
    # XXX: Now go through the TS
    tts = [i/pred.DAYS for i in range(pred.LT, pred.UT+pred.TSTEP,
                                      pred.TSTEP)]
    MS = len(mms)
    TS = len(tts)

    data1 = blosc2.load_array(name1)
    # XXX: These are the dates
    dates = data1[:, 0].astype(np.datetime64, copy=False)

    y = data1[:, 1:MS*TS+1].astype(float, copy=False)
    yp = data1[:, MS*TS+1:].astype(float, copy=False)

    # XXX: Across time
    return (dates, y.reshape(dates.shape[0], MS, TS),
            yp.reshape(dates.shape[0], MS, TS))


def trade(dates, y, yp, otype, strat, eps=0.05, lags=5):

    def getTC(data, P=0.25):
        # XXX: See: Options Trading Costs Are Lower than You
        # Think Dmitriy Muravyev (page-4)
        return 2.5/100
        # return 2.5/100

    def c_position_s(sign, CP, PP, TC, N):
        """This is trading a straddle
        """
        tc = (CP * TC) + (PP * TC)
        if sign > 0:
            return ((CP + PP) - tc)*N
        else:
            return ((-(CP + PP)) - tc)*N

    def o_position_s(sign, CP, PP, TC, N):
        tc = CP * TC + PP * TC
        if sign > 0:
            return ((-(CP + PP)) - tc)*N
        else:
            return ((CP + PP) - tc)*N

    N = 5                     # number of contracts to buy/sell every time
    mms = np.arange(pred.LM, pred.UM+pred.MSTEP, pred.MSTEP)
    # XXX: Now go through the TS
    tts = [i/pred.DAYS for i in range(pred.LT, pred.UT+pred.TSTEP,
                                      pred.TSTEP)]
    # XXX: First read all the real prices from database for the test
    # dates
    data = dict()
    for d in dates:
        df = pd.read_csv('./interest_rates/%s.csv' % str(d))
        df = df[df['Type'] == otype].reset_index()
        df = df[['IV', 'm', 'tau', 'Mid', 'Delta', 'UnderlyingPrice', 'Strike',
                 'InterestR', 'Ask', 'Bid', 'Last']]
        data[d] = df

    cash = 10000               # starting cash position 100K
    ip = list()                 # list of traded indices (moneyness)
    jp = list()                 # list of traded indices (term structure)
    mp = list()                 # list of traded moneyness
    tp = list()                 # list of traded term structure
    dl = list()                 # list of traded deltas
    Kp = list()
    Sp = list()
    Rp = list()
    Cp = list()
    signl = list()              # list of traded volatility
    cashl = list()              # list of cash positions
    trade_date = list()   # list of trade dates
    pos_date = list()
    open_position = False       # do we have a current open position?

    # XXX: Attach the underlying price
    mDates = list()
    marketPrice = list()

    # XXX: Actual trade
    for t in range(0, dates.shape[0]-1):
        open_p = False
        # XXX: Get all the points from the real dataset
        tdata = data[dates[t]]
        # XXX: These are the changes
        dhat = yp[t+1] - y[t]
        ddhat = np.abs(dhat)
        R = np.mean(np.abs(tdata['InterestR'].values))
        R = Rp[-1] if np.isnan(R) else R
        UP = tdata['UnderlyingPrice'].values[0]
        mDates.append(dates[t])
        marketPrice.append(UP)
        # XXX: Now get the highest dhat point
        i, j = np.unravel_index(np.argmax(ddhat, axis=None), ddhat.shape)

        # XXX: Another technique to use min instead of max
        # i, j = np.unravel_index(np.argmin(ddhat, axis=None), ddhat.shape)

        # XXX: Only if the change is greater than some filter (eps) --
        # trade.
        if ddhat[i, j]/y[t][i, j] >= eps and yp[t+1][i, j] > 0:
            m = mms[i]
            tau = tts[j]
            S = UP
            K = m*S
            if otype == 'call':
                ecall = BlackScholesCall(S=S, K=K, T=tau, r=R,
                                         sigma=y[t][i, j])
                ecallp = BlackScholesCall(S=S, K=K, T=tau-(1/365), r=R,
                                          sigma=yp[t+1][i, j])
            else:
                ecall = BlackScholesPut(S=S, K=K, T=tau, r=R,
                                        sigma=y[t][i, j])
                ecallp = BlackScholesPut(S=S, K=K, T=tau-(1/365), r=R,
                                         sigma=yp[t+1][i, j])
            if otype == 'put':
                PP = BlackScholesCall(S=S, K=K, T=tau, r=R,
                                      sigma=y[t][i, j]).price()
                PPp = BlackScholesCall(S=S, K=K, T=tau-(1/365), r=R,
                                       sigma=yp[t+1][i, j]).price()
            else:
                PP = BlackScholesPut(S=S, K=K, T=tau, r=R,
                                     sigma=y[t][i, j]).price()
                PPp = BlackScholesPut(S=S, K=K, T=tau-(1/365), r=R,
                                      sigma=yp[t+1][i, j]).price()

            Delta = ecall.delta()
            CP = ecall.price()
            CPp = ecallp.price()

            # XXX: Uncomment the line below for real trading
            if (np.abs((CPp + PPp) - (CP + PP))/(CP+PP) >= 0.05 and
                # XXX: Only long positions taken, no shorts
                dhat[i, j] > 0):
                open_p = True       # open a position later

        if open_position:   # is position already open?
            # XXX: Get the days that have passed by
            trd = pd.to_datetime(str(pos_date[-1]), format='%Y%m%d')
            today = pd.to_datetime(str(dates[t]), format='%Y%m%d')
            # print('Days to maturity: ', tp[-1]*365)
            # print('Today: ', today, 'Trad day: ', trd)
            # print('days gone: ', (today - trd).days)
            DTM = tp[-1]*365 - (today-trd).days
            # print(mms[i], mp[-1], tp[-1]*365, tts[j]*365,
            #       tts[j]-((today-trd).days/365))

            # XXX: Maturity has reached. Maturity might be a weekend,
            # because of abstract tau.
            if DTM <= 0:
                # print('Maturity today: ', int(DTM))
                # XXX: Close the position in a different way
                DTM_UP = data[dates[t-int(DTM)]]['UnderlyingPrice'].values[0]
                K = Kp[-1]
                TC = getTC(tdata, 1)
                if signl[-1] > 0:
                    # FIXME: Need to correctly add the transaction costs
                    # here on maturity dates.
                    res = (max(DTM_UP-K, 0) + max(K-DTM_UP, 0)) - TC
                else:
                    res = -(max(DTM_UP-K, 0) + max(K-DTM_UP, 0)) - TC
                cash += res
                if not open_p:
                    cashl.append(cash)
                    trade_date.append(dates[t])
                open_position = False
                # print('Position closed on Maturity')

            # XXX: We have to redo all training with 1 day tts.
            elif (mms[i] == mp[-1] and tp[-1] == tts[j]):
                # elif i == ip[-1] and j == jp[-1]:
                # print('holding!')
                open_p = False  # just hold

            else:
                UPo = Kp[-1]
                # UPo = mp[-1]*UP

                days_gone = (today - trd).days//pred.TSTEP
                days_left = (today - trd).days/365  # days to subtract
                # print('closing the position')

                # T = 1e-2 if tp[-1]-days_left <= 0 else tp[-1]-days_left
                # J = 0 if jp[-1]-days_gone < 0 else jp[-1]-days_gone
                T = tp[-1]-days_left
                J = jp[-1]-days_gone

                if otype == 'call':
                    ecall = BlackScholesCall(S=UP, K=UPo, T=T,
                                             r=R, sigma=y[t][ip[-1], J])
                else:
                    ecall = BlackScholesPut(S=UP, K=UPo, T=T,
                                            r=R, sigma=y[t][ip[-1], J])
                CPo = ecall.price()

                if otype == 'call':
                    UPo = BlackScholesPut(S=UP, K=UPo, T=T,
                                          r=R, sigma=y[t][ip[-1], J]).price()
                else:
                    UPo = BlackScholesCall(S=UP, K=UPo, T=T,
                                           r=R, sigma=y[t][ip[-1], J]).price()

                # XXX: Get transaction costs for this day as % of
                # bid-ask spread.
                TC = getTC(tdata, 1)
                cash += c_position_s(signl[-1], CPo, UPo, TC, N)
                # cash += c_position(signl[-1], CPo, UPo, dl[-1], TC)
                # XXX: Only append if we are not going to open a new
                # position immediately
                if not open_p:
                    cashl.append(cash)
                    trade_date.append(dates[t])
                open_position = False

        # if open_p and (not open_position):
        # XXX: You can open multiple (same or different) positions at
        # once.
        if open_p:
            TC = getTC(tdata, 1)
            ccash = o_position_s(dhat[i, j], CP, PP, TC, N)
            # ccash = o_position(dhat[i, j], CP, S, Delta, TC)
            if cash + ccash >= 0:
                open_position = True  # for closing the position later
                # XXX: Make the cash updates
                mp.append(m)
                tp.append(tau)
                dl.append(Delta)
                signl.append(dhat[i, j])
                ip.append(i)
                jp.append(j)
                Sp.append(S)
                Kp.append(K)
                Rp.append(R)
                Cp.append(CP)
                cash += ccash
                cashl.append(cash)
                trade_date.append(dates[t])
                pos_date.append(dates[t])
                # print('Opened position on: ',
                #       pd.to_datetime(str(dates[t]), format='%Y%m%d'))
            open_p = False      # The position is now opened
        else:
            cashl.append(cash)
            trade_date.append(dates[t])
            open_p = False      # Kinda done!

    import os
    if not os.path.exists('./trades/marketPrice.csv'):
        mdates = pd.to_datetime(mDates, format='%Y%m%d')
        df = pd.DataFrame({'dates': mdates, 'Price': marketPrice})
        df.to_csv('./trades/marketPrice.csv')

    trade_date = pd.to_datetime(trade_date, format='%Y%m%d')
    res = pd.DataFrame({'cash': cashl, 'dates': trade_date})
    res = res.dropna()
    res.to_csv('./trades/%s_%s_%s.csv' % (strat, otype, lags))


def analyse_trades(otype, model, lags, alpha, betas,
                   cagrs, wins, maxs, mins, avgs, medians,
                   sds, srs, ns, rf=3.83/(100), Y=9):

    import warnings
    warnings.filterwarnings("ignore")
    # print('Model %s_%s_%s: ' % (model, otype, lags))
    mrdf = pd.read_csv('./trades/marketPrice.csv')
    prdf = pd.read_csv('./trades/%s_%s_%s.csv' % (model, otype, lags))

    prdf['pcash'] = prdf['cash'].expanding().apply(
        lambda x:
        (x.values[-1]-x.values[0])/x.values[0]*100)

    # XXX: Plot the portfolio
    prdf.plot.line(x='dates', y='pcash', label='%s_%s' % (model, otype))
    plt.xlabel('Years')
    plt.ylabel('% P/L')
    plt.xticks(rotation=45)
    plt.legend()
    plt.savefig('./trades/%s_%s_%s_portfolio.pdf' % (model, otype, lags),
                bbox_inches='tight')
    plt.close()

    prdf['Date'] = pd.to_datetime(prdf['dates'], format='%Y-%m-%d')
    mrdf['Date'] = pd.to_datetime(mrdf['dates'], format='%Y-%m-%d')
    # prdf['pct_chg'] = prdf.groupby(prdf.Date.dt.year)['cash'].pct_change()
    # mrdf['pct_chg'] = mrdf.groupby(mrdf.Date.dt.year)['Price'].pct_change()

    # XXX: Expected portfolio return
    rp = prdf.groupby(prdf.Date.dt.year)['cash'].apply(
        lambda x: (x.values[-1]-x.values[0])/x.values[0])
    # rp = prdf['cash'].pct_change().dropna()
    # print(rp*100)
    # rp = np.log(prdf['cash']).diff().dropna()
    # XXX: Expected market return
    rm = mrdf.groupby(mrdf.Date.dt.year)['Price'].apply(
        lambda x: (x.values[-1]-x.values[0])/x.values[0])
    # rm = mrdf['Price'].pct_change().dropna()
    # rm = np.log(mrdf['Price']).diff().dropna()
    # print(rp*100)

    trp = pd.DataFrame({'Years': rp.index, '% Return': rp.values*100})
    trp.plot.bar(x='Years', y='% Return')
    plt.savefig('./trades/%s_%s_%s.pdf' % (model, otype, lags),
                bbox_inches='tight')
    plt.close()

    gn = (np.log((1+rm).mean()) - np.log(1+rf))
    gd = np.log((1+rm)).replace([np.inf, -np.inf], np.nan).dropna().var()
    # print('gn: ', gn)
    # print('gd: ', gd)
    gamma = gn/gd
    # print('gamma: ', gamma)

    drprm = pd.DataFrame({'rm': rm, 'rp': rp})
    drprm['rr'] = -(1+rm)**(-gamma)
    drprm = drprm.replace([np.inf, -np.inf], np.nan).dropna()
    # print(drprm.describe())

    bn = np.cov(drprm['rp'], drprm['rr'])[0, 1]
    bd = np.cov(drprm['rm'], drprm['rr'])[0, 1]

    beta = bn/bd
    # print('beta: ', beta)

    if prdf['cash'].values[-1] >= 0:
        cagr_rp = (prdf['cash'].values[-1]/prdf['cash'].values[0])**(1/Y) - 1
    else:
        cagr_rp = -np.inf       # infinitely worse!
    cagr_mp = (mrdf['Price'].values[-1]/mrdf['Price'].values[0])**(1/Y) - 1
    # print('CAGR Portfolio (%), CAGR Market (%): ', cagr_rp*100, cagr_mp*100)

    if cagr_rp >= 0:
        alpha = cagr_rp - beta*(cagr_mp - rf).mean() - rf
    else:
        alpha = -np.inf

    # XXX: Get the 10 year return (risk free rate)
    dgs = pd.read_csv('DGS10.csv')
    for i, r in enumerate(dgs['DGS10']):
        if r == '.':
            dgs['DGS10'][i] = dgs['DGS10'][i-1]
        dgs['DGS10'][i] = float(dgs['DGS10'][i])/(100*365)

    # XXX: Now get the rate for the same dates as the prdf
    prdf['rf'] = [np.nan]*prdf.shape[0]
    for i, d in enumerate(prdf['dates']):
        prdf['rf'][i] = dgs[dgs['DATE'].isin([d])]['DGS10'].values[0]
    # print(dgs.shape, prdf.shape)
    # print(rfrs)
    # print(prdf)

    # XXX: Statsitic data for trades
    K = 252                     # number of trading days/year (approx)
    # XXX: Append to lists
    dprdf = prdf['cash'].pct_change().dropna()
    # dmrdf = mrdf['Price'].pct_change().dropna()
    alphas.append(alpha*100)
    betas.append(beta)
    ns.append(dprdf[dprdf != 0].shape[0])
    cagrs.append(cagr_rp*100)
    maxs.append(dprdf.max()*100)
    mins.append(dprdf.min()*100)
    medians.append(dprdf.median()*100)
    sds.append(dprdf.std()*100)
    # XXX: This is the quant sharpe at the end of the whole thing
    srs.append(((dprdf-prdf['rf'][1:]).mean() /
                (dprdf - prdf['rf'][1:]).std())*np.sqrt(K))
    wins.append(dprdf[dprdf > 0].shape[0]/ns[-1]*100)
    avgs.append(dprdf.mean()*100)

    # XXX: Now the rolling sharpe ratio
    rsr = (dprdf - prdf['rf'][1:]).rolling(K).apply(
        lambda x: x.mean()/x.std()*np.sqrt(K))
    plt.plot(range(len(rsr)), rsr)
    plt.savefig('./trades/%s_%s_%s_rsr.pdf' % (model, otype, lags),
                bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    plt.style.use('seaborn-v0_8-whitegrid')

    def setup_trade(model):
        for otype in ['call', 'put']:
            # XXX: Change or add to the loops as needed
            for dd in ['figs']:
                for ts in [5]:
                    name = ('./final_results/%s_%s_ts_%s_model_%s.npy.gz' %
                            (otype, dd, ts, model))
                    print('Doing model: ', name)
                    dates, y, yp = getpreds_trading(name, otype)
                    trade(dates, y, yp, otype, model, lags=ts)

    # XXX: The trading part, only for the best model
    models = ['ridge', 'ssviridge', 'plsridge', 'ctridge',
              'tskridge', 'tskplsridge', 'tsknsridge',
              'mskridge', 'msknsridge', 'mskplsridge',
              'pmridge', 'pmplsridge']
    # from joblib import Parallel, delayed
    # Parallel(n_jobs=-1)(delayed(setup_trade)(model)
    #                     for model in models)

    for otype in ['call', 'put']:
        alphas = list()
        betas = list()
        cagrs = list()
        wins = list()
        maxs = list()
        mins = list()
        avgs = list()
        medians = list()
        sds = list()
        srs = list()
        ns = list()
        # XXX: Change or add to the loops as needed
        for dd in ['figs']:
            for ts in [5]:
                for model in models:
                    analyse_trades(otype, model, ts,
                                   alphas, betas,
                                   cagrs, wins,
                                   maxs, mins,
                                   avgs, medians, sds, srs, ns)
        df = pd.DataFrame({'alpha': alphas, 'beta': betas,
                           'cagr (%)': cagrs, 'win (%)': wins,
                           'max (%)': maxs, 'min (%)': mins,
                           'mean (%)': avgs, 'median (%)': medians,
                           'std (%)': sds, 'sharpe ratio': srs,
                           'N': ns},
                          index=models)
        df.to_csv('./trades/%s_trades.csv' % otype)
