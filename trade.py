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


def trade(dates, y, yp, otype, strat, eps=0.01, lags=5):

    def getTC(data, P=0.25):
        # XXX: See: Options Trading Costs Are Lower than You
        # Think Dmitriy Muravyev (page-4)
        # return 0/100
        return 1.5/100

    def c_position_s(sign, CP, PP, TC, N):
        """This is trading a straddle
        """
        tc = (CP * TC) + (PP * TC)
        # print('transaction cost: ', tc)
        if sign > 0:
            # print('Close selling straddle')
            return ((CP + PP) - tc)*N
        else:
            # print('Close buying straddle')
            return ((-(CP + PP)) - tc)*N

    def o_position_s(sign, CP, PP, TC, N):
        tc = CP * TC + PP * TC
        # print('transaction cost: ', tc)
        if sign > 0:
            # print('Open buying straddle')
            return ((-(CP + PP)) - tc)*N
        else:
            # print('Open selling straddle')
            return ((CP + PP) - tc)*N

    N = 10                     # number of contracts to buy/sell every time
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

    cash = 100000               # starting cash position 100K
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

    # XXX: Harvest profit
    HARVEST = 0.05
    MAX_DAYS_GONE = 400          # max days holding allowed

    # XXX: Actual trade
    MM = 1
    for t in range(0, dates.shape[0]-MM):
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
        # i, j = np.unravel_index(np.argmax(ddhat, axis=None), ddhat.shape)

        # XXX: Just the closest date only
        j = 0
        i = np.argmax(ddhat[:, j])
        # print('Max change index: ', i, j)

        # XXX: Another technique to use min instead of max
        # i, j = np.unravel_index(np.argmin(ddhat, axis=None), ddhat.shape)

        # XXX: Only if the change is greater/lesser than some filter (eps) --
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

            # XXX: Selling straddle -- dangerous
            if (np.abs((CPp + PPp) - (CP + PP))/(CP+PP) < 0.01 and
               (t % MM == 0 and dhat[i, j] < 0)):
                open_p = True       # open a position later
            # XXX: Buying straddle -- limited risk
            elif (np.abs((CPp + PPp) - (CP + PP))/(CP+PP) > 0 and
                  (t % MM == 0 and dhat[i, j] > 0)):
                open_p = True       # open a position later

        if open_position and t % MM == 0:   # is position already open?
            # XXX: Get the days that have passed by
            trd = pd.to_datetime(str(pos_date[-1]), format='%Y%m%d')
            today = pd.to_datetime(str(dates[t]), format='%Y%m%d')
            # print('Days to maturity: ', tp[-1]*365)
            # print('Today: ', today, 'Trad day: ', trd)
            # print('days gone: ', (today - trd).days)
            DTM = int(tp[-1]*365 - (today-trd).days)
            # print(mms[i], mp[-1], tp[-1]*365, tts[j]*365,
            #       tts[j]-((today-trd).days/365))

            # XXX: Maturity has reached. Maturity might be a weekend,
            # because of abstract tau.
            if DTM <= 0:
                # print('Maturity today: ', int(DTM))
                # xxx: Close the position in a different way
                DTM_UP = data[dates[t-int(DTM)]]['UnderlyingPrice'].values[0]
                K = Kp[-1]
                TC = getTC(tdata, 1)
                # print('Underlying price on Maturity: ', DTM_UP)
                # print('Straddle Strike price: ', K)
                if signl[-1] > 0:
                    # FIXME: Need to correctly add the transaction costs
                    # here on maturity dates.
                    tc = max(DTM_UP-K, 0)*TC + max(K-DTM_UP, 0)*TC
                    res = ((max(DTM_UP-K, 0) + max(K-DTM_UP, 0)) - tc)*N
                    # print('transaction cost: ', tc)
                else:
                    tc = max(DTM_UP-K, 0)*TC + max(K-DTM_UP, 0)*TC
                    res = (-(max(DTM_UP-K, 0) + max(K-DTM_UP, 0)) - tc)*N
                    # print('transaction cost: ', tc)
                cash += res
                if not open_p:
                    cashl.append(cash)
                    trade_date.append(dates[t])
                open_position = False
                # print('Position closed on Maturity, cash: ', cash)

            else:
                # XXX: Calculate the change for this day
                UPo = Kp[-1]
                # UPo = mp[-1]*UP

                days_gone = (today - trd).days//pred.TSTEP
                days_left = (today - trd).days/365  # days to subtract

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
                CLOSE_POSITION = c_position_s(signl[-1], CPo, UPo, TC, N)
                # CCP = cashl[-1] + CLOSE_POSITION

                # print('cashl[-1]: ', cashl[-1])
                # print('CLOSE_POSITION: ', CLOSE_POSITION)
                # print('Close CP: ', CCP)
                # XXX: We have to redo all training with 1 day tts.
                # elif (mms[i] == mp[-1] and tp[-1] == tts[j]):
                if CLOSE_POSITION >= 0:
                    # print('CP: ', CCP, 'cashl[-1]', cashl[-1],
                    #       '% change: ', CLOSE_POSITION/cashl[-1])
                    if (CLOSE_POSITION/cashl[-1] >= HARVEST or
                       (t + 1 >= dates.shape[0]-MM) or
                       (today-trd).days >= MAX_DAYS_GONE):
                        # print('closing and harvesting the profit')
                        cash += CLOSE_POSITION
                        # print('cash: ', cash)
                        # cash += c_position(signl[-1], CPo, UPo, dl[-1], TC)
                        # XXX: Only append if we are not going to open a new
                        # position immediately
                        if not open_p:
                            cashl.append(cash)
                            trade_date.append(dates[t])
                        open_position = False
                    else:
                        # elif i == ip[-1] and j == jp[-1]:
                        # print('holding!')
                        open_p = False  # just hold

                elif signl[-1] < 0:
                    # print('Loss but continuing')
                    open_p = False
                elif signl[-1] > 0:
                    # print('closing the position on loss')
                    cash += CLOSE_POSITION
                    # print('cash: ', cash)
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
        if open_p and (t + 1 < dates.shape[0]-MM):
            TC = getTC(tdata, 1)
            # print('Opening CP:%s, PP:%s' % (CP, PP))
            # print('Opening Next CP:%s, Next PP:%s' % (CPp, PPp))
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
                # print('cash when opening position: ', cashl[-1])
            open_p = False      # The position is now opened
        else:
            # FIXME: This cash position should be mark to market
            cashl.append(cash)
            trade_date.append(dates[t])
            open_p = False      # Kinda done!

    # XXX: Close any open positions

    import os
    if not os.path.exists('./trades/marketPrice.csv'):
        tempDates = list(map(lambda x: str(x), mDates))
        mdates = pd.to_datetime(tempDates, format='%Y%m%d')
        df = pd.DataFrame({'dates': mdates, 'Price': marketPrice})
        df.to_csv('./trades/marketPrice.csv')

    temptradedate = list(map(lambda x: str(x), trade_date))
    trade_date = pd.to_datetime(temptradedate, format='%Y%m%d')
    res = pd.DataFrame({'cash': cashl, 'dates': trade_date})
    res = res.dropna()
    res.to_csv('./trades/%s_%s_%s.csv' % (strat, otype, lags))


def analyse_trades(otype, model, lags, alpha, betas,
                   cagrs, wins, maxs, mins, avgs, medians,
                   sds, srs, ns, model_label, ax1, ax2, marker,
                   rf=3.83/(100), Y=9):

    import warnings
    warnings.filterwarnings("ignore")
    # print('Model %s_%s_%s: ' % (model, otype, lags))
    mrdf = pd.read_csv('./trades/marketPrice.csv')
    prdf = pd.read_csv('./trades/%s_%s_%s.csv' % (model, otype, lags))

    prdf['pcash'] = prdf['cash'].expanding().apply(
        lambda x:
        (x.values[-1]-x.values[0])/x.values[0]*100)

    # XXX: For the paper
    ax1.plot(prdf['dates'].values, prdf['pcash'].values,
             label='%s' % model_label,
             marker=marker, markevery=0.5)
    # XXX: Plot the portfolio
    f, ax = plt.subplots(nrows=1, ncols=1)
    prdf.plot.line(x='dates', y='pcash', label='%s' % (model_label),
                   ax=ax)
    ax.set_xlabel('Years')
    ax.set_ylabel('% P/L')
    plt.xticks(rotation=45)
    ax.legend()
    plt.savefig('./trades/%s_%s_%s_portfolio.pdf' % (model, otype, lags),
                bbox_inches='tight')
    plt.close(f)

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

    f, ax = plt.subplots(nrows=1, ncols=1)
    trp = pd.DataFrame({'Years': rp.index, '% Return': rp.values*100})
    trp.plot.bar(x='Years', y='% Return', ax=ax)
    plt.savefig('./trades/%s_%s_%s.pdf' % (model, otype, lags),
                bbox_inches='tight')
    plt.close(f)

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
    # srs.append(((dprdf-prdf['rf'][1:]).mean() /
    #             (dprdf - prdf['rf'][1:]).std())*np.sqrt(K))
    if ns[-1] != 0:
        wins.append(dprdf[dprdf > 0].shape[0]/ns[-1]*100)
    else:
        wins.append(0)
    avgs.append(dprdf.mean()*100)

    # XXX: Now the rolling sharpe ratio
    rsr = (dprdf - prdf['rf'][1:]).rolling(K).apply(
        lambda x: x.mean()/x.std()*np.sqrt(K))
    dfrsrs = pd.DataFrame({'Date': prdf['Date'][1:], 'sr': rsr})
    # XXX: This is the quant sharpe at the end of the whole thing
    srs.append(rsr.mean())      # expected sharpe overall
    # XXX: For the paper
    ax2.plot(dfrsrs['Date'], dfrsrs['sr'], label=model_label,
             marker=marker, markevery=0.5)
    # XXX: Just plotting
    f, ax = plt.subplots(nrows=1, ncols=1)
    dfrsrs.plot.line(x='Date', y='sr', label='%s' % (model_label), ax=ax)
    ax.set_ylabel('Sharpe Ratio')
    ax.legend()
    plt.savefig('./trades/%s_%s_%s_rsr.pdf' % (model, otype, lags),
                bbox_inches='tight')
    plt.close(f)


if __name__ == '__main__':
    plt.style.use('seaborn-v0_8-whitegrid')

    def setup_trade(model):
        for otype in ['call', 'put']:
            # XXX: Change or add to the loops as needed
            for dd in ['figs']:
                for ts in [5]:
                    if not (model in ['har', 'mskhar']):
                        name = ('./final_results/%s_%s_ts_%s_model_%s.npy.gz' %
                                (otype, dd, ts, model))
                    else:
                        name = ('./final_results/%s_%s_ts_21_model_%s.npy.gz' %
                                (otype, dd, model))
                    print('Doing model: ', name)
                    dates, y, yp = getpreds_trading(name, otype)
                    trade(dates, y, yp, otype, model, lags=ts)

    # XXX: The trading part, only for the best model
    # models = ['har', 'mskhar', 'ridge', 'ssviridge',
    #           'plsridge', 'ctridge',
    #           'tskridge', 'tskplsridge', 'tsknsridge',
    #           'mskridge', 'msknsridge', 'mskplsridge',
    #           'pmridge', 'pmplsridge']
    models = ['pmridge', 'mskhar', 'tskridge', 'har']
    model_labels = models
    # model_labels = ['Point-SAM', 'MS-HAR', 'TS-SAM', 'Surface-HAR']
    from joblib import Parallel, delayed
    Parallel(n_jobs=-1)(delayed(setup_trade)(model)
                        for model in models)

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
        markers = ['*', 'o', '^', 'x', '.', '1', '2', '3', '4', '8', 's',
                   'X', 'd', 'D']
        f1, ax1 = plt.subplots(nrows=1, ncols=1)
        f2, ax2 = plt.subplots(nrows=1, ncols=1)
        for dd in ['figs']:
            for ts in [5]:
                count = 0
                for model, model_label in zip(models, model_labels):
                    analyse_trades(otype, model, ts,
                                   alphas, betas,
                                   cagrs, wins,
                                   maxs, mins,
                                   avgs, medians, sds, srs, ns,
                                   model_label, ax1, ax2, markers[count])
                    count += 1
        ax1.legend()
        ax1.set_xlabel('Years')
        ax1.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax1.set_ylabel('% P/L')
        ax2.legend()
        ax2.set_xlabel('Years')
        ax2.set_ylabel('Sharpe Ratio')
        f1.savefig('./trades/portfolio_%s.pdf' %
                   otype, bbox_inches='tight')
        f2.savefig('./trades/srs_%s.pdf' %
                   otype, bbox_inches='tight')
        # f1.savefig('../feature_paper/elsarticle/figs/portfolio_%s.pdf' %
        #            otype, bbox_inches='tight')
        # f2.savefig('../feature_paper/elsarticle/figs/srs_%s.pdf' %
        #            otype, bbox_inches='tight')
        df = pd.DataFrame({
            '# Trades': ns,
            'Wins (%)': wins,
            'Alpha (%)': alphas,
            'Beta': betas,
            'CAGR (%)': cagrs,
            'Avg. SR': srs,
            # 'max (%)': maxs, 'min (%)': mins,
            # 'mean (%)': avgs, 'median (%)': medians,
            # 'std (%)': sds,
        }, index=model_labels)
        df.to_csv('./trades/%s_trades.csv' % otype)
        # df.to_latex('../feature_paper/elsarticle/figs/traderes_%s.tex' %
        #             otype)
