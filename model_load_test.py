#!/usr/bin/env python

import shutil
import keras
import pred
import numpy as np
import matplotlib.pyplot as plt
import glob
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score
import dmtest
# from mpl_toolkits.mplot3d import Axes3D
# import gzip
import pandas as pd
import blosc2
from pred import cr2_score, cr2_score_pval
from pred import MPls
from pred import NS
from pred import CT
from pred import SSVI
from feature_extraction import VAE, Sampling, VaeRegression, PcaRegression, HarRegression, Encoder, Decoder
from scipy import stats
# from blackscholes import BlackScholesCall, BlackScholesPut

# For plotting style
import scienceplots


def date_to_num(otype, date, dd='./figs'):
    ff = sorted(glob.glob('./'+otype+'_'+dd.split('/')[1]+'/*.npy'))
    count = 0
    for i in ff:
        if i.split('/')[-1].split('.')[0] == date:
            break
        count += 1
    return count


def num_to_date(otype, num, dd='./figs'):
    ff = sorted(glob.glob('./'+otype+'_'+dd.split('/')[1]+'/*.npy'))
    return ff[num].split('/')[-1].split('.')[0]


def load_image(num, dd='./figs'):
    ff = sorted(glob.glob(dd+'/*.npy'))
    img = np.load(ff[num])
    return img


def extract_features(X, model, dd, TSTEPS, feature_res, m=0, t=0, type='mskew',
                     otype='call'):
    """
    Returns the features extracted from the input data ready for prediction
    using corrisponding model
    """

    import feature_extraction as fe
    # transform_type = type[1:]
    model = model[-3:]
    if (model == 'pca'):
        if type == 'point':
            n_components = (TSTEPS//2)
        else:
            num_points = X.shape[1]//TSTEPS
            n_components = (TSTEPS//2) * num_points

        X = fe.pca_transform(X, n_components)
    elif (model == 'autoencoder' or model == 'vae'):

        # Load the encoder model
        if type == 'point':
            if dd == './figs':
                toopen = './%s_feature_models/%s_ts_%s_%s_%s_%s_encoder.keras' % (type, model, TSTEPS, m, t, otype)
            else:
                toopen = './%s_feature_models/%s_ts_%s_%s_%s_%s_encoder_gfigs.keras' % (type, model, TSTEPS, m, t, otype)
        elif type == 'surf':
            if dd == './figs':
                toopen = './%s_feature_models/%s_ts_%s_%s_encoder.keras' % (type, model, TSTEPS, otype)
            else:
                toopen = './%s_feature_models/%s_ts_%s_%s_encoder_gfigs.keras' % (type, model, TSTEPS, otype)
        else:
            if dd == './figs':
                toopen = './%s_feature_models/%s_ts_%s_%s_%s_encoder.keras' % (type, model, TSTEPS, m, otype)
            else:
                toopen = './%s_feature_models/%s_ts_%s_%s_%s_encoder_gfigs.keras' % (type, model, TSTEPS, m, otype)

        vae = True
        # Get encoder
        if model == 'autoencoder':
            encoder = keras.saving.load_model(toopen)
            vae = False
            encoder = keras.Model(inputs=encoder.input,
                                  outputs=encoder.layers[1].output)
        else:
            encoder = keras.saving.load_model(
                toopen,
                custom_objects={'VAE': fe.VAE, 'Sampling': fe.Sampling})

        X = fe.autoencoder_transform(encoder, X, vae=vae)
        del encoder
    else:
        if type[1:] == 'skew' or type == 'surf':
            # Reshape to 3D
            X = X.reshape(X.shape[0], TSTEPS, X.shape[1]//TSTEPS)
        X = fe.har_transform(X, TSTEPS=TSTEPS, type=type)
    return X


def plotme(valY, Ydates, out):
    print(valY.shape, out.shape)
    # XXX: The moneyness
    MS = np.arange(pred.LM, pred.UM+pred.MSTEP, pred.MSTEP)
    # XXX: The term structure
    TS = np.array([i/pred.DAYS
                   for i in
                   range(pred.LT, pred.UT+pred.TSTEP, pred.TSTEP)])

    for y, yd, yp in zip(valY, Ydates, out):
        y *= 100
        yp *= 100
        ynum = date_to_num(yd)
        pyd = num_to_date(ynum-1)
        if (int(yd) - int(pyd)) == 1:
            # XXX: Then we can get this figure
            fig, axs = plt.subplots(1, 3,
                                    subplot_kw=dict(projection='3d'))
            axs[0].title.set_text('Truth: ' + yd)
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
            # axs[0, 0].imshow(np.transpose(y), cmap='hot')
            axs[1].title.set_text('Predicted: ' + yd)
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
            # axs[0, 1].imshow(np.transpose(yp), cmap='hot')

            # XXX: Previous day' volatility
            ximg = load_image(ynum-1)
            ximg *= 100
            xdf = list()
            for cm, m in enumerate(MS):
                for ct, t in enumerate(TS):
                    xdf.append([m, t, ximg[cm, ct]])
            xdf = np.array(xdf)
            axs[2].plot_trisurf(xdf[:, 0], xdf[:, 1], xdf[:, 2],
                                cmap='afmhot', linewidth=0.2,
                                antialiased=True)
            axs[2].set_xlabel('Moneyness')
            axs[2].set_ylabel('Term structure')
            axs[2].set_zlabel('Vol %')
            axs[2].title.set_text(pyd)
            # dimgt = y - ximg
            # axs[1, 0].imshow(np.transpose(dimgt), cmap='hot')
            # axs[1, 0].title.set_text('True diff : %s_%s' % (yd, pyd))
            # dimg = yp - ximg
            # axs[1, 1].imshow(np.transpose(dimg), cmap='hot')
            # axs[1, 1].title.set_text('diff dates :%s-%s' % (yd, pyd))
            plt.show()
            plt.close(fig)


def plot_hmap_features(vv, X, Y, name):
    # print(name)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs, ys = np.meshgrid(Y, X)
    for i in range(vv.shape[0]):
        im = ax.plot_surface(xs, ys, vv[i],
                             antialiased=False, linewidth=0,
                             cmap='viridis')
    ax.view_init(elev=30, azim=-125)
    ax.set_zticks([])
    ax.set_xlabel('Term structure')
    ax.set_ylabel('Moneyness')
    plt.colorbar(im)
    plt.savefig(name, bbox_inches='tight')
    plt.close(fig)
    # plt.show()


def main(otype, dd='./figs', model='Ridge', plot=True, TSTEPS=5, NIMAGES=1000,
         get_features=False, feature_res=10):
    print('Doing model: ', dd, TSTEPS, model)
    # XXX: Num 3000 == '20140109'
    START_DATE = '20140109'
    END_DATE = '20221230'

    START = date_to_num(otype, START_DATE, dd=dd)
    END = date_to_num(otype, END_DATE, dd=dd) - TSTEPS
    # print(num_to_date(END))
    # assert (False)
    # print("START:", START)
    NIMAGES = END-START
    # print(NIMAGES)

    # XXX: Load the test dataset
    pred.TSTEPS = TSTEPS
    if pred.TSTEPS == 5:
        bs = 50
    elif pred.TSTEPS == 10:
        bs = 40
    elif pred.TSTEPS == 20:
        bs = 20
    else:
        raise Exception("TSTEPS not correct")

    nf = 64

    valX, valY, Ydates = pred.load_data_for_keras(otype, START=START, dd=dd,
                                                  NUM_IMAGES=NIMAGES,
                                                  TSTEP=pred.TSTEPS)
    # print(valX.shape, valY.shape)
    valX = valX.reshape(valX.shape[0]//pred.TSTEPS, pred.TSTEPS,
                        *valX.shape[1:])
    # print(valX.shape, valY.shape)

    # XXX: Reshape the data for testing
    MS = valY.shape[1]
    TS = valY.shape[2]
    # XXX: Moneyness
    MONEYNESS = [0, 43, 1, 44, 1, 55]
    # XXX: Some terms
    TERM = [29, 10, 49, 9, 71, 10]

    # XXX: Clean the data
    if dd == './gfigs':
        valX, valY = pred.clean_data(valX, valY)

    # XXX: Make a prediction for images
    if model == 'keras':
        raise Exception("Keras support removed")
        # XXX: Load the model
        if dd == './figs':
            toopen = './modelcr_bs_%s_ts_%s_filters_%s.keras' % (bs,
                                                                 TSTEPS,
                                                                 nf)
        else:
            toopen = './modelcr_bs_%s_ts_%s_filters_%s_gfigs.keras' % (bs,
                                                                       TSTEPS,
                                                                       nf)
        print('Doing model: ', toopen)
        m1 = keras.saving.load_model(toopen)
        print(m1.summary())
        # out = m1(valX, training=False).numpy()
        out = m1.predict(valX)
        if not plot:
            # XXX: Reshape data for measurements
            out = out.reshape(out.shape[0], out.shape[1]*out.shape[2])
            valY = valY.reshape(valY.shape[0], valY.shape[1]*valY.shape[2])

    elif (model == 'pmridge' or model == 'pmlasso' or model == 'pmenet' or
          model == 'pmplsridge' or model == 'pmplslasso' or (
              model == 'pmplsenet')):
        # XXX: The output vector
        out = np.array([0.0]*(valY.shape[0]*valY.shape[1]*valY.shape[2]))
        out = out.reshape(*valY.shape)
        # XXX: Now go through the MS and TS
        mms = np.arange(pred.LM, pred.UM+pred.MSTEP, pred.MSTEP)
        tts = [i/pred.DAYS
               for i in range(pred.LT, pred.UT+pred.TSTEP, pred.TSTEP)]
        import pickle
        for i, s in enumerate(mms):
            for j, t in enumerate(tts):
                if dd == './figs':
                    with open('./point_models/%s_ts_%s_%s_%s_%s.pkl' %
                              (model, TSTEPS, s, t, otype), 'rb') as f:
                        m1 = pickle.load(f)
                else:
                    with open('./point_models/%s_ts_%s_%s_%s_%s_gfigs.pkl' %
                              (model, TSTEPS, s, t, otype), 'rb') as f:
                        m1 = pickle.load(f)
                # XXX: Now make the prediction
                k = np.array([s, t]*valX.shape[0]).reshape(valX.shape[0], 2)
                val_vec = np.append(valX[:, :, i, j], k, axis=1)
                if model == 'pmplsridge' or model == 'pmplslasso' or (
                        model == 'pmplsenet'):
                    vvo = m1.predict(val_vec)
                    out[:, i, j] = vvo.reshape(vvo.shape[0])
                else:
                    out[:, i, j] = m1.predict(val_vec)

                # XXX: Feature vector plot
                if get_features and model == 'pmridge':
                    if i in MONEYNESS and j in TERM:
                        fig, ax = plt.subplots()
                        xaxis = ['t-%s' % (i+1) for i in (range(TSTEPS))[::-1]]
                        xaxis.append(r'$m$')
                        xaxis.append(r'$\tau$')
                        ax.bar(xaxis, m1.coef_, color='b')
                        ax.set_ylabel('Coefficient magnitudes')
                        plt.xticks(fontsize=9, rotation=45)
                        dfname = dd.split('/')[1]
                        fname = './plots/%s_m_%s_t_%s_lags_%s_%s_%s.pdf' % (
                            model, s, t, TSTEPS, otype, dfname)
                        plt.savefig(fname, bbox_inches='tight')
                        plt.close(fig)
        if not plot:
            out = out.reshape(out.shape[0], out.shape[1]*out.shape[2])
            valY = valY.reshape(valY.shape[0], valY.shape[1]*valY.shape[2])

    elif (model == 'tskenet' or model == 'tskridge' or model == 'tsklasso' or
          model == 'tskplsridge' or model == 'tskplslasso' or
          model == 'tskplsenet' or model == 'tsknsridge' or
          model == 'tsknslasso' or model == 'tsknsenet'):
        # XXX: The output vector
        out = np.array([0.0]*(valY.shape[0]*valY.shape[1]*valY.shape[2]))
        out = out.reshape(*valY.shape)
        mms = np.arange(pred.LM, pred.UM+pred.MSTEP, pred.MSTEP)
        # XXX: Go through the MS
        import pickle
        for j, m in enumerate(mms):
            if dd == './figs':
                with open('./tskew_models/%s_ts_%s_%s_%s.pkl' %
                          (model, TSTEPS, m, otype), 'rb') as f:
                    m1 = pickle.load(f)
            else:
                with open('./tskew_models/%s_ts_%s_%s_%s_gfigs.pkl' %
                          (model, TSTEPS, m, otype), 'rb') as f:
                    m1 = pickle.load(f)
            tskew = valX[:, :, j]
            tskew = tskew.reshape(tskew.shape[0],
                                  tskew.shape[1]*tskew.shape[2])
            # XXX: Add m to the sample set
            ms = np.array([m]*tskew.shape[0]).reshape(tskew.shape[0], 1)
            tskew = np.append(tskew, ms, axis=1)
            # XXX: Predict the output
            out[:, j] = m1.predict(tskew)

            # XXX: Features plot
            if get_features and model == 'tskridge':
                if j in MONEYNESS:
                    X = [i/pred.DAYS
                         for i in range(pred.LT, pred.UT+pred.TSTEP,
                                        pred.TSTEP)]
                    labels = ['t-%s' % (i+1) for i in range(TSTEPS)[::-1]]
                    import itertools
                    markers = itertools.cycle(('o', '+', '*', 'x', 'p'))
                    # markers = [(3+i, 1, 0) for i in range(TSTEPS)]
                    for mts in TERM:
                        # XXX: The term structure weights
                        ws = m1.coef_[mts][:-1].reshape(TSTEPS, TS)
                        # XXX: The moneyness weight
                        wms = m1.coef_[mts][-1]
                        # XXX: Make sure that the moneyness weight is nothing
                        assert (abs(wms) < 1e-4)
                        # XXX: Now just plot the 2d curves
                        fig, ax = plt.subplots()
                        for i in range(TSTEPS):
                            ax.plot(X, ws[i], marker=next(markers),
                                    label=labels[i], markevery=0.1)
                        ax.set_ylabel('Coefficient magnitudes')
                        ax.set_xlabel('Term structure')
                        ax.legend(ncol=3)
                        dfname = dd.split('/')[1]
                        fname = './plots/%s_m_%s_t_%s_lags_%s_%s_%s.pdf' % (
                            model, m, X[mts], TSTEPS, otype, dfname)
                        plt.savefig(fname, bbox_inches='tight')
                        plt.close(fig)

        if not plot:
            out = out.reshape(out.shape[0], out.shape[1]*out.shape[2])
            valY = valY.reshape(valY.shape[0], valY.shape[1]*valY.shape[2])
    elif (model == 'mskpca' or model == 'mskhar'
          or model == 'mskvae' or model == 'mskenethar' or
          model == 'msklassohar'
          or model == 'mskenetpca' or model == 'msklassopca' or
          model == 'mskenetvae' or model == 'msklassovae'):
        # XXX: The output vector
        out = np.array([0.0]*(valY.shape[0]*valY.shape[1]*valY.shape[2]))
        out = out.reshape(*valY.shape)
        mms = np.arange(pred.LM, pred.UM+pred.MSTEP, pred.MSTEP)
        # XXX: Now go through the TS
        tts = [i/pred.DAYS for i in range(pred.LT, pred.UT+pred.TSTEP,
                                          pred.TSTEP)]
        import pickle
        model_name = model[3:]

        # XXX: Special case for har due to switching to 20 tstep from 21
        load_tsteps = TSTEPS
        if model_name[-3:] == 'har':
            load_tsteps = 21

        for j, t in enumerate(tts):
            if dd == './figs':
                with open('./mskew_feature_models/%s_ts_%s_%s_%s.pkl' %
                          (model_name, load_tsteps, t, otype), 'rb') as f:
                    m1 = pickle.load(f)
            else:
                with open('./mskew_feature_models/%s_ts_%s_%s_%s_gfigs.pkl' %
                          (model_name, load_tsteps, t, otype), 'rb') as f:
                    m1 = pickle.load(f)

            vYY = valY[:, :, j]
            mskew = valX[:, :, :, j]
            mskew = mskew.reshape(mskew.shape[0],
                                  mskew.shape[1]*mskew.shape[2])

            # # Extract features before prediction
            # mskew = extract_features(mskew, model_name, dd, TSTEPS,
            #                          feature_res, m=t, type='mskew',
            #                          otype=otype)

            # XXX: Add t to the sample set
            ts = np.array([t]*mskew.shape[0]).reshape(mskew.shape[0], 1)
            mskew = np.append(mskew, ts, axis=1)

            # out[:, :, j] = m1.predict(mskew, vYY, TSTEPS, 'skew')

            if get_features and model != 'mskvae':
                if j in TERM:
                    X = np.arange(pred.LM, pred.UM+pred.MSTEP, pred.MSTEP)
                    if model == 'mskhar':
                        labels = ['lag-20', 'lag-5', 'lag-1']
                    else:
                        labels = ['t-%s' % (i+1) for i in range(feature_res)[::-1]]

                    import itertools
                    markers = itertools.cycle(('o', '+', '*')) 
                    # markers = [(3+i, 1, 0) for i in range(feature_res)]
                    for mts in MONEYNESS:
                        # XXX: The term structure weights
                        ws = m1.coef_[mts][:-1].reshape(feature_res, MS)
                        # XXX: The term structure weight
                        wms = m1.coef_[mts][-1]
                        # XXX: Make sure that the term structure weight=nil
                        assert (abs(wms) < 1e-4)
                        # XXX: Now just plot the 2d curves
                        fig, ax = plt.subplots()
                        for i in range(feature_res):
                            ax.plot(X, ws[i], marker=next(markers),
                                    label=labels[i], markevery=0.1)
                        ax.set_ylabel('Coefficient magnitudes')
                        ax.set_xlabel('Moneyness')
                        ax.legend(ncol=3)
                        dfname = dd.split('/')[1]
                        fname = './plots/%s_m_%s_t_%s_lags_%s_%s_%s.pdf' % (
                            model, X[mts], t, TSTEPS, otype, dfname)
                        plt.savefig(fname, bbox_inches='tight')
                        plt.close(fig)
        if not plot:
            out = out.reshape(out.shape[0], out.shape[1]*out.shape[2])
            valY = valY.reshape(valY.shape[0], valY.shape[1]*valY.shape[2])

    elif (model == 'tskpca' or
          model == 'tskhar' or
          model == 'tskvae'
          or model == 'tskenethar' or
          model == 'tsklassohar' or model == 'tskenetpca' or
          model == 'tsklassopca'
          or model == 'tskenetvae' or model == 'tsklassovae'):
        # XXX: The output vector
        out = np.array([0.0]*(valY.shape[0]*valY.shape[1]*valY.shape[2]))
        out = out.reshape(*valY.shape)
        mms = np.arange(pred.LM, pred.UM+pred.MSTEP, pred.MSTEP)
        # XXX: Go through the MS
        import pickle
        model_name = model[3:]

        # XXX: Special case for har due to switching to 20 tstep from 21
        load_tsteps = TSTEPS
        if model_name[-3:] == 'har':
            load_tsteps = 21

        for j, m in enumerate(mms):
            if dd == './figs':
                with open('./tskew_feature_models/%s_ts_%s_%s_%s.pkl' %
                          (model_name, load_tsteps, m, otype), 'rb') as f:
                    m1 = pickle.load(f)
            else:
                with open('./tskew_feature_models/%s_ts_%s_%s_%s_gfigs.pkl' %
                          (model_name, load_tsteps, m, otype), 'rb') as f:
                    m1 = pickle.load(f)

            vYY = valY[:, j]
            tskew = valX[:, :, j]
            tskew = tskew.reshape(tskew.shape[0],
                                  tskew.shape[1]*tskew.shape[2])

            # # Extract features before prediction
            # tskew = extract_features(tskew, model_name, dd, TSTEPS,
            #                          feature_res,
            #                          m=m, type='tskew', otype=otype)

            # XXX: Add m to the sample set
            ms = np.array([m]*tskew.shape[0]).reshape(tskew.shape[0], 1)
            tskew = np.append(tskew, ms, axis=1)

            # XXX: Predict the output
            out[:, j] = m1.predict(tskew, vYY, TSTEPS, 'skew')

            # XXX: Features plot
            if get_features and model != 'tskvae':
                if j in MONEYNESS:
                    X = [i/pred.DAYS
                         for i in range(pred.LT, pred.UT+pred.TSTEP,
                                        pred.TSTEP)]
                    if model == 'tskhar':
                        labels = ['lag-1', 'lag-5', 'lag-20']
                    else:
                        labels = ['t-%s' % (i+1) for i in range(feature_res)[::-1]]
                    import itertools
                    markers = itertools.cycle(('o', '+', '*', 'x', 'p', '.', '>',
                                               '1', '2', '3', '4', 'd', 's', '<'))
                    # markers = [(3+i, 1, 0) for i in range(feature_res)]
                    for mts in TERM:
                        # XXX: The term structure weights
                        ws = m1.coef_[mts][:-1].reshape(feature_res, TS)
                        # XXX: The moneyness weight
                        wms = m1.coef_[mts][-1]
                        # XXX: Make sure that the moneyness weight is nothing
                        assert (abs(wms) < 1e-4)
                        # XXX: Now just plot the 2d curves
                        fig, ax = plt.subplots()
                        for i in range(feature_res):
                            ax.plot(X, ws[i], marker=next(markers),
                                    label=labels[i], markevery=0.1)
                        ax.set_ylabel('Coefficient magnitudes')
                        ax.set_xlabel('Term structure')
                        ax.legend(ncol=3)
                        dfname = dd.split('/')[1]
                        fname = './plots/%s_m_%s_t_%s_lags_%s_%s_%s.pdf' % (
                            model, m, X[mts], TSTEPS, otype, dfname)
                        plt.savefig(fname, bbox_inches='tight')
                        plt.close(fig)

        if not plot:
            out = out.reshape(out.shape[0], out.shape[1]*out.shape[2])
            valY = valY.reshape(valY.shape[0], valY.shape[1]*valY.shape[2])

    elif (model == 'pmhar' or model == 'pmpca' or
          model == 'pmvae'
          or model == 'pmenethar' or model == 'pmlassohar' or
          model == 'pmenetpca' or model == 'pmlassopca'
          or model == 'pmenetvae' or model == 'pmlassovae'):
        # XXX: The output vector
        out = np.array([0.0]*(valY.shape[0]*valY.shape[1]*valY.shape[2]))
        out = out.reshape(*valY.shape)
        # XXX: Now go through the MS and TS
        mms = np.arange(pred.LM, pred.UM+pred.MSTEP, pred.MSTEP)
        tts = [i/pred.DAYS
               for i in range(pred.LT, pred.UT+pred.TSTEP, pred.TSTEP)]
        import pickle
        model_name = model[2:]

        # XXX: Special case for har due to switching to 20 tstep from 21
        load_tsteps = TSTEPS
        if model_name[-3:] == 'har':
            load_tsteps = 21

        for i, s in enumerate(mms):
            for j, t in enumerate(tts):
                if dd == './figs':
                    with open('./point_feature_models/%s_ts_%s_%s_%s_%s.pkl' %
                              (model_name, load_tsteps,
                               s, t, otype), 'rb') as f:
                        m1 = pickle.load(f)
                else:
                    with open('./point_feature_models/%s_ts_%s_%s_%s_%s_gfigs.pkl' %
                              (model_name, load_tsteps,
                               s, t, otype), 'rb') as f:
                        m1 = pickle.load(f)

                # XXX: Now make the prediction
                vYY = valY[:, i, j]
                val_vec = valX[:, :, i, j]

                # # XXX: Extract features before prediction
                # val_vec = extract_features(val_vec, model_name, dd, TSTEPS,
                #                            feature_res, m=s, t=t,
                #                            type='point', otype=otype)

                # XXX: Add t to the sample set
                k = np.array([s, t]*val_vec.shape[0]).reshape(val_vec.shape[0], 2)
                val_vec = np.append(val_vec, k, axis=1)

                vYY = vYY.reshape(vYY.shape[0], 1)
                out[:, i, j] = m1.predict(val_vec, vYY, TSTEPS, 'point').reshape(vYY.shape[0],)

                # XXX: Feature vector plot
                if get_features and model != 'pmvae':
                    if i in MONEYNESS and j in TERM:
                        fig, ax = plt.subplots()
                        xaxis = ['t-%s' % (i+1)
                                 for i in (range(feature_res))[::-1]]
                        # if model == 'pmhar':
                        xaxis.append(r'$\mu$')
                        xaxis.append(r'$\tau$')
                        ax.bar(xaxis, m1.coef_, color='b')
                        ax.set_ylabel('Coefficient magnitudes')
                        plt.xticks(fontsize=9, rotation=45)
                        dfname = dd.split('/')[1]
                        fname = './plots/%s_m_%s_t_%s_lags_%s_%s_%s.pdf' % (
                            model, s, t, TSTEPS, otype, dfname)
                        plt.savefig(fname, bbox_inches='tight')
                        plt.close(fig)
        if not plot:
            out = out.reshape(out.shape[0], out.shape[1]*out.shape[2])
            valY = valY.reshape(valY.shape[0], valY.shape[1]*valY.shape[2])

    elif (model == 'mskenet' or model == 'mskridge' or model == 'msklasso' or
          model == 'mskplsridge' or model == 'mskplslasso' or
          model == 'mskplsenet' or model == 'msknsridge' or
          model == 'msknslasso' or model == 'msknsenet'):
        # XXX: The output vector
        out = np.array([0.0]*(valY.shape[0]*valY.shape[1]*valY.shape[2]))
        out = out.reshape(*valY.shape)
        mms = np.arange(pred.LM, pred.UM+pred.MSTEP, pred.MSTEP)
        # XXX: Now go through the TS
        tts = [i/pred.DAYS for i in range(pred.LT, pred.UT+pred.TSTEP,
                                          pred.TSTEP)]
        import pickle
        for j, t in enumerate(tts):
            if dd == './figs':
                with open('./mskew_models/%s_ts_%s_%s_%s.pkl' %
                          (model, TSTEPS, t, otype), 'rb') as f:
                    m1 = pickle.load(f)
            else:
                with open('./mskew_models/%s_ts_%s_%s_%s_gfigs.pkl' %
                          (model, TSTEPS, t, otype), 'rb') as f:
                    m1 = pickle.load(f)
            mskew = valX[:, :, :, j]
            mskew = mskew.reshape(mskew.shape[0],
                                  mskew.shape[1]*mskew.shape[2])
            # XXX: Add t to the sample set
            ts = np.array([t]*mskew.shape[0]).reshape(mskew.shape[0], 1)
            mskew = np.append(mskew, ts, axis=1)
            # print('./mskew_models/%s_ts_%s_%s_%s.pkl' % (model, TSTEPS,
            #                                              t, otype))
            # print(mskew.shape)
            out[:, :, j] = m1.predict(mskew)

            if get_features and model == 'mskridge':
                if j in TERM:
                    X = np.arange(pred.LM, pred.UM+pred.MSTEP, pred.MSTEP)
                    labels = ['t-%s' % (i+1) for i in range(TSTEPS)[::-1]]
                    markers = [(3+i, 1, 0) for i in range(TSTEPS)]
                    for mts in MONEYNESS:
                        # XXX: The term structure weights
                        ws = m1.coef_[mts][:-1].reshape(TSTEPS, MS)
                        # XXX: The term structure weight
                        wms = m1.coef_[mts][-1]
                        # XXX: Make sure that the term structure weight=nil
                        assert (abs(wms) < 1e-4)
                        # XXX: Now just plot the 2d curves
                        fig, ax = plt.subplots()
                        for i in range(TSTEPS):
                            ax.plot(X, ws[i], marker=markers[i],
                                    label=labels[i], markevery=0.1)
                        ax.set_ylabel('Coefficient magnitudes')
                        ax.set_xlabel('Moneyness')
                        ax.legend(ncol=3)
                        dfname = dd.split('/')[1]
                        fname = './plots/%s_m_%s_t_%s_lags_%s_%s_%s.pdf' % (
                            model, X[mts], t, TSTEPS, otype, dfname)
                        plt.savefig(fname, bbox_inches='tight')
                        plt.close(fig)
        if not plot:
            out = out.reshape(out.shape[0], out.shape[1]*out.shape[2])
            valY = valY.reshape(valY.shape[0], valY.shape[1]*valY.shape[2])

    elif (model == 'vae' or model == 'pca' or
          model == 'har' or model == 'enethar'
          or model == 'lassohar' or
          model == 'enetpca' or model == 'lassopca'
          or model == 'enetvae' or model == 'lassovae'):
        import pickle
        valX = valX.reshape(valX.shape[0],
                            valX.shape[1]*valX.shape[2]*valX.shape[3])
        if not plot:
            valY = valY.reshape(valY.shape[0], valY.shape[1]*valY.shape[2])

        # XXX: Special case for har due to switching to 20 tstep from 21
        load_tsteps = TSTEPS
        if model[-3:] == 'har':
            load_tsteps = 21

        # XXX: Load the model and then carry it out
        if dd == './gfigs':
            toopen = r"./surf_feature_models/%s_ts_%s_%s_gfigs.pkl" % (
                model.lower(), load_tsteps, otype)
        else:
            toopen = r"./surf_feature_models/%s_ts_%s_%s.pkl" % (
                model.lower(), load_tsteps, otype)

        # print('Doing model: ', toopen)
        with open(toopen, "rb") as input_file:
            m1 = pickle.load(input_file)

        # XXX: Get the feature importances
        if get_features and model != 'vaesurf':
            ws = m1.coef_.reshape(MS, TS, feature_res, MS, TS)
            # XXX: Just get the top 10 results
            X = np.arange(pred.LM, pred.UM+pred.MSTEP, pred.MSTEP)
            Y = [i/pred.DAYS for i in range(pred.LT, pred.UT+pred.TSTEP,
                                            pred.TSTEP)]
            for i in MONEYNESS:
                for j in TERM:
                    wsurf = ws[i][j]  # 1 IV point
                    if dd == './figs':
                        name = './plots/%s_m_%s_t_%s_%s_%s.pdf' % (
                            model, X[i], Y[j], TSTEPS, otype)
                    else:
                        name = './plots/%s_m_%s_t_%s_%s_%s_gfigs.pdf' % (
                            model, X[i], Y[j], TSTEPS, otype)
                    plot_hmap_features(wsurf, X, Y, name)

        # # Extract features before prediction
        # valX = extract_features(valX, model, dd, TSTEPS, feature_res,
        #                         type='surf', otype=otype)

        # XXX: Predict the output
        out = m1.predict(valX, valY, TSTEPS, 'surf')
        if plot:
            # XXX: Reshape the data for plotting
            out = out.reshape(out.shape[0], MS, TS)

    else:
        import pickle
        valX = valX.reshape(valX.shape[0],
                            valX.shape[1]*valX.shape[2]*valX.shape[3])
        if not plot:
            valY = valY.reshape(valY.shape[0], valY.shape[1]*valY.shape[2])

        # XXX: Load the model and then carry it out
        if dd == './gfigs':
            toopen = r"./surf_models/model_%s_ts_%s_%s_gfigs.pkl" % (
                model.lower(), pred.TSTEPS, otype)
        else:
            toopen = r"./surf_models/model_%s_ts_%s_%s.pkl" % (
                model.lower(), pred.TSTEPS, otype)

        # print('Doing model: ', toopen)
        with open(toopen, "rb") as input_file:
            m1 = pickle.load(input_file)

        # XXX: Get the feature importances
        if get_features:
            if model == 'ridge':
                ws = m1.coef_.reshape(MS, TS, TSTEPS, MS, TS)
                # XXX: Just get the top 10 results
                X = np.arange(pred.LM, pred.UM+pred.MSTEP, pred.MSTEP)
                Y = [i/pred.DAYS for i in range(pred.LT, pred.UT+pred.TSTEP,
                                                pred.TSTEP)]
                for i in MONEYNESS:
                    for j in TERM:
                        wsurf = ws[i][j]  # 1 IV point
                        if dd == './figs':
                            name = './plots/%s_m_%s_t_%s_%s_%s.pdf' % (
                                model, X[i], Y[j], TSTEPS, otype)
                        else:
                            name = './plots/%s_m_%s_t_%s_%s_%s_gfigs.pdf' % (
                                model, X[i], Y[j], TSTEPS, otype)
                        plot_hmap_features(wsurf, X, Y, name)

        # XXX: Predict the output
        out = m1.predict(valX)
        if plot:
            # XXX: Reshape the data for plotting
            out = out.reshape(out.shape[0], MS, TS)

    if plot:
        plotme(valY, Ydates, out)
        return Ydates, valY, out
    else:
        # XXX: We want to do analysis (quant measures)

        # XXX: RMSE (mean and std-dev of RMSE)
        # rmses = root_mean_squared_error(valY, out, multioutput='raw_values')
        # mapes = mean_absolute_percentage_error(valY, out,
        #                                        multioutput='raw_values')
        # r2sc = r2_score(valY, out, multioutput='raw_values')
        # print('RMSE mean: ', np.mean(rmses), 'RMSE std-dev: ', np.std(rmses))
        # print('MAPE mean: ', np.mean(mapes), 'MAPE std-dev: ', np.std(mapes))
        # print('R2 score mean:', np.mean(r2sc), 'R2 score std-dev: ',
        #       np.std(r2sc))
        Ydates = np.array(Ydates)
        # print(Ydates.shape, valY.shape, out.shape)
        return Ydates, valY, out


def save_results(otype, models, fp, cache):
    # XXX: f = gzip.GzipFile('file.npy.gz', "r"); np.load(f) -- to read
    # XXX: Save all the dates and outputs
    for dd in ['./figs', './gfigs']:
        for t in fp.keys():
            for m in models:
                ddf = dd.split('/')[1]
                tosave = './final_results/%s_%s_ts_%s_model_%s.npy.gz' % (
                    otype, ddf, t, m)
                dates, y, o = cache[dd][t][m]
                dates = dates.reshape(y.shape[0], 1)
                res = np.append(dates, y, axis=1)
                res = np.append(res, o, axis=1)
                blosc2.save_array(res, tosave, mode='w')
                # with gzip.open(filename=tosave, mode='wb',
                #                compresslevel=3) as f:
                #     np.save(file=f, arr=res)


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


def getpreds(name1):
    mms = np.arange(pred.LM, pred.UM+pred.MSTEP, pred.MSTEP)
    # XXX: Now go through the TS
    tts = [i/pred.DAYS for i in range(pred.LT, pred.UT+pred.TSTEP,
                                      pred.TSTEP)]
    MS = len(mms)
    TS = len(tts)

    data1 = blosc2.load_array(name1)
    # XXX: These are the dates
    # dates = data1[:, 0].astype(np.datetime64, copy=False)
    # with gzip.open(filename=name1, mode='rb') as f:
    #     data1 = np.load(f)

    y = data1[:, 1:MS*TS+1].astype(float, copy=False)
    yp = data1[:, MS*TS+1:].astype(float, copy=False)

    # XXX: Across time
    return np.transpose(y), np.transpose(yp)


def rmse_r2_time_series(fname, ax1, ax2, mm, m1, em, bottom):
    mms = np.arange(pred.LM, pred.UM+pred.MSTEP, pred.MSTEP)
    # XXX: Now go through the TS
    tts = [i/pred.DAYS for i in range(pred.LT, pred.UT+pred.TSTEP,
                                      pred.TSTEP)]
    MS = len(mms)
    TS = len(tts)

    otype = fname.split('/')[2].split('_')[0]
    figss = fname.split('/')[2].split('_')[1]
    mname = fname.split('/')[2].split('_')[5].split('.')[0]
    mlags = fname.split('/')[2].split('_')[3]

    print('Doing model: %s %s %s, Lags: %s' % (otype, figss, mname, mlags))

    data = blosc2.load_array(fname)
    # with gzip.open(filename=fname, mode='rb') as f:
    #     data = np.load(f)

    # XXX: Attach the date time for each y and yp
    date = pd.to_datetime(data[:, 0], format='%Y%m%d')
    date = date.date
    y = data[:, 1:MS*TS+1].astype(float, copy=False)
    yp = data[:, MS*TS+1:].astype(float, copy=False)

    years = [np.datetime64('%s-01-01' % i) for i in range(2014, 2024)]
    ylabels = [str(i.astype('datetime64[Y]')) for i in years][:-1]
    r2sc = [np.nan]*(len(years)-1)
    for i in range(len(years)-1):
        h = (date >= years[i]) & (date < years[i+1])
        # XXX: Get the y and yp that are true
        yi = y[h]
        ypi = yp[h]
        r2sc[i] = r2_score(yi, ypi)*100
    # XXX: Clean it up to have a min value of 0
    r2sc = [0 if i < 0 else i for i in r2sc]

    # XXX: Get the rolling RMSE
    rmses = root_mean_squared_error(np.transpose(y), np.transpose(yp),
                                    multioutput='raw_values')

    mname = 'sridge' if mname == 'ridge' else mname
    mname = 'slasso' if mname == 'lasso' else mname
    mname = 'senet' if mname == 'enet' else mname

    ax1.plot(rmses, label='%s' % mname, marker=m1, markevery=em,
             linewidth=0.6)
    width = 0.55
    bb = ax2.bar(ylabels, r2sc, label='%s' % mname, width=width, bottom=bottom)
    lr2sc = ['']*len(r2sc)
    for e, i in enumerate(r2sc):
        lr2sc[e] = '%0.2f' % i if i > 0 else ''
    ax2.bar_label(bb, labels=lr2sc, label_type='center')
    return date, [r2sc[i]+bottom[i] if r2sc[i] >= 0 else bottom[i]
                  for i in range(len(r2sc))]


def overall(fname):
    mms = np.arange(pred.LM, pred.UM+pred.MSTEP, pred.MSTEP)
    # XXX: Now go through the TS
    tts = [i/pred.DAYS for i in range(pred.LT, pred.UT+pred.TSTEP,
                                      pred.TSTEP)]
    MS = len(mms)
    TS = len(tts)

    # otype = fname.split('/')[2].split('_')[0]
    # figss = fname.split('/')[2].split('_')[1]
    # mname = fname.split('/')[2].split('_')[5].split('.')[0]
    # mlags = fname.split('/')[2].split('_')[3]

    # print('Doing model: %s %s %s, Lags: %s' % (otype, figss, mname, mlags))

    data = blosc2.load_array(fname)
    # with gzip.open(filename=fname, mode='rb') as f:
    #     data = np.load(f)

    # XXX: Attach the date time for each y and yp
    # date = pd.to_datetime(data[:, 0], format='%Y%m%d')
    y = data[:, 1:MS*TS+1].astype(float, copy=False)
    yp = data[:, MS*TS+1:].astype(float, copy=False)
    # XXX: This is being done for handling nan values
    mask = np.isnan(yp)
    if (mask.sum() > 0):
        print('Nan values in prediction: ', mask.sum())
        yp[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask),
                             yp[~mask])
    # print(date.shape, y.shape, yp.shape)

    # XXX: Get the rolling RMSE
    rmses = root_mean_squared_error(np.transpose(y), np.transpose(yp),
                                    multioutput='raw_values')
    # mapes = mean_absolute_percentage_error(y, yp, multioutput='raw_values')
    r2sc = r2_score(y, yp, multioutput='raw_values')*100

    return np.mean(rmses), np.std(rmses), np.mean(r2sc), np.std(r2sc)


def model_v_model(otype):
    TTS = [20]
    models = [
        # r'ctridge', r'ctlasso', r'ctenet',
        # r'ssviridge', r'ssvilasso', r'ssvienet',

        # 'msknsridge', 'mskenet', 'msknslasso', 'msknsenet',
        # 'tsknsridge', 'tsknslasso', 'tsknsenet',
        # 'mskplslasso', 'mskplsridge', 'mskplsenet',
        # 'msklasso', 'ridge', 'lasso',
        # 'enet', 'plsridge', 'plslasso', 'plsenet',
        # 'pmridge', 'pmlasso', 'pmenet', 'pmplsridge',
        # 'pmplslasso', 'pmplsenet',
        # 'mskridge', 'tskridge',
        # 'tsklasso', 'tskenet', 'tskplsridge', 'tskplslasso',
        # 'tskplsenet'
        # 'har', 'lassohar', 'enethar',
        # 'pmhar', 'pmlassohar', 'pmenethar',
        # 'tskhar', 'tsklassohar', 'tskenethar',
        # 'mskhar', 'msklassohar', 'mskenethar'
        # 'vae', 'mskvae', 'tskvae',
        # 'lassovae', 'msklassovae', 'tsklassovae',
        # 'enetvae', 'mskenetvae', 'tskenetvae',
        # 'pca', 'mskpca', 'tskpca', 'pmpca',
        # 'lassopca', 'msklassopca', 'tsklassopca', 'pmlassopca',
        # 'enetpca', 'mskenetpca', 'tskenetpca', 'pmenetpca',
        # 'har', 
        'mskhar', 
        # 'pmridge', 
        # 'tskridge'
    ]
    fp = {t: np.array([0.0]*len(models)*len(models)).reshape(len(models),
                                                             len(models))
          for t in TTS}

    for dd in ['./figs']:
        for t in fp.keys():
            for i in range(len(models)):
                if models[i][-3:] == 'har':
                    feature_res = 3
                dates, y, o = main(otype, plot=False, TSTEPS=t,
                                   model=models[i],
                                   feature_res=feature_res,
                                   get_features=True,
                                   dd=dd)
                # XXX: Save all the results
                # tosave = './final_results/%s_%s_ts_%s_model_%s.npy.gz' % (
                #     otype, dd.split('/')[1], t, models[i])
                # print('Saving result: %s' % tosave)
                # dates = dates.reshape(y.shape[0], 1)
                # res = np.append(dates, y, axis=1)
                # res = np.append(res, o, axis=1)
                # blosc2.save_array(res, tosave, mode='w')


def call_dmtest(otype, mmodel, models):
    TTS = [5]
    #models = ['ssviridge', 'lasso', 'enet', 'plsridge', 'plslasso',
    # 'plsenet',
    #           'pmlasso', 'pmenet', 'pmplsridge', 'pmplslasso',
    #           'pmplsenet', 'msklasso', 'mskenet', 'mskplsridge',
    # 'mskplslasso',
    #           'mskplsenet', 'msknsridge', 'msknslasso', 'msknsenet',
    #           'ctridge', 'ctlasso', 'ctenet',
    #           'ssvilasso', 'ssvienet',
    #           'tsklasso', 'tskenet', 'tskplsridge', 'tskplslasso',
    #           'tskplsenet', 'tsknsridge', 'tsknslasso', 'tsknsenet',
    #           'ridge', 'mskridge', 'pmridge', 'tskridge']
    fp = {t: np.array([0.0]*len(models)*len(models)).reshape(len(models),
                                                             len(models))
          for t in TTS}
    fd = {t: np.array([0.0]*len(models)*len(models)).reshape(len(models),
                                                             len(models))
          for t in TTS}

    r2p = {t: np.array([0.0]*len(models)*len(models)).reshape(len(models),
                                                              len(models))
           for t in TTS}
    r2v = {t: np.array([0.0]*len(models)*len(models)).reshape(len(models),
                                                              len(models))
           for t in TTS}

    for dd in ['figs']:
        for ts in TTS:
            # XXX: Moved the cache inside to reduce memory consumption
            cache = {i: {j: {k: None for k in models} for j in TTS}
                     for i in ['figs', 'gfigs']}
            # fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
            for i in range(len(models)-1):

                # XXX: Modification for 20-lag HAR
                tstep = ts
                if models[i][-3:] == 'har':
                    tstep = 20

                name1 = ('./final_results/%s_%s_ts_%s_model_%s.npy.gz' %
                         (otype, dd, tstep, models[i]))
                print('Doing %d: %s' % (i, name1))
                if cache[dd][ts][models[i]] is None:
                    y, yp = getpreds(name1)
                    cache[dd][ts][models[i]] = (y, yp)
                else:
                    y, yp = cache[dd][ts][models[i]]
                for j in range(i+1, len(models)):

                    # XXX: Modification for 20-lag HAR
                    tstep = ts
                    if models[j][-3:] == 'har':
                        tstep = 20

                    # XXX: Do only the best ones
                    name2 = ('./final_results/%s_%s_ts_%s_model_%s.npy.gz' %
                             (otype, dd, tstep, models[j]))
                    print('Doing %d: %s' % (j, name2))
                    if cache[dd][ts][models[j]] is None:
                        yk, ypk = getpreds(name2)
                        cache[dd][ts][models[j]] = (yk, ypk)
                    else:
                        yk, ypk = cache[dd][ts][models[j]]
                    # XXX: Remove the NaN values
                    mask = np.isnan(yp)
                    if (mask.sum() > 0):
                        print('Nan values in YP prediction: ', mask.sum())
                        yp[mask] = np.interp(np.flatnonzero(mask),
                                             np.flatnonzero(~mask),
                                             yp[~mask])
                    mask = np.isnan(ypk)
                    if (mask.sum() > 0):
                        print('Nan values in YPK prediction: ', mask.sum())
                        ypk[mask] = np.interp(np.flatnonzero(mask),
                                              np.flatnonzero(~mask),
                                              ypk[~mask])
                                            
                    # XXX: Modifications for 20-lag HAR
                    # remove first few values
                    if not np.array_equal(y.shape, yk.shape):
                        print('Shapes not equal: ', y.shape, yp.shape, yk.shape, ypk.shape)
                        if y.shape[1] > yk.shape[1]:
                            y = y[:, -yk.shape[1]:]
                            yp = yp[:, -yk.shape[1]:]
                        else:
                            yk = yk[:, -y.shape[1]:]
                            ypk = ypk[:, -y.shape[1]:]

                    # assert (np.array_equal(y, yk))
                    # XXX: Now we can do Diebold mariano test
                    try:
                        dstat, pval = dmtest.dm_test(y, yk, yp, ypk)
                    except dmtest.ZeroVarianceException:
                        dstat, pval = np.nan, np.nan
                    fd[ts][i][j] = dstat
                    fp[ts][i][j] = pval
                    # XXX: We have to transpose these, because they are
                    # transposed in getpreds!
                    r2v[ts][i][j] = cr2_score(y.T, yk.T, yp.T, ypk.T)*100
                    r2p[ts][i][j] = cr2_score_pval(y.T, yk.T, yp.T, ypk.T,
                                                   greater=False)
        # XXX: Save the results of dmtest
        header = ','.join(models)
        for t in fp.keys():
            np.savetxt('./plots/pval_%s_%s_%s_%s_rmse.csv' % (otype, t, dd,
                                                              mmodel),
                       fp[t], delimiter=',', header=header)
            np.savetxt('./plots/dstat_%s_%s_%s_%s_rmse.csv' % (otype, t, dd,
                                                               mmodel),
                       fd[t], delimiter=',', header=header)
            np.savetxt('./plots/r2cmp_%s_%s_%s_%s.csv' % (otype, t, dd,
                                                          mmodel),
                       r2v[t], delimiter=',', header=header)
            np.savetxt('./plots/r2cmp_pval_%s_%s_%s_%s.csv' % (otype, t, dd,
                                                               mmodel),
                       r2p[t], delimiter=',', header=header)


def call_timeseries(otype):
    # XXX: This is many models vs many other models
    # model_v_model()
    lmodels = ['Point-SAM', 'MS-HAR', 'TS-Ridge', 'Surface-HAR']
    models = ['pmridge', 'mskhar' , 'tskridge', 'har']
    m1 = ['*', 'P', 'd', '8']
    for dd in ['figs']:
        for ts in [5]:
            fig1, ax1 = plt.subplots(nrows=1, ncols=1)
            fig2, ax2 = plt.subplots(nrows=1, ncols=1)
            bottom = [0]*9
            for i, model in enumerate(models):
                # XXX: Modification for 20-lag HAR
                tstep = ts
                if model[-3:] == 'har':
                    tstep = 20

                # XXX: Do only the best ones
                name = './final_results/%s_%s_ts_%s_model_%s.npy.gz' % (
                    otype, dd, tstep, model)
                dates, bottom = rmse_r2_time_series(name, ax1, ax2,
                                                    model, m1[i],
                                                    em=0.999,
                                                    bottom=bottom)
            EVERY = 400
            ax1.set_xticks(range(0, dates.shape[0], EVERY),
                           labels=[dates[i]
                                   for i in range(0, dates.shape[0], EVERY)])
            # ax2.set_xlabel('Dates')
            ax1.set_ylabel('RMSE (avg)')
            ax2.set_ylabel(r'$R^2$ (avg)')
            ax2.yaxis.set_ticklabels([])
            ax1.legend(lmodels, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.05),
                       fancybox=True, shadow=True)
            ax2.legend(lmodels, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.05),
                       fancybox=True, shadow=True)
            plt.xticks(fontsize=9, rotation=40)
            fig1.savefig('./plots/%s_%s_rmse_time_series_best_models_%s.pdf'
                         % (otype, dd, ts))
            fig2.savefig('./plots/%s_%s_r2_time_series_best_models_%s.pdf'
                         % (otype, dd, ts))
            plt.close(fig1)
            plt.close(fig2)


def call_overall(otype):
    # XXX: Now plot the overall RMSE, MAPE, and R2 for all models
    # average across all results.
    TTS = [5, 10, 20]
    # lmodels = ['ssviridge', 'ctridge', 'ctlasso', 'ctenet',
    #            'ssvilasso', 'ssvienet',
    #            'sridge', 'slasso', 'senet',
    #            'splsridge', 'splslasso', 'splsenet',
    #            'pmridge', 'pmlasso',
    #            'pmenet', 'pmplsridge', 'pmplslasso', 'pmplsenet',
    #            'mskridge', 'msklasso', 'mskenet', 'mskplsridge', 'mskplslasso',
    #            'mskplsenet', 'msknsridge', 'msknslasso', 'msknsenet',
    #            'tskridge', 'tsklasso', 'tskenet', 'tskplsridge', 'tskplslasso',
    #            'tskplsenet', 'tsknsridge', 'tsknslasso', 'tsknsenet']
    # models = ['ssviridge', 'ctridge', 'ctlasso', 'ctenet',
    #           'ssvilasso', 'ssvienet',
    #           'ridge', 'lasso', 'enet', 'plsridge', 'plslasso', 'plsenet',
    #           'pmridge', 'pmlasso',
    #           'pmenet', 'pmplsridge', 'pmplslasso', 'pmplsenet',
    #           'mskridge', 'msklasso', 'mskenet', 'mskplsridge', 'mskplslasso',
    #           'mskplsenet', 'msknsridge', 'msknslasso', 'msknsenet',
    #           'tskridge', 'tsklasso', 'tskenet', 'tskplsridge', 'tskplslasso',
    #           'tskplsenet', 'tsknsridge', 'tsknslasso', 'tsknsenet']
    lmodels = ['Point-SAM', 'MS-HAR' , 'TS-Ridge', 'Surface-HAR']
    models = ['pmridge', 'mskhar', 'tskridge', 'har']
    ylabels = ['RMSE', 'RMSE (std)', r'$R^2$ (%)', r'$R^2$ (std)']
 
    # XXX: Do the overall RMSE, MAPE, and R2
    for dd in ['figs']:
        data = {key: {
            'Lags 5': [],
            'Lags 10': [],
            'Lags 20': []
        } for key in ['rmse', 'rmsestd', 'r2', 'r2std']}

        for i, model in enumerate(models):
            rmsemeans = []
            # mapemeans = []
            r2means = []
            rmsestds = []
            # mapestds = []
            r2stds = []
            for ts in TTS:
                # XXX: Modification for 20-lag HAR
                tstep = ts
                if model[-3:] == 'har':
                    tstep = 20

                name = './final_results/%s_%s_ts_%s_model_%s.npy.gz' % (
                    otype, dd, tstep, model)
                print('Doing model: %s' % name)
                rmsem, rmsestd, r2m, r2std = overall(name)
                # XXX: MEAN
                rmsemeans.append(rmsem)
                # mapemeans.append(mapesm)
                r2means.append(r2m)
                # XXX: STD
                rmsestds.append(rmsestd)
                # mapestds.append(mapestd)
                r2stds.append(r2std)

                data['rmse']['Lags %d' % ts].append(rmsem)
                # data['mape']['Lags %d' % ts].append(mapesm)
                data['r2']['Lags %d' % ts].append(r2m)
                data['rmsestd']['Lags %d' % ts].append(rmsestd)
                # data['mapestd']['Lags %d' % ts].append(mapestd)
                data['r2std']['Lags %d' % ts].append(r2std)

            # Plot grouped bar chat for RMSE and R2

            
            # df = pd.DataFrame({'models': lmodels, 'rmse': rmsemeans,
            #                    'rmsestd': rmsestds, 'r2:': r2means,
            #                    'r2std': r2stds})
            # df.to_csv('./plots/%s_%s_rmse_r2_avg_std_models_%s.csv'
            #           % (otype, dd, ts))
        i = 0
        for measurement, _ in data.items():

            fig, ax = plt.subplots(nrows=1, ncols=1)
            x = np.arange(len(lmodels))
            width = 0.25
            multiplier = 0

            for key, value in data[measurement].items():
                offset = width * multiplier
                ax.bar(x + offset, value, width, label=key)
                multiplier += 1
            
            # ax.set_title('Total absolute %s for best models' % data[i])
            ax.set_xticks(x + width, lmodels)
            ax.legend(loc='upper left', ncols=4)
            # set y-label horizontally
            ax.set_ylabel('%s' % ylabels[i])
            # Set y-limit
            ax.set_ylim(0, np.max([np.max(data[measurement][key]) for key in data[measurement].keys()])*1.5)

            fig.savefig('./plots/%s_%s_%s_avg_models.pdf' % (otype, dd, measurement))
            i += 1


def moneyness_term(fname, m, mi, t, tn, df, df2, y=None, yp=None):
    if y is None:
        mms = np.arange(pred.LM, pred.UM+pred.MSTEP, pred.MSTEP)
        # XXX: Now go through the TS
        tts = [i/pred.DAYS for i in range(pred.LT, pred.UT+pred.TSTEP,
                                          pred.TSTEP)]
        MS = len(mms)
        TS = len(tts)

        otype = fname.split('/')[2].split('_')[0]
        figss = fname.split('/')[2].split('_')[1]
        mname = fname.split('/')[2].split('_')[5].split('.')[0]
        mlags = fname.split('/')[2].split('_')[3]

        print('Doing model: %s %s %s, Lags: %s' % (otype, figss, mname, mlags))

        data = blosc2.load_array(fname)
        # with gzip.open(filename=fname, mode='rb') as f:
        #     data = np.load(f)

        # XXX: Attach the date time for each y and yp
        y = data[:, 1:MS*TS+1].astype(float, copy=False)
        y = y.reshape(y.shape[0], MS, TS)
        yp = data[:, MS*TS+1:].astype(float, copy=False)
        yp = yp.reshape(yp.shape[0], MS, TS)

    # XXX: Now get the square matrix that you want
    yr = y[:, m[0]:m[1], t[0]:t[1]]
    ypr = yp[:, m[0]:m[1], t[0]:t[1]]

    # XXX: Now do r2_score for these and add to the df
    yr = yr.reshape(yr.shape[0], yr.shape[1]*yr.shape[2])
    ypr = ypr.reshape(ypr.shape[0], ypr.shape[1]*ypr.shape[2])
    df[tn][mi] = r2_score(yr, ypr)
    df2[tn][mi] = root_mean_squared_error(np.transpose(yr),
                                          np.transpose(ypr))
    return y, yp


def r2_rmse_score_mt(otype, models=['pmridge', 'tskridge']):
    assert (len(models) == 2)
    TS = [5]
    dds = ['figs']
    mms = np.arange(pred.LM, pred.UM+pred.MSTEP, pred.MSTEP)
    # XXX: Now go through the TS
    tts = [i/pred.DAYS for i in range(pred.LT, pred.UT+pred.TSTEP,
                                      pred.TSTEP)]

    itms = [i for i in mms if i > 0.8 and i <= 0.99]
    atms = [i for i in mms if i > 0.99 and i <= 1.01]
    otms = [i for i in mms if i > 1.01 and i <= 1.21]
    ITM = (0, len(itms))
    ATM = (ITM[1], ITM[1]+len(atms))
    OTM = (ATM[1], ATM[1]+len(otms))
    st = [i for i in tts if i <= 90/pred.DAYS]
    STI = (0, len(st))
    mt = [i for i in tts if i > 90/pred.DAYS and i <= 180/pred.DAYS]
    MTI = (STI[1], STI[1]+len(mt))
    lt = [i for i in tts if i > 180/pred.DAYS]
    LTI = (MTI[1], MTI[1]+len(lt))
    for dd in dds:
        for ts in TS:
            # for i, model in enumerate(models):
            model1 = models[0]
            model2 = models[1]
            # XXX: Modification for 20-lag HAR
            tstep1 = ts
            tstep2 = ts
            if model1[-3:] == 'har':
                tstep1 = 20
            
            if model2[-3:] == 'har':
                tstep2 = 20
            
            name1 = './final_results/%s_%s_ts_%s_model_%s.npy.gz' % (
                otype, dd, tstep1, model1)
            name2 = './final_results/%s_%s_ts_%s_model_%s.npy.gz' % (
                otype, dd, tstep2, model2)
            # XXX: Make the different square matrices
            df1 = pd.DataFrame({'m': ['itm', 'atm', 'otm'],
                                'st': [np.nan]*3, 'mt': [np.nan]*3,
                                'lt': [np.nan]*3})
            df2 = pd.DataFrame({'m': ['itm', 'atm', 'otm'],
                                'st': [np.nan]*3, 'mt': [np.nan]*3,
                                'lt': [np.nan]*3})
            df3 = pd.DataFrame({'m': ['itm', 'atm', 'otm'],
                                'st': [np.nan]*3, 'mt': [np.nan]*3,
                                'lt': [np.nan]*3})
            df4 = pd.DataFrame({'m': ['itm', 'atm', 'otm'],
                                'st': [np.nan]*3, 'mt': [np.nan]*3,
                                'lt': [np.nan]*3})
            dm = pd.DataFrame({'m': ['itm', 'atm', 'otm'],
                               'st': [np.nan]*3, 'mt': [np.nan]*3,
                               'lt': [np.nan]*3})
            dmp = pd.DataFrame({'m': ['itm', 'atm', 'otm'],
                                'st': [np.nan]*3, 'mt': [np.nan]*3,
                                'lt': [np.nan]*3})

            rm = pd.DataFrame({'m': ['itm', 'atm', 'otm'],
                               'st': [np.nan]*3, 'mt': [np.nan]*3,
                               'lt': [np.nan]*3})
            rmp = pd.DataFrame({'m': ['itm', 'atm', 'otm'],
                                'st': [np.nan]*3, 'mt': [np.nan]*3,
                                'lt': [np.nan]*3})
            y1 = None
            yp1 = None
            y2 = None
            yp2 = None
            for mi, m in enumerate([ITM, ATM, OTM]):
                for t, tn in [(STI, 'st'), (MTI, 'mt'), (LTI, 'lt')]:
                    y1, yp1 = moneyness_term(name1, m, mi, t, tn, df1, df2,
                                             y1, yp1)
                    y2, yp2 = moneyness_term(name2, m, mi, t, tn, df3, df4,
                                             y2, yp2)
                    # XXX: Perform Diebold mariano test for yp1 and yp2
                    # assert (np.array_equal(y1, y2))
                    print(y1.shape, yp1.shape, y2.shape, yp2.shape)
                    # XXX: Modifications for 20-lag HAR
                    if not np.array_equal(y1.shape, y2.shape):
                        print('Shapes not equal: ', y1.shape, yp1.shape, y2.shape, yp2.shape)
                        if y1.shape[0] > y2.shape[0]:
                            y1 = y1[-y2.shape[0]:, :]
                            yp1 = yp1[-y2.shape[0]:, :]
                        else:
                            y2 = y2[-y1.shape[0]:, :]
                            yp2 = yp2[-y1.shape[0]:, :]
                    print(y1.shape, yp1.shape, y2.shape, yp2.shape)
                    yr = y1[:, m[0]:m[1], t[0]:t[1]]
                    yr2 = y2[:, m[0]:m[1], t[0]:t[1]]
                    ypr1 = yp1[:, m[0]:m[1], t[0]:t[1]]
                    ypr2 = yp2[:, m[0]:m[1], t[0]:t[1]]
                    yr = yr.reshape(yr.shape[0], yr.shape[1]*yr.shape[2])
                    yr2 = yr2.reshape(yr2.shape[0], yr2.shape[1]*yr2.shape[2])
                    ypr1 = ypr1.reshape(ypr1.shape[0],
                                        ypr1.shape[1]*ypr1.shape[2])
                    ypr2 = ypr2.reshape(ypr2.shape[0],
                                        ypr2.shape[1]*ypr2.shape[2])
                    dstat, pval = dmtest.dm_test(np.transpose(yr),
                                                 np.transpose(yr2),
                                                 np.transpose(ypr1),
                                                 np.transpose(ypr2))
                    dm[tn][mi] = dstat
                    dmp[tn][mi] = pval
                    rm[tn][mi] = cr2_score(yr, yr2, ypr1, ypr2)*100
                    rmp[tn][mi] = cr2_score_pval(yr, yr2, ypr1, ypr2, greater=False)

            # XXX: Write the data frame to file
            # df1.to_csv('./plots/mt_r2_%s_%s_%s_%s.csv' % (model1, otype, ts,
            #                                               dd))
            # df2.to_csv('./plots/mt_rmse_%s_%s_%s_%s.csv' % (model1, otype,
            #                                                 ts, dd))
            # df3.to_csv('./plots/mt_r2_%s_%s_%s_%s.csv' % (model2, otype, ts,
            #                                               dd))
            # df4.to_csv('./plots/mt_rmse_%s_%s_%s_%s.csv' % (model2, otype,
            #                                                 ts, dd))
            dm.to_csv('./plots/mt_rmse_dstat_%s_%s_%s_%s_%s.csv' % (
                model1, model2, otype, ts, dd))
            dmp.to_csv('./plots/mt_rmse_pval_%s_%s_%s_%s_%s.csv' % (
                model1, model2, otype, ts, dd))
            rm.to_csv('./plots/mt_r2_stat_%s_%s_%s_%s_%s.csv' % (
                model1, model2, otype, ts, dd))
            rmp.to_csv('./plots/mt_r2_pval_%s_%s_%s_%s_%s.csv' % (
                model1, model2, otype, ts, dd))


def pttest(y, yhat):
    """Given NumPy arrays with predictions and with true values, return
    Directional Accuracy Score, Pesaran-Timmermann statistic and its
    p-value

    """
    size = y.shape[0]
    pyz = np.sum(np.sign(y) == np.sign(yhat))/size
    py = np.sum(y > 0)/size
    qy = py*(1 - py)/size
    pz = np.sum(yhat > 0)/size
    qz = pz*(1 - pz)/size
    p = py*pz + (1 - py)*(1 - pz)
    v = p*(1 - p)/size
    w = ((2*py - 1)**2) * qz + ((2*pz - 1)**2) * qy + 4*qy*qz
    pt = (pyz - p) / (np.sqrt(v - w))
    pval = 1 - stats.norm.cdf(pt, 0, 1)
    return pyz, pt, pval


def direction(otype, models=['mskhar']):
    mms = np.arange(pred.LM, pred.UM+pred.MSTEP, pred.MSTEP)
    # XXX: Now go through the TS
    tts = [i/pred.DAYS for i in range(pred.LT, pred.UT+pred.TSTEP,
                                      pred.TSTEP)]
    MS = len(mms)
    TS = len(tts)
    for dd in ['figs']:
        for ts in [5]:
            for i, model in enumerate(models):
                # XXX: Modification for 20-lag HAR
                tstep = ts
                if model[-3:] == 'har':
                    tstep = 20

                name = './final_results/%s_%s_ts_%s_model_%s.npy.gz' % (
                    otype, dd, tstep, model)
                print('Doing model: %s' % name)
                data = blosc2.load_array(name)
                y = data[:, 1:MS*TS+1].astype(float, copy=False)
                yp = data[:, MS*TS+1:].astype(float, copy=False)
                d = np.diff(y, axis=0)
                dhat = (yp[1:] - y[:-1])
                print(d.shape, dhat.shape)
                res = [pttest(d[:, i], dhat[:, i])
                       for i in range(d.shape[1])]
                pyz = [i[0] for i in res]
                res = [i[2] for i in res]
                pyz = np.array(pyz).reshape(MS, TS)
                res = np.array(res).reshape(MS, TS)

                # fig, (axp, ax) = plt.subplots(nrows=1, ncols=1, sharex=True)
                fig, axp = plt.subplots(nrows=1, ncols=1, sharex=True)

                imp = axp.imshow(pyz, cmap='Greys')
                # im = ax.imshow(res, cmap='binary')

                axp.set_xlabel('Term structure', fontsize=12)
                axp.set_ylabel('Moneyness', fontsize=12)
                axp.set_yticks([0, MS-1], labels=['%0.2f' % mms[0],
                                                  '%0.2f' % mms[-1]],
                               fontsize=10)
                axp.set_xticks([0, TS-1], labels=['%0.2f' % tts[0],
                                                    '%0.2f' % tts[-1]],
                                 fontsize=10)
                fig.colorbar(imp, orientation='vertical', ax=axp)

                # ax.set_xlabel('Term structure', fontsize=16)
                # ax.set_ylabel('Moneyness', fontsize=16)
                # ax.set_yticks([])
                # ax.set_yticks([0, MS-1], labels=['%0.2f' % mms[0],
                #                                  '%0.2f' % mms[-2]],
                #               fontsize=16)
                # ax.set_xticks([0, TS-1], labels=['%0.2f' % tts[0],
                #                                  '%0.2f' % tts[-1]],
                #               fontsize=16)
                # fig.colorbar(im, orientation='vertical', ax=ax)

                print('For model: %s' % model)
                # # Get the index of the max and min pyz values
                idx = np.unravel_index(np.argmax(pyz, axis=None), pyz.shape)
                # print('Max: %0.5f at %0.5f, %0.5f' % (pyz[idx], mms[idx[0]], tts[idx[1]]))
                print('Max: Co-or: ', idx)
                max_file = ('%s_m_%s_t_%s_lags_%s_call_figs.pdf' % (model, mms[idx[0]], tts[idx[1]], tstep))
                # print('Max: ', max_file)
                idx = np.unravel_index(np.argmin(pyz, axis=None), pyz.shape)
                # print('Min: %0.5f at %0.5f, %0.5f' % (pyz[idx], mms[idx[0]], tts[idx[1]]))
                print('Min: Co-or: ', idx)
                min_file = ('%s_m_%s_t_%s_lags_%s_call_figs.pdf' % (model, mms[idx[0]], tts[idx[1]], tstep))
                # print('Min: ', min_file)

                # Copy min and max files to ../feature_paper/figs/
                shutil.copyfile('./plots/%s' % max_file, '../feature_paper/figs/%s_max_lags_%s_call_figs.pdf' % (model, tstep))
                shutil.copyfile('./plots/%s' % min_file, '../feature_paper/figs/%s_min_lags_%s_call_figs.pdf' % (model, tstep))


                # plt.savefig('./plots/dir_score_pval_%s_%s_ts_%s_model_%s.pdf' %
                #             (otype, dd, ts, model), bbox_inches='tight')
                # plt.close(fig)


def lag_test(otype):
    def load(fname, get_date=False, dt=None):
        mms = np.arange(pred.LM, pred.UM+pred.MSTEP, pred.MSTEP)
        # XXX: Now go through the TS
        tts = [i/pred.DAYS for i in range(pred.LT, pred.UT+pred.TSTEP,
                                          pred.TSTEP)]
        MS = len(mms)
        TS = len(tts)
        data = blosc2.load_array(fname)
        date = pd.to_datetime(data[:, 0], format='%Y%m%d')
        date = date.date
        if get_date:
            return date[0]

        assert (dt is not None)
        y = data[:, 1:MS*TS+1].astype(float, copy=False)
        yp = data[:, MS*TS+1:].astype(float, copy=False)
        toget = date[date >= dt].shape[0]
        start = y.shape[0] - toget
        return y[start:], yp[start:]

    TTS = [20, 10, 5]
    models = ['pmridge', 'tskridge']
    dds = ['figs']
    for dd in dds:
        for i, model in enumerate(models):
            dstatf = np.array([0.0]*len(TTS)*len(TTS)).reshape(len(TTS), len(TTS))
            dstatpf = np.array([0.0]*len(TTS)*len(TTS)).reshape(len(TTS), len(TTS))
            r2f = np.array([0.0]*len(TTS)*len(TTS)).reshape(len(TTS), len(TTS))
            r2pf = np.array([0.0]*len(TTS)*len(TTS)).reshape(len(TTS), len(TTS))

            for i, ts in enumerate(TTS[:-1]):
                name = './final_results/%s_%s_ts_%s_model_%s.npy.gz' % (
                    otype, dd, ts, model)
                print('Doing model: %s' % name)
                # XXX: Get the date first
                dt = load(name, get_date=True)
                y1, yp1 = load(name, dt=dt)
                for j in range(i+1, len(TTS)):
                    name = './final_results/%s_%s_ts_%s_model_%s.npy.gz' % (
                        otype, dd, TTS[j], model)
                    y2, yp2 = load(name, dt=dt)
                    assert (np.array_equal(y1, y2))
                    try:
                        dstat, pval = dmtest.dm_test(y1.T, y2.T, yp1.T, yp2.T)
                    except dmtest.ZeroVarianceException:
                        dstat, pval = np.nan, np.nan
                    rval = cr2_score(y1, y2, yp1, yp2)
                    rpval = cr2_score_pval(y1, y2, yp1, yp2, greater=False)
                    # XXX: Append to the dict list
                    dstatf[i][j] = dstat
                    dstatpf[i][j] = pval
                    r2f[i][j] = rval
                    r2pf[i][j] = rpval
            header = ','.join([str(i) for i in TTS[:]])

            np.savetxt('./plots/lag_dstat_%s_%s_%s.csv' % (otype, dd, model),
                       dstatf, delimiter=',',
                       header=header)
            np.savetxt('./plots/lag_dstat_pval_%s_%s_%s.csv' % (otype, dd,
                                                               model),
                       dstatpf, delimiter=',',
                       header=header)
            np.savetxt('./plots/lag_r2_%s_%s_%s.csv' % (otype, dd, model),
                       r2f, delimiter=',',
                       header=header)
            np.savetxt('./plots/lag_r2_pval_%s_%s_%s.csv' % (otype, dd, model),
                       r2pf, delimiter=',',
                       header=header)
            
            # print(model, ' DM:', dstatf, dstatpf)
            # print(model, r' R^2: ', r2f, r2pf)


def trade(dates, y, yp, otype, strat, eps=0.05, lags=5):

    def getTC(data, P=0.25):
        # XXX: See: Options Trading Costs Are Lower than You
        # Think Dmitriy Muravyev (page-4)
        # return 0/100
        return 2.5/100

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

    def c_position(sign, CP, UP, delta, TC):
        """This is delta hedged
        """

        tc = (CP * TC) + (UP * np.abs(delta)) * TC
        if sign > 0:
            return (CP - (UP * delta)) - tc
        else:
            return (-CP + (UP * delta)) - tc

    def o_position(sign, CP, UP, delta, TC):
        tc = (CP * TC) + (UP * np.abs(delta)) * TC
        if sign > 0:
            return (-CP + (UP * delta)) - tc
        else:
            return (CP - (UP * delta)) - tc

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
            if np.abs((CPp + PPp) - (CP + PP))/(CP+PP) >= 0.05:
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

    # plt.plot(trade_date, cashl[1:])
    # plt.xlabel('Years')
    # plt.ylabel('$ cash position')
    # plt.show()
    # # plt.savefig('/tmp/trade.pdf', bbox_inches='tight')
    # plt.close()


def analyse_trades(otype, model, lags, alpha, betas,
                   cagrs, wins, maxs, mins, avgs, medians,
                   sds, srs, ns, rf=3.83/(100), Y=9):

    import warnings
    warnings.filterwarnings("ignore")
    # print('Model %s_%s_%s: ' % (model, otype, lags))
    mrdf = pd.read_csv('./trades/marketPrice.csv')
    prdf = pd.read_csv('./trades/%s_%s_%s.csv' % (model, otype, lags))

    # XXX: Plot the portfolio
    prdf.plot.line(x='dates', y='cash', label='%s_%s' % (model, otype))
    plt.xlabel('Years')
    plt.ylabel('$ Portfolio P/L')
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
    # print(rp)
    # rp = np.log(prdf['cash']).diff().dropna()
    # XXX: Expected market return
    rm = mrdf.groupby(mrdf.Date.dt.year)['Price'].apply(
        lambda x: (x.values[-1]-x.values[0])/x.values[0])
    # rm = mrdf['Price'].pct_change().dropna()
    # rm = np.log(mrdf['Price']).diff().dropna()
    # print(rp)

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
        cagr_rp = (prdf['cash'].values[-1]/prdf['cash'].values[0])**(1/9) - 1
    else:
        cagr_rp = -np.inf       # infinitely worse!
    cagr_mp = (mrdf['Price'].values[-1]/mrdf['Price'].values[0])**(1/9) - 1
    # print('CAGR Portfolio (%), CAGR Market (%): ', cagr_rp*100, cagr_mp*100)

    if cagr_rp >= 0:
        alpha = cagr_rp - beta*(cagr_mp - rf).mean() - rf
    else:
        alpha = -np.inf

    # XXX: Statsitic data for trades
    dprdf = prdf['cash'].pct_change().dropna()
    # print('Number of trades: ', dprdf[dprdf != 0].shape[0])
    # print('% Wins: ', dprdf[dprdf >= 0].shape[0]/dprdf.shape[0]*100)
    # print('Mean % return: ', dprdf.mean()*100)
    # print('Median % return: ', dprdf.median()*100)
    # print('Max % return: ', dprdf.max()*100)
    # print('Min % return: ', dprdf.min()*100)
    # print('SD % return: ', dprdf.std()*100)
    # # print('Trade description: ', dprdf.describe())
    # print('Sharpe Ratio: ', (cagr_rp-rf)/dprdf.std())
    # print('\n')

    # XXX: Append to lists
    dprdf = prdf['cash'].pct_change().dropna()
    alphas.append(alpha)
    betas.append(beta)
    ns.append(dprdf[dprdf != 0].shape[0])
    cagrs.append(cagr_rp*100)
    maxs.append(dprdf.max()*100)
    mins.append(dprdf.min()*100)
    medians.append(dprdf.median()*100)
    sds.append(dprdf.std()*100)
    srs.append(((cagr_rp-rf)/dprdf.std()))
    wins.append(dprdf[dprdf > 0].shape[0]/ns[-1]*100)
    avgs.append(dprdf.mean()*100)


if __name__ == '__main__':
    plt.style.use('seaborn-v0_8-whitegrid')

    # XXX: model vs model
    # for otype in ['call']:
    #    model_v_model(otype)

    # # # XXX: Point ridge models -- Peter add other models here
    # point_ridge = {'point_ridge': ['pmridge','pmpca', 'pmplsridge', 'pmhar']}
    # point_lasso = {'point_lasso': ['pmlasso','pmlassopca', 'pmplslasso',
    # 'pmlassohar']}
    # point_enet = {'point_enet': ['pmenet', 'pmenetpca', 'pmplsenet',
    # 'pmenethar']}

    # # XXX: Skew models -- Peter add other models here
    # skew_ridge = {'skew_ridge': ['mskridge','mskpca', 'mskplsridge',
    # 'msknsridge', 'mskhar', 'mskvae']}
    # skew_lasso = {'skew_lasso': ['msklasso','msklassopca', 'mskplslasso',
    # 'msknslasso', 'msklassohar', 'msklassovae']}
    # skew_enet = {'skew_enet': ['mskenet', 'mskenetpca', 'mskplsenet',
    # 'msknsenet', 'mskenethar', 'mskenetvae']}

    # # XXX: Term structure models -- Peter add other models here
    # ts_ridge = {'termstructure_ridge':
    #             ['tskridge', 'tskpca', 'tskplsridge', 'tsknsridge',
    # 'tskhar', 'tskvae']}
    # ts_lasso = {'termstructure_lasso':
    #             ['tsklasso', 'tsklassopca', 'tskplslasso', 'tsknslasso',
    # 'tsklassohar', 'tsklassovae']}
    # ts_enet = {'termstructure_enet':
    #            ['tskenet', 'tskenetpca', 'tskplsenet', 'tsknsenet',
    # 'tskenethar', 'tskenetvae']}

    # # XXX: Surface models -- Peter add other models here
    # surf_ridge = {'surf_ridge': ['ridge', 'pca', 'plsridge', 'ctridge',
    # 'ssviridge', 'har', 'vae']}
    # surf_lasso = {'surf_lasso': ['lasso', 'lassopca', 'plslasso', 'ctlasso',
    # 'ssvilasso', 'lassohar', 'lassovae']}
    # surf_enet = {'surf_enet': ['enet', 'enetpca', 'plsenet', 'ctenet',
    # 'ssvienet', 'enethar', 'enetvae']}

    # XXX: Best models for each regressor
    # point = {'point_all': ['pmridge', 'pmplslasso', 'pmplsenet']}
    # skew = {'skew_all': ['mskhar', 'mskplslasso', 'mskplsenet']}
    # ts = {'termstructure_all': ['tskridge', 'tskplslasso', 'tskplsenet']}
    # surf = {'surf_all': ['har', 'plslasso', 'plsenet']}

    # XXX: Best models for each feature
    # models = {'best': ['pmridge', 'mskhar', 'tskridge', 'har']}


    for otype in ['call']:
    # # #     #  XXX: Plot the bar graph for overall results
    #     call_overall(otype)
        #  XXX: Plot the best time series RMSE and MAPE
        call_timeseries(otype)

    # for otype in ['call']:
    #     for i in [ point_ridge, point_lasso, point_enet]:
    #         # XXX: DM test across time (RMSE)
    #         model, model_names = list(i.keys())[0], list(i.values())[0]
    #         call_dmtest(otype, model, model_names)

    # XXX: r2_score for moneyness and term structure
    # for otype in ['call']:
    #     for i in range(len(models['best'])):
    #         for j in range(i+1, len(models['best'])):
                # r2_rmse_score_mt(otype, [models['best'][i], models['best'][j]])

    # for otype in ['call']:
    #     direction(otype)

    # # XXX: Creating feature importance graphs for the best and worst directional accuracy
    # for otype in ['put']:
    #     bestAndWorst = direction(otype, models['best'])



    # XXX: This test compares different lag lengths for the same model,
    # usually there is literally no difference.
    # for otype in ['call']:
    #     lag_test(otype)
    
    # Copying over direction accuracy graphs
    # for otype in ['put']:
    #     for model in models['best']:
    #         shutil.copyfile('./plots/dir_score_pval_%s_figs_ts_5_model_%s.pdf' % (otype, model), '../feature_paper/figs/dir_score_pval_%s_figs_ts_5_model_%s.pdf' % (otype, model))

    # Copying over the timeseries and stacked bar graph
    shutil.copyfile('./plots/call_figs_r2_time_series_best_models_5.pdf', '../feature_paper/figs/call_figs_r2_time_series_best_models_5.pdf')
    shutil.copyfile('./plots/call_figs_rmse_time_series_best_models_5.pdf', '../feature_paper/figs/call_figs_rmse_time_series_best_models_5.pdf')

    # # XXX: Generate latex table for best absolute r2 and rmse results
    # # Copy over best absolute r2 and rmse graphs
    # for otype in ['put']:
    #     shutil.copyfile('./plots/%s_figs_r2_avg_models.pdf' % (otype), '../feature_paper/figs/best_abs_r2_%s.pdf' % (otype))
    #     shutil.copyfile('./plots/%s_figs_r2std_avg_models.pdf' % (otype), '../feature_paper/figs/best_abs_r2std_%s.pdf' % (otype))
    #     shutil.copyfile('./plots/%s_figs_rmse_avg_models.pdf' % (otype), '../feature_paper/figs/best_abs_rmse_%s.pdf' % (otype))
    #     shutil.copyfile('./plots/%s_figs_rmsestd_avg_models.pdf' % (otype), '../feature_paper/figs/best_abs_rmsestd_%s.pdf' % (otype))

    # # df = pd.read_csv('./plots/call_figs_rmse_r2_avg_std_models_5.csv')

    # # columns = ['Index', 'Models', 'RMSE', 'RMSE STD', '$R^2$', '$R^2$ STD']
    # # df.columns = columns

    # # df.drop('Index', axis=1, inplace=True)

    # # df['Models'] = ['Point', 'Skew', 'Term Structure', 'Surface']

    # # df.set_index('Models', inplace=True)
# # # Bold the max r2 and min rmse
    # # def highlight_max(s):
    # #     is_max = s == s.max()
    # #     return ['font-weight: bold' if v else None for v in is_max]
    
    # # def highlight_min(s):
    # #     is_min = s == s.min()
    # #     return ['font-weight: bold' if v else None for v in is_min]
    
    # # s = df.style.apply(highlight_max, subset=['$R^2$'])
    # # s = s.apply(highlight_min, subset=['RMSE'])

    # # s.to_latex(buf='../feature_paper/figs/best/rmse_r2_avg_std_models_5.tex', column_format='l'*df.columns.size+'l', hrules=True, convert_css=True)

    # XXX: Generate latex tables for best dmtest rmse and r2 
    # for otype in ['put']:
    #     for ts in [5]:
    #         for dd in ['figs']:
    #             df = pd.read_csv('./plots/dstat_%s_%s_%s_best_rmse.csv' % (
    #                 otype, ts, dd))
    #             dfp = pd.read_csv('./plots/pval_%s_%s_%s_best_rmse.csv' % (
    #                 otype, ts, dd))
                
    #             dfr = pd.read_csv('./plots/r2cmp_%s_%s_%s_best.csv' % (
    #                 otype, ts, dd))
    #             dfrp = pd.read_csv('./plots/r2cmp_pval_%s_%s_%s_best.csv' % (
    #                 otype, ts, dd))
                
                
    #             # rename the column headers
    #             columns = ['Point-SAM', 'Skew-HAR', '\\ac{TS}-SAM', 'Surface-HAR']
    #             index = ['Point-SAM', 'Skew-HAR', '\\ac{TS}-SAM', 'Surface-HAR']

    #             df.columns = columns
    #             df.index = index
    #             dfp.columns = columns
    #             dfp.index = index

    #             dfr.columns = columns
    #             dfr.index = index
    #             dfrp.columns = columns
    #             dfrp.index = index

    #             # Remove the surface index
    #             df = df.drop('Surface-HAR', axis=0)
    #             dfp = dfp.drop('Surface-HAR', axis=0)

    #             dfr = dfr.drop('Surface-HAR', axis=0)
    #             dfrp = dfrp.drop('Surface-HAR', axis=0)

    #             # set the values past the diagonal to nan
    #             for i in range(df.shape[0]):
    #                 for j in range(df.shape[1]):
    #                     if i >= j:
    #                         df.iloc[i, j] = np.nan
    #                         dfr.iloc[i, j] = np.nan

    #             # Remove the Point column
    #             df = df.drop('Point-SAM', axis=1)
    #             dfp = dfp.drop('Point-SAM', axis=1)

    #             dfr = dfr.drop('Point-SAM', axis=1)
    #             dfrp = dfrp.drop('Point-SAM', axis=1)

    #             # Map function to check if the corrisponding p-value is significant
    #             # Bold insignificant values
    #             def check_significance(x, xp):
    #                 result = np.where(xp > 0.05 ,f"font-weight: bold;" , None)

    #                 # if result.all():
    #                 #     # set all values to None
    #                 #     result = np.where(xp < 0.05, None, None)

    #                 # # Set value to None if x is nan 
    #                 # result = np.where(x != x, None, result)
                    
    #                 return result
                
    #             s = df.style.apply(check_significance, xp=dfp, axis=None)
    #             sr = dfr.style.apply(check_significance, xp=dfrp, axis=None)

    #             s.format('{:.1f}', na_rep=" ")
    #             sr.format('{:.1f}', na_rep=" ")

    #             # print(s.to_latex(column_format='l'*df.columns.size+'l', hrules=True, convert_css=True))
    #             # print(sr.to_latex(column_format='l'*dfr.columns.size+'l', hrules=True, convert_css=True))

    #             s.to_latex(buf='../feature_paper/figs/best/dstat-%s-%s-%s-best-rmse.tex' % (
    #                 otype, ts, dd), column_format='l'*df.columns.size+'l', hrules=True, convert_css=True) 
    #             sr.to_latex(buf='../feature_paper/figs/best/r2cmp-%s-%s-%s-best.tex' % (
    #                 otype, ts, dd), column_format='l'*dfr.columns.size+'l', hrules=True, convert_css=True)

                # for i in range(len(models['best'])):
                #     for j in range(i+1, len(models['best'])):
                #         for stat in ['r2_stat', 'rmse_dstat']:
                #             df = pd.read_csv('./plots/mt_%s_%s_%s_%s_%s_%s.csv' % (
                #                 stat, models['best'][i], models['best'][j], otype, ts, dd))
                            
                #             dfp = pd.read_csv('./plots/mt_%s_pval_%s_%s_%s_%s_%s.csv' % (
                #                 stat.split('_')[0], models['best'][i], models['best'][j], otype, ts, dd))

                #             # rename the column headers
                #             columns = ['Index', 'm', 'Short-Term', 'Medium-Term', 'Long-Term']
                #             df.columns = columns
                #             dfp.columns = columns
                #             df['m'] = ['\\ac{ITM}', '\\ac{ATM}', '\\ac{OTM}']
                #             dfp['m'] = ['\\ac{ITM}', '\\ac{ATM}', '\\ac{OTM}']
                            
                #             # Remove column Index
                #             df = df.drop('Index', axis=1)
                #             df.set_index('m', inplace=True)
                #             dfp = dfp.drop('Index', axis=1)
                #             dfp.set_index('m', inplace=True)
                #             print(df)

                #             # Map function to check if the corrisponding p-value is significant
                #             def check_significance(x, xp):
                #                 result = np.where(xp > 0.05 ,f"font-weight: bold;" , None)

                #                 # if result.all():
                #                 #     # set all values to None
                #                 #     result = np.where(xp < 0.05, None, None)

                #                 # # Set value to None if x is nan 
                #                 # result = np.where(x != x, None, result)
                                
                #                 return result
                            
                #             s = df.style.apply(check_significance, xp=dfp, axis=None)
                #             s.format('{:.2f}' )
                            
                #             s.to_latex(buf='../feature_paper/figs/mt/mt-%s-%s-%s-%s-%s-%s-%s.tex' % (
                #                 stat, models['best'][i], models['best'][j], otype, ts, dd, 'best'), column_format='l'*df.columns.size+'l', hrules=True, convert_css=True)

    # # # XXX: Generate latex table for dmtest rmse
    # for otype in ['call']:
    #     for ts in [5, 10, 20]:
    #         for dd in ['figs']:
    #             for feature in ['surf', 'point', 'skew', 'termstructure']:
    #                 for mmodel in ['ridge', 'lasso', 'enet']:
    #                     df = pd.read_csv('./plots/dstat_%s_%s_%s_%s_%s_rmse.csv' % (
    #                         otype, ts, dd, feature, mmodel))
    #                     dfp = pd.read_csv('./plots/pval_%s_%s_%s_%s_%s_rmse.csv' % (
    #                         otype, ts, dd, feature, mmodel))
                        
    #                     dfr = pd.read_csv('./plots/r2cmp_%s_%s_%s_%s_%s.csv' % (
    #                         otype, ts, dd, feature, mmodel))
    #                     dfrp = pd.read_csv('./plots/r2cmp_pval_%s_%s_%s_%s_%s.csv' % (
    #                         otype, ts, dd, feature, mmodel))
                        
                        
    #                     # rename the column headers
    #                     columns = ['SAM', '\\ac{PCA}', '\\ac{CCA}', '\\ac{NS}', '\\ac{ADNS}', '\\ac{SSVI}', '\\ac{HAR}', '\\ac{VAE}']
    #                     index = ['SAM', '\\ac{PCA}', '\\ac{CCA}', '\\ac{NS}', '\\ac{ADNS}', '\\ac{SSVI}', '\\ac{HAR}', '\\ac{VAE}']

    #                     if feature == 'surf':
    #                         columns.remove('\\ac{NS}')
    #                         index.remove('\\ac{NS}')
    #                     elif feature == 'point':
    #                         columns.remove('\\ac{SSVI}')
    #                         columns.remove('\\ac{ADNS}')
    #                         columns.remove('\\ac{NS}')
    #                         columns.remove('\\ac{VAE}')
    #                         index.remove('\\ac{SSVI}')
    #                         index.remove('\\ac{ADNS}')
    #                         index.remove('\\ac{NS}')
    #                         index.remove('\\ac{VAE}')

    #                     else:
    #                         columns.remove('\\ac{SSVI}')
    #                         columns.remove('\\ac{ADNS}')
    #                         index.remove('\\ac{SSVI}')
    #                         index.remove('\\ac{ADNS}')


    #                     df.columns = columns
    #                     df.index = index
    #                     dfp.columns = columns
    #                     dfp.index = index

    #                     dfr.columns = columns
    #                     dfr.index = index
    #                     dfrp.columns = columns
    #                     dfrp.index = index

    #                     # Remove the last index 
    #                     if feature == 'point':
    #                         df = df.drop('\\ac{HAR}', axis=0)
    #                         dfp = dfp.drop('\\ac{HAR}', axis=0)

    #                         dfr = dfr.drop('\\ac{HAR}', axis=0)
    #                         dfrp = dfrp.drop('\\ac{HAR}', axis=0)
    #                     else:
    #                         df = df.drop('\\ac{VAE}', axis=0)
    #                         dfp = dfp.drop('\\ac{VAE}', axis=0)

    #                         dfr = dfr.drop('\\ac{VAE}', axis=0)
    #                         dfrp = dfrp.drop('\\ac{VAE}', axis=0)


    #                     # remove the har index if ts != 20
    #                     # if ts != 20:
    #                     #     dfp = dfp.drop('\\ac{HAR}', axis=0)
    #                     #     df = df.drop('\\ac{HAR}', axis=0)

    #                     #     dfrp = dfrp.drop('\\ac{HAR}', axis=0)
    #                     #     dfr = dfr.drop('\\ac{HAR}', axis=0)

    #                     #     #convert har to nan
    #                     #     df['\\ac{HAR}'] = np.nan
    #                     #     dfr['\\ac{HAR}'] = np.nan

    #                     # set the values past the diagonal to nan
    #                     for i in range(df.shape[0]):
    #                         for j in range(df.shape[1]):
    #                             if i >= j:
    #                                 df.iloc[i, j] = np.nan
    #                                 dfr.iloc[i, j] = np.nan

    #                     # Drop the first column
    #                     df = df.drop('SAM', axis=1)
    #                     dfp = dfp.drop('SAM', axis=1)

    #                     dfr = dfr.drop('SAM', axis=1)
    #                     dfrp = dfrp.drop('SAM', axis=1)

    #                     # Map function to check if the corrisponding p-value is significant
    #                     def check_significance(x, xp):
    #                         result = np.where(xp > 0.05 ,f"font-weight: bold;" , None)

    #                         # if result.all():
    #                         #     # set all values to None
    #                         #     result = np.where(xp < 0.05, None, None)

    #                         # # Set value to None if x is nan 
    #                         # result = np.where(x != x, None, result)
                            
    #                         return result
                        
    #                     s = df.style.apply(check_significance, xp=dfp, axis=None)
    #                     sr = dfr.style.apply(check_significance, xp=dfrp, axis=None)
                        
    #                     # only style the dstat values if they are significant
    #                     # for i in range(df.shape[0]):
    #                     #     for j in range(df.shape[1]):
    #                     #         if dfp.iloc[i, j] > 0.05:
    #                     #             df.iloc[i, j] = np.nan

    #                     s.format('{:.1f}', na_rep=" ")
    #                     sr.format('{:.1f}', na_rep=" ")

    #                     # print(s.to_latex(column_format='l'*df.columns.size+'l', hrules=True, convert_css=True))
    #                     # print(sr.to_latex(column_format='l'*dfr.columns.size+'l', hrules=True, convert_css=True))

    #                     s.to_latex(buf='../feature_paper/figs/dstat/dstat-%s-%s-%s-%s-%s-rmse.tex' % (
    #                         otype, ts, dd, feature, mmodel), column_format='l'*df.columns.size+'l', hrules=True, convert_css=True) 
    #                     sr.to_latex(buf='../feature_paper/figs/r2cmp/r2cmp-%s-%s-%s-%s-%s.tex' % (
    #                         otype, ts, dd, feature, mmodel), column_format='l'*dfr.columns.size+'l', hrules=True, convert_css=True)

    # #                     # df.to_latex(buf='./plots/%s_%s_%s_%s_rmse.tex' % (
    # #                     #     otype, ts, dd, mmodel), header=True)

    # for otype in ['call']:
    #         for ts in [5, 10, 20]:
    #             for dd in ['figs']:
    #                 for feature in ['surf', 'point', 'skew', 'termstructure']:
    #                     mmodel = 'all'
    #                     df = pd.read_csv('./plots/dstat_%s_%s_%s_%s_%s_rmse.csv' % (
    #                             otype, ts, dd, feature, mmodel))
    #                     dfp = pd.read_csv('./plots/pval_%s_%s_%s_%s_%s_rmse.csv' % (
    #                         otype, ts, dd, feature, mmodel))
                        
    #                     dfr = pd.read_csv('./plots/r2cmp_%s_%s_%s_%s_%s.csv' % (
    #                         otype, ts, dd, feature, mmodel))
    #                     dfrp = pd.read_csv('./plots/r2cmp_pval_%s_%s_%s_%s_%s.csv' % (
    #                         otype, ts, dd, feature, mmodel))

    #                     if feature == 'surf' or feature ==  'skew':
    #                         ridge_col = 'HAR'
    #                     else: 
    #                         ridge_col = 'SAM'
                        
    #                     # rename the column headers
    #                     columns = ['Ridge-%s' % ridge_col, 'Lasso-\\ac{CCA}', 'Elastic Net-\\ac{CCA}']
    #                     index = ['Ridge-%s' % ridge_col, 'Lasso-\\ac{CCA}', 'Elastic Net-\\ac{CCA}']

    #                     df.columns = columns
    #                     df.index = index
    #                     dfp.columns = columns
    #                     dfp.index = index

    #                     dfr.columns = columns
    #                     dfr.index = index
    #                     dfrp.columns = columns
    #                     dfrp.index = index

    #                     # Remove the elastic net cca index
    #                     df = df.drop('Elastic Net-\\ac{CCA}', axis=0)
    #                     dfp = dfp.drop('Elastic Net-\\ac{CCA}', axis=0)

    #                     dfr = dfr.drop('Elastic Net-\\ac{CCA}', axis=0)
    #                     dfrp = dfrp.drop('Elastic Net-\\ac{CCA}', axis=0)


    #                     # set the values past the diagonal to nan
    #                     for i in range(df.shape[0]):
    #                         for j in range(df.shape[1]):
    #                             if i >= j:
    #                                 df.iloc[i, j] = np.nan
    #                                 dfr.iloc[i, j] = np.nan

    #                     # Drop the first column
    #                     df = df.drop('Ridge-%s' % ridge_col, axis=1)
    #                     dfp = dfp.drop('Ridge-%s' % ridge_col, axis=1)

    #                     dfr = dfr.drop('Ridge-%s' % ridge_col, axis=1)
    #                     dfrp = dfrp.drop('Ridge-%s' % ridge_col, axis=1)


    #                     # Map function to check if the corrisponding p-value is significant
    #                     def check_significance(x, xp):
    #                         result = np.where(xp > 0.05 ,f"font-weight: bold;" , None)

    #                         # if result.all():
    #                         #     # set all values to None
    #                         #     result = np.where(xp < 0.05, None, None)

    #                         # # Set value to None if x is nan 
    #                         # result = np.where(x != x, None, result)
                            
    #                         return result
                        
    #                     s = df.style.apply(check_significance, xp=dfp, axis=None)
    #                     sr = dfr.style.apply(check_significance, xp=dfrp, axis=None)
                        
    #                     s.format('{:.1f}', na_rep=" ")
    #                     sr.format('{:.1f}', na_rep=" ")

    #                     # print(s.to_latex(column_format='l'*df.columns.size+'l', hrules=True, convert_css=True))
    #                     # print(sr.to_latex(column_format='l'*dfr.columns.size+'l', hrules=True, convert_css=True))

    #                     s.to_latex(buf='../feature_paper/figs/dstat/dstat-%s-%s-%s-%s-%s-rmse.tex' % (
    #                         otype, ts, dd, feature, mmodel), column_format='l'*df.columns.size+'l', hrules=True, convert_css=True) 
    #                     sr.to_latex(buf='../feature_paper/figs/r2cmp/r2cmp-%s-%s-%s-%s-%s.tex' % (
    #                         otype, ts, dd, feature, mmodel), column_format='l'*dfr.columns.size+'l', hrules=True, convert_css=True)

    #                     # df.to_latex(buf='./plots/%s_%s_%s_%s_rmse.tex' % (
    #                     #     otype, ts, dd, mmodel), header=True) 

    # # # XXX: Generate latex table for dmtest and rmse cmp acrross lags
    # for otype in ['call']:
    #     for dd in ['figs']:
    #         for models in ['pmridge', 'tskridge']:
    #             df = pd.read_csv('./plots/lag_dstat_%s_%s_%s.csv' % (
    #                     otype, dd, models))
    #             dfp = pd.read_csv('./plots/lag_dstat_pval_%s_%s_%s.csv' % (
    #                 otype, dd, models))

    #             dfr = pd.read_csv('./plots/lag_r2_%s_%s_%s.csv' % (
    #                 otype, dd, models))
    #             dfrp = pd.read_csv('./plots/lag_r2_pval_%s_%s_%s.csv' % (
    #                 otype, dd, models))

    #             # rename the column headers
    #             columns = ['Lag 20', 'Lag 10', 'Lag 5']
    #             index = ['Lag 20', 'Lag 10', 'Lag 5']

    #             df.columns = columns
    #             df.index = index
    #             dfp.columns = columns
    #             dfp.index = index

    #             dfr.columns = columns
    #             dfr.index = index
    #             dfrp.columns = columns
    #             dfrp.index = index

    #             # Remove lag 5 index
    #             df = df.drop('Lag 5', axis=0)
    #             dfp = dfp.drop('Lag 5', axis=0)

    #             dfr = dfr.drop('Lag 5', axis=0)
    #             dfrp = dfrp.drop('Lag 5', axis=0)


    #             # set the values past the diagonal to nan
    #             for i in range(df.shape[0]):
    #                 for j in range(df.shape[1]):
    #                     if i >= j:
    #                         df.iloc[i, j] = np.nan
    #                         dfr.iloc[i, j] = np.nan

    #             # Remove the lag 20 column
    #             df = df.drop('Lag 20', axis=1)
    #             dfp = dfp.drop('Lag 20', axis=1)

    #             dfr = dfr.drop('Lag 20', axis=1)
    #             dfrp = dfrp.drop('Lag 20', axis=1)

    #             # Map function to check if the corrisponding p-value is significant
    #             def check_significance(x, xp):
    #                 result = np.where(xp > 0.05 ,f"font-weight: bold;" , None)

    #                 # if result.all():
    #                 #     # set all values to None
    #                 #     result = np.where(xp < 0.05, None, None)

    #                 # # Set value to None if x is nan 
    #                 # result = np.where(x != x, None, result)
    
    #                 return result

    #             s = df.style.apply(check_significance, xp=dfp, axis=None)
    #             sr = dfr.style.apply(check_significance, xp=dfrp, axis=None)

    #             s.format('{:.1f}', na_rep=" ")
    #             sr.format('{:.1f}', na_rep=" ")

    #             # print(s.to_latex(column_format='l'*df.columns.size+'l', hrules=True, convert_css=True))
    #             # print(sr.to_latex(column_format='l'*dfr.columns.size+'l', hrules=True, convert_css=True))

    #             s.to_latex(buf='../feature_paper/figs/dstat/lag-dstat-%s-%s-%s-rmse.tex' % (
    #                 otype, dd, models), column_format='l'*df.columns.size+'l', hrules=True, convert_css=True) 
    #             sr.to_latex(buf='../feature_paper/figs/r2cmp/lag-r2cmp-%s-%s-%s.tex' % (
    #                 otype, dd, models), column_format='l'*dfr.columns.size+'l', hrules=True, convert_css=True)

                # df.to_latex(buf='./plots/%s_%s_%s_%s_rmse.tex' % (
                #     otype, ts, dd, mmodel), header=True)  

    # df.to_latex(buf='./plots/%s_%s_%s_%s_rmse.tex' % (
    #     otype, ts, dd, mmodel), header=True)

    # def setup_trade(model):
    #     for otype in ['call', 'put']:
    #         # XXX: Change or add to the loops as needed
    #         for dd in ['figs']:
    #             for ts in [5]:
    #                 name = ('./final_results/%s_%s_ts_%s_model_%s.npy.gz' %
    #                         (otype, dd, ts, model))
    #                 print('Doing model: ', name)
    #                 dates, y, yp = getpreds_trading(name, otype)
    #                 trade(dates, y, yp, otype, model)

    # # XXX: The trading part, only for the best model
    # models = ['ridge', 'ssviridge', 'plsridge', 'ctridge',
    #           'tskridge', 'tskplsridge', 'tsknsridge',
    #           'mskridge', 'msknsridge', 'mskplsridge',
    #           'pmridge', 'pmplsridge']
    # from joblib import Parallel, delayed
    # Parallel(n_jobs=-1)(delayed(setup_trade)(model)
    #                     for model in models)

    # for otype in ['call', 'put']:
    #     alphas = list()
    #     betas = list()
    #     cagrs = list()
    #     wins = list()
    #     maxs = list()
    #     mins = list()
    #     avgs = list()
    #     medians = list()
    #     sds = list()
    #     srs = list()
    #     ns = list()
    #     # XXX: Change or add to the loops as needed
    #     for dd in ['figs']:
    #         for ts in [5]:
    #             for model in models:
    #                 analyse_trades(otype, model, ts,
    #                                alphas, betas,
    #                                cagrs, wins,
    #                                maxs, mins,
    #                                avgs, medians, sds, srs, ns)
    #     df = pd.DataFrame({'alpha': alphas, 'beta': betas,
    #                        'cagr (%)': cagrs, 'win (%)': wins,
    #                        'max (%)': maxs, 'min (%)': mins,
    #                        'mean (%)': avgs, 'median (%)': medians,
    #                        'std (%)': sds, 'sharpe ratio': srs,
    #                        'N': ns},
    #                       index=models)
    #     df.to_csv('./trades/%s_trades.csv' % otype)
    #     df = pd.DataFrame(resa, index=[0])
    #     df.to_csv('./trades/alpha_%s.csv' % otype)
    #     df = pd.DataFrame(resb, index=[0])
    #     df.to_csv('./trades/beta_%s.csv' % otype)

    # XXX: The statistics for the complete dataset
    # cdfm = list()
    # t = list()
    # cdfiv = list()
    # pdft = list()
    # pdfm = list()
    # pdfiv = list()
    # for i, f in enumerate(sorted(glob.glob('./interest_rates/*.csv'))):
    #     df = pd.read_csv(f)
    #     dfc = df[df['Type'] == 'call']
    #     # XXX: Print statistics for call options
    #     cdfm.append(list(dfc['m'].values))
    #     cdft.append(list(dfc['tau'].values))
    #     cdfiv.append(list(dfc['IV'].values))
    #     # XXX: Print statistics for put options
    #     dfp = df[df['Type'] == 'put']
    #     pdft.append(list(dfp['tau'].values))
    #     pdfm.append(list(dfp['m'].values))
    #     pdfiv.append(list(dfp['IV'].values))

    # XXX: Flatten the lists
    # cdfm = [j for i in cdfm for j in i]
    # cdft = [j for i in cdft for j in i]
    # cdfiv = [j for i in cdfiv for j in i]
    # pdfm = [j for i in pdfm for j in i]
    # pdft = [j for i in pdft for j in i]
    # pdfiv = [j for i in pdfiv for j in i]

    # XXX: Turn into dataframe
    # cdft = pd.DataFrame(cdft)
    # cdfm = pd.DataFrame(cdfm)
    # cdfiv = pd.DataFrame(cdfiv)
    # cdf = pd.DataFrame({'Moneyness': cdfm, 'Time-to-maturity': cdft,
    #                     'Implied Volatility': cdfiv})
    # cdf = cdf[cdf['Implied Volatility'] > 0]
    # pdfm = pd.DataFrame(pdfm)
    # pdft = pd.DataFrame(pdft)
    # pdfiv = pd.DataFrame(pdfiv)
    # pdf = pd.DataFrame({'Moneyness': pdfm, 'Time-to-maturity': pdft,
    #                     'Implied Volatility': pdfiv})
    # pdf = pdf[pdf['Implied Volatility'] > 0]
    # XXX: Print the statistics
    # print('Call moneyness: ', cdfm.describe())
    # print('Call maturity: ', cdft.describe())
    # print('Call IV: ', cdfiv.describe())
    # print('Put moneyness: ', pdfm.describe())
    # print('Put maturity: ', pdft.describe())
    # print('Put IV: ', pdfiv.describe())
    # cdf.describe().T.to_latex(buf='../feature_paper/elsarticle/figs/callstats.tex',
    #                           header=['Count', 'Mean',
    #                                   'Std', 'Min', '25%', '50%', '75%',
    #                                   'Max'])
    # pdf.describe().T.to_latex(buf='../feature_paper/elsarticle/figs/putstats.tex',
    #                           header=['Count', 'Mean',
    #                                   'Std', 'Min', '25%', '50%', '75%',
    #                                   'Max'])

    # XXX: ITM/ATM/OTM
    # citm = cdf[cdf['Moneyness'] <= 0.99]
    # catm = cdf[((cdf['Moneyness'] > 0.99) & (cdf['Moneyness'] <= 1.01))]
    # cotm = cdf[cdf['Moneyness'] > 1.01]
    # print('Call ITM', citm.shape[0])
    # print('Call ATM', catm.shape[0])
    # print('Call OTM', cotm.shape[0])

    # XXX: Term structure
    # cst = cdf[cdf['Time-to-maturity'] < 90/365]
    # cmt = cdf[((cdf['Time-to-maturity'] >= 90/365) &
    #            (cdf['Time-to-maturity'] < 180/365))]
    # clt = cdf[cdf['Time-to-maturity'] >= 180/365]
    # print('Call ST', cst.shape[0])
    # print('Call MT', cmt.shape[0])
    # print('Call LT', clt.shape[0])

    # XXX: ITM/ATM/OTM
    # pitm = pdf[pdf['Moneyness'] <= 0.99]
    # patm = pdf[((pdf['Moneyness'] > 0.99) & (pdf['Moneyness'] <= 1.01))]
    # potm = pdf[pdf['Moneyness'] > 1.01]
    # print('Put ITM', pitm.shape[0])
    # print('Put ATM', patm.shape[0])
    # print('Put OTM', potm.shape[0])

    # XXX: Term structure
    # pst = pdf[pdf['Time-to-maturity'] < 90/365]
    # pmt = pdf[((pdf['Time-to-maturity'] >= 90/365) &
    #            (pdf['Time-to-maturity'] < 180/365))]
    # plt = pdf[pdf['Time-to-maturity'] >= 180/365]
    # print('Put ST', pst.shape[0])
    # print('Put MT', pmt.shape[0])
    # print('Put LT', plt.shape[0])
