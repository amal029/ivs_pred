#!/usr/bin/env python

import keras
import pred
import numpy as np
import matplotlib.pyplot as plt
import glob
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score
import dmtest
# from mpl_toolkits.mplot3d import Axes3D
import gzip
import pandas as pd


def date_to_num(date, dd='./figs'):
    ff = sorted(glob.glob(dd+'/*.npy'))
    count = 0
    for i in ff:
        if i.split('/')[-1].split('.')[0] == date:
            break
        count += 1
    return count


def num_to_date(num, dd='./figs'):
    ff = sorted(glob.glob(dd+'/*.npy'))
    return ff[num].split('/')[-1].split('.')[0]


def load_image(num, dd='./figs'):
    ff = sorted(glob.glob(dd+'/*.npy'))
    img = np.load(ff[num])
    return img


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

def extract_features(X, t, model, dd, TSTEPS, feature_res, type='mskew'):
    """
    Returns the features extracted from the input data ready for prediction
    using corrisponding model
    """

    import feature_extraction as fe
    transform_type = type[1:]
    if(model == 'pca'):
        X = fe.pca_transform(X, feature_res, TSTEPS=TSTEPS, type=transform_type)
    elif(model == 'autoencoder'):
        X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])

        # Load the encoder model
        if dd == './figs':
            toopen = './%s_feature_models/%s_ts_%s_%s_encoder.keras' % (type, model, TSTEPS, t)
        else:
            toopen = './%s_feature_models/%s_ts_%s_%s_encoder_gfigs.keras' % (type, model, TSTEPS, t)

        encoder = keras.saving.load_model(toopen) 
        # Get encoder
        encoder = keras.Model(inputs=encoder.input, outputs=encoder.layers[1].output)
        X = encoder.predict(X)
    else:
        X = fe.har_transform(X, TSTEPS=TSTEPS, type=transform_type)
    return X

def main(dd='./figs', model='Ridge', plot=True, TSTEPS=5, NIMAGES=1000,
        get_features=False, feature_res=10):
    # Har feature extraction model required 32 lags
    if (model == 'mskhar' or model == 'tskhar' or model == 'pmkhar') and TSTEPS != 32:
        feature_res = 3
        TSTEPS = 32

    # XXX: Num 3000 == '20140109'
    START_DATE = '20140109'
    END_DATE = '20221230'

    START = date_to_num(START_DATE)
    END = date_to_num(END_DATE) - TSTEPS
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
    elif pred.TSTEPS == 32: # Exception for har feature extraction
        bs = 10
    else:
        raise Exception("TSTEPS not correct")

    nf = 64

    valX, valY, Ydates = pred.load_data_for_keras(START=START, dd=dd,
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
    MONEYNESS = [0, MS//2, MS-1]
    # XXX: Some terms
    TERM = [0, TS//2, TS-1]

    # XXX: Clean the data
    if dd == './gfigs':
        valX, valY = pred.clean_data(valX, valY)

    # XXX: Make a prediction for images
    if model == 'keras':
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

    elif model == 'pmridge' or model == 'pmlasso' or model == 'pmenet':
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
                    with open('./point_models/%s_ts_%s_%s_%s.pkl' %
                              (model, TSTEPS, s, t), 'rb') as f:
                        m1 = pickle.load(f)
                else:
                    with open('./point_models/%s_ts_%s_%s_%s_gfigs.pkl' %
                              (model, TSTEPS, s, t), 'rb') as f:
                        m1 = pickle.load(f)
                # XXX: Now make the prediction
                k = np.array([s, t]*valX.shape[0]).reshape(valX.shape[0], 2)
                val_vec = np.append(valX[:, :, i, j], k, axis=1)
                out[:, i, j] = m1.predict(val_vec)

                # XXX: Feature vector plot
                if get_features and model == 'pmridge':
                    if i in MONEYNESS and j in TERM:
                        fig, ax = plt.subplots()
                        xaxis = ['t-%s' % (i+1) for i in (range(TSTEPS))[::-1]]
                        xaxis.append(r'$\mu$')
                        xaxis.append(r'$\tau$')
                        ax.bar(xaxis, m1.coef_, color='b')
                        ax.set_ylabel('Coefficient magnitudes')
                        plt.xticks(fontsize=9, rotation=45)
                        dfname = dd.split('/')[1]
                        fname = './plots/%s_m_%s_t_%s_lags_%s_%s.pdf' % (
                            model, s, t, TSTEPS, dfname)
                        plt.savefig(fname, bbox_inches='tight')
                        plt.close(fig)
        if not plot:
            out = out.reshape(out.shape[0], out.shape[1]*out.shape[2])
            valY = valY.reshape(valY.shape[0], valY.shape[1]*valY.shape[2])

    elif model == 'tskenet' or model == 'tskridge' or model == 'tsklasso':
        # XXX: The output vector
        out = np.array([0.0]*(valY.shape[0]*valY.shape[1]*valY.shape[2]))
        out = out.reshape(*valY.shape)
        mms = np.arange(pred.LM, pred.UM+pred.MSTEP, pred.MSTEP)
        # XXX: Go through the MS
        import pickle
        for j, m in enumerate(mms):
            if dd == './figs':
                with open('./tskew_models/%s_ts_%s_%s.pkl' %
                          (model, TSTEPS, m), 'rb') as f:
                    m1 = pickle.load(f)
            else:
                with open('./tskew_models/%s_ts_%s_%s_gfigs.pkl' %
                          (model, TSTEPS, m), 'rb') as f:
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
                    markers = [(3+i, 1, 0) for i in range(TSTEPS)]
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
                            ax.plot(X, ws[i], marker=markers[i],
                                    label=labels[i], markevery=0.1)
                        ax.set_ylabel('Coefficient magnitudes')
                        ax.set_xlabel('Term structure')
                        ax.legend(ncol=3)
                        dfname = dd.split('/')[1]
                        fname = './plots/%s_m_%s_t_%s_lags_%s_%s.pdf' % (
                            model, m, X[mts], TSTEPS, dfname)
                        plt.savefig(fname, bbox_inches='tight')
                        plt.close(fig)

        if not plot:
            out = out.reshape(out.shape[0], out.shape[1]*out.shape[2])
            valY = valY.reshape(valY.shape[0], valY.shape[1]*valY.shape[2])
    elif model == 'mskautoencoder' or model == 'mskpca' or model == 'mskhar':
        import feature_extraction as fe
        # XXX: The output vector
        out = np.array([0.0]*(valY.shape[0]*valY.shape[1]*valY.shape[2]))
        out = out.reshape(*valY.shape)
        mms = np.arange(pred.LM, pred.UM+pred.MSTEP, pred.MSTEP)
        # XXX: Now go through the TS
        tts = [i/pred.DAYS for i in range(pred.LT, pred.UT+pred.TSTEP,
                                          pred.TSTEP)]
        import pickle
        model_name = model[3:]
        for j, t in enumerate(tts):
            if dd == './figs':
                with open('./mskew_feature_models/%s_ts_%s_%s.pkl' %
                          (model_name, TSTEPS, t), 'rb') as f:
                    m1 = pickle.load(f)
            else:
                with open('./mskew_feature_models/%s_ts_%s_%s_gfigs.pkl' %
                          (model_name, TSTEPS, t), 'rb') as f:
                    m1 = pickle.load(f)
                    
            mskew = valX[:, :, :, j]

            # Extract features before prediction
            mskew = extract_features(mskew, t, model_name, dd, TSTEPS, feature_res, type='mskew')

            out[:, :, j] = m1.predict(mskew)

            if get_features:
                if j in TERM:
                    X = np.arange(pred.LM, pred.UM+pred.MSTEP, pred.MSTEP)
                    labels = ['t-%s' % (i+1) for i in range(feature_res)[::-1]]
                    markers = [(3+i, 1, 0) for i in range(feature_res)]
                    for mts in MONEYNESS:
                        # XXX: The term structure weights
                        ws = m1.coef_[mts][:].reshape(feature_res, MS)
                        # # # XXX: The term structure weight
                        # wms = m1.coef_[mts][-1]
                        # # XXX: Make sure that the term structure weight=nil
                        # assert (abs(wms) < 1e-4)
                        # XXX: Now just plot the 2d curves
                        fig, ax = plt.subplots()
                        for i in range(feature_res):
                            ax.plot(X, ws[i], marker=markers[i],
                                    label=labels[i], markevery=0.1)
                        ax.set_ylabel('Coefficient magnitudes')
                        ax.set_xlabel('Moneyness')
                        ax.legend(ncol=3)
                        dfname = dd.split('/')[1]
                        fname = './plots/%s_m_%s_t_%s_lags_%s_%s.pdf' % (
                            model, X[mts], t, TSTEPS, dfname)
                        plt.savefig(fname, bbox_inches='tight')
                        plt.close(fig)
        if not plot:
            out = out.reshape(out.shape[0], out.shape[1]*out.shape[2])
            valY = valY.reshape(valY.shape[0], valY.shape[1]*valY.shape[2])



    elif model == 'tskautoencoder' or model == 'tskpca' or model == 'tskhar':
        import feature_extraction as fe
        # XXX: The output vector
        out = np.array([0.0]*(valY.shape[0]*valY.shape[1]*valY.shape[2]))
        out = out.reshape(*valY.shape)
        mms = np.arange(pred.LM, pred.UM+pred.MSTEP, pred.MSTEP)
        # XXX: Go through the MS
        import pickle
        model_name = model[3:]
        for j, m in enumerate(mms):
            if dd == './figs':
                with open('./tskew_feature_models/%s_ts_%s_%s.pkl' %
                          (model_name, TSTEPS, m), 'rb') as f:
                    m1 = pickle.load(f)
            else:
                with open('./tskew_feature_models/%s_ts_%s_%s_gfigs.pkl' %
                          (model_name, TSTEPS, m), 'rb') as f:
                    m1 = pickle.load(f)
            tskew = valX[:, :, j]
            
            # Extract features before prediction
            tskew = extract_features(tskew, m, model_name, dd, TSTEPS, feature_res, type='tskew')

            # XXX: Predict the output
            out[:, j] = m1.predict(tskew)

            # XXX: Features plot
            if get_features:
                if j in MONEYNESS:
                    X = [i/pred.DAYS
                         for i in range(pred.LT, pred.UT+pred.TSTEP,
                                        pred.TSTEP)]
                    labels = ['t-%s' % (i+1) for i in range(feature_res)[::-1]]
                    markers = [(3+i, 1, 0) for i in range(feature_res)]
                    for mts in TERM:
                        # XXX: The term structure weights
                        ws = m1.coef_[mts][:].reshape(feature_res, TS)
                        # # XXX: The moneyness weight
                        # wms = m1.coef_[mts][-1]
                        # # XXX: Make sure that the moneyness weight is nothing
                        # assert (abs(wms) < 1e-4)
                        # XXX: Now just plot the 2d curves
                        fig, ax = plt.subplots()
                        for i in range(feature_res):
                            ax.plot(X, ws[i], marker=markers[i],
                                    label=labels[i], markevery=0.1)
                        ax.set_ylabel('Coefficient magnitudes')
                        ax.set_xlabel('Term structure')
                        ax.legend(ncol=3)
                        dfname = dd.split('/')[1]
                        fname = './plots/%s_m_%s_t_%s_lags_%s_%s.pdf' % (
                            model, m, X[mts], TSTEPS, dfname)
                        plt.savefig(fname, bbox_inches='tight')
                        plt.close(fig)

        if not plot:
            out = out.reshape(out.shape[0], out.shape[1]*out.shape[2])
            valY = valY.reshape(valY.shape[0], valY.shape[1]*valY.shape[2])

    elif model == 'pmpca' or model == 'pmhar':
        import feature_extraction as fe
        # XXX: The output vector
        out = np.array([0.0]*(valY.shape[0]*valY.shape[1]*valY.shape[2]))
        out = out.reshape(*valY.shape)
        # XXX: Now go through the MS and TS
        mms = np.arange(pred.LM, pred.UM+pred.MSTEP, pred.MSTEP)
        tts = [i/pred.DAYS
               for i in range(pred.LT, pred.UT+pred.TSTEP, pred.TSTEP)]
        import pickle
        model_name = model[2:]
        for i, s in enumerate(mms):
            for j, t in enumerate(tts):
                if dd == './figs':
                    with open('./point_feature_models/%s_ts_%s_%s_%s.pkl' %
                              (model_name, TSTEPS, s, t), 'rb') as f:
                        m1 = pickle.load(f)
                else:
                    with open('./point_feature_models/%s_ts_%s_%s_%s_gfigs.pkl' %
                              (model_name, TSTEPS, s, t), 'rb') as f:
                        m1 = pickle.load(f)
                # XXX: Now make the prediction
                k = np.array([s, t]*valX.shape[0]).reshape(valX.shape[0], 2)
                val_vec = np.append(valX[:, :, i, j], k, axis=1)
                # Extract features before prediction
                val_vec = extract_features(val_vec, t, model_name, dd, TSTEPS, feature_res, type='point')
                out[:, i, j] = m1.predict(val_vec)

                # XXX: Feature vector plot
                if get_features and model == 'pmridge':
                    if i in MONEYNESS and j in TERM:
                        fig, ax = plt.subplots()
                        xaxis = ['t-%s' % (i+1) for i in (range(feature_res))[::-1]]
                        xaxis.append(r'$\mu$')
                        xaxis.append(r'$\tau$')
                        ax.bar(xaxis, m1.coef_, color='b')
                        ax.set_ylabel('Coefficient magnitudes')
                        plt.xticks(fontsize=9, rotation=45)
                        dfname = dd.split('/')[1]
                        fname = './plots/%s_m_%s_t_%s_lags_%s_%s.pdf' % (
                            model, s, t, TSTEPS, dfname)
                        plt.savefig(fname, bbox_inches='tight')
                        plt.close(fig)
        if not plot:
            out = out.reshape(out.shape[0], out.shape[1]*out.shape[2])
            valY = valY.reshape(valY.shape[0], valY.shape[1]*valY.shape[2])

    elif model == 'mskenet' or model == 'mskridge' or model == 'msklasso':
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
                with open('./mskew_models/%s_ts_%s_%s.pkl' %
                          (model, TSTEPS, t), 'rb') as f:
                    m1 = pickle.load(f)
            else:
                with open('./mskew_models/%s_ts_%s_%s_gfigs.pkl' %
                          (model, TSTEPS, t), 'rb') as f:
                    m1 = pickle.load(f)
            mskew = valX[:, :, :, j]
            mskew = mskew.reshape(mskew.shape[0],
                                  mskew.shape[1]*mskew.shape[2])
            # XXX: Add t to the sample set
            ts = np.array([t]*mskew.shape[0]).reshape(mskew.shape[0], 1)
            mskew = np.append(mskew, ts, axis=1)
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
                        fname = './plots/%s_m_%s_t_%s_lags_%s_%s.pdf' % (
                            model, X[mts], t, TSTEPS, dfname)
                        plt.savefig(fname, bbox_inches='tight')
                        plt.close(fig)
        if not plot:
            out = out.reshape(out.shape[0], out.shape[1]*out.shape[2])
            valY = valY.reshape(valY.shape[0], valY.shape[1]*valY.shape[2])

    else:
        import pickle
        valX = valX.reshape(valX.shape[0],
                            valX.shape[1]*valX.shape[2]*valX.shape[3])
        if not plot:
            valY = valY.reshape(valY.shape[0], valY.shape[1]*valY.shape[2])

        # XXX: Load the model and then carry it out
        if dd == './gfigs':
            toopen = r"./surf_models/model_%s_ts_%s_gfigs.pkl" % (
                model.lower(), pred.TSTEPS)
        else:
            toopen = r"./surf_models/model_%s_ts_%s.pkl" % (
                model.lower(), pred.TSTEPS)

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
                            name = './plots/%s_m_%s_t_%s_%s_fm.pdf' % (
                                model, X[i], Y[j], TSTEPS)
                        else:
                            name = './plots/%s_m_%s_t_%s_%s_gfigs_fm.pdf' % (
                                model, X[i], Y[j], TSTEPS)
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


def save_results(models, fp, cache):
    # XXX: f = gzip.GzipFile('file.npy.gz', "r"); np.load(f) -- to read
    # XXX: Save all the dates and outputs
    for dd in ['./figs', './gfigs']:
        for t in fp.keys():
            for m in models:
                ddf = dd.split('/')[1]
                tosave = './final_results/%s_ts_%s_model_%s.npy.gz' % (ddf,
                                                                       t, m)
                dates, y, o = cache[dd][t][m]
                dates = dates.reshape(y.shape[0], 1)
                res = np.append(dates, y, axis=1)
                res = np.append(res, o, axis=1)
                with gzip.open(filename=tosave, mode='wb',
                               compresslevel=6) as f:
                    np.save(file=f, arr=res)


def getpreds(name1):
    mms = np.arange(pred.LM, pred.UM+pred.MSTEP, pred.MSTEP)
    # XXX: Now go through the TS
    tts = [i/pred.DAYS for i in range(pred.LT, pred.UT+pred.TSTEP,
                                      pred.TSTEP)]
    MS = len(mms)
    TS = len(tts)

    with gzip.open(filename=name1, mode='rb') as f:
        data1 = np.load(f)

    y = data1[:, 1:MS*TS+1].astype(float, copy=False)
    yp = data1[:, MS*TS+1:].astype(float, copy=False)

    # XXX: Across time
    return np.transpose(y), np.transpose(yp)


def rmse_r2_time_series(fname, ax1, ax2, mm, m1, em):
    mms = np.arange(pred.LM, pred.UM+pred.MSTEP, pred.MSTEP)
    # XXX: Now go through the TS
    tts = [i/pred.DAYS for i in range(pred.LT, pred.UT+pred.TSTEP,
                                      pred.TSTEP)]
    MS = len(mms)
    TS = len(tts)

    figss = fname.split('/')[2].split('_')[0]
    mname = fname.split('/')[2].split('_')[4].split('.')[0]
    mlags = fname.split('/')[2].split('_')[2]

    print('Doing model: %s %s, Lags: %s' % (figss, mname, mlags))

    with gzip.open(filename=fname, mode='rb') as f:
        data = np.load(f)

    # XXX: Attach the date time for each y and yp
    date = pd.to_datetime(data[:, 0], format='%Y%m%d')
    date = date.date
    y = data[:, 1:MS*TS+1].astype(float, copy=False)
    yp = data[:, MS*TS+1:].astype(float, copy=False)
    print(date.shape, y.shape, yp.shape)

    # XXX: Get the rolling RMSE
    rmses = root_mean_squared_error(np.transpose(y), np.transpose(yp),
                                    multioutput='raw_values')
    r2cs = r2_score(np.transpose(y), np.transpose(yp),
                    multioutput='raw_values')

    ax1.plot(rmses, label='%s' % mname, marker=m1, markevery=em,
             linewidth=0.6)
    ax2.plot(r2cs, label='%s' % mname, marker=m1, markevery=em,
             linewidth=0.6)
    return date


def overall(fname):
    mms = np.arange(pred.LM, pred.UM+pred.MSTEP, pred.MSTEP)
    # XXX: Now go through the TS
    tts = [i/pred.DAYS for i in range(pred.LT, pred.UT+pred.TSTEP,
                                      pred.TSTEP)]
    MS = len(mms)
    TS = len(tts)

    figss = fname.split('/')[2].split('_')[0]
    mname = fname.split('/')[2].split('_')[4].split('.')[0]
    mlags = fname.split('/')[2].split('_')[2]

    print('Doing model: %s %s, Lags: %s' % (figss, mname, mlags))

    with gzip.open(filename=fname, mode='rb') as f:
        data = np.load(f)

    # XXX: Attach the date time for each y and yp
    date = pd.to_datetime(data[:, 0], format='%Y%m%d')
    y = data[:, 1:MS*TS+1].astype(float, copy=False)
    yp = data[:, MS*TS+1:].astype(float, copy=False)
    print(date.shape, y.shape, yp.shape)

    # XXX: Get the rolling RMSE
    rmses = root_mean_squared_error(np.transpose(y), np.transpose(yp),
                                    multioutput='raw_values')
    # mapes = mean_absolute_percentage_error(y, yp, multioutput='raw_values')
    r2cs = r2_score(np.transpose(y), np.transpose(yp),
                    multioutput='raw_values')

    return (np.mean(rmses), np.std(rmses), np.mean(r2cs), np.std(r2cs))


def model_v_model():
    TTS = [20, 10, 5]
    # models = ['ridge', 'lasso',  # 'rf',
    #           'enet',  # 'keras',
    #           'pmridge', 'pmlasso', 'pmenet', 'mskridge',
    #           'msklasso', 'mskenet', 'tskridge', 'tsklasso', 'tskenet']
    models = [
              'mskridge', 'mskautoencoder', 'mskpca',
              'tskridge', 'tskautoencoder', 'tskpca', 
              'pmridge', 'pmpca'
              ]
    fp = {t: np.array([0.0]*len(models)*len(models)).reshape(len(models),
                                                             len(models))
          for t in TTS}
    fd = {t: np.array([0.0]*len(models)*len(models)).reshape(len(models),
                                                             len(models))
          for t in TTS}

    cache = {i: {j: {k: None for k in models} for j in TTS}
             for i in ['./figs', './gfigs']}


    for dd in ['./figs', './gfigs']:
        for t in fp.keys():
            for i in range(0, len(models)-1):
                print('Comparing models with: ', dd, t, models[i])
                for j in range(i+1, len(models)):
                    feature_res = t//2
                    if cache[dd][t][models[i]] is None:
                        dates, y, yp = main(plot=False, TSTEPS=t,
                                            model=models[i],
                                            get_features=True,
                                            dd=dd,
                                            feature_res=feature_res)
                        cache[dd][t][models[i]] = (dates, y, yp)
                    else:
                        _, y, yp = cache[dd][t][models[i]]
                    if cache[dd][t][models[j]] is None:
                        dates, yk, ypk = main(plot=False, TSTEPS=t,
                                              model=models[j],
                                              get_features=True,
                                              dd=dd,
                                              feature_res=feature_res)
                        cache[dd][t][models[j]] = (dates, yk, ypk)
                    else:
                        _, yk, ypk = cache[dd][t][models[j]]
                    assert (np.array_equal(y, yk))
                    # XXX: Now we can do Diebold mariano test
                    try:
                        dstat, pval = dmtest.dm_test(y, yp, ypk)
                    except dmtest.ZeroVarianceException:
                        dstat, pval = np.nan, np.nan
                    fd[t][i][j] = dstat
                    fp[t][i][j] = pval
        # XXX: Save the results
        header = ','.join(models)
        for t in fp.keys():
            np.savetxt('pval_%s_%s.csv' % (t, dd.split('/')[1]), fp[t],
                       delimiter=',', header=header)
            np.savetxt('dstat_%s_%s.csv' % (t, dd.split('/')[1]), fd[t],
                       delimiter=',', header=header)

    # XXX: Save all the results
    print('Saving results')
    save_results(models, fp, cache)


def call_dmtest():
    TTS = [20, 10, 5]
    models = ['ridge', 'lasso', 'enet',
              'pmridge', 'pmlasso', 'pmenet', 'mskridge',
              'msklasso', 'mskenet', 'tskridge', 'tsklasso', 'tskenet']
    fp = {t: np.array([0.0]*len(models)*len(models)).reshape(len(models),
                                                             len(models))
          for t in TTS}
    fd = {t: np.array([0.0]*len(models)*len(models)).reshape(len(models),
                                                             len(models))
          for t in TTS}
    cache = {i: {j: {k: None for k in models} for j in TTS}
             for i in ['figs', 'gfigs']}
    for dd in ['figs', 'gfigs']:
        for ts in TTS:
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
            for i in range(len(models)-1):
                name1 = ('./final_results/%s_ts_%s_model_%s.npy.gz' %
                         (dd, ts, models[i]))
                print('Doing %d: %s' % (i, name1))
                if cache[dd][ts][models[i]] is None:
                    y, yp = getpreds(name1)
                    cache[dd][ts][models[i]] = (y, yp)
                else:
                    y, yp = cache[dd][ts][models[i]]
                for j in range(i+1, len(models)):
                    # XXX: Do only the best ones
                    name2 = ('./final_results/%s_ts_%s_model_%s.npy.gz' %
                             (dd, ts, models[j]))
                    print('Doing %d: %s' % (j, name2))
                    if cache[dd][ts][models[j]] is None:
                        yk, ypk = getpreds(name2)
                        cache[dd][ts][models[j]] = (yk, ypk)
                    else:
                        yk, ypk = cache[dd][ts][models[j]]
                    assert (np.array_equal(y, yk))
                    # XXX: Now we can do Diebold mariano test
                    try:
                        dstat, pval = dmtest.dm_test(y, yp, ypk)
                    except dmtest.ZeroVarianceException:
                        dstat, pval = np.nan, np.nan
                    fd[ts][i][j] = dstat
                    fp[ts][i][j] = pval
        # XXX: Save the results of dmtest
        header = ','.join(models)
        for t in fp.keys():
            np.savetxt('./plots/pval_%s_%s_rmse.csv' % (t, dd), fp[t],
                       delimiter=',', header=header)
            np.savetxt('./plots/dstat_%s_%s_rmse.csv' % (t, dd), fd[t],
                       delimiter=',', header=header)


def call_timeseries():
    # XXX: This is many models vs many other models
    # model_v_model()
    m1 = ['*', 'P', 'd', '8']
    for dd in ['figs', 'gfigs']:
        for ts in [5, 10, 20]:
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
            for i, model in enumerate(['pmridge', 'ridge', 'tskridge',
                                       'mskridge']):
                # XXX: Do only the best ones
                name = './final_results/%s_ts_%s_model_%s.npy.gz' % (dd, ts,
                                                                     model)
                dates = rmse_r2_time_series(name, ax1, ax2, model, m1[i],
                                            em=0.999)
            EVERY = 400
            ax2.set_xticks(range(0, dates.shape[0], EVERY),
                           labels=[dates[i]
                                   for i in range(0, dates.shape[0], EVERY)])
            ax2.set_xlabel('Dates')
            ax1.set_ylabel('RMSE')
            ax2.set_ylabel(r'$R^2$')
            ax2.legend()
            plt.xticks(fontsize=9, rotation=40)
            plt.savefig('./plots/%s_rmse_r2_time_series_best_models_%s.pdf'
                        % (dd, ts))
            plt.close(fig)


def call_overall():
    # XXX: Now plot the overall RMSE, MAPE, and R2 for all models
    # average across all results.
    TTS = [20, 10, 5]
    models = ['ridge', 'lasso', 'enet', 'pmridge', 'pmlasso',
              'pmenet', 'mskridge', 'msklasso', 'mskenet', 'tskridge',
              'tsklasso', 'tskenet']

    # XXX: Do the overall RMSE, MAPE, and R2
    for dd in ['figs', 'gfigs']:
        for ts in TTS:
            rmsemeans = []
            # mapemeans = []
            r2means = []
            rmsestds = []
            # mapestds = []
            r2stds = []
            for i, model in enumerate(models):
                name = './final_results/%s_ts_%s_model_%s.npy.gz' % (dd, ts,
                                                                     model)
                rmsem, rmsestd, r2m, r2std = overall(name)
                # XXX: MEAN
                rmsemeans.append(rmsem)
                # mapemeans.append(mapesm)
                r2means.append(r2m)
                # XXX: STD
                rmsestds.append(rmsestd)
                # mapestds.append(mapestd)
                r2stds.append(r2std)

            # XXX: plot means
            fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
            bar0 = axs[0].bar(models, rmsemeans, width=0.2)
            axs[0].bar_label(bar0, fmt='%3.3f')
            # bar1 = axs[1].bar(models, mapemeans, width=0.2, color='r')
            # if dd != 'gfigs':
            #     axs[1].bar_label(bar1, fmt='%3.3f')
            bar2 = axs[1].bar(models, r2means*100, width=0.2, color='g')
            axs[1].bar_label(bar2, fmt='%3.3f')
            axs[0].set_ylabel('RMSE (avg)')
            # axs[1].set_ylabel('MAPE (avg)')
            axs[1].set_ylabel(r'$R^2$ (avg)')
            plt.xticks(fontsize=9, rotation=45)
            plt.savefig('./plots/%s_rmse_r2_avg_models_%s.pdf'
                        % (dd, ts))
            plt.close(fig)

            # XXX: Plot the std deviation
            fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
            bar0 = axs[0].bar(models, rmsestds, width=0.2)
            axs[0].bar_label(bar0, fmt='%3.3f')
            # bar1 = axs[1].bar(models, mapestds, width=0.2, color='r')
            # axs[1].bar_label(bar1, fmt='%3.3f')
            bar2 = axs[1].bar(models, r2stds, width=0.2, color='g')
            axs[1].bar_label(bar2, fmt='%3.3f')
            axs[0].set_ylabel('RMSE (std-dev)')
            # axs[1].set_ylabel('MAPE (std-dev)')
            axs[1].set_ylabel(r'$R^2$ (std-dev)')
            plt.xticks(fontsize=9, rotation=45)
            plt.savefig('./plots/%s_rmse_r2_std_models_%s.pdf'
                        % (dd, ts))
            plt.close(fig)


if __name__ == '__main__':
    plt.style.use('seaborn-v0_8-whitegrid')
    # XXX: Plot the best time series RMSE and MAPE
    # call_timeseries()

    # XXX: Plot the bar graph for overall results
    # call_overall()
    model_v_model()

    # XXX: DM test across time (RMSE and R2)
    # call_dmtest()
