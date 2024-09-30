#!/usr/bin/env python

import pandas as pd
import os
import fnmatch
import zipfile as zip
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from statsmodels.nonparametric.kernel_regression import KernelReg
# import tensorflow.math as K
# import tensorflow as tf
import xgboost as xgb
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import glob
from keras.layers import Input, ConvLSTM2D, BatchNormalization, Conv2D
from keras.models import Model
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
import keras
# from PIL import Image
# XXX: For plotting only
import matplotlib.pyplot as plt
# from sklearn.linear_model import SGDRegressor

# XXX: Moneyness Bounds inclusive
LM = 0.9
UM = 1.1
MSTEP = 0.00333

# XXX: Tau Bounds inclusive
LT = 14
UT = 366
TSTEP = 5                       # days

DAYS = 365


def preprocess_ivs_df(dfs: dict, otype):
    toret = dict()
    for k in dfs.keys():
        df = dfs[k]
        # XXX: First only get those that have volume > 0
        df = df[(df['Volume'] > 0) &
                (df['Type'] == otype)].reset_index(drop=True)
        # XXX: Make the log of K/UnderlyingPrice
        df['m'] = (df['Strike']/df['UnderlyingPrice'])
        # XXX: Moneyness is not too far away from ATM
        df = df[(df['m'] >= LM) & (df['m'] <= UM)]
        # XXX: Make the days to expiration
        df['Expiration'] = pd.to_datetime(df['Expiration'])
        df['DataDate'] = pd.to_datetime(df['DataDate'])
        df['tau'] = (df['Expiration'] - df['DataDate']).dt.days
        # XXX: Only those that are greater than at least 2 weeks ahead
        # and also not too ahead
        df = df[(df['tau'] >= LT) & (df['tau'] <= UT)]
        df['tau'] = df['tau']/DAYS
        df['m2'] = df['m']**2
        df['tau2'] = df['tau']**2
        df['mtau'] = df['m']*df['tau']

        # XXX: This is the final dataframe
        dff = df[['IV', 'm', 'tau', 'm2', 'tau2', 'mtau']]
        toret[k] = dff.reset_index(drop=True)
    return toret


def plot_hmap(ivs_hmap, mrows, tcols, otype, dd='figs'):
    for k in ivs_hmap.keys():
        np.save('/tmp/%s_%s/%s.npy' % (otype, dd, k), ivs_hmap[k])


def plot_ivs(ivs_surface, IVS='IVS', view='XY'):
    for k in ivs_surface.keys():
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        X = ivs_surface[k]['m']
        Y = ivs_surface[k]['tau']
        if IVS == 'IVS':
            Z = ivs_surface[k][IVS]*100
        else:
            Z = ivs_surface[k][IVS]
            # viridis = cm.get_cmap('gist_gray', 256)
        _ = ax.plot_trisurf(X, Y, Z, cmap='afmhot',
                            linewidth=0.2, antialiased=True)
        # ax.set_xlabel('m')
        # ax.set_ylabel('tau')
        # ax.set_zlabel(IVS)
        # ax.view_init(azim=-45, elev=30)
        # ax.invert_xaxis()
        # ax.invert_yaxis()
        if view == 'XY':
            ax.view_init(elev=90, azim=-90)
        elif view == 'XZ':
            ax.view_init(elev=0, azim=-90)
        elif view == 'YZ':
            ax.view_init(elev=0, azim=0)
            ax.axis('off')
            # ax.zaxis.set_major_formatter('{x:.02f}')
            # fig.colorbar(surf, shrink=0.5, aspect=5)
            # plt.show()
            # fig.subplots_adjust(bottom=0)
            # fig.subplots_adjust(top=0.00001)
            # fig.subplots_adjust(right=1)
            # fig.subplots_adjust(left=0)
        plt.savefig('/tmp/figs/{k}_{v}.png'.format(k=k, v=view),
                    bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        # XXX: Convert to gray scale 1 channel only
        # img = Image.open('/tmp/figs/{k}.png'.format(k=k)).convert('LA')
        # img.save('/tmp/figs/{k}.png'.format(k=k))


def load_data_for_keras(otype, dd='./figs', START=0, NUM_IMAGES=1000, TSTEP=1):

    Xs = list()               # Training inputs [0..TEST_IMAGES-1]
    Ys = list()               # Training outputs [1..TEST_IMAGES]
    Ysdates = list()
    ff = sorted(glob.glob('./'+otype+'_'+dd.split('/')[1]+'/*.npy'))
    # XXX: Load the first TEST_IMAGES for training
    # print('In load image!')
    for i in range(START, START+NUM_IMAGES):
        # print('i is: ', i)
        for j in range(TSTEP):
            # print('j is: ', j)
            img = np.load(ff[i+j])   # PIL image
            np.all((img > 0) & (img <= 1))
            # print('loaded i, j: X(i+j)', i, j,
            #       ff[i+j].split('/')[-1].split('.')[0])
            Xs += [img]

        # XXX: Just one output image to compare against
        # XXX: Now do the same thing for the output label image
        # img = Image.open(ff[i+TSTEP]).convert('LA')
        img = np.load(ff[i+TSTEP])   # PIL image
        np.all((img > 0) & (img <= 1))
        # print('loaded Y: i, TSTEP: (i+TSTEP)', i, TSTEP,
        #       ff[(i+TSTEP)].split('/')[-1].split('_')[0])
        Ysdates.append(ff[(i+TSTEP)].split('/')[-1].split('.')[0])
        Ys += [img]

    # XXX: Convert the lists to np.array
    Xs = np.array(Xs)
    Ys = np.array(Ys)
    np.expand_dims(Xs, axis=-1)
    np.expand_dims(Ys, axis=-1)
    return Xs, Ys, Ysdates


def load_image_for_keras(dd='./figs', START=0, NUM_IMAGES=1000, NORM=255,
                         RESIZE_FACTOR=8, TSTEP=1):

    Xs = list()               # Training inputs [0..TEST_IMAGES-1]
    Ys = list()               # Training outputs [1..TEST_IMAGES]
    Ysdates = list()
    ff = sorted(glob.glob(dd+'/*.png'))
    # XXX: Load the first TEST_IMAGES for training
    # print('In load image!')
    for i in range(START, START+NUM_IMAGES):
        # print('i is: ', i)
        for j in range(TSTEP):
            # print('j is: ', j)
            # img = Image.open(ff[i+j]).convert('LA')
            img = load_img(ff[i+j])   # PIL image
            # print('loaded i, j: X(i+j)', i, j,
            #       ff[i+j].split('/')[-1].split('_')[0])
            w, h = img.size
            img = img.resize((w//RESIZE_FACTOR, h//RESIZE_FACTOR))
            img_gray = img.convert('L')
            img_array = img_to_array(img_gray)
            # img_array = rgb2gray(img_array)  # make it gray scale
            Xs += [img_array/NORM]

        # XXX: Just one output image to compare against
        # XXX: Now do the same thing for the output label image
        # img = Image.open(ff[i+TSTEP]).convert('LA')
        img = load_img(ff[i+TSTEP])   # PIL image
        # print('loaded Y: i, TSTEP: (i+TSTEP)', i, TSTEP,
        #       ff[(i+TSTEP)].split('/')[-1].split('_')[0])
        Ysdates.append(ff[(i+TSTEP)].split('/')[-1].split('_')[0])
        w, h = img.size
        img = img.resize((w//RESIZE_FACTOR, h//RESIZE_FACTOR))
        img_gray = img.convert('L')
        img_array1 = img_to_array(img_gray)
        # img_array = rgb2gray(img_array1)
        Ys += [img_array1/NORM]

        # DEBUG
        # XXX: Plot the images
        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # ax1.title.set_text('color')
        # ax2.title.set_text('gray')
        # ax1.imshow(img, cmap='gray')
        # ax2.imshow(img_gray, cmap='gray')
        # plt.show()
        # plt.close(fig)

    # XXX: Convert the lists to np.array
    Xs = np.array(Xs)
    Ys = np.array(Ys)
    np.expand_dims(Xs, axis=-1)
    np.expand_dims(Ys, axis=-1)
    return Xs, Ys, Ysdates


def main(mdir, years, months, instrument, dfs: dict):
    ff = []
    for f in os.listdir(mdir):
        for y in years:
            for m in months:
                # XXX: Just get the year and month needed
                tosearch = "*_{y}_{m}*.zip".format(y=y, m=m)
                if fnmatch.fnmatch(f, tosearch):
                    ff += [f]
                    # print(ff)
                    # XXX: Read the csvs
    for f in ff:
        z = zip.ZipFile(mdir+f)
        ofs = [i for i in z.namelist() if 'options_' in i]
        # print(ofs)
        # assert (1 == 2)
        # XXX: Now read just the option data files
        for f in ofs:
            key = f.split(".csv")[0].split("_")[2]
            df = pd.read_csv(z.open(f))
            df = df[df['UnderlyingSymbol'] == instrument].reset_index(
                drop=True)
            dfs[key] = df


def build_gird_and_images_gaussian(df, otype):
    # print('building grid and fitting')
    # XXX: Now fit a multi-variate linear regression to the dataset
    # one for each day.
    df = dict(sorted(df.items()))
    fitted_dict = dict()
    grid = dict()
    scores = list()
    for k in df.keys():
        # print('doing key: ', k)
        y = df[k]['IV']
        X = df[k][['m', 'tau']]
        # print('fitting')
        reg = KernelReg(endog=y, exog=X, var_type='cc',
                        reg_type='lc')
        # reg.fit()               # fit the model
        # print('fitted')
        fitted_dict[k] = reg
        scores += [reg.r_squared()]
        # XXX: Now make the grid
        ss = []
        mms = np.arange(LM, UM+MSTEP, MSTEP)
        tts = [i/DAYS for i in range(LT, UT+TSTEP, TSTEP)]
        # print('making grid: ', len(mms), len(tts))
        for mm in mms:
            for tt in tts:
                # XXX: Make the feature vector
                ss.append([mm, tt])

        grid[k] = pd.DataFrame(ss, columns=['m', 'tau'])
        # print('made grid and output')

    print("average fit score: ", sum(scores)/len(scores))
    # XXX: Now make the smooth ivs surface for each day
    ivs_surf_hmap = dict()
    ivs_surface = dict()
    for k in grid.keys():
        # XXX: This ivs goes m1,t1;m1,t2... then
        # m2,t1;m2,t2,m2,t3.... this is why reshape for heat map as
        # m, t, so we get m rows and t cols. Hence, x-axis is t and
        # y-axis is m.
        pivs, _ = fitted_dict[k].fit(grid[k])
        ivs_surface[k] = pd.DataFrame({'IVS': pivs,
                                       'm': grid[k]['m'],
                                       'tau': grid[k]['tau']})
        ivs_surface[k]['IVS'] = ivs_surface[k]['IVS']
        # print('IVS len:', len(ivs_surface[k]['IVS']))
        mcount = len(mms)
        tcount = len(tts)
        # print('mcount%s, tcount%s: ' % (mcount, tcount))
        ivs_surf_hmap[k] = ivs_surface[k]['IVS'].values.reshape(mcount,
                                                                tcount)
        # print('ivs hmap shape: ', ivs_surf_hmap[k].shape)

    # XXX: Plot the heatmap
    plot_hmap(ivs_surf_hmap, mms, tts, otype, dd='gfigs')


def build_gird_and_images(df, otype):
    # print('building grid and fitting')
    # XXX: Now fit a multi-variate linear regression to the dataset
    # one for each day.
    df = dict(sorted(df.items()))
    fitted_dict = dict()
    grid = dict()
    scores = list()
    for k in df.keys():
        # print('doing key: ', k)
        y = df[k]['IV']
        X = df[k][['m', 'tau', 'm2', 'tau2', 'mtau']]
        # print('fitting')
        reg = LinearRegression(n_jobs=-1).fit(X, y)
        # print('fitted')
        fitted_dict[k] = reg
        scores += [reg.score(X, y)]

        # XXX: Now make the grid
        ss = []
        mms = np.arange(LM, UM+MSTEP, MSTEP)
        tts = [i/DAYS for i in range(LT, UT+TSTEP, TSTEP)]
        # print('making grid: ', len(mms), len(tts))
        for mm in mms:
            for tt in tts:
                # XXX: Make the feature vector
                ss.append([mm, tt, mm**2, tt**2, mm*tt])

        grid[k] = pd.DataFrame(ss, columns=['m', 'tau', 'm2', 'tau2', 'mtau'])
        # print('made grid and output')

    print("average fit score: ", sum(scores)/len(scores))
    # XXX: Now make the smooth ivs surface for each day
    ivs_surf_hmap = dict()
    ivs_surface = dict()
    for k in grid.keys():
        # XXX: This ivs goes m1,t1;m1,t2... then
        # m2,t1;m2,t2,m2,t3.... this is why reshape for heat map as
        # m, t, so we get m rows and t cols. Hence, x-axis is t and
        # y-axis is m.
        pivs = fitted_dict[k].predict(grid[k])
        ivs_surface[k] = pd.DataFrame({'IVS': pivs,
                                       'm': grid[k]['m'],
                                       'tau': grid[k]['tau']})
        ivs_surface[k]['IVS'] = ivs_surface[k]['IVS'].clip(0.01, None)
        # print('IVS len:', len(ivs_surface[k]['IVS']))
        mcount = len(mms)
        tcount = len(tts)
        # print('mcount%s, tcount%s: ' % (mcount, tcount))
        ivs_surf_hmap[k] = ivs_surface[k]['IVS'].values.reshape(mcount,
                                                                tcount)
        # print('ivs hmap shape: ', ivs_surf_hmap[k].shape)

    # XXX: Plot the heatmap
    plot_hmap(ivs_surf_hmap, mms, tts, otype)

    # XXX: Plot the ivs surface
    # plot_ivs(ivs_surface, view='XY')


def excel_to_images(dvf=True, otype='call'):
    dir = '../../HistoricalOptionsData/'
    years = [str(i) for i in range(2002, 2024)]
    months = ['January', 'February',  'March', 'April', 'May', 'June', 'July',
              'August', 'September', 'October', 'November', 'December'
              ]
    instrument = ["SPX"]
    dfs = dict()
    # XXX: The dictionary of all the dataframes with the requires
    # instrument ivs samples
    for i in instrument:
        # XXX: Load the excel files
        main(dir, years, months, i, dfs)

        # XXX: Now make ivs surface for each instrument
        df = preprocess_ivs_df(dfs, otype)

        if dvf:
            # XXX: Build the 2d matrix with DVF
            build_gird_and_images(df, otype)
        else:
            # XXX: Build the 2d matrix with NW
            build_gird_and_images_gaussian(df, otype)


def build_keras_model(shape, inner_filters, bs, LR=1e-3):
    inp = Input(shape=shape[1:], batch_size=bs)
    x = ConvLSTM2D(
        filters=32,
        kernel_size=(7, 7),
        padding="same",
        data_format='channels_last',
        activation='relu',
        # dropout=0.2,
        # recurrent_dropout=0.1,
        # stateful=True,
        return_sequences=True)(inp)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(
        filters=64,
        kernel_size=(5, 5),
        data_format='channels_last',
        padding='same',
        # dropout=0.2,
        # recurrent_dropout=0.1,
        activation='relu',
        # stateful=True,
        return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(
        filters=inner_filters,
        kernel_size=(3, 3),
        data_format='channels_last',
        padding='same',
        # dropout=0.2,
        # recurrent_dropout=0.1,
        activation='relu',
        # stateful=True,
        return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(
        filters=inner_filters,
        kernel_size=(1, 1),
        data_format='channels_last',
        padding='same',
        # dropout=0.1,
        # recurrent_dropout=0.1,
        activation='relu',
        # stateful=True
        )(x)
    # XXX: 3D layer for images, 1 for each timestep
    x = Conv2D(
        filters=1, kernel_size=(1, 1), activation="relu",
        padding="same")(x)
    # XXX: Take the average in depth
    # x = keras.layers.AveragePooling3D(pool_size=(shape[1], 1, 1),
    #                                   padding='same',
    #                                   data_format='channels_last')(x)
    # XXX: Flatten the output
    # x = keras.layers.Flatten()(x)
    # # XXX: Dense layer for 1 output image
    # tot = 1
    # for i in shape[2:]:
    #     tot *= i
    # # print('TOT:', tot)
    # x = keras.layers.Dense(units=tot, activation='relu')(x)

    # # XXX: Reshape the output
    x = keras.layers.Reshape(shape[2:4])(x)

    # XXX: The complete model and compiled
    model = Model(inp, x)
    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=keras.optimizers.Adam(learning_rate=LR))

    # def loss(ytrue, ypred):
    #     ytrue = tf.reshape(ytrue, (ytrue.shape[0],
    #                                ytrue.shape[1]*ytrue.shape[2]))
    #     ypred = tf.reshape(ypred, (ypred.shape[0],
    #                                ypred.shape[1]*ypred.shape[2]))
    #     num = K.reduce_sum(K.square(ytrue-ypred), axis=-1)
    #     den = K.reduce_sum(K.square(ytrue - K.reduce_mean(ytrue)),
    #                        axis=-1)
    #     res = 1 - (num/den)
    #     return (-res)

    # model.compile(loss=loss,
    #               optimizer=keras.optimizers.Adam(learning_rate=LR))
    return model


def keras_model_fit(model, trainX, trainY, valX, valY, batch_size):
    # Define some callbacks to improve training.
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss",
                                                   patience=10,
                                                   restore_best_weights=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                  patience=5)

    # Define modifiable training hyperparameters.
    epochs = 500
    # batch_size = 2

    # Fit the model to the training data.
    history = model.fit(
        trainX,                 # this is not a 5D tensor right now!
        trainY,                 # this is not a 5D tensor right now!
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(valX, valY),
        verbose=1,
        callbacks=[early_stopping, reduce_lr])
    # callbacks=[reduce_lr])
    return history


def load_data(otype, dd='./figs', TSTEPS=10):
    NIMAGES1 = 2000
    # XXX: This is very important. If too long then changes are not
    # shown. If too short then too much influence from previous lags.
    TSTEPS = TSTEPS
    START = 0

    # Load, process and learn a ConvLSTM2D network
    trainX, trainY, _ = load_data_for_keras(otype, dd=dd,
                                            START=START, NUM_IMAGES=NIMAGES1,
                                            TSTEP=TSTEPS)
    # print(trainX.shape, trainY.shape)
    trainX = trainX.reshape(trainX.shape[0]//TSTEPS, TSTEPS,
                            *trainX.shape[1:], 1)
    # trainY = trainY.reshape(trainY.shape[0]//TSTEPS, TSTEPS,
    #                         *trainY.shape[1:])
    # print(trainX.shape, trainY.shape)

    NIMAGES2 = 1000
    START = START+NIMAGES1

    valX, valY, _ = load_data_for_keras(otype, dd=dd, START=START,
                                        NUM_IMAGES=NIMAGES2,
                                        TSTEP=TSTEPS)
    # print(valX.shape, valY.shape)
    valX = valX.reshape(valX.shape[0]//TSTEPS, TSTEPS, *valX.shape[1:], 1)
    # valY = valY.reshape(valY.shape[0]//TSTEPS, TSTEPS, *valY.shape[1:])
    # print(valX.shape, valY.shape)
    return (trainX, trainY, valX, valY, TSTEPS)


def plot_predicted_outputs_reg(vY, vYP, TSTEPS):

    # XXX: The moneyness
    MS = np.arange(LM, UM+MSTEP, MSTEP)
    # XXX: The term structure
    TS = np.array([i/DAYS
                   for i in
                   range(LT, UT+TSTEP, TSTEP)])
    # XXX: Reshape the outputs
    vY = vY.reshape(vY.shape[0], len(MS), len(TS))
    vYP = vYP.reshape(vYP.shape[0], len(MS), len(TS))
    print(vY.shape, vYP.shape)

    for i in range(vY.shape[0]):
        y = vY[i]*100
        yp = vYP[i]*100
        fig, axs = plt.subplots(1, 2,
                                subplot_kw=dict(projection='3d'))
        axs[0].title.set_text('Truth')
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
        axs[1].title.set_text('Predicted')
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

        plt.show()
        plt.close()


def clean_data(tX, tY):
    print("Cleaning data for gfigs")
    mask = np.isnan(tX)
    tX[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask),
                         tX[~mask])
    mask = np.isnan(tY)
    tY[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask),
                         tY[~mask])
    return tX, tY


def regression_predict(otype, dd='./figs', model='Ridge', TSTEPS=10):
    # XXX: We will need to do steps 5, 10 and 20
    tX, tY, vX, vY, lags = load_data(otype, dd=dd, TSTEPS=TSTEPS)
    tX = tX.reshape(tX.shape[:-1])
    vX = vX.reshape(vX.shape[:-1])
    # tX = np.append(tX, vX, axis=0)
    # tY = np.append(tY, vY, axis=0)
    # print('tX, tY: ', tX.shape, tY.shape)
    tX = tX.reshape(tX.shape[0], tX.shape[1]*tX.shape[2]*tX.shape[3])
    tY = tY.reshape(tY.shape[0], tY.shape[1]*tY.shape[2])
    # print('tX, tY:', tX.shape, tY.shape)

    # XXX: Validation set
    vX = vX.reshape(vX.shape[0], vX.shape[1]*vX.shape[2]*vX.shape[3])
    vY = vY.reshape(vY.shape[0], vY.shape[1]*vY.shape[2])
    # print('vX, vY:', vX.shape, vY.shape)

    # XXX: Intercept?
    intercept = True

    # Fill in NaN's... required for non-parametric regression
    if dd == './gfigs':
        tX, tY = clean_data(tX, tY)
        vX, vY = clean_data(vX, vY)

    # XXX: Make a LinearRegression
    if model == 'Lasso':
        treg = 'lasso'  # overfits
        reg = Lasso(fit_intercept=intercept, alpha=1,
                    selection='random')

    if model == 'Ridge':        # gives the best results
        treg = 'ridge'
        reg = Ridge(fit_intercept=intercept, alpha=1)

    if model == 'OLS':
        treg = 'ols'
        reg = LinearRegression(fit_intercept=intercept, n_jobs=-1)

    if model == 'ElasticNet':
        treg = 'enet'               # overfits
        reg = ElasticNet(fit_intercept=intercept, alpha=1,
                         selection='random')

    if model == 'RF':
        treg = 'rf'
        reg = RandomForestRegressor(n_jobs=10, max_features='sqrt',
                                    n_estimators=150,
                                    bootstrap=True, verbose=1)

    if model == 'XGBoost':
        treg = 'xgboost'
        reg = MultiOutputRegressor(
            xgb.XGBRegressor(n_jobs=12,
                             tree_method='hist',
                             multi_strategy='multi_output_tree',
                             n_estimators=100,
                             verbosity=2))

    reg.fit(tX, tY)
    print('Train set R2: ', reg.score(tX, tY))

    # XXX: Predict (Validation)
    vYP = reg.predict(vX)
    print(vY.shape, vYP.shape)
    rmses = root_mean_squared_error(vY, vYP, multioutput='raw_values')
    mapes = mean_absolute_percentage_error(vY, vYP, multioutput='raw_values')
    r2sc = r2_score(vY, vYP, multioutput='raw_values')
    print('RMSE mean: ', np.mean(rmses), 'RMSE std-dev: ', np.std(rmses))
    print('MAPE mean: ', np.mean(mapes), 'MAPE std-dev: ', np.std(mapes))
    print('R2 score mean:', np.mean(r2sc), 'R2 score std-dev: ', np.std(r2sc))

    # XXX: Plot some outputs
    # plot_predicted_outputs_reg(vY, vYP, TSTEPS)

    # XXX: Save the model
    import pickle
    if dd != './gfigs':
        with open('./surf_models/model_%s_ts_%s_%s.pkl' %
                  (treg, lags, otype), 'wb') as f:
            pickle.dump(reg, f)
    else:
        with open('./surf_models/model_%s_ts_%s_%s_gfigs.pkl' %
                  (treg, lags, otype), 'wb') as f:
            pickle.dump(reg, f)


def convlstm_predict(dd='./figs'):
    TSTEPS = 20
    trainX, trainY, valX, valY, _ = load_data(dd=dd, TSTEPS=TSTEPS)

    # Fill in NaN's... required for non-parametric regression
    if dd == './gfigs':
        print("fixing gfigs data")
        mask = np.isnan(trainX)
        trainX[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask),
                                 trainX[~mask])
        mask = np.isnan(trainY)
        trainY[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask),
                                 trainY[~mask])

        print('tX, tY:', trainX.shape, trainY.shape)
        mask = np.isnan(valX)
        valX[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask),
                               valX[~mask])
        mask = np.isnan(valY)
        valY[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask),
                               valY[~mask])
        print('vX, vY:', valX.shape, valY.shape)

    # XXX: Inner number of filters
    inner_filters = 64

    # XXX: Now fit the model
    batch_size = 20

    # XXX: Now build the keras model
    model = build_keras_model(trainX.shape, inner_filters, batch_size)
    print(model.summary())

    history = keras_model_fit(model, trainX, trainY, valX, valY, batch_size)

    # XXX: Save the model after training
    if dd == './gfigs':
        model.save('modelcr_bs_%s_ts_%s_filters_%s_%s.keras' %
                   (batch_size, TSTEPS, inner_filters, 'gfigs'))
    else:
        model.save('modelcr_bs_%s_ts_%s_filters_%s_%s.keras' %
                   (batch_size, TSTEPS, inner_filters, 'figs'))

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('history_bs_%s_ts_%s_filters_%s.pdf' % (batch_size, TSTEPS,
                                                        inner_filters))


def mskew_pred(otype, dd='./figs', model='mskridge', TSTEPS=5):
    # XXX: We will need to do steps 5, 10 and 20
    tX, tY, vX, vY, lags = load_data(otype, dd=dd, TSTEPS=TSTEPS)
    tX = tX.reshape(tX.shape[:-1])
    vX = vX.reshape(vX.shape[:-1])

    # Fill in NaN's... required for non-parametric regression
    if dd == './gfigs':
        tX, tY = clean_data(tX, tY)
        vX, vY = clean_data(vX, vY)

    # XXX: Now go through the MS and TS
    tts = [i/DAYS for i in range(LT, UT+TSTEP, TSTEP)]

    count = 0

    # XXX: Now we go moneyness skew
    for j, t in enumerate(tts):
        if count % 10 == 0:
            print('Done: ', count)
        count += 1
        # XXX: shape = samples, TSTEPS, moneyness, term structure
        mskew = tX[:, :, :, j]
        tYY = tY[:, :, j]
        mskew = mskew.reshape(mskew.shape[0], mskew.shape[1]*mskew.shape[2])
        # XXX: Add t to the sample set
        ts = np.array([t]*mskew.shape[0]).reshape(mskew.shape[0], 1)
        mskew = np.append(mskew, ts, axis=1)
        # XXX: Fit the ridge model
        if model == 'mskridge':
            reg = Ridge(fit_intercept=True, alpha=1)
        elif model == 'msklasso':
            reg = Lasso(fit_intercept=True, alpha=1)
        else:
            reg = ElasticNet(fit_intercept=True, alpha=1)
        reg.fit(mskew, tYY)
        import pickle
        if dd != './gfigs':
            with open('./mskew_models/%s_ts_%s_%s_%s.pkl' %
                      (model, lags, t, otype), 'wb') as f:
                pickle.dump(reg, f)
        else:
            with open('./mskew_models/%s_ts_%s_%s_%s_gfigs.pkl' %
                      (model, lags, t, otype), 'wb') as f:
                pickle.dump(reg, f)


def tskew_pred(otype, dd='./figs', model='tskridge', TSTEPS=5):
    # XXX: We will need to do steps 5, 10 and 20
    tX, tY, vX, vY, lags = load_data(otype, dd=dd, TSTEPS=TSTEPS)
    tX = tX.reshape(tX.shape[:-1])
    vX = vX.reshape(vX.shape[:-1])

    # Fill in NaN's... required for non-parametric regression
    if dd == './gfigs':
        tX, tY = clean_data(tX, tY)
        vX, vY = clean_data(vX, vY)

    # XXX: Now go through the MS and TS
    mms = np.arange(LM, UM+MSTEP, MSTEP)

    count = 0

    # XXX: Now we go term structure skew
    for j, m in enumerate(mms):
        if count % 10 == 0:
            print('Done: ', count)
        count += 1
        # XXX: shape = samples, TSTEPS, moneyness, term structure
        tskew = tX[:, :, j]
        tskew = tskew.reshape(tskew.shape[0], tskew.shape[1]*tskew.shape[2])
        # XXX: Add m to the sample set
        ms = np.array([m]*tskew.shape[0]).reshape(tskew.shape[0], 1)
        tskew = np.append(tskew, ms, axis=1)
        tYY = tY[:, j]
        # XXX: Fit the ridge model
        if model == 'tskridge':
            reg = Ridge(fit_intercept=True, alpha=1)
        elif model == 'tsklasso':
            reg = Lasso(fit_intercept=True, alpha=1)
        else:
            reg = ElasticNet(fit_intercept=True, alpha=1)
        reg.fit(tskew, tYY)
        import pickle
        if dd != './gfigs':
            with open('./tskew_models/%s_ts_%s_%s_%s.pkl' %
                      (model, lags, m, otype), 'wb') as f:
                pickle.dump(reg, f)
        else:
            with open('./tskew_models/%s_ts_%s_%s_%s_gfigs.pkl' %
                      (model, lags, m, otype), 'wb') as f:
                pickle.dump(reg, f)


def point_pred(otype, dd='./figs', model='pmridge', TSTEPS=10):
    # XXX: We will need to do steps 5, 10 and 20
    tX, tY, vX, vY, lags = load_data(otype, dd=dd, TSTEPS=TSTEPS)
    tX = tX.reshape(tX.shape[:-1])
    vX = vX.reshape(vX.shape[:-1])
    # tX = np.append(tX, vX, axis=0)
    # tY = np.append(tY, vY, axis=0)
    # print('tX, tY: ', tX.shape, tY.shape)

    # XXX: Validation set
    # print('vX, vY:', vX.shape, vY.shape)

    # Fill in NaN's... required for non-parametric regression
    if dd == './gfigs':
        tX, tY = clean_data(tX, tY)
        vX, vY = clean_data(vX, vY)

    # XXX: Now go through the MS and TS
    mms = np.arange(LM, UM+MSTEP, MSTEP)
    tts = [i/DAYS for i in range(LT, UT+TSTEP, TSTEP)]

    count = 0
    for i, s in enumerate(mms):
        for j, t in enumerate(tts):
            if count % 50 == 0:
                print('Done: ', count)
            count += 1
            # XXX: Make the vector for training
            k = np.array([s, t]*tX.shape[0]).reshape(tX.shape[0], 2)
            train_vec = np.append(tX[:, :, i, j], k, axis=1)
            # print(train_vec.shape, tY[:, i, j].shape)

            # XXX: Fit the ridge model
            if model == 'pmridge':
                reg = Ridge(fit_intercept=True, alpha=1)
            elif model == 'pmlasso':
                reg = Lasso(fit_intercept=True, alpha=1)
            else:
                reg = ElasticNet(fit_intercept=True, alpha=1,
                                 selection='random')
            reg.fit(train_vec, tY[:, i, j])
            # print('Train set R2: ', reg.score(train_vec, tY[:, i, j]))

            # XXX: Predict (Validation)
            # print(vX.shape, vY.shape)
            # k = np.array([s, t]*vX.shape[0]).reshape(vX.shape[0], 2)
            # val_vec = np.append(vX[:, :, i, j], k, axis=1)
            # vYP = reg.predict(val_vec)
            # vvY = vY[:, i, j]
            # r2sc = r2_score(vvY, vYP, multioutput='raw_values')
            # print('Test R2:', np.mean(r2sc))

            # XXX: Save the model
            import pickle
            if dd != './gfigs':
                with open('./point_models/%s_ts_%s_%s_%s_%s.pkl' %
                          (model, lags, s, t, otype), 'wb') as f:
                    pickle.dump(reg, f)
            else:
                with open('./point_models/%s_ts_%s_%s_%s_%s_gfigs.pkl' %
                          (model, lags, s, t, otype), 'wb') as f:
                    pickle.dump(reg, f)


def linear_fit(otype):
    # Surface regression prediction (RUN THIS WITH OMP_NUM_THREADS=10 on
    # command line)
    # XXX: Point regression
    for j in ['./figs', './gfigs']:
        for k in ['pmridge', 'pmlasso', 'pmenet']:
            for i in [5, 10, 20]:
                print('Doing: %s_%s_%s' % (k, j, i))
                point_pred(otype, dd=j, model=k, TSTEPS=i)

    # XXX: Moneyness skew regression
    for j in ['./figs', './gfigs']:
        for k in ['mskridge', 'msklasso', 'mskenet']:
            for i in [5, 10, 20]:
                print('Doing: %s_%s_%s' % (k, j, i))
                mskew_pred(otype, dd=j, model=k, TSTEPS=i)

    # XXX: Term structure skew regression
    for j in ['./figs', './gfigs']:
        for k in ['tskridge', 'tsklasso', 'tskenet']:
            for i in [5, 10, 20]:
                print('Doing: %s_%s_%s' % (k, j, i))
                tskew_pred(otype, dd=j, model=k, TSTEPS=i)

    for k in ['Ridge', 'Lasso', 'ElasticNet']:
        for j in ['./figs', './gfigs']:
            for i in [5, 10, 20]:
                print('Doing: %s_%s_%s' % (k, j, i))
                regression_predict(otype, model=k, dd=j, TSTEPS=i)


if __name__ == '__main__':
    # XXX: Excel data to images
    # excel_to_images(otype='call')  # call options with linear fit
    # excel_to_images(otype='put')  # put options with linear fit

    # XXX: Non-parametric regression for calls
    # excel_to_images(otype='call', dvf=False)
    # XXX: Non-parametric regression for puts
    # excel_to_images(otype='put', dvf=False)

    # XXX: Fit the linear models
    for otype in ['call', 'put']:
        linear_fit(otype)

    # XXX: ConvLSTM2D prediction
    # convlstm_predict(dd='./gfigs')
