#!/usr/bin/env python

import keras
import pred
import numpy as np
import matplotlib.pyplot as plt
import glob
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import dmtest
from mpl_toolkits.mplot3d import Axes3D


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
    print(name)
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
    # plt.show()


def main(dd='./figs', model='Ridge', plot=True, TSTEPS=5, NIMAGES=1000,
         get_features=False, feature_res=10):
    # XXX: Important date: 20201014

    START = date_to_num('20190102')
    # print("START:", START)
    NIMAGES = NIMAGES

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

    valX, valY, Ydates = pred.load_data_for_keras(START=START, dd=dd,
                                                  NUM_IMAGES=NIMAGES,
                                                  TSTEP=pred.TSTEPS)
    # print(valX.shape, valY.shape)
    valX = valX.reshape(valX.shape[0]//pred.TSTEPS, pred.TSTEPS,
                        *valX.shape[1:])
    # print(valX.shape, valY.shape)

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
        print('predicted shape: ', out.shape)
        if not plot:
            # XXX: Reshape data for measurements
            out = out.reshape(out.shape[0], out.shape[1]*out.shape[2])
            valY = valY.reshape(valY.shape[0], valY.shape[1]*valY.shape[2])
            print('predicted reshaped: ', out.shape)

    elif model == 'pmodel':
        # XXX: The output vector
        out = np.array([0.0]*(valY.shape[0]*valY.shape[1]*valY.shape[2]))
        out = out.reshape(*valY.shape)
        # XXX: Now go through the MS and TS
        mms = np.arange(pred.LM, pred.UM+pred.MSTEP, pred.MSTEP)
        tts = [i/pred.DAYS
               for i in range(pred.LT, pred.UT+pred.TSTEP, pred.TSTEP)]
        for i, s in enumerate(mms):
            for j, t in enumerate(tts):
                if dd == './figs':
                    import pickle
                    with open('./point_models/pm_%s_ts_%s_%s_%s.pkl' %
                              ("ridge", TSTEPS, s, t), 'rb') as f:
                        m1 = pickle.load(f)
                else:
                    with open('./point_models/pm_%s_ts_%s_%s_%s_gfigs.pkl' %
                              ("ridge", TSTEPS, s, t), 'rb') as f:
                        m1 = pickle.load(f)
                # XXX: Now make the prediction
                k = np.array([s, t]*valX.shape[0]).reshape(valX.shape[0], 2)
                val_vec = np.append(valX[:, :, i, j], k, axis=1)
                out[:, i, j] = m1.predict(val_vec)
        if not plot:
            out = out.reshape(out.shape[0], out.shape[1]*out.shape[2])
            valY = valY.reshape(valY.shape[0], valY.shape[1]*valY.shape[2])

    else:
        import pickle
        # XXX: Reshape the data for testing
        MS = valY.shape[1]
        TS = valY.shape[2]
        valX = valX.reshape(valX.shape[0],
                            valX.shape[1]*valX.shape[2]*valX.shape[3])
        if not plot:
            valY = valY.reshape(valY.shape[0], valY.shape[1]*valY.shape[2])

        # XXX: Load the model and then carry it out
        if dd == './gfigs':
            toopen = r"model_%s_ts_%s_gfigs.pkl" % (model.lower(), pred.TSTEPS)
        else:
            toopen = r"model_%s_ts_%s.pkl" % (model.lower(), pred.TSTEPS)

        print('Doing model: ', toopen)
        with open(toopen, "rb") as input_file:
            m1 = pickle.load(input_file)

        # XXX: Get the feature importances
        if get_features:
            # XXX: Moneyness
            MONEYNESS = [0, MS//2, MS-1]
            # XXX: Some terms
            TERM = [0, TS//2, TS-1]
            if model == 'ridge':
                ws = m1.coef_.reshape(MS, TS, TSTEPS, MS, TS)
                print(ws.shape)
                # XXX: Just get the top 10 results
                X = np.arange(pred.LM, pred.UM+pred.MSTEP, pred.MSTEP)
                Y = [i/pred.DAYS for i in range(pred.LT, pred.UT+pred.TSTEP,
                                                pred.TSTEP)]
                for i in MONEYNESS:
                    for j in TERM:
                        wsurf = ws[i][j]  # 1 IV point
                        if dd == './figs':
                            name = 'm_%s_t_%s_%s_fm.pdf' % (X[i],
                                                            int(365*Y[j]),
                                                            TSTEPS)
                        else:
                            name = 'm_%s_t_%s_%s_gfigs_fm.pdf' % (
                                X[i],
                                int(365*Y[j]),
                                TSTEPS)
                        plot_hmap_features(wsurf, X, Y, name)

        # XXX: Predict the output
        out = m1.predict(valX)
        if plot:
            # XXX: Reshape the data for plotting
            out = out.reshape(out.shape[0], MS, TS)

    if plot:
        plotme(valY, Ydates, out)
        return None, None, None, valY, out
    else:
        # XXX: We want to do analysis (quant measures)

        # XXX: RMSE (mean and std-dev of RMSE)
        rmses = root_mean_squared_error(valY, out, multioutput='raw_values')
        mapes = mean_absolute_percentage_error(valY, out,
                                               multioutput='raw_values')
        r2sc = r2_score(valY, out, multioutput='raw_values')
        print('RMSE mean: ', np.mean(rmses), 'RMSE std-dev: ', np.std(rmses))
        print('MAPE mean: ', np.mean(mapes), 'MAPE std-dev: ', np.std(mapes))
        print('R2 score mean:', np.mean(r2sc), 'R2 score std-dev: ',
              np.std(r2sc))
        return rmses, mapes, r2sc, valY, out


def model_v_model():
    # FIXME: This can be made better by running each model just once and
    # then caching it.
    TTS = [5, 10, 20]
    models = ['ridge', 'lasso', 'rf', 'enet', 'keras']
    fp = {t: np.array([0.0]*len(models)*len(models)).reshape(len(models),
                                                             len(models))
          for t in TTS}
    fd = {t: np.array([0.0]*len(models)*len(models)).reshape(len(models),
                                                             len(models))
          for t in TTS}

    for dd in ['./figs', './gfigs']:
        for t in fp.keys():
            for i in range(0, len(models)-1):
                for j in range(i+1, len(models)):
                    _, _, _, y, yp = main(plot=False, TSTEPS=t,
                                          model=models[i], get_features=True)
                    _, _, _, yk, ypk = main(plot=False, TSTEPS=t,
                                            model=models[j], get_features=True)
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


def model_surf_v_point_model():
    TTS = [5, 10, 20]
    # XXX: Only Ridge model(s)
    for dd in ['./figs', './gfigs']:
        for t in TTS:
            _, _, _, y, yp = main(plot=False, TSTEPS=t, model="ridge")
            _, _, _, yk, ypk = main(plot=False, TSTEPS=t, model="pmodel")
            assert (np.array_equal(y, yk))
            # XXX: Now we can do Diebold mariano test
            try:
                dstat, pval = dmtest.dm_test(y, yp, ypk)
            except dmtest.ZeroVarianceException:
                dstat, pval = np.nan, np.nan
                # XXX: save the dstat and pvals
            with open('%s_pmodel_v_model_%s.csv' %
                      (t, dd.split('/')[1]), 'w') as f:
                f.write('#ridge, pmodel\n')
                f.write('dstat,pval\n')
                f.write('%s,%s' % (dstat, pval))


if __name__ == '__main__':
    # XXX: This is many models vs many other models
    # model_v_model()

    # XXX: This is ridge surf vs ridge point estimate (base line paper)
    model_surf_v_point_model()
