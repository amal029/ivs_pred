#!/usr/bin/env python

import keras
import pred
import numpy as np
import matplotlib.pyplot as plt
import glob
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import dmtest


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
    pass


def main(dd='./figs', model='Ridge', plot=True, TSTEPS=5, NIMAGES=1000):
    # XXX: Important date: 20201014

    START = date_to_num('20190102')
    print("START:", START)
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
    print(valX.shape, valY.shape)
    valX = valX.reshape(valX.shape[0]//pred.TSTEPS, pred.TSTEPS,
                        *valX.shape[1:])
    print(valX.shape, valY.shape)

    # XXX: Clean the data
    if dd == './gfigs':
        valX, valY = pred.clean_data(valX, valY)

    # XXX: Make a prediction for 10 images
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


if __name__ == '__main__':
    rmses, mapes, r2sc, y, yp = main(plot=False, TSTEPS=10, model='ridge')
    rmsesk, mapesk, r2sck, yk, ypk = main(plot=False, TSTEPS=10, model='rf')

    assert (np.array_equal(y, yk))

    # XXX: Now we can do Diebold mariano test
    dstat, pval = dmtest.dm_test(y, yp, ypk)  # ridge >= keras?
    print('Diebold-Mariano test results. dstat: %s, pval: %s' % (dstat, pval))

    # dir_to_num()
