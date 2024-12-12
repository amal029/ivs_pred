# import pandas as pd
import pickle
import numpy as np
import os
# import fnmatch
# import zipfile as zip
# import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from pred import load_data 
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from keras.layers import Input, Dense
from keras.models import Model
from keras import ops
from keras import layers
import tensorflow as tf
import keras
# import multiprocessing

from non_linear_models import SVM, LSTM

import pred


# VAE implementation

# Sampling layer
@keras.saving.register_keras_serializable()
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch,dim), mean=0., stddev=1.)
        return z_mean + tf.exp(0.5*z_log_var)*epsilon

# Encoder
@keras.saving.register_keras_serializable()
class Encoder(layers.Layer):

    def __init__(self, latent_dim, intermediate_dim, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation='tanh')
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()
    
    def call(self, inputs):
        x = self.dense_proj(inputs)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z

# Decoder
@ keras.saving.register_keras_serializable()
class Decoder(layers.Layer):
    def __init__(self, original_dim, intermediate_dim, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation='tanh')
        self.dense_output = layers.Dense(original_dim)
    
    def call(self, inputs):
        x = self.dense_proj(inputs)
        return self.dense_output(x)
    
# Vae model
@keras.saving.register_keras_serializable()
class VAE(keras.Model):
    def __init__(self, original_dim, intermediate_dim, latent_dim, name="vae", **kwargs):
        super().__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim, intermediate_dim)
        self.decoder = Decoder(original_dim, intermediate_dim)
        self.istrained = False
    
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Calculate the KL divergence loss
        kl_loss = -0.5*tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        # Calculate the reconstruction loss
        self.add_loss(kl_loss)
        return reconstructed
    
    def encode(self, inputs):
        return self.encoder(inputs)
    
    def decode(self, z):
        return self.decoder(z)
    
    def get_config(self):
        base_config = super().get_config()
        config = {
            'original_dim': self.original_dim,
            'intermediate_dim': self.intermediate_dim,
            'latent_dim': self.latent_dim
        }
        return {**base_config, **config} 
    
def train_vae(tX, vX=None, epochs=20, batch_size=64, shuffle=True):
    # reduce learning rate
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=2, min_lr=0.00001)
    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

    mse_loss = keras.losses.MeanSquaredError()    
    loss_metric = keras.metrics.Mean()
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    vae = VAE(original_dim=tX.shape[1], intermediate_dim=64, latent_dim=2)

    train_dataset = tf.data.Dataset.from_tensor_slices(tX).batch(batch_size)

    for epoch in range(epochs):
        print("Start of epoch %d" % (epoch,))
        for step, x_batch_train in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                reconstructed = vae(x_batch_train)
                loss_mse = mse_loss(x_batch_train, reconstructed) 
                loss = sum(vae.losses) + loss_mse
            grads = tape.gradient(loss, vae.trainable_weights)
            optimizer.apply_gradients(zip(grads, vae.trainable_weights))

            loss_metric(loss)
            if step % 100 == 0:
                print("step %d: mean loss = %.4f, mse_loss = %.4f, kl_loss = %.4f" % (step, loss_metric.result(), loss_mse, sum(vae.losses)))

    # vae.compile(optimizer=optimizer, loss=keras.losses.MeanSquaredError())
    # vae.fit(tX, tX, validation_data=(vX, vX), epochs=epochs, batch_size=batch_size, shuffle=shuffle, callbacks=[reduce_lr, early_stopping], verbose=1)
    vae.istrained = True

    # # Validate
    # vX_encoded = vae.encode(vX)
    # vX_decoded = vae.decode(vX_encoded[2])
    # from sklearn.metrics import mean_squared_error
    # print('Validation Loss: ', mean_squared_error(vX, vX_decoded))

    # import matplotlib.pyplot as plt

    # # Plot the skew from reconstructed data
    # # Reshape
    # vX_decoded = vX_decoded.numpy().reshape(vX_decoded.shape[0], 72)
    # vx = vX.reshape(vX.shape[0], 72)
    # plt.figure(figsize=(6,6))
    # # plot surf
    # # plt.imshow(vX_decoded[0], cmap='viridis')
    # plt.plot(vX_decoded[0])
    # # plt.colorbar()
    # plt.savefig('vae_surf.png')
    # plt.figure(figsize=(6,6))
    # # plt.imshow(vx[0], cmap='viridis')
    # plt.plot(vx[0])
    # # plt.colorbar()
    # plt.savefig('surf.png')

    return vae

# Regression model for vae
class VaeRegression:
    def __init__(self, reg='', vae_name=''):
        self.vae_name = vae_name
        if os.path.isfile(vae_name):
            self.load()
        else:
            self.vae = None
        if reg == '':
            self._reg = Ridge()
        elif reg == 'lasso':
            self._reg = Lasso()
        elif reg == 'enet':
            self._reg = ElasticNet()
    
    def load(self, vae_name=None):
        if vae_name is not None:
            self.vae_name = vae_name
        self.vae = keras.models.load_model(self.vae_name)
    
    def fit(self, tX, tY, TSTEPS, type):
        if type != 'surf':
            tX = tX[:, :-1]

        # reshape data
        tX = tX.reshape(tX.shape[0], TSTEPS, tX.shape[1]//TSTEPS)
        tX = tX.reshape(tX.shape[0]*tX.shape[1], tX.shape[2])

        if self.vae is None:
            self.vae = train_vae(tX)
        
        tX_encoded = self.vae.encode(tX)[0]
        tY_encoded = self.vae.encode(tY)[0]
        tX_encoded = tX_encoded.numpy()
        tY_encoded = tY_encoded.numpy()

        # reshape back
        tX_encoded = tX_encoded.reshape(tX_encoded.shape[0]//TSTEPS, TSTEPS, tX_encoded.shape[1])
        tX_encoded = tX_encoded.reshape(tX_encoded.shape[0], tX_encoded.shape[1]*tX_encoded.shape[2])

        self._reg.fit(tX_encoded, tY_encoded)
    
    def predict(self, vX, TSTEPS, type):
        if type != 'surf':
            vX = vX[:, :-1]

        if self.vae is None:
            self.load()
            if self.vae is None:
                raise ValueError('Model not trained')
        elif self.vae.istrained == False:
            raise ValueError('Model not trained')

        vX = vX.reshape(vX.shape[0], TSTEPS, vX.shape[1]//TSTEPS)
        vX = vX.reshape(vX.shape[0]*vX.shape[1], vX.shape[2])
        
        vX_encoded = self.vae.encode(vX)[0]
        vX_encoded = vX_encoded.numpy()
        vX_encoded = vX_encoded.reshape(vX_encoded.shape[0]//TSTEPS, TSTEPS, vX_encoded.shape[1])
        vX_encoded = vX_encoded.reshape(vX_encoded.shape[0], vX_encoded.shape[1]*vX_encoded.shape[2])

        vYp = self._reg.predict(vX_encoded)
        return self.vae.decode(vYp)

# PCA regression model
class PcaRegression:
    def __init__(self, reg=''):
        if reg == '':
            self._reg = Ridge()
        elif reg == 'lasso':
            self._reg = Lasso()
        elif reg == 'enet':
            self._reg = ElasticNet()
    
    def fit(self, tX, tY, TSTEPS, type):

        pca = PCA(n_components=0.95)
        tX = pca.fit_transform(tX)
        tY = pca.fit_transform(tY)

        self._reg.fit(tX, tY)
    
    def predict(self, vX, TSTEPS, type):
        
        pca = PCA(n_components=0.95)
        vX_transform = pca.fit_transform(vX)
        vYp = self._reg.predict(vX_transform)
        return pca.inverse_transform(vYp)
    
class HarRegression:
    def __init__(self, reg=''):
        if reg == '':
            self._reg = Ridge()
        elif reg == 'lasso':
            self._reg = Lasso()
        elif reg == 'enet':
            self._reg = ElasticNet()
    
    def har_transform(self, tX, TSTEPS=20, type='skew'):
        """
        Transform the given input data to the HAR method of feature extraction
        Expects 2D input shape

        Input shape: (samples, TSTEP, moneyness)
        TSTEP must be of size 21 as a month of IV data is required for the HAR method

        Output shape: (samples, tx.shape[-1]*3)
        """
        if TSTEPS != 20:
            raise ValueError('TSTEP must be 20 for HAR method of feature extraction')

        keepdims = type == 'point' 

        # Get average skew for 21 days
        skew1 = np.mean(tX[:, :], axis=1, keepdims=keepdims)
        skew2 = np.mean(tX[:, -5:], axis=1, keepdims=keepdims)
        skew3 = np.mean(tX[:, -1:], axis=1, keepdims=keepdims)
        tX = np.concatenate([skew1, skew2, skew3], axis=1)

        return tX
    
    def fit(self, tX, tY, TSTEPS, type):

        if type[1:] =='skew':
            tX = tX[:, :-1]
            # Reshape to 3D
            tX = tX.reshape(tX.shape[0], TSTEPS, tX.shape[1]//TSTEPS)
        elif type == 'surf':
            tX = tX.reshape(tX.shape[0], TSTEPS, tX.shape[1]//TSTEPS)
        else:
            tX = tX[:, :-2]

        tX = self.har_transform(tX, TSTEPS=TSTEPS, type='skew')
        self._reg.fit(tX, tY)
    
    def predict(self, vX, TSTEPS, type):
        if type[1:] =='skew':
            vX = vX[:, :-1]
            # Reshape to 3D
            vX = vX.reshape(vX.shape[0], TSTEPS, vX.shape[1]//TSTEPS)
        elif type == 'surf':
            vX = vX.reshape(vX.shape[0], TSTEPS, vX.shape[1]//TSTEPS)
        else:
            vX = vX[:, :-2]

        vX = self.har_transform(vX, TSTEPS=TSTEPS, type='skew')
        vYp = self._reg.predict(vX)
        return vYp


def surf_pred(data, otype, model_name, TSTEPS, dd, learn="ridge"):
    if not os.path.exists('./surf_feature_models'):
        os.makedirs('./surf_feature_models')

    tX, tY, vX, vY, _ = data
    # removing extra dimension
    tX = tX.reshape(tX.shape[:-1])
    vX = vX.reshape(vX.shape[:-1])

    if dd == './gfigs':
        tX, tY = pred.clean_data(tX, tY)
        vX, vY = pred.clean_data(vX, vY)
    
    # Flatten the time step term structure and moneyness
    tX = tX.reshape(tX.shape[0], tX.shape[1]*tX.shape[2]*tX.shape[3])
    tY = tY.reshape(tY.shape[0], tY.shape[1]*tY.shape[2])

    if model_name == 'har' and TSTEPS != 21:
        return

    if model_name == 'har':
        reg = HarRegression(learn)
    elif model_name == 'vae':
        reg = VaeRegression(learn, './surf_feature_models/%s_ts_%s_encoder.keras' % (model_name, otype))
    elif model_name == 'pca':
        reg = PcaRegression(learn)
    
    reg.fit(tX, tY, TSTEPS, 'surf')

    # Validate the model
    # validation(tX, tY, model)

    if dd != './gfigs':
        with open('./surf_feature_models/%s%s_ts_%s_%s.pkl' % (learn, model_name, TSTEPS, otype), 'wb') as f:
            pickle.dump(reg, f)
    else:
        with open('./surf_feature_models/%s%s_ts_%s_%s_gfigs.pkl' % (learn, model_name, TSTEPS, otype), 'wb') as f:
            pickle.dump(reg, f)

def tskew_pred(data, otype, model_name='pca', TSTEPS=10, dd='./figs', learn="ridge"):
    # Check if directory exists
    if not os.path.exists('./tskew_feature_models'):
        os.makedirs('./tskew_feature_models')

    # Load data
    tX, tY, vX, vY, _ = data 
    # removing extra dimension
    tX = tX.reshape(tX.shape[:-1]) 
    vX = vX.reshape(vX.shape[:-1])

    if dd == './gfigs':
        tX, tY = pred.clean_data(tX, tY)
        vX, vY = pred.clean_data(vX, vY)

    # Moneyness range to iterate over
    mms = np.arange(pred.LM, pred.UM+pred.MSTEP, pred.MSTEP)

    count = 0

    for j, m in enumerate(mms):
        if count % 10 == 0:
            print('Done: ', count)
        count += 1

        # XXX: shape = samples, TSTEPS, moneyness, term structure
        tskew = tX[:, :, j]
        tYY = tY[:, j]
        tskew = tskew.reshape(tskew.shape[0], tskew.shape[1]*tskew.shape[2])

        if model_name == 'har' and TSTEPS != 21:
            continue 

        # add the moneyness to the sample set
        k = np.array([m]*tX.shape[0]).reshape(tX.shape[0], 1)
        tskew = np.append(tskew, k, axis=1)

        if model_name == 'har':
            reg = HarRegression(learn)
        elif model_name == 'vae':
            reg = VaeRegression(learn, './tskew_feature_models/%s_ts_%s_%s_encoder.keras' % (model_name, m, otype))
        elif model_name == 'pca':
            reg = PcaRegression(learn)
        
        reg.fit(tskew, tYY, TSTEPS, 'tskew')
        
        # Validate the model
        # validation(tskew, tYY, model)

        if dd != './gfigs':
            with open('./tskew_feature_models/%s%s_ts_%s_%s_%s.pkl' % (learn, model_name, TSTEPS, m, otype), 'wb') as f:
                pickle.dump(reg, f)
        else:
            with open('./tskew_feature_models/%s%s_ts_%s_%s_%s_gfigs.pkl' % (learn, model_name, TSTEPS, m, otype), 'wb') as f:
                pickle.dump(reg, f)


def mskew_pred(data, otype, model_name='pca', TSTEPS=10, dd='./figs', learn="ridge"):
    # Check if directory exists
    if not os.path.exists('./mskew_feature_models'):
        os.makedirs('./mskew_feature_models')

    # Load data
    tX, tY, vX, vY, _ = data 
    # removing extra dimension
    tX = tX.reshape(tX.shape[:-1]) 
    vX = vX.reshape(vX.shape[:-1])

    if dd == './gfigs':
        tX, tY = pred.clean_data(tX, tY)
        vX, vY = pred.clean_data(vX, vY)

    # Term structure range to iterate over
    tts = [i/pred.DAYS for i in range(pred.LT, pred.UT+pred.TSTEP, pred.TSTEP)]

    count = 0

    for j, t in enumerate(tts):
        if count % 10 == 0:
            print('Done: ', count)
        count += 1
        # XXX: shape = samples, TSTEPS, moneyness, term structure
        mskew = tX[:, :, :, j]
        tYY = tY[:, :, j]
        mskew = mskew.reshape(mskew.shape[0], mskew.shape[1]*mskew.shape[2])

        # add the term structure to the sample set
        k = np.array([t]*tX.shape[0]).reshape(tX.shape[0], 1)
        mskew = np.append(mskew, k, axis=1)

        if model_name == 'har':
            reg = HarRegression(learn)
        elif model_name == 'vae':
            reg = VaeRegression(learn, './mskew_feature_models/%s_ts_%s_%s_encoder.keras' % (model_name, t, otype))
        elif model_name == 'pca':
            reg = PcaRegression(learn)
        
        reg.fit(mskew, tYY, TSTEPS, 'mskew')

        # Validate the model
        # validation(mskew, tYY, model)

        if dd != './gfigs':
            with open('./mskew_feature_models/%s%s_ts_%s_%s_%s.pkl' % (learn, model_name, TSTEPS, t, otype), 'wb') as f:
                pickle.dump(reg, f)
        else:
            with open('./mskew_feature_models/%s%s_ts_%s_%s_%s_gfigs.pkl' % (learn, model_name, TSTEPS, t, otype), 'wb') as f:
                pickle.dump(reg, f)

def point_pred(data, otype, model_name='pca', TSTEPS=10, dd='./figs', learn="ridge"):
    print('Running point pred for: ', otype, model_name, TSTEPS, dd, learn)
    # Check if directory exists
    if not os.path.exists('./point_feature_models'):
        os.makedirs('./point_feature_models')

    # Load data
    tX, tY, vX, vY, _ = data 
    # removing extra dimension
    tX = tX.reshape(tX.shape[:-1]) 
    vX = vX.reshape(vX.shape[:-1])

    if dd == './gfigs':
        tX, tY = pred.clean_data(tX, tY)
        vX, vY = pred.clean_data(vX, vY)

    tts = [i/pred.DAYS for i in range(pred.LT, pred.UT+pred.TSTEP, pred.TSTEP)]
    mms = np.arange(pred.LM, pred.UM+pred.MSTEP, pred.MSTEP)

    for i, s in enumerate(mms):
        for j, t in enumerate(tts):
                train_vec = tX[:, :, i, j]
                tYY = tY[:, i, j]

                if dd == './gfigs':
                    tosave = './point_feature_models/%s%s_ts_%s_%s_%s_%s_gfigs.pkl' % (learn, model_name, TSTEPS, s, t, otype)
                else:
                    tosave = './point_feature_models/%s%s_ts_%s_%s_%s_%s.pkl' % (learn, model_name, TSTEPS, s, t, otype)

                # XXX: Add the moneyness and term structure to the sample set
                k = np.array([s, t]*tX.shape[0]).reshape(tX.shape[0], 2)
                train_vec = np.append(train_vec, k, axis=1)
                
                if model_name == 'har':
                    reg = HarRegression(learn)
                elif model_name == 'pca':
                    reg = PcaRegression(learn)
                
                # Add dimension for point prediction
                tYY = tYY.reshape(tYY.shape[0], 1)
                reg.fit(train_vec, tYY, TSTEPS, 'point')


                # Validate the model
                # validation(train_vec, tY[:, i, j], reg)

                # XXX: Save the model
                with open(tosave, 'wb') as f:
                    pickle.dump(reg, f) 
            

    print("Done")

if __name__ == "__main__":

    # otype = 'call'
    # dd = './figs'
    # TSTEPS = 20
    # model_name = 'vae'

    # Train vae models for each feature type surf, tskew, mskew
    # for x in ['call']:
    #     for j in ['./figs']:
    #         for i in ['skew']:
    #             data = load_data(otype=x, dd=j, TSTEPS=5)
    #             tX, tY, vX, vY, _ = data
    #             # removing extra dimension
    #             tX = tX.reshape(tX.shape[:-1]) 
    #             vX = vX.reshape(vX.shape[:-1])

    #             # reshape to 2d
    #             tX = tX.reshape(tX.shape[0]*tX.shape[1], np.prod(tX.shape[2:]))
    #             vX = vX.reshape(vX.shape[0]*vX.shape[1], np.prod(vX.shape[2:]))

    #             vae_encoder = train_vae(tX, vX, epochs=10, batch_size=64, shuffle=True) 

    #             # tosave = './%s_feature_models/%s_ts_%s_encoder.keras' % (i, 'vae', x)
    #             tosave = 'test.keras'

    #             save = True 
    #             if save:
    #                 vae_encoder.save(tosave)




    # Set environment variable to use cpu instead of gpu
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)
    # surf_pred(data, otype=otype, model_name=model_name, TSTEPS=TSTEPS, dd=dd)
    # mskew_pred(data, otype=otype, model_name=model_name, TSTEPS=TSTEPS, dd=dd)
    # tskew_pred(data, otype=otype, model_name=model_name, TSTEPS=TSTEPS, dd=dd)
    # point_pred(data, otype=otype, model_name=model_name, TSTEPS=TSTEPS, dd=dd)
    for n in ['call']:
        
        # for i in ['./figs']:
        #     for x in ['', 'enet', 'lasso']:
        #         for j in [5, 10, 20]:
        #             data = load_data(otype=n, dd=i, TSTEPS=j)
        #             for k in ['pca']:
        #                 print('Running for surf: ', n, i, x, j, k)
        #                 surf_pred(data, n, k, j, i, x)
        #                 # p = multiprocessing.Process(target=surf_pred, args=(data, n, k, j, i, x))
        #                 # p.start()
        #                 # p.join()

        # for i in ['./figs']:
        #     for x in ['', 'enet', 'lasso']:
        #         for j in [5, 10, 20]:
        #             data = load_data(otype=n, dd=i, TSTEPS=j)
        #             for k in ['pca']:
        #                 print('Running for mskew pred: ', x, n, i, k, j)
        #                 mskew_pred(data, n, k, j, i, x)
        #                 # p = multiprocessing.Process(target=mskew_pred, args=(data, n, k, j, i, x))
        #                 # p.start()
        #                 # p.join()

        for i in ['./figs']:
            for j in [5, 10, 20]:
                data = load_data(otype=n, dd=i, TSTEPS=j)
                for x in ['', 'enet', 'lasso']:
                    for k in ['pca']:
                        print('Running for point pred: ', x, n, i, k, j)
                        point_pred(data, n, k, j, i, x)
                        # p = multiprocessing.Process(target=point_pred, args=(data, n, k, j, i, x))
                        # p.start()
                        # p.join()

        for i in ['./figs']:
            for x in ['', 'enet', 'lasso']:
                for j in [5, 10, 20]:
                    data = load_data(otype=n, dd=i, TSTEPS=j)
                    for k in ['pca']:
                        print('Running for tskew pred: ', x, n, i, k, j)
                        # p = multiprocessing.Process(target=tskew_pred, args=(data, n, k, j, i, x))
                        # p.start()
                        # p.join()
                        # print("Done")
                        tskew_pred(data, otype=n, model_name=k, TSTEPS=j, dd=i, learn=x)
  
# #         # For HAR 
#         for i in ['./figs', './gfigs']:
#             for x in ['ridge', 'enet', 'lasso']:
#                 data = load_data(otype=n, dd=i, TSTEPS=20)
#                 print('Running for all : ', x, n, i, 'har')
#                 surf_pred(data, otype=n, model_name='har', TSTEPS=20, dd=i, learn=x)
#                 print("Done surf")
#                 tskew_pred(data, otype=n, model_name='har', TSTEPS=20, dd=i, learn=x)
#                 print("Done tskew")
#                 mskew_pred(data, otype=n, model_name='har', TSTEPS=20, dd=i, learn=x)
#                 print("Done mskew")
#                 point_pred(data, otype=n, model_name='har', TSTEPS=20, dd=i, learn=x)
#                 print("Done point")
    #PCA experement
    # for i in ['./figs']:
    #     for x in ['ridge']:
    #         data = load_data(otype='call', dd=i, TSTEPS=5)
    #         tX, tY, vX, vY, _ = data 
    #         # removing extra dimension
    #         tX = tX.reshape(tX.shape[:-1]) 
    #         vX = vX.reshape(vX.shape[:-1])

    #         if i == './gfigs':
    #             tX, tY = pred.clean_data(tX, tY)
    #             vX, vY = pred.clean_data(vX, vY)

    #         tts = [i/pred.DAYS for i in range(pred.LT, pred.UT+pred.TSTEP, pred.TSTEP)]
    #         mms = np.arange(pred.LM, pred.UM+pred.MSTEP, pred.MSTEP)

    #         # for i, s in enumerate(mms):
    #         #     for j, t in enumerate(tts):
    #         #         train_vec = tX[:, :, i, j]
    #         #         # Fit and transform the data
    #         #         pca = PCA(n_components=train_vec.shape[1])
    #         #         tX_transform = pca.fit_transform(train_vec)
    #         #         print(pca.explained_variance_ratio_)

    #         # Now do for skew
    #         for j, t in enumerate(tts):
    #             train_vec = tX[:, :, :, j]
    #             tYY = tY[:, :, j]
    #             train_vec = train_vec.reshape(train_vec.shape[0], train_vec.shape[1]*train_vec.shape[2])
                
    #             # Fit and transform the data
    #             pca = PCA(n_components=124)
    #             tX_transform = pca.fit_transform(train_vec)
                
    #             # XXX: Add t to the sample set
    #             ts = np.array([t]*tX_transform.shape[0]).reshape(tX_transform.shape[0], 1)
    #             tX_transform = np.append(tX_transform, ts, axis=1)

    #             # Fit Regression model
    #             model = Ridge() 
    #             model.fit(tX_transform, tYY)

    #             val_x = vX[:, :, :, j]
    #             val_y = vY[:, :, j]
    #             val_x = val_x.reshape(val_x.shape[0], val_x.shape[1]*val_x.shape[2])
    #             val_x_transform = pca.transform(val_x)
    #             ts = np.array([t]*val_x_transform.shape[0]).reshape(val_x_transform.shape[0], 1)
    #             val_x_transform = np.append(val_x_transform, ts, axis=1)

    #             # Validate the model
    #             print(model.score(val_x_transform, val_y))


