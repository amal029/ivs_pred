import pandas as pd
import pickle
import numpy as np
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import fnmatch
# import zipfile as zip
import keras
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from pred import load_data, load_data_for_keras
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from keras.layers import Input, Dense
from keras.models import Model
from keras import ops
from keras import layers
import tensorflow as tf

import pred


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon

def vae_loss(z_mean, z_log_var): 
    def vae_reconstruction_loss(y_true, y_pred):
        reconstruction_ration = 1000
        reconstruction_loss = ops.mean(ops.square(y_true-y_pred), axis=1)
        return reconstruction_loss*reconstruction_ration
    def vae_kl_loss(z_mean, z_log_var):
        kl_loss = -0.5 * ops.sum(1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var), axis=1)
        return kl_loss
    
    def vae_kl_loss_metric(y_true, y_pred):
        kl_loss = -0.5 * ops.sum(1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var), axis=1)
        return kl_loss
    
    def vae_loss(y_true, y_pred):
        reconstruction_loss = vae_reconstruction_loss(y_true, y_pred)
        kl_loss = vae_kl_loss(y_true, y_pred)
        return reconstruction_loss + kl_loss
    
    return vae_loss
    

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
    
    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data[0])
            reconstruction = self.decoder(z)
            # reconstruction_loss = keras.losses.binary_crossentropy(data[0], reconstruction)
            reconstruction_loss = ops.mean(ops.square(data[1]-reconstruction), axis=1)
            reconstruction_loss *= data[1].shape[1]
            kl_loss = -0.5 * ops.sum(1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var), axis=1)
            total_loss = reconstruction_loss + kl_loss
        
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result()
        }

def build_vae(input_shape, intermediate_dim=512, latent_dim=2):
    """
    Variational Autoencoder model
    """
    # Encoder
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # Sample from the latent space
    z = Sampling()([z_mean, z_log_var]) 

    # Encoder
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()

    # Decoder
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = Dense(input_shape[0], activation='relu')(x)
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()

    # VAE
    decoder_output = decoder(encoder(inputs)[2])
    vae = Model(inputs, decoder_output, name='vae')
    vae.compile(optimizer='adam', loss=vae_loss(z_mean, z_log_var))

    return vae


class Autoencoder:
    def __init__(self, encoding_dim, input_shape):
        self.encoding_dim = encoding_dim
        self.input_shape = input_shape
        self.model = self.autoencoder_build()
        pass

    def fit(self, tX, epochs=100, batch_size=256, shuffle=True, validation_split=0.2):
        tX = tX.reshape(tX.shape[0], tX.shape[1]*tX.shape[2])
        self.model.fit(tX, tX, epochs=epochs, batch_size=batch_size, 
                       shuffle=shuffle, validation_split=validation_split)
    
    def save(self, path):
        self.model.save(path)

    def autoencoder_build(self):
        encoding_dim= self.encoding_dim*self.input_shape[-1]

        # Reshape the data
        combined_shape = self.input_shape[1]*self.input_shape[2]

        # Fit and transform the data
        input_layer = Input(shape=(combined_shape,))
        encoded = Dense(encoding_dim, activation='relu')(input_layer)
        decoded = Dense(combined_shape, activation='sigmoid')(encoded)

        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        print(autoencoder.summary())

        return autoencoder 

    def predict(self, valX):
        """
        Transforms the input into expected shape and passes it through the encoder

        Input shape: (samples, TSTEP, moneyness)
        Output shape: (samples, encoding_dim*tX.shape[-1])
        """
        # Transform the data to predict
        encoder = Model(inputs=self.model.input, outputs=self.model.layers[1].output)
        valX = valX.reshape(valX.shape[0], valX.shape[1]*valX.shape[2])
        return encoder.predict(valX)


def autoencoder_fit(tX, ty, encoding_dim, TSTEPS=32):
    """
    Uses an autoencoder to extract features from the data 
    and then uses a regression model to predict the implied volatility

    Features extracted will be in the shape of (samples, components*tX.shape[-1])
    i.e. component number of skews for each sample

    Output: encoder, ridge model 
    """
    encoder = Autoencoder(encoding_dim, tX.shape)
    encoder.fit(tX)
    tX_transform = encoder.predict(tX)
    # Fit regression model
    model = Ridge()
    model.fit(tX_transform, ty)
    return encoder, model

def vae_fit(tX, tY, encoding_dim):
    """
    Uses a VAE to extract features from the data 
    and then uses a regression model to predict the implied volatility

    Features extracted will be in the shape of (samples, components*tX.shape[-1])
    i.e. component number of skews for each sample

    Output: encoder, ridge model 
    """
    # XXX latent_dim = encoding_dim * num points in skew
    latent_dim = encoding_dim*tX.shape[-1]
    # Intermediate dim is slightly larger than the latent dim
    intermediate_dim = (encoding_dim+1)*tX.shape[-1]

    tX = tX.reshape(tX.shape[0], tX.shape[1]*tX.shape[2])
    vae = build_vae(input_shape=(tX.shape[1],), intermediate_dim=intermediate_dim,
                                 latent_dim=latent_dim)
    vae.summary()
    # reduce learning rate
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.0001)
    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
    vae.fit(tX, tX, epochs=100, batch_size=5, shuffle=True, callbacks=[reduce_lr, early_stopping])

    # XXX Transform the data
    # Get the encoder model
    encoder = Model(inputs=vae.get_layer('encoder').input, outputs=vae.get_layer('encoder').output) 

    tX_transform = encoder.predict(tX)[2]
    # Fit regression model
    model = Ridge()
    model.fit(tX_transform, tY)
    return vae, model

def pca_predict(valX, model, n_components, TSTEPS):
    valX_transform = pca_transform(valX, n_components, TSTEPS)
    return model.predict(valX_transform)

def pca_transform(tX, components=3, TSTEPS=32, type='skew'):
    """
    uses PCA to extract features from the data

    Input shape: (samples, TSTEP, moneyness)

    Ouput shape: (samples, components*tX.shape[-1])
    """
    if type == 'skew':
        n_components = components*tX.shape[-1]
        # Reshape the data
        tX = tX.reshape(tX.shape[0], tX.shape[1]*tX.shape[2])
    else: # point model
        n_components = components

    # Fit and transform the data
    pca = PCA(n_components=n_components)
    tX_transform = pca.fit_transform(tX)

    return tX_transform
    
def pca_fit(tX, ty, components=3 , TSTEPS=32, type='skew'):
    """
    Uses PCA to extract features from the data 
    and then uses a regression model to predict the implied volatility

    Features extracted will be in the shape of (samples, components*tX.shape[-1])
    i.e. component number of skews for each sample
    """
    tX_transform = pca_transform(tX, components=components, TSTEPS=TSTEPS, type=type) 
    # Fit regression model
    model = Ridge()
    model.fit(tX_transform, ty)
    return model


def har_transform(tX, TSTEPS=32, type='skew'):
    """
    Transform the given input data to the HAR method of feature extraction

    Input shape: (samples, TSTEP, moneyness)
    TSTEP must be of size 32 as a month of IV data is required for the HAR method

    Output shape: (samples, tx.shape[-1]*3)
    """
    if TSTEPS != 32:
        raise ValueError('TSTEP must be 32 for HAR method of feature extraction')

    # Get average skew for 32 days
    if type == 'skew':
        skew1 = np.mean(tX[:, :, :], axis=1)
        skew2 = np.mean(tX[:, -5:, :], axis=1)
        skew3 = np.mean(tX[:, -1:, :], axis=1)
        tX = np.concatenate([skew1, skew2, skew3], axis=1)
    else: # Point model
        # remove the moneyness and term structure features
        moneyness = tX[:, -2:-1]
        term_structure = tX[:, -1:]
        skew1 = np.mean(tX[:, :-2], axis=1).reshape(-1, 1)
        skew2 = np.mean(tX[:, -7:-2], axis=1).reshape(-1, 1)
        skew3 = np.mean(tX[:, -3:-2], axis=1).reshape(-1, 1)
        tX = np.concatenate([skew1, skew2, skew3, moneyness, term_structure], axis=1)


    return tX 


def har_features(tX, tY, TSTEPS=32, type='skew'):
    """
    Extracts har features of implied volatility which includes an averaged skew for the 1 month, 1 week and 1 day lagging features
    """
    tX = har_transform(tX, TSTEPS=TSTEPS, type=type)

    # XXX: New shape is samples, 3 features concatenated
    
    # Fit regression model
    model = Ridge()
    model.fit(tX, tY)
    return model


def har_predict(valX, model, TSTEPS=32):
    valX_transform = har_transform(valX, TSTEPS=TSTEPS)
    return model.predict(valX_transform)    

def tskew_pred(otype, model_name='pca', TSTEPS=10, dd='./figs'):
    # Check if directory exists
    if not os.path.exists('./tskew_feature_models'):
        os.makedirs('./tskew_feature_models')

    # Load data
    tX, tY, vX, vY, _ = load_data(otype, TSTEPS=TSTEPS, dd=dd)
    tX = tX.reshape(tX.shape[:-1]) 
    vX = vX.reshape(vX.shape[:-1])

    if dd == './gfigs':
        tX, tY = pred.clean_data(tX, tY)
        vX, vY = pred.clean_data(vX, vY)

    mms = np.arange(pred.LM, pred.UM+pred.MSTEP, pred.MSTEP)

    count = 0

    for j, m in enumerate(mms):
        if count % 10 == 0:
            print('Done: ', count)
        count += 1

        # XXX: shape = samples, TSTEPS, moneyness, term structure
        # No need to reshape as har requires the tstep feature
        tskew = tX[:, :, j]
        tYY = tY[:, j]

        vtskew= vX[:, :, j]
        
        # Fit the model
        if model_name == 'har':
            if TSTEPS != 32:
                continue
            model = har_features(tskew, tYY, TSTEPS=TSTEPS)
            ypred = har_predict(vtskew, model, TSTEPS=TSTEPS)
            # Fit the model
        elif model_name == 'autoencoder':
            encoding_dim = TSTEPS//2 
            encoder, model = autoencoder_fit(tskew, tYY, encoding_dim=encoding_dim, TSTEPS=TSTEPS)
            # Save encoder model
            if dd != './gfigs':
                encoder.save('./tskew_feature_models/%s_ts_%s_%s_encoder.keras' % (model_name, TSTEPS, m))
            else:
                encoder.save('./tskew_feature_models/%s_ts_%s_%s_encoder_gfigs.keras' % (model_name, TSTEPS, m))
            # transform and validate 
            valX_transform = encoder.predict(vtskew) 
            ypred = model.predict(valX_transform)
        else: # PCA
            n_components = TSTEPS//2 
            model = pca_fit(tskew, tYY, components=n_components, TSTEPS=TSTEPS)
            ypred = pca_predict(vtskew, model, n_components=n_components, TSTEPS=TSTEPS)
            pass

        # Evaluate the model
        rmse = root_mean_squared_error(vY[:, j], ypred, multioutput='raw_values')
        mapes = mean_absolute_percentage_error(vY[:, j], ypred, multioutput='raw_values')
        r2sc = r2_score(vY[:, j], ypred, multioutput='raw_values')

        print('RMSE mean: ', np.mean(rmse), 'RMSE std: ', np.std(rmse))
        print('MAPE mean: ', np.mean(mapes), 'MAPE std: ', np.std(mapes))
        print('R2 mean: ', np.mean(r2sc), 'R2 std: ', np.std(r2sc))

        if dd != './gfigs':
            with open('./tskew_feature_models/%s_ts_%s_%s.pkl' % (model_name, TSTEPS, m), 'wb') as f:
                pickle.dump(model, f)
        else:
            with open('./tskew_feature_models/%s_ts_%s_%s_gfigs.pkl' % (model_name, TSTEPS, m), 'wb') as f:
                pickle.dump(model, f)


def mskew_pred(otype, model_name='pca', TSTEPS=10, dd='./figs'):
    # Check if directory exists
    if not os.path.exists('./mskew_feature_models'):
        os.makedirs('./mskew_feature_models')

    # Load data
    tX, tY, vX, vY, _ = load_data(otype, TSTEPS=TSTEPS, dd=dd)
    tX = tX.reshape(tX.shape[:-1]) 
    vX = vX.reshape(vX.shape[:-1])

    if dd == './gfigs':
        tX, tY = pred.clean_data(tX, tY)
        vX, vY = pred.clean_data(vX, vY)

    tts = [i/pred.DAYS for i in range(pred.LT, pred.UT+pred.TSTEP, pred.TSTEP)]

    count = 0

    for j, t in enumerate(tts):
        if count % 10 == 0:
            print('Done: ', count)
        count += 1
        # XXX: shape = samples, TSTEPS, moneyness, term structure
        mskew = tX[:, :, :, j]
        tYY = tY[:, :, j]

        vmskew= vX[:, :, :, j]
        
        # Fit the model
        if model_name == 'har':
            if TSTEPS != 32:
                continue
            model = har_features(mskew, tYY, TSTEPS=TSTEPS)
            ypred = har_predict(vmskew, model, TSTEPS=TSTEPS)
            # Fit the model
        elif model_name == 'autoencoder':
            encoding_dim = TSTEPS//2
            encoder, model = autoencoder_fit(mskew, tYY, encoding_dim=encoding_dim, TSTEPS=TSTEPS)
            # Save encoder model
            if dd != './gfigs':
                encoder.save('./mskew_feature_models/%s_ts_%s_%s_encoder.keras' % (model_name, TSTEPS, t))
            else:
                encoder.save('./mskew_feature_models/%s_ts_%s_%s_encoder_gfigs.keras' % (model_name, TSTEPS, t))
            # transform and validate 
            valX_transform = encoder.predict(vmskew) 
            ypred = model.predict(valX_transform)
        else: # PCA
            n_components = TSTEPS//2 
            model = pca_fit(mskew, tYY, components=n_components, TSTEPS=TSTEPS)
            ypred = pca_predict(vmskew, model, n_components=n_components, TSTEPS=TSTEPS)
            pass

        # Evaluate the model
        rmse = root_mean_squared_error(vY[:, :, j], ypred, multioutput='raw_values')
        mapes = mean_absolute_percentage_error(vY[:, :, j], ypred, multioutput='raw_values')
        r2sc = r2_score(vY[:, :, j], ypred, multioutput='raw_values')

        print('RMSE mean: ', np.mean(rmse), 'RMSE std: ', np.std(rmse))
        print('MAPE mean: ', np.mean(mapes), 'MAPE std: ', np.std(mapes))
        print('R2 mean: ', np.mean(r2sc), 'R2 std: ', np.std(r2sc))

        if dd != './gfigs':
            with open('./mskew_feature_models/%s_ts_%s_%s.pkl' % (model_name, TSTEPS, t), 'wb') as f:
                pickle.dump(model, f)
        else:
            with open('./mskew_feature_models/%s_ts_%s_%s_gfigs.pkl' % (model_name, TSTEPS, t), 'wb') as f:
                pickle.dump(model, f)

def point_pred(otype, model_name='pca', TSTEPS=10, dd='./figs'):
    # Check if directory exists
    if not os.path.exists('./point_feature_models'):
        os.makedirs('./point_feature_models')

    # Load data
    tX, tY, vX, vY, _ = load_data(otype, TSTEPS=TSTEPS, dd=dd)
    tX = tX.reshape(tX.shape[:-1]) 
    vX = vX.reshape(vX.shape[:-1])

    if dd == './gfigs':
        tX, tY = pred.clean_data(tX, tY)
        vX, vY = pred.clean_data(vX, vY)

    tts = [i/pred.DAYS for i in range(pred.LT, pred.UT+pred.TSTEP, pred.TSTEP)]
    mms = np.arange(pred.LM, pred.UM+pred.MSTEP, pred.MSTEP)

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

            # Fit the model
            if model_name == 'har':
                if TSTEPS != 32:
                    continue
                reg = har_features(train_vec, tY[:, i, j], TSTEPS=TSTEPS, type='point')
                tX_transform = har_transform(train_vec, TSTEPS=TSTEPS, type='point')
                print('Train set R2: ', reg.score(tX_transform, tY[:, i, j]))
                # ypred = har_predict(vmskew, reg, TSTEPS=TSTEPS)
                # Fit the model
            elif model_name == 'autoencoder':
                encoding_dim = TSTEPS//2
                encoder, reg = autoencoder_fit(train_vec, tY[:, i, j], encoding_dim=encoding_dim, TSTEPS=TSTEPS)
                # Save encoder model
                if dd != './gfigs':
                    encoder.save('./point_feature_models/%s_ts_%s_%s_encoder.keras' % (model_name, TSTEPS, t))
                else:
                    encoder.save('./point_feature_models/%s_ts_%s_%s_encoder_gfigs.keras' % (model_name, TSTEPS, t))
                # # transform and validate 
                # valX_transform = encoder.predict(vmskew) 
                # ypred = reg.predict(valX_transform)
            else: # PCA
                n_components = TSTEPS//2 
                reg = pca_fit(train_vec, tY[:, i, j], components=n_components, TSTEPS=TSTEPS, type='point')
                tX_transform = pca_transform(train_vec, components=n_components, TSTEPS=TSTEPS, type='point') 
                print('Train set R2: ', reg.score(tX_transform, tY[:, i, j]))
                # ypred = pca_predict(vmskew, reg, n_components=n_components, TSTEPS=TSTEPS)
                pass


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
                with open('./point_feature_models/%s_ts_%s_%s_%s.pkl' %
                          (model_name, TSTEPS, s, t), 'wb') as f:
                    pickle.dump(reg, f)
            else:
                with open('./point_feature_models/%s_ts_%s_%s_%s_gfigs.pkl' %
                          (model_name, TSTEPS, s, t), 'wb') as f:
                    pickle.dump(reg, f)

def run_all_models():
    for k in ['pca', 'autoencoder']:
        for j in [5, 10, 20]:
            mskew_pred(model_name=k, TSTEPS=j)
            tskew_pred(model_name=k, TSTEPS=j)
    # Do a run for the HAR method
    tskew_pred(model_name='har', TSTEPS=32)
    mskew_pred(model_name='har', TSTEPS=32)


if __name__ == "__main__":
    # for n in ['call', 'put']:
    #     for i in ['./figs', './gfigs']:
    #         for k in ['pca', 'autoencoder']:
    #             for j in [5, 10, 20]:
    #                 mskew_pred(otype=n, model_name=k, TSTEPS=j, dd=i)
    #                 tskew_pred(otype=n, model_name=k, TSTEPS=j, dd=i)
    #         # Do a run for the HAR method
    #         # tskew_pred(model_name='har', TSTEPS=32, dd=i)
    #         # mskew_pred(model_name='har', TSTEPS=32, dd=i)

    # for i in ['./gfigs']:
    #     print('Running for: ', i)
    #     for k in ['pca']:
    #         print('Running for: ', k)
    #         for j in [5, 10, 20]:
    #             print('Running for: ', j)
    #             point_pred(model_name=k, TSTEPS=j, dd=i)
    #     # Do a run for the HAR method
    #     print('Running for: ', 'har')
    #     point_pred(model_name='har', TSTEPS=32, dd=i)
    # Load data

    tX, tY, vX, vY, _ = load_data(otype='call', TSTEPS=5, dd='./figs')
    tX = tX.reshape(tX.shape[:-1]) 
    vX = vX.reshape(vX.shape[:-1])
    mskew = tX[:, :, :, 1]
    tYY = tY[:, :, 1]

    vmskew= vX[:, :, :, 1]
    vmskew = vmskew.reshape(vmskew.shape[0], vmskew.shape[1]*vmskew.shape[2])
    vae, model = vae_fit(mskew, tYY, encoding_dim=2)
    encoder = Model(inputs=vae.get_layer('encoder').input, outputs=vae.get_layer('encoder').output) 
    vX_transform = encoder.predict(vmskew)[2]
    ypred =  model.predict(vX_transform)
    # Evaluate the model
    rmse = root_mean_squared_error(vY[:, :, 1], ypred, multioutput='raw_values')
    mapes = mean_absolute_percentage_error(vY[:, :, 1], ypred, multioutput='raw_values')
    r2sc = r2_score(vY[:, :, 1], ypred, multioutput='raw_values')

    print('RMSE mean: ', np.mean(rmse), 'RMSE std: ', np.std(rmse))
    print('MAPE mean: ', np.mean(mapes), 'MAPE std: ', np.std(mapes))
    print('R2 mean: ', np.mean(r2sc), 'R2 std: ', np.std(r2sc))

