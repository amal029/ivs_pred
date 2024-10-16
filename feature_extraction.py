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
import multiprocessing

import pred

@keras.saving.register_keras_serializable()
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim))
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon

@keras.saving.register_keras_serializable()
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

    def vae_loss(self,y_true, y_pred, z_mean, z_log_var): 
        """
        Loss function for the VAE model that computes the reconstruction loss and the KL divergence loss
        """
        def vae_reconstruction_loss(y_true, y_pred):
            reconstruction_ratio = 1000
            reconstruction_loss = tf.reduce_mean(ops.square(y_true-y_pred), axis=1)
            return reconstruction_loss*reconstruction_ratio
        def vae_kl_loss(z_mean, z_log_var):
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var), axis=1)
            return kl_loss
        
        def vae_loss(y_true, y_pred, z_mean, z_log_var):
            reconstruction_loss = vae_reconstruction_loss(y_true, y_pred)
            kl_loss = vae_kl_loss(z_mean, z_log_var)
            return (reconstruction_loss + kl_loss), reconstruction_loss, kl_loss
        
        return vae_loss(y_true, y_pred, z_mean, z_log_var)

    def train_step(self, data):
        with tf.GradientTape() as tape: 
            z_mean, z_log_var, z = self.encoder(data[0])
            reconstruction = self.decoder(z)
            # reconstruction_loss = keras.losses.binary_crossentropy(data[0], reconstruction)
            total_loss, reconstruction_loss, kl_loss = self.vae_loss(data[0], reconstruction, z_mean, z_log_var)

        gradient = tape.gradient(total_loss, self.trainable_weights) 

        self.optimizer.apply_gradients(zip(gradient, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result()
        }
    
    def test_step(self, data):
        z_mean, z_log_var, z = self.encoder(data[0])
        reconstruction = self.decoder(z)
        # reconstruction_loss = keras.losses.binary_crossentropy(data[0], reconstruction)
        total_loss, reconstruction_loss, kl_loss = self.vae_loss(data[0], reconstruction, z_mean, z_log_var)
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss
        }

class Autoencoder:
    def __init__(self, encoding_dim, input_shape, vae=False, intermediate_dim=None):
        self.encoding_dim = encoding_dim
        self.input_shape = input_shape
        self.vae = vae
        if vae:
            self.model = self.build_vae(self.input_shape, latent_dim=encoding_dim, intermediate_dim=intermediate_dim)
        else: 
            self.model = self.autoencoder_build()

    def fit(self, tX, epochs=100, batch_size=125, shuffle=True, validation_split=0.2):
        # reduce learning rate
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.0001)
        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
        # tX = tX.reshape(tX.shape[0], tX.shape[1]*tX.shape[2])
        self.model.fit(tX, tX, epochs=epochs, batch_size=batch_size, 
                       shuffle=shuffle, validation_split=validation_split, callbacks=[reduce_lr, early_stopping], verbose=0)
    
    def save(self, path):
        if self.vae:
            self.model.encoder.save(path)
        else:
            self.model.save(path)


    def autoencoder_build(self) :

        # Fit and transform the data
        input_layer = Input(shape=(self.input_shape,))
        encoded = Dense(self.encoding_dim, activation='relu')(input_layer)
        decoded = Dense(self.input_shape, activation='sigmoid')(encoded)

        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        # print(autoencoder.summary())

        return autoencoder 

    def build_vae(self, input_shape, intermediate_dim=512, latent_dim=2):
        """
        Variational Autoencoder model
        """
        # Encoder
        inputs = Input(shape=(input_shape,), name='encoder_input')
        x = Dense(input_shape, activation='relu')(inputs)
        z_mean = Dense(latent_dim, activation='relu', name='z_mean')(x)
        z_log_var = Dense(latent_dim, activation='relu', name='z_log_var')(x)

        # Sample from the latent space
        z = Sampling(name='sampling')([z_mean, z_log_var]) 

        # Encoder
        encoder = Model(inputs,  [z_mean, z_log_var, z], name='encoder')
        # encoder.summary()

        # Decoder
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        x = Dense(input_shape, activation='relu')(latent_inputs)
        outputs = Dense(input_shape, activation='sigmoid')(x)
        decoder = Model(latent_inputs, outputs, name='decoder')
        # decoder.summary()

        # VAE
        vae = VAE(encoder, decoder)
        vae.compile(optimizer='adam')

        return vae


    def predict(self, valX):
        """
        Transforms the input into expected shape and passes it through the encoder

        Input shape: (samples, TSTEP, moneyness)
        Output shape: (samples, encoding_dim*tX.shape[-1])
        """
        if self.vae:
            encoder = self.model.encoder 
        else:
            encoder = Model(inputs=self.model.input, outputs=self.model.layers[1].output)
        return encoder.predict(valX)

def autoencoder_transform(encoder , tX, type='skew', vae=False):
    # Strip the moneyness or term structure feature
    if type == 'skew':
        structure = tX[:, -1:]
        tX = tX[:, :-1]
    else: # point model
        structure = tX[:, -2:]
        tX = tX[:, :-2]
    
    tX_transformed = encoder.predict(tX)

    # Because vae is outputing the z_mean, z_log_var and z
    if vae:
        tX_transformed = tX_transformed[2]

    # Add the structure back So we end up with our encoded features + structure at the end of each sample
    tX_transformed = np.concatenate([tX_transformed, structure], axis=1)

    return tX_transformed



def autoencoder_fit(tX, ty, encoding_dim, TSTEPS=21, type='skew', vae=False):
    """
    Uses an autoencoder to extract features from the data 
    and then uses a regression model to predict the implied volatility

    Features extracted will be in the shape of (samples, components*tX.shape[-1])
    i.e. component number of skews for each sample

    Output: encoder, ridge model 
    """
    # Strip the moneyness or term structure feature
    if type == 'skew':
        structure = tX[:, -1:]
        tX = tX[:, :-1]
    else: # point model
        structure = tX[:, -2:]
        tX = tX[:, :-2]

    if type == 'skew':
        encoding_dim = encoding_dim*(tX.shape[1]//TSTEPS)

    if vae:
        intermediate_dim = (encoding_dim+1)*(tX.shape[1]//TSTEPS)
    else:
        intermediate_dim = None
    

    encoder = Autoencoder(encoding_dim, tX.shape[1], vae=vae, intermediate_dim=intermediate_dim)
    encoder.fit(tX)
    if vae:
        tX_transform = encoder.predict(tX)[2]
    else:
        tX_transform = encoder.predict(tX)
    # Add the structure back So we end up with our encoded features + structure at the end of each sample
    tX_transform = np.concatenate([tX_transform, structure], axis=1)
    # Fit regression model
    model = Ridge()
    model.fit(tX_transform, ty)
    return encoder, model

def pca_predict(valX, model, n_components, TSTEPS):
    valX_transform = pca_transform(valX, n_components, TSTEPS)
    return model.predict(valX_transform)

def pca_transform(tX, components=3, TSTEPS=21, type='skew'):
    """
    uses PCA to extract features from the data

    Input shape: (samples, TSTEP, moneyness)

    Ouput shape: (samples, components*tX.shape[-1])
    """
    # Strip the moneyness or term structure feature
    if type == 'skew':
        structure = tX[:, -1:]
        tX = tX[:, :-1]
    else: # point model
        structure = tX[:, -2:]
        tX = tX[:, :-2]

    if type == 'skew':
        n_components = components*(tX.shape[1]//TSTEPS)
    else: # point model
        n_components = components

    # Fit and transform the data
    pca = PCA(n_components=n_components)
    try:
        tX_transform = pca.fit_transform(tX)
    except:
        print("Error in PCA trying again")

    tX_transform = pca.fit_transform(tX)

    # Add the structure back So we end up with our encoded features + structure at the end of each sample
    tX_transform = np.concatenate([tX_transform, structure], axis=1)

    return tX_transform
    
def pca_fit(tX, ty, components=3 , TSTEPS=21, type='skew'):
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


def har_transform(tX, TSTEPS=21, type='skew'):
    """
    Transform the given input data to the HAR method of feature extraction
    Expects 2D input shape

    Input shape: (samples, TSTEP, moneyness)
    TSTEP must be of size 21 as a month of IV data is required for the HAR method

    Output shape: (samples, tx.shape[-1]*3)
    """
    if TSTEPS != 21:
        raise ValueError('TSTEP must be 21 for HAR method of feature extraction')
    # remove the moneyness and term structure features


    # Get average skew for 21 days
    if type == 'skew':
        # Reshape the data
        skew = tX[:, :-1]
        structure = tX[:, -1:]
        tX = skew.reshape(skew.shape[0], TSTEPS, skew.shape[1]//TSTEPS)
        skew1 = np.mean(tX[:, :, :], axis=1)
        skew2 = np.mean(tX[:, -5:, :], axis=1)
        skew3 = np.mean(tX[:, -1:, :], axis=1)
        tX = np.concatenate([skew1, skew2, skew3, structure], axis=1)
    else: # Point model
        # remove the moneyness and term structure features
        moneyness = tX[:, -2:-1]
        term_structure = tX[:, -1:]
        skew1 = np.mean(tX[:, :-2], axis=1).reshape(-1, 1)
        skew2 = np.mean(tX[:, -7:-2], axis=1).reshape(-1, 1)
        skew3 = np.mean(tX[:, -3:-2], axis=1).reshape(-1, 1)
        tX = np.concatenate([skew1, skew2, skew3, moneyness, term_structure], axis=1)


    return tX 


def har_features(tX, tY, TSTEPS=21, type='skew'):
    """
    Extracts har features of implied volatility which includes an averaged skew for the 1 month, 1 week and 1 day lagging features
    """
    tX = har_transform(tX, TSTEPS=TSTEPS, type=type)

    # XXX: New shape is samples, 3 features concatenated
    
    # Fit regression model
    model = Ridge()
    model.fit(tX, tY)
    return model


def har_predict(valX, model, TSTEPS=21):
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
        tskew = tskew.reshape(tskew.shape[0], tskew.shape[1]*tskew.shape[2])

        # XXX: Add m to the sample set
        ms = np.array([m]*tskew.shape[0]).reshape(tskew.shape[0], 1)
        tskew = np.append(tskew, ms, axis=1)

        vtskew= vX[:, :, j]
        vtskew = vtskew.reshape(vtskew.shape[0], vtskew.shape[1]*vtskew.shape[2])
        vms = np.array([m]*vtskew.shape[0]).reshape(vtskew.shape[0], 1)
        vtskew = np.append(vtskew, vms, axis=1)

        
        # Fit the model
        if model_name == 'har':
            if TSTEPS != 21:
                continue
            model = har_features(tskew, tYY, TSTEPS=TSTEPS)
            ypred = har_predict(vtskew, model, TSTEPS=TSTEPS)
        elif model_name == 'autoencoder':
            encoding_dim = TSTEPS//2 
            encoder, model = autoencoder_fit(tskew, tYY, encoding_dim=encoding_dim, TSTEPS=TSTEPS)
            # Save encoder model
            if dd != './gfigs':
                encoder.save('./tskew_feature_models/%s_ts_%s_%s_%s_encoder.keras' % (model_name, TSTEPS, m, otype))
            else:
                encoder.save('./tskew_feature_models/%s_ts_%s_%s_%s_encoder_gfigs.keras' % (model_name, TSTEPS, m, otype))
            # # transform and validate 
            # structure = vtskew[:, -1:]
            # vtskew = vtskew[:, :-1]
            # valX_transform = encoder.predict(vtskew) 
            # valX_transform = np.concatenate([valX_transform, structure], axis=1)
            # ypred = model.predict(valX_transform)
        elif model_name == 'vae':
            encoding_dim = TSTEPS//2
            vae_encoder, model = autoencoder_fit(tskew, tYY, encoding_dim=encoding_dim, TSTEPS=TSTEPS, vae=True)
            # Save encoder model
            if dd != './gfigs':
                vae_encoder.save('./tskew_feature_models/%s_ts_%s_%s_%s_encoder.keras' % (model_name, TSTEPS, m, otype))
            else:
                vae_encoder.save('./tskew_feature_models/%s_ts_%s_%s_%s_encoder_gfigs.keras' % (model_name, TSTEPS, m, otype))
            # # transform and validate 
            # # Strip the moneyness or term structure feature
            # structure = vtskew[:, -1:]
            # vtskew = vtskew[:, :-1]
            # vX_transform = vae_encoder.predict(vtskew)[2]
            # vX_transform = np.concatenate([vX_transform, structure], axis=1)
            # ypred = model.predict(vX_transform)
        else: # PCA
            n_components = TSTEPS//2 
            model = pca_fit(tskew, tYY, components=n_components, TSTEPS=TSTEPS)
            # ypred = pca_predict(vtskew, model, n_components=n_components, TSTEPS=TSTEPS)
            pass

        # Evaluate the model
        # rmse = root_mean_squared_error(vY[:, j], ypred, multioutput='raw_values')
        # mapes = mean_absolute_percentage_error(vY[:, j], ypred, multioutput='raw_values')
        # r2sc = r2_score(vY[:, j], ypred, multioutput='raw_values')

        # print('RMSE mean: ', np.mean(rmse), 'RMSE std: ', np.std(rmse))
        # print('MAPE mean: ', np.mean(mapes), 'MAPE std: ', np.std(mapes))
        # print('R2 mean: ', np.mean(r2sc), 'R2 std: ', np.std(r2sc))

        if dd != './gfigs':
            with open('./tskew_feature_models/%s_ts_%s_%s_%s.pkl' % (model_name, TSTEPS, m, otype), 'wb') as f:
                pickle.dump(model, f)
        else:
            with open('./tskew_feature_models/%s_ts_%s_%s_%s_gfigs.pkl' % (model_name, TSTEPS, m, otype), 'wb') as f:
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
        mskew = mskew.reshape(mskew.shape[0], mskew.shape[1]*mskew.shape[2])

        # XXX: Add t to the sample set
        ts = np.array([t]*mskew.shape[0]).reshape(mskew.shape[0], 1)
        mskew = np.append(mskew, ts, axis=1)

        # vmskew= vX[:, :, :, j]

        # vts = np.array([t]*vmskew.shape[0]).reshape(vmskew.shape[0], 1)
        # vmskew = np.append(vmskew, vts, axis=1)
        
        # Fit the model
        if model_name == 'har':
            if TSTEPS != 21:
                continue
            model = har_features(mskew, tYY, TSTEPS=TSTEPS)
            # ypred = har_predict(vmskew, model, TSTEPS=TSTEPS)
            # Fit the model
        elif model_name == 'autoencoder':
            encoding_dim = TSTEPS//2
            encoder, model = autoencoder_fit(mskew, tYY, encoding_dim=encoding_dim, TSTEPS=TSTEPS)
            # Save encoder model
            if dd != './gfigs':
                encoder.save('./mskew_feature_models/%s_ts_%s_%s_%s_encoder.keras' % (model_name, TSTEPS, t, otype))
            else:
                encoder.save('./mskew_feature_models/%s_ts_%s_%s_%s_encoder_gfigs.keras' % (model_name, TSTEPS, t, otype))
            # # transform and validate 
            # valX_transform = encoder.predict(vmskew) 
            # ypred = model.predict(valX_transform)
        elif model_name == 'vae':
            encoding_dim = TSTEPS//2
            def fit_and_save():
                vae_encoder, model = autoencoder_fit(mskew, tYY, encoding_dim=encoding_dim,TSTEPS=TSTEPS, vae=True)
                
                # Save encoder model
                if dd != './gfigs':
                    vae_encoder.save('./mskew_feature_models/%s_ts_%s_%s_%s_encoder.keras' % (model_name, TSTEPS, t, otype))
                else:
                    vae_encoder.save('./mskew_feature_models/%s_ts_%s_%s_%s_encoder_gfigs.keras' % (model_name, TSTEPS, t, otype))
                return model
            model = fit_and_save() 
            # # transform and validate 
            # valX_transform = vae_encoder.predict(vmskew)
            # ypred = model.predict(valX_transform)
        else: # PCA
            n_components = TSTEPS//2 
            model = pca_fit(mskew, tYY, components=n_components, TSTEPS=TSTEPS)
            # ypred = pca_predict(vmskew, model, n_components=n_components, TSTEPS=TSTEPS)
            pass

        # # Evaluate the model
        # rmse = root_mean_squared_error(vY[:, :, j], ypred, multioutput='raw_values')
        # mapes = mean_absolute_percentage_error(vY[:, :, j], ypred, multioutput='raw_values')
        # r2sc = r2_score(vY[:, :, j], ypred, multioutput='raw_values')

        # print('RMSE mean: ', np.mean(rmse), 'RMSE std: ', np.std(rmse))
        # print('MAPE mean: ', np.mean(mapes), 'MAPE std: ', np.std(mapes))
        # print('R2 mean: ', np.mean(r2sc), 'R2 std: ', np.std(r2sc))

        if dd != './gfigs':
            with open('./mskew_feature_models/%s_ts_%s_%s_%s.pkl' % (model_name, TSTEPS, t, otype), 'wb') as f:
                pickle.dump(model, f)
        else:
            with open('./mskew_feature_models/%s_ts_%s_%s_%s_gfigs.pkl' % (model_name, TSTEPS, t, otype), 'wb') as f:
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
                if TSTEPS != 21:
                    continue
                reg = har_features(train_vec, tY[:, i, j], TSTEPS=TSTEPS, type='point')
            elif model_name == 'autoencoder':
                encoding_dim = TSTEPS//2
                encoder, reg = autoencoder_fit(train_vec, tY[:, i, j], encoding_dim=encoding_dim, TSTEPS=TSTEPS)
                # Save encoder model
                if dd != './gfigs':
                    encoder.save('./point_feature_models/%s_ts_%s_%s_%s_encoder.keras' % (model_name, TSTEPS, t, otype))
                else:
                    encoder.save('./point_feature_models/%s_ts_%s_%s_%s_encoder_gfigs.keras' % (model_name, TSTEPS, t, otype))

            elif model_name == 'vae':
                encoding_dim = TSTEPS//2
                vae, reg = autoencoder_fit(train_vec, tY[:, i, j], encoding_dim=encoding_dim, TSTEPS=TSTEPS, vae=True)
                # Save encoder model
                if dd != './gfigs':
                    vae.save('./point_feature_models/%s_ts_%s_%s_%s_encoder.keras' % (model_name, TSTEPS, t, otype))
                else:
                    vae.save('./point_feature_models/%s_ts_%s_%s_%s_encoder_gfigs.keras' % (model_name, TSTEPS, t, otype))
            else: # PCA
                n_components = TSTEPS//2 
                reg = pca_fit(train_vec, tY[:, i, j], components=n_components, TSTEPS=TSTEPS, type='point')
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
                with open('./point_feature_models/%s_ts_%s_%s_%s_%s.pkl' %
                          (model_name, TSTEPS, s, t, otype), 'wb') as f:
                    pickle.dump(reg, f)
            else:
                with open('./point_feature_models/%s_ts_%s_%s_%s_%s_gfigs.pkl' %
                          (model_name, TSTEPS, s, t, otype), 'wb') as f:
                    pickle.dump(reg, f)

def run_all_models():
    for k in ['pca', 'autoencoder']:
        for j in [5, 10, 20]:
            mskew_pred(model_name=k, TSTEPS=j)
            tskew_pred(model_name=k, TSTEPS=j)
    # Do a run for the HAR method
    tskew_pred(model_name='har', TSTEPS=21)
    mskew_pred(model_name='har', TSTEPS=21)


if __name__ == "__main__":
    for n in ['call', 'put']:
        for i in ['./figs', './gfigs']:
            for k in ['vae']:
                for j in [5, 10, 20]:
                    if j == 20 and n == 'call':
                        continue
                    if (j == 10 or j == 5) and n == 'call' and i == './figs':
                        continue
                    print('Running for mskew: ', n, i, k, j)
                    p = multiprocessing.Process(target=mskew_pred, args=(n, k, j, i))
                    p.start()
                    p.join()

        # for i in ['./figs', './gfigs']:
        #     for k in ['vae']:
        #         for j in [20]:
        #             print('Running for point: ', n, i, k, j)
        #             p = multiprocessing.Process(target=point_pred, args=(n, k, j, i))
        #             p.start()
        #             p.join()
        #             # point_pred(otype=n, model_name=k, TSTEPS=j, dd=i)
        
        for i in ['./figs', './gfigs']:
            for k in ['vae']:
                for j in [5, 10, 20]:
                    if j == 20:
                        continue
                    if (j == 10 or j == 5) and i == './figs' and n == 'call':
                        continue
                    print('Running for tskew: ', n, i, k, j)
                    p = multiprocessing.Process(target=tskew_pred, args=(n, k, j, i))
                    p.start()
                    p.join()
                    # tskew_pred(otype=n, model_name=k, TSTEPS=j, dd=i)
  
        # For HAR 
        # for i in ['./figs', './gfigs']:
        #     print('Running for: ', n, i, 'har')
        #     tskew_pred(otype=n, model_name='har', TSTEPS=21, dd=i)
        #     mskew_pred(otype=n, model_name='har', TSTEPS=21, dd=i)
        #     point_pred(otype=n, model_name='har', TSTEPS=21, dd=i)
